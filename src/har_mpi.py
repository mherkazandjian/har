"""
MPI parallel HDF5 I/O for har --bagit mode.

Uses h5py with driver='mpio' so multiple MPI ranks write concurrently
to a single HDF5 file. Requires: mpi4py, h5py built with parallel HDF5.

IMPORTANT: With parallel HDF5, ALL metadata operations (group creation,
dataset creation, attribute writes) must be called by ALL ranks collectively.
Only data writes can be rank-specific (independent I/O).
"""

import os
import sys
import time
import hashlib
import numpy as np

from har_bagit import (
    build_inventory, parse_batch_size, HAR_FORMAT_ATTR, HAR_FORMAT_VALUE,
    HAR_VERSION_ATTR, HAR_VERSION_VALUE, INDEX_DTYPE, DEFAULT_BATCH_SIZE,
    _human_size, _sha256_bytes, _generate_bagit_txt, _generate_bag_info,
    _generate_manifest, _generate_tagmanifest,
)


def _check_mpi():
    try:
        from mpi4py import MPI
    except ImportError:
        print("Error: mpi4py is not available.", file=sys.stderr)
        sys.exit(1)
    import h5py
    if not h5py.get_config().mpi:
        print("Error: h5py was not built with MPI support.", file=sys.stderr)
        sys.exit(1)
    return MPI, h5py


# ---------------------------------------------------------------------------
# MPI Pack
# ---------------------------------------------------------------------------

def mpi_pack_bagit(sources, output_h5, batch_size=DEFAULT_BATCH_SIZE, verbose=False):
    """Create a BagIt-mode HDF5 archive using MPI parallel I/O."""
    MPI, h5py = _check_mpi()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    output_h5 = os.path.expanduser(output_h5)
    t0 = time.time()

    # Phase 1: Rank 0 inventories, stats all files, broadcasts
    if rank == 0:
        file_entries, empty_dirs = build_inventory(sources)
        file_entries.sort(key=lambda x: x[1])
        file_info = []
        for file_path, rel_path in file_entries:
            fsize = os.path.getsize(file_path)
            fmode = os.stat(file_path).st_mode
            bagit_path = "data/" + rel_path
            file_info.append((file_path, bagit_path, fsize, fmode))
        empty_dirs_bagit = ["data/" + d for d in empty_dirs]
    else:
        file_info = None
        empty_dirs_bagit = None

    file_info = comm.bcast(file_info, root=0)
    empty_dirs_bagit = comm.bcast(empty_dirs_bagit, root=0)

    # Phase 2: Compute batch assignments (all ranks, deterministic)
    sorted_paths = [fi[1] for fi in file_info]
    file_sizes = {fi[1]: fi[2] for fi in file_info}
    file_modes = {fi[1]: fi[3] for fi in file_info}

    batches = []  # list of (batch_id, [(bagit_path, offset), ...], total_bytes)
    current_files = []
    current_bytes = 0
    batch_id = 0

    for bagit_path in sorted_paths:
        fsize = file_sizes[bagit_path]
        if current_bytes + fsize > batch_size and current_files:
            batches.append((batch_id, current_files, current_bytes))
            batch_id += 1
            current_files = []
            current_bytes = 0
        current_files.append((bagit_path, current_bytes))
        current_bytes += fsize

    if current_files:
        batches.append((batch_id, current_files, current_bytes))

    # Assign batches to ranks
    batch_owner = {bid: bid % size for bid, _, _ in batches}

    # Phase 3: Each rank reads files for its OWN batches
    my_data = {}  # bagit_path -> (content, sha256)
    # Build reverse lookup: file_path from bagit_path
    path_to_file = {fi[1]: fi[0] for fi in file_info}

    for bid, batch_files, _ in batches:
        if batch_owner[bid] == rank:
            for bagit_path, _ in batch_files:
                file_path = path_to_file[bagit_path]
                with open(file_path, 'rb') as f:
                    content = f.read()
                sha = hashlib.sha256(content).hexdigest()
                my_data[bagit_path] = (content, sha)

    # Phase 4: Allgather checksums (needed for manifests)
    local_shas = {p: sha for p, (_, sha) in my_data.items()}
    all_sha_list = comm.allgather(local_shas)
    all_shas = {}
    for d in all_sha_list:
        all_shas.update(d)

    total_bytes = sum(file_sizes[p] for p in sorted_paths)
    file_count = len(sorted_paths)

    if verbose and rank == 0:
        print(f"[rank 0] {file_count} files, {len(batches)} batches, "
              f"{_human_size(total_bytes)} payload, {size} ranks")

    # Phase 5: Build manifests (all ranks compute identically)
    manifest_records = [(p, all_shas[p]) for p in sorted_paths]
    bagit_txt = _generate_bagit_txt()
    bag_info_txt = _generate_bag_info(total_bytes, file_count)
    manifest_txt = _generate_manifest(manifest_records)
    tagmanifest_txt = _generate_tagmanifest({
        'bagit.txt': bagit_txt,
        'bag-info.txt': bag_info_txt,
        'manifest-sha256.txt': manifest_txt,
    })

    # Phase 6: Build index array (all ranks compute identically)
    index_arr = np.zeros(file_count, dtype=INDEX_DTYPE)
    idx = 0
    for bid, batch_files, _ in batches:
        for bagit_path, offset in batch_files:
            index_arr[idx]['path'] = bagit_path.encode('utf-8')
            index_arr[idx]['batch_id'] = bid
            index_arr[idx]['offset'] = offset
            index_arr[idx]['length'] = file_sizes[bagit_path]
            index_arr[idx]['mode'] = file_modes[bagit_path]
            index_arr[idx]['sha256'] = all_shas[bagit_path].encode('ascii')
            idx += 1

    # Phase 7: Collective HDF5 writes
    h5f = h5py.File(output_h5, 'w', driver='mpio', comm=comm)

    # Root attributes — collective
    h5f.attrs[HAR_FORMAT_ATTR] = HAR_FORMAT_VALUE
    h5f.attrs[HAR_VERSION_ATTR] = HAR_VERSION_VALUE

    # Index — collective create, rank 0 writes
    ds_index = h5f.create_dataset('index', shape=(file_count,), dtype=INDEX_DTYPE)
    if rank == 0:
        ds_index[()] = index_arr

    # Batch datasets — collective create, owner writes
    batches_grp = h5f.create_group('batches')
    for bid, batch_files, total in batches:
        ds = batches_grp.create_dataset(str(bid), shape=(total,), dtype=np.uint8)
        if batch_owner[bid] == rank:
            buf = bytearray(total)
            for bagit_path, offset in batch_files:
                content = my_data[bagit_path][0]
                buf[offset:offset + len(content)] = content
            ds[()] = np.frombuffer(bytes(buf), dtype=np.uint8)
            if verbose:
                print(f"[rank {rank}] Wrote batch {bid}: "
                      f"{len(batch_files)} files, {_human_size(total)}")

    # BagIt tag files — collective create, rank 0 writes
    bagit_grp = h5f.create_group('bagit')
    for name, content in [('bagit.txt', bagit_txt),
                          ('bag-info.txt', bag_info_txt),
                          ('manifest-sha256.txt', manifest_txt),
                          ('tagmanifest-sha256.txt', tagmanifest_txt)]:
        ds = bagit_grp.create_dataset(name, shape=(len(content),), dtype=np.uint8)
        if rank == 0:
            ds[()] = np.frombuffer(content, dtype=np.uint8)

    # Empty dirs — collective create, rank 0 writes
    if empty_dirs_bagit:
        max_len = max(len(d) for d in empty_dirs_bagit)
        dt = f'S{max_len + 1}'
        ds = h5f.create_dataset('empty_dirs', shape=(len(empty_dirs_bagit),), dtype=dt)
        if rank == 0:
            ds[()] = np.array([d.encode('utf-8') for d in empty_dirs_bagit], dtype=dt)

    h5f.close()

    t1 = time.time()
    if rank == 0:
        archive_size = os.path.getsize(output_h5)
        print(f"\nOperation completed in {t1 - t0:.2f} seconds.")
        print(f"  {file_count} files in {len(batches)} batches across {size} ranks, "
              f"{_human_size(total_bytes)} payload, archive: {_human_size(archive_size)}")


# ---------------------------------------------------------------------------
# MPI Extract
# ---------------------------------------------------------------------------

def mpi_extract_bagit(h5_path, extract_dir, file_key=None, validate=False,
                      bagit_raw=False, verbose=False):
    """Extract a BagIt-mode HDF5 archive using MPI parallel I/O."""
    MPI, h5py = _check_mpi()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    h5_path = os.path.expanduser(h5_path)
    extract_dir = os.path.expanduser(extract_dir)

    if not os.path.exists(h5_path):
        if rank == 0:
            print(f"Error: archive '{h5_path}' not found.", file=sys.stderr)
        sys.exit(1)

    h5f = h5py.File(h5_path, 'r', driver='mpio', comm=comm)

    index = h5f['index'][()]

    empty_dirs = []
    if 'empty_dirs' in h5f:
        empty_dirs = [d.decode('utf-8') for d in h5f['empty_dirs'][()]]

    if file_key:
        lookup = file_key if file_key.startswith('data/') else 'data/' + file_key
        matches = [r for r in index if r['path'].decode('utf-8') == lookup]
        if not matches:
            if rank == 0:
                print(f"Error: '{file_key}' not found in archive.", file=sys.stderr)
            h5f.close()
            sys.exit(1)

        rec = matches[0]
        bid = int(rec['batch_id'])
        batch_data = h5f[f'batches/{bid}'][()]
        h5f.close()

        if rank == 0:
            off = int(rec['offset'])
            length = int(rec['length'])
            mode = int(rec['mode'])
            sha = rec['sha256'].decode('ascii')
            file_bytes = batch_data[off:off + length].tobytes()
            out_path = rec['path'].decode('utf-8')
            if not bagit_raw:
                out_path = out_path.removeprefix('data/')
            dest = os.path.join(extract_dir, out_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, 'wb') as fout:
                fout.write(file_bytes)
            if mode:
                os.chmod(dest, mode)
            if validate and _sha256_bytes(file_bytes) != sha:
                print(f"CHECKSUM MISMATCH: {out_path}", file=sys.stderr)
                sys.exit(1)
            print("Extraction complete!")
        return

    # Full extraction
    from collections import defaultdict
    by_batch = defaultdict(list)
    for i, rec in enumerate(index):
        by_batch[int(rec['batch_id'])].append(i)

    errors = []
    for bid in sorted(by_batch.keys()):
        indices = by_batch[bid]
        batch_data = h5f[f'batches/{bid}'][()]
        my_file_indices = [indices[j] for j in range(rank, len(indices), size)]

        for i in my_file_indices:
            rec = index[i]
            off = int(rec['offset'])
            length = int(rec['length'])
            mode = int(rec['mode'])
            sha = rec['sha256'].decode('ascii')
            path = rec['path'].decode('utf-8')
            out_path = path if bagit_raw else path.removeprefix('data/')

            file_bytes = batch_data[off:off + length].tobytes()
            dest = os.path.join(extract_dir, out_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, 'wb') as fout:
                fout.write(file_bytes)
            if mode:
                os.chmod(dest, mode)
            if validate and _sha256_bytes(file_bytes) != sha:
                errors.append(out_path)

    comm.Barrier()

    if rank == 0:
        for d in empty_dirs:
            out = d if bagit_raw else d.removeprefix('data/')
            os.makedirs(os.path.join(extract_dir, out), exist_ok=True)
        if bagit_raw and 'bagit' in h5f:
            for name in h5f['bagit']:
                content = h5f[f'bagit/{name}'][()].tobytes()
                with open(os.path.join(extract_dir, name), 'wb') as fout:
                    fout.write(content)

    h5f.close()

    all_errors = comm.gather(errors, root=0)
    if rank == 0:
        flat_errors = [e for errs in all_errors for e in errs]
        if flat_errors:
            print(f"CHECKSUM MISMATCH on {len(flat_errors)} file(s):", file=sys.stderr)
            for e in flat_errors[:10]:
                print(f"  {e}", file=sys.stderr)
            sys.exit(1)
        print("Extraction complete!")


# ---------------------------------------------------------------------------
# MPI List
# ---------------------------------------------------------------------------

def mpi_list_bagit(h5_path):
    MPI, h5py = _check_mpi()
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        from har_bagit import list_bagit
        list_bagit(h5_path)
    comm.Barrier()
