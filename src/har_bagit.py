"""
BagIt-mode for har: batched HDF5 storage with RFC 8493 manifests.

Instead of one HDF5 dataset per file (slow metadata), files are concatenated
into a small number of large batch datasets with a compound index for lookup.
BagIt manifests (SHA-256 checksums, bag-info) are embedded in the archive.
"""

import os
import sys
import base64
import hashlib
import json
import datetime
import time
import concurrent.futures

import h5py
import numpy as np



class ProgressBar:
    """Simple terminal progress bar with last-5-files display."""
    def __init__(self, total):
        self.total = total
        self.current = 0
        self.recent = []
        self.prev_lines = 0
        self.is_tty = sys.stderr.isatty() and total > 0

    def update(self, filename):
        self.current += 1
        self.recent.append(filename)
        if len(self.recent) > 5:
            self.recent.pop(0)
        if self.is_tty:
            self._render()

    def _render(self):
        if self.prev_lines > 0:
            sys.stderr.write(f'\x1b[{self.prev_lines}A')
        pct = self.current * 100 // max(self.total, 1)
        bar_width = 40
        filled = bar_width * self.current // max(self.total, 1)
        empty = bar_width - filled
        sys.stderr.write(f'\x1b[2K[{"█" * filled}{"░" * empty}] {pct}% ({self.current}/{self.total})\n')
        for f in self.recent:
            sys.stderr.write(f'\x1b[2K  {f}\n')
        self.prev_lines = 1 + len(self.recent)
        sys.stderr.flush()

    def finish(self):
        if not self.is_tty or self.prev_lines == 0:
            return
        sys.stderr.write(f'\x1b[{self.prev_lines}A')
        for _ in range(self.prev_lines):
            sys.stderr.write('\x1b[2K\n')
        sys.stderr.write(f'\x1b[{self.prev_lines}A')
        sys.stderr.flush()
        self.prev_lines = 0


def compute_checksum(data, algo):
    """Compute a hex digest using the specified algorithm."""
    if algo == 'md5':
        return hashlib.md5(data).hexdigest()
    elif algo == 'sha256':
        return hashlib.sha256(data).hexdigest()
    elif algo == 'blake3':
        import blake3
        return blake3.blake3(data).hexdigest()
    else:
        raise ValueError(f"Unsupported checksum algorithm: {algo}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HAR_FORMAT_ATTR = "har_format"
HAR_FORMAT_VALUE = "bagit-v1"
HAR_VERSION_ATTR = "har_version"
HAR_VERSION_VALUE = "1.0.0"

DEFAULT_BATCH_SIZE = 64 * 1024 * 1024  # 64 MiB

INDEX_DTYPE = np.dtype([
    ('path',     'S4096'),
    ('batch_id', np.uint32),
    ('offset',   np.uint64),
    ('length',   np.uint64),
    ('mode',     np.uint32),
    ('sha256',   'S64'),
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_batch_size(s):
    """Parse a human-readable size string like '64M' or '1G' into bytes."""
    s = s.strip().upper()
    multipliers = {'K': 1024, 'M': 1024**2, 'G': 1024**3}
    if s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


def _human_size(nbytes):
    """Return a human-readable size string."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _read_and_hash(file_path, algo='sha256'):
    """Read file bytes, compute checksum, return (content, mode, hash_hex)."""
    with open(file_path, 'rb') as f:
        content = f.read()
    mode = os.stat(file_path).st_mode
    hash_hex = compute_checksum(content, algo)
    return content, mode, hash_hex


def _checksum_bytes(data, algo='sha256'):
    """Hex digest of a bytes object using the specified algorithm."""
    return compute_checksum(data, algo)


def build_inventory(sources):
    """Walk sources, return (file_entries, empty_dirs).

    file_entries: list of (abs_file_path, rel_path)
    empty_dirs:   list of rel_path strings
    """
    file_entries = []
    empty_dirs = []
    for source in sources:
        source = os.path.expanduser(source)
        if os.path.isdir(source):
            source = os.path.normpath(source)
            base_dir = os.path.dirname(os.path.abspath(source))
            for root, dirs, files in os.walk(source):
                for fname in files:
                    file_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(file_path, base_dir)
                    file_entries.append((file_path, rel_path))
                if not files and not dirs:
                    rel_dir = os.path.relpath(root, base_dir)
                    empty_dirs.append(rel_dir)
        elif os.path.isfile(source):
            rel_path = os.path.basename(source)
            file_entries.append((source, rel_path))
        else:
            print(f"Warning: '{source}' is not a valid file or directory; skipping.",
                  file=sys.stderr)
    return file_entries, empty_dirs


def is_bagit_archive(h5_path):
    """Check if an HDF5 file is a bagit-v1 archive."""
    try:
        with h5py.File(h5_path, 'r') as h5f:
            return h5f.attrs.get(HAR_FORMAT_ATTR, None) == HAR_FORMAT_VALUE
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def _assign_batches(inventory, batch_size):
    """Assign files to batches.

    inventory: list of (rel_path, content_bytes, mode, sha256_hex)
               sorted by rel_path.

    Returns list of batch dicts:
      [{'id': int,
        'files': [(rel_path, content, mode, sha256, offset_in_batch), ...],
        'total_bytes': int}, ...]
    """
    batches = []
    current_files = []
    current_bytes = 0
    batch_id = 0

    for rel_path, content, mode, sha256 in inventory:
        fsize = len(content)
        # Start a new batch if adding this file would exceed target
        # (unless current batch is empty — always accept at least one file).
        if current_bytes + fsize > batch_size and current_files:
            batches.append({
                'id': batch_id,
                'files': current_files,
                'total_bytes': current_bytes,
            })
            batch_id += 1
            current_files = []
            current_bytes = 0

        current_files.append((rel_path, content, mode, sha256, current_bytes))
        current_bytes += fsize

    # Finalize last batch
    if current_files:
        batches.append({
            'id': batch_id,
            'files': current_files,
            'total_bytes': current_bytes,
        })

    return batches


# ---------------------------------------------------------------------------
# BagIt manifest generation
# ---------------------------------------------------------------------------

def _generate_bagit_txt():
    """Return the contents of bagit.txt as bytes."""
    return b"BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n"


def _generate_bag_info(total_bytes, file_count):
    """Return bag-info.txt contents as bytes."""
    lines = [
        f"Bagging-Date: {datetime.date.today().isoformat()}",
        f"Bag-Software-Agent: har/{HAR_VERSION_VALUE}",
        f"Payload-Oxum: {total_bytes}.{file_count}",
        f"Bag-Size: {_human_size(total_bytes)}",
    ]
    return ("\n".join(lines) + "\n").encode('utf-8')


def _generate_manifest(index_records):
    """Generate manifest-sha256.txt contents from index records.

    index_records: list of (bagit_path, sha256_hex) — paths already have data/ prefix.
    """
    lines = []
    for path, sha256 in sorted(index_records):
        lines.append(f"{sha256}  {path}")
    return ("\n".join(lines) + "\n").encode('utf-8')


def _generate_tagmanifest(tag_files, algo='sha256'):
    """Generate tagmanifest-{algo}.txt from dict of {filename: bytes_content}."""
    lines = []
    for name in sorted(tag_files):
        h = _checksum_bytes(tag_files[name], algo)
        lines.append(f"{h}  {name}")
    return ("\n".join(lines) + "\n").encode('utf-8')


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------

def pack_bagit(sources, output_h5, compression=None, compression_opts=None,
               shuffle=False, batch_size=DEFAULT_BATCH_SIZE, parallel=1,
               verbose=False, checksum=None, xattr_flag=False):
    """Create a BagIt-mode HDF5 archive with batched storage."""
    output_h5 = os.path.expanduser(output_h5)
    t0 = time.time()
    hash_algo = checksum or 'sha256'

    # Phase 1: Inventory
    file_entries, empty_dirs = build_inventory(sources)
    if verbose:
        print(f"Inventory: {len(file_entries)} files, {len(empty_dirs)} empty dirs")

    # Phase 2: Read + checksum (parallelizable)
    progress = ProgressBar(len(file_entries) if verbose else 0)
    verbose_file = verbose and not progress.is_tty

    user_metadata_map = {}

    if parallel <= 1:
        inventory = []
        for file_path, rel_path in sorted(file_entries, key=lambda x: x[1]):
            content, mode, hash_hex = _read_and_hash(file_path, hash_algo)
            bagit_path = "data/" + rel_path
            inventory.append((bagit_path, content, mode, hash_hex))
            if xattr_flag:
                from har import _read_xattrs
                xattrs = _read_xattrs(file_path)
                if xattrs:
                    xmap = {}
                    for name, value in xattrs.items():
                        xmap[name] = 'base64:' + base64.b64encode(value).decode('ascii')
                    user_metadata_map[rel_path] = {'xattrs': xmap}
            progress.update(rel_path)
            if verbose_file:
                print(f"  Read: {rel_path}")
    else:
        sorted_entries = sorted(file_entries, key=lambda x: x[1])
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(_read_and_hash, fp, hash_algo) for fp, _ in sorted_entries]
            inventory = []
            for (file_path, rel_path), future in zip(sorted_entries, futures):
                content, mode, hash_hex = future.result()
                bagit_path = "data/" + rel_path
                inventory.append((bagit_path, content, mode, hash_hex))
                if xattr_flag:
                    from har import _read_xattrs
                    xattrs = _read_xattrs(file_path)
                    if xattrs:
                        xmap = {}
                        for name, value in xattrs.items():
                            xmap[name] = 'base64:' + base64.b64encode(value).decode('ascii')
                        user_metadata_map[rel_path] = {'xattrs': xmap}
                progress.update(rel_path)
                if verbose_file:
                    print(f"  Read: {rel_path}")
    progress.finish()

    # Phase 3: Batch assignment
    batches = _assign_batches(inventory, batch_size)
    if verbose:
        print(f"Batches: {len(batches)} (target {_human_size(batch_size)})")

    # Phase 4: Build BagIt manifests
    total_bytes = sum(len(content) for _, content, _, _ in inventory)
    file_count = len(inventory)

    manifest_records = [(path, hash_hex) for path, _, _, hash_hex in inventory]
    manifest_name = f'manifest-{hash_algo}.txt'
    tagmanifest_name = f'tagmanifest-{hash_algo}.txt'
    bagit_txt = _generate_bagit_txt()
    bag_info_txt = _generate_bag_info(total_bytes, file_count)
    manifest_txt = _generate_manifest(manifest_records)
    tagmanifest_txt = _generate_tagmanifest({
        'bagit.txt': bagit_txt,
        'bag-info.txt': bag_info_txt,
        manifest_name: manifest_txt,
    }, hash_algo)

    # Phase 5: Write HDF5
    with h5py.File(output_h5, 'w') as h5f:
        # Root attributes
        h5f.attrs[HAR_FORMAT_ATTR] = HAR_FORMAT_VALUE
        h5f.attrs[HAR_VERSION_ATTR] = HAR_VERSION_VALUE
        h5f.attrs['har_checksum_algo'] = hash_algo

        # Index dataset
        index_arr = np.zeros(file_count, dtype=INDEX_DTYPE)
        i = 0
        for batch in batches:
            for rel_path, content, mode, sha256, offset in batch['files']:
                index_arr[i]['path'] = rel_path.encode('utf-8')
                index_arr[i]['batch_id'] = batch['id']
                index_arr[i]['offset'] = offset
                index_arr[i]['length'] = len(content)
                index_arr[i]['mode'] = mode
                index_arr[i]['sha256'] = sha256.encode('ascii')
                i += 1

        h5f.create_dataset('index', data=index_arr, compression='gzip',
                           compression_opts=9)

        # Batch datasets
        batches_grp = h5f.create_group('batches')
        for batch in batches:
            buf = bytearray(batch['total_bytes'])
            for _, content, _, _, offset in batch['files']:
                buf[offset:offset + len(content)] = content
            ds_kwargs = {}
            if compression:
                ds_kwargs['compression'] = compression
                ds_kwargs['compression_opts'] = compression_opts
                ds_kwargs['shuffle'] = shuffle
            batches_grp.create_dataset(
                str(batch['id']),
                data=np.frombuffer(bytes(buf), dtype=np.uint8),
                dtype=np.uint8,
                **ds_kwargs,
            )
            if verbose:
                print(f"  Wrote batch {batch['id']}: "
                      f"{len(batch['files'])} files, {_human_size(batch['total_bytes'])}")

        # BagIt tag files
        bagit_grp = h5f.create_group('bagit')
        for name, content in [('bagit.txt', bagit_txt),
                              ('bag-info.txt', bag_info_txt),
                              (manifest_name, manifest_txt),
                              (tagmanifest_name, tagmanifest_txt)]:
            bagit_grp.create_dataset(
                name, data=np.frombuffer(content, dtype=np.uint8), dtype=np.uint8)

        # Empty directories
        if empty_dirs:
            bagit_empty = ["data/" + d for d in empty_dirs]
            max_len = max(len(d) for d in bagit_empty)
            dt = f'S{max_len + 1}'
            h5f.create_dataset(
                'empty_dirs',
                data=np.array([d.encode('utf-8') for d in bagit_empty], dtype=dt))

        # User metadata (xattrs)
        if user_metadata_map:
            json_blob = json.dumps(user_metadata_map, indent=2, sort_keys=True).encode('utf-8')
            h5f.create_dataset('user_metadata',
                               data=np.frombuffer(json_blob, dtype=np.uint8),
                               dtype=np.uint8)

    t1 = time.time()
    print(f"\nOperation completed in {t1 - t0:.2f} seconds.")
    print(f"  {file_count} files in {len(batches)} batches, "
          f"{_human_size(total_bytes)} payload, "
          f"archive: {_human_size(os.path.getsize(output_h5))}")


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract_bagit(h5_path, extract_dir, file_key=None, validate=False,
                  bagit_raw=False, parallel=1, verbose=False,
                  metadata_json=False, xattr_flag=False):
    """Extract a BagIt-mode HDF5 archive."""
    h5_path = os.path.expanduser(h5_path)
    extract_dir = os.path.expanduser(extract_dir)

    if not os.path.exists(h5_path):
        print(f"Error: archive '{h5_path}' not found.", file=sys.stderr)
        sys.exit(1)
    try:
        h5f = h5py.File(h5_path, 'r')
    except OSError as e:
        print(f"Error: cannot open '{h5_path}': {e}", file=sys.stderr)
        sys.exit(1)

    with h5f:
        fmt = h5f.attrs.get(HAR_FORMAT_ATTR, None)
        if fmt != HAR_FORMAT_VALUE:
            print(f"Error: not a bagit-v1 archive (har_format={fmt!r}).",
                  file=sys.stderr)
            sys.exit(1)

        # Read checksum algorithm from archive (defaults to sha256)
        hash_algo = h5f.attrs.get('har_checksum_algo', 'sha256')

        # Read user metadata
        user_metadata = {}
        if (xattr_flag or metadata_json) and 'user_metadata' in h5f:
            blob = h5f['user_metadata'][()].tobytes()
            try:
                user_metadata = json.loads(blob)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # Read index
        index = h5f['index'][()]

        # Read empty dirs
        empty_dirs = []
        if 'empty_dirs' in h5f:
            empty_dirs = [d.decode('utf-8') for d in h5f['empty_dirs'][()]]

        # Single-file extraction
        if file_key:
            # Try with and without data/ prefix
            lookup = file_key
            if not lookup.startswith('data/'):
                lookup = 'data/' + lookup
            matches = [r for r in index if r['path'].decode('utf-8') == lookup]
            if not matches:
                print(f"Error: '{file_key}' not found in archive.", file=sys.stderr)
                sys.exit(1)
            rec = matches[0]
            batch_id = int(rec['batch_id'])
            offset = int(rec['offset'])
            length = int(rec['length'])
            mode = int(rec['mode'])
            sha256 = rec['sha256'].decode('ascii')

            batch_data = h5f[f'batches/{batch_id}'][()]
            file_bytes = batch_data[offset:offset + length].tobytes()

            if bagit_raw:
                out_path = rec['path'].decode('utf-8')
            else:
                out_path = rec['path'].decode('utf-8').removeprefix('data/')
            dest = os.path.join(extract_dir, out_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, 'wb') as fout:
                fout.write(file_bytes)
            if mode:
                os.chmod(dest, mode)
            if validate:
                actual = _checksum_bytes(file_bytes, hash_algo)
                if actual != sha256:
                    print(f"CHECKSUM MISMATCH: {out_path}", file=sys.stderr)
                    sys.exit(1)
            # Restore xattrs
            rel_key = rec['path'].decode('utf-8').removeprefix('data/')
            if xattr_flag and rel_key in user_metadata:
                xattrs = user_metadata[rel_key].get('xattrs', {})
                if xattrs:
                    from har import _restore_xattrs_from_json
                    _restore_xattrs_from_json(dest, xattrs)
            # Write metadata.json
            if metadata_json and rel_key in user_metadata:
                meta = user_metadata[rel_key]
                xattrs_json = meta.get('xattrs', {})
                if xattrs_json:
                    from har import _write_metadata_json
                    _write_metadata_json(extract_dir, {rel_key: ({}, xattrs_json, {})})
            if verbose:
                print(f"Extracted: {out_path}")
            print("Extraction complete!")
            return

        # Full extraction
        # Group index records by batch_id for efficient batch-at-a-time reading
        from collections import defaultdict
        by_batch = defaultdict(list)
        for rec in index:
            by_batch[int(rec['batch_id'])].append(rec)

        errors = []
        batch_ids = sorted(by_batch.keys())
        file_count = len(index)
        progress = ProgressBar(file_count if verbose else 0)
        verbose_file = verbose and not progress.is_tty

        def _extract_from_batch(batch_id, records, batch_data):
            """Extract all files from one batch buffer."""
            for rec in records:
                offset = int(rec['offset'])
                length = int(rec['length'])
                mode = int(rec['mode'])
                sha256 = rec['sha256'].decode('ascii')
                path = rec['path'].decode('utf-8')

                if bagit_raw:
                    out_path = path
                else:
                    out_path = path.removeprefix('data/')

                file_bytes = batch_data[offset:offset + length].tobytes()
                dest = os.path.join(extract_dir, out_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, 'wb') as fout:
                    fout.write(file_bytes)
                if mode:
                    os.chmod(dest, mode)
                # Restore xattrs
                rel_key = path.removeprefix('data/')
                if xattr_flag and rel_key in user_metadata:
                    xattrs = user_metadata[rel_key].get('xattrs', {})
                    if xattrs:
                        from har import _restore_xattrs_from_json
                        _restore_xattrs_from_json(dest, xattrs)
                if validate:
                    actual = _checksum_bytes(file_bytes, hash_algo)
                    if actual != sha256:
                        errors.append(out_path)
                progress.update(out_path)
                if verbose_file:
                    print(f"Extracted: {out_path}")

        if parallel <= 1:
            for bid in batch_ids:
                batch_data = h5f[f'batches/{bid}'][()]
                _extract_from_batch(bid, by_batch[bid], batch_data)
        else:
            # Read all batches sequentially from HDF5, then write files in parallel
            for bid in batch_ids:
                batch_data = h5f[f'batches/{bid}'][()]
                records = by_batch[bid]
                work = []
                for rec in records:
                    offset = int(rec['offset'])
                    length = int(rec['length'])
                    mode = int(rec['mode'])
                    path = rec['path'].decode('utf-8')
                    out_path = path if bagit_raw else path.removeprefix('data/')
                    file_bytes = batch_data[offset:offset + length].tobytes()
                    work.append((out_path, file_bytes, mode, extract_dir))
                with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
                    ex.map(_write_extracted_file_bagit, work)
                # Restore xattrs after parallel write
                if xattr_flag:
                    from har import _restore_xattrs_from_json
                    for rec in records:
                        path = rec['path'].decode('utf-8')
                        rel_key = path.removeprefix('data/')
                        if rel_key in user_metadata:
                            xattrs = user_metadata[rel_key].get('xattrs', {})
                            if xattrs:
                                out_path = path if bagit_raw else rel_key
                                dest = os.path.join(extract_dir, out_path)
                                _restore_xattrs_from_json(dest, xattrs)
                for out_path, _, _, _ in work:
                    progress.update(out_path)
                if validate:
                    for rec in records:
                        offset = int(rec['offset'])
                        length = int(rec['length'])
                        sha256 = rec['sha256'].decode('ascii')
                        path = rec['path'].decode('utf-8')
                        out_path = path if bagit_raw else path.removeprefix('data/')
                        file_bytes = batch_data[offset:offset + length].tobytes()
                        actual = _checksum_bytes(file_bytes, hash_algo)
                        if actual != sha256:
                            errors.append(out_path)
        progress.finish()

        # Empty directories
        for d in empty_dirs:
            out_path = d if bagit_raw else d.removeprefix('data/')
            dest = os.path.join(extract_dir, out_path)
            os.makedirs(dest, exist_ok=True)
            if verbose_file:
                print(f"Created empty dir: {out_path}")

        # BagIt tag files (only with --bagit-raw)
        if bagit_raw and 'bagit' in h5f:
            for name in h5f['bagit']:
                content = h5f[f'bagit/{name}'][()].tobytes()
                dest = os.path.join(extract_dir, name)
                with open(dest, 'wb') as fout:
                    fout.write(content)
                if verbose:
                    print(f"Wrote tag file: {name}")

    # Write metadata.json
    if metadata_json and user_metadata:
        from har import _write_metadata_json
        all_meta = {}
        for path, meta in user_metadata.items():
            xattrs_json = meta.get('xattrs', {})
            if xattrs_json:
                all_meta[path] = ({}, xattrs_json, {})
        if all_meta:
            _write_metadata_json(extract_dir, all_meta)

    if errors:
        print(f"CHECKSUM MISMATCH on {len(errors)} file(s):", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more", file=sys.stderr)
        sys.exit(1)

    print("Extraction complete!")


def _write_extracted_file_bagit(args):
    """Write a single file to disk (for parallel extraction)."""
    out_path, file_bytes, mode, extract_dir = args
    dest = os.path.join(extract_dir, out_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as fout:
        fout.write(file_bytes)
    if mode:
        os.chmod(dest, mode)


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

def list_bagit(h5_path, bagit_raw=False):
    """List contents of a BagIt-mode HDF5 archive."""
    h5_path = os.path.expanduser(h5_path)
    if not os.path.exists(h5_path):
        print(f"Error: archive '{h5_path}' not found.", file=sys.stderr)
        sys.exit(1)
    try:
        h5f = h5py.File(h5_path, 'r')
    except OSError as e:
        print(f"Error: cannot open '{h5_path}': {e}", file=sys.stderr)
        sys.exit(1)

    with h5f:
        fmt = h5f.attrs.get(HAR_FORMAT_ATTR, None)
        if fmt != HAR_FORMAT_VALUE:
            print(f"Error: not a bagit-v1 archive.", file=sys.stderr)
            sys.exit(1)

        tag_files = []
        if bagit_raw and 'bagit' in h5f:
            tag_files = sorted(h5f['bagit'].keys())

        index = h5f['index'][()]
        if bagit_raw:
            paths = sorted('data/' + rec['path'].decode('utf-8') for rec in index)
        else:
            paths = sorted(rec['path'].decode('utf-8') for rec in index)

        total = len(paths) + len(tag_files)
        print(f"Contents of {h5_path} ({total} entries)")
        for t in tag_files:
            print(t)
        for p in paths:
            print(p)
