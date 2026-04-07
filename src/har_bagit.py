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
    """Multi-phase terminal progress bar based on bytes processed."""
    def __init__(self, verbose):
        self.is_tty = sys.stderr.isatty() and verbose
        self.completed = []
        self.label = None
        self.total = 0
        self.current = 0
        self.recent = []
        self.prev_lines = 0

    def start_phase(self, label, total_bytes):
        if self.label is not None:
            self.completed.append((self.label, self.total))
        self.label = label
        self.total = total_bytes
        self.current = 0
        self.recent = []
        if self.is_tty:
            self._render()

    def update(self, filename, nbytes):
        self.current += nbytes
        self.recent.append(filename)
        if len(self.recent) > 5:
            self.recent.pop(0)
        if self.is_tty:
            self._render()

    def _render(self):
        if self.prev_lines > 0:
            sys.stderr.write(f'\x1b[{self.prev_lines}A')
        lines = 0
        bar_width = 40
        for label, total in self.completed:
            bar = '\u2588' * bar_width
            tot_h = _human_size(total)
            sys.stderr.write(f'\x1b[2K\u2713 {label:12s} [{bar}] 100.0% ({tot_h} / {tot_h})\n')
            lines += 1
        if self.label and self.total > 0:
            pct = self.current * 100.0 / max(self.total, 1)
            filled = int(bar_width * self.current / max(self.total, 1))
            empty = bar_width - filled
            cur_h = _human_size(self.current)
            tot_h = _human_size(self.total)
            sys.stderr.write(f'\x1b[2K  {self.label:12s} [{"█" * filled}{"░" * empty}] {pct:.1f}% ({cur_h} / {tot_h})\n')
            lines += 1
            for f in self.recent:
                sys.stderr.write(f'\x1b[2K    \u2713 {f}\n')
                lines += 1
        self.prev_lines = lines
        sys.stderr.flush()

    def finish(self):
        if self.label is not None:
            self.completed.append((self.label, self.total))
            self.label = None
        if not self.is_tty or self.prev_lines == 0:
            return
        sys.stderr.write(f'\x1b[{self.prev_lines}A')
        bar_width = 40
        bar = '\u2588' * bar_width
        lines = 0
        for label, total in self.completed:
            tot_h = _human_size(total)
            sys.stderr.write(f'\x1b[2K\u2713 {label:12s} [{bar}] 100.0% ({tot_h} / {tot_h})\n')
            lines += 1
        for _ in range(self.prev_lines - lines):
            sys.stderr.write('\x1b[2K\n')
        extra = self.prev_lines - lines
        if extra > 0:
            sys.stderr.write(f'\x1b[{extra}A')
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
    ('uid',      np.uint32),
    ('gid',      np.uint32),
    ('mtime',    np.float64),
    ('owner',    'S256'),
    ('group_name', 'S256'),
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
    """Read file bytes, compute checksum, return (content, mode, hash_hex, uid, gid, mtime, owner, group_name)."""
    with open(file_path, 'rb') as f:
        content = f.read()
    st = os.stat(file_path)
    mode = st.st_mode
    uid = st.st_uid
    gid = st.st_gid
    mtime = st.st_mtime
    import pwd, grp as grp_mod
    try:
        owner = pwd.getpwuid(uid).pw_name
    except KeyError:
        owner = str(uid)
    try:
        group_name = grp_mod.getgrgid(gid).gr_name
    except KeyError:
        group_name = str(gid)
    hash_hex = compute_checksum(content, algo)
    return content, mode, hash_hex, uid, gid, mtime, owner, group_name


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


def is_split_archive(h5_path):
    """Check if an HDF5 file is a split bagit archive."""
    try:
        with h5py.File(h5_path, 'r') as h5f:
            return h5f.attrs.get('har_split_count', 0) > 1
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def parse_split_arg(value):
    """Parse --split argument.

    '100M' or 'size=100M' → {'mode': 'size', 'size': 104857600}
    'n=4'                 → {'mode': 'count', 'count': 4}
    """
    value = value.strip()
    if value.startswith('n='):
        n = int(value[2:])
        if n < 1:
            print("Error: --split count must be >= 1.", file=sys.stderr)
            sys.exit(1)
        return {'mode': 'count', 'count': n}
    if value.startswith('size='):
        value = value[5:]
    sz = parse_batch_size(value)
    if sz <= 0:
        print("Error: --split size must be > 0.", file=sys.stderr)
        sys.exit(1)
    return {'mode': 'size', 'size': sz}


def split_filename(base, index, total):
    """Generate filename for split index. Split 0 = base name."""
    if index == 0:
        return base
    width = max(3, len(str(total - 1)))
    stem, ext = os.path.splitext(base)
    return f"{stem}.{str(index).zfill(width)}{ext}"


def distribute_files(sized_entries, n_splits):
    """Bin-pack file entries into n_splits balanced groups.

    sized_entries: list of (abs_path, rel_path, size)
    Returns list of n_splits lists, each containing (abs_path, rel_path, size).
    Uses first-fit-decreasing by file size.
    """
    import heapq
    if n_splits <= 1:
        return [sized_entries]
    # Sort by size descending
    sorted_items = sorted(sized_entries, key=lambda x: x[2], reverse=True)
    # Min-heap of (current_total_size, split_index)
    heap = [(0, i) for i in range(n_splits)]
    heapq.heapify(heap)
    assignments = [[] for _ in range(n_splits)]
    for entry in sorted_items:
        total, idx = heapq.heappop(heap)
        assignments[idx].append(entry)
        heapq.heappush(heap, (total + entry[2], idx))
    return assignments


def _assign_batches_streaming(file_entries, batch_size):
    """Assign files to batches based on file size, without reading content.

    file_entries: list of (abs_path, rel_path, size)

    Returns list of batch dicts:
      [{'id': int,
        'files': [(abs_path, rel_path, size, offset_in_batch), ...],
        'total_bytes': int}, ...]
    """
    batches = []
    current_files = []
    current_bytes = 0
    batch_id = 0

    for abs_path, rel_path, fsize in file_entries:
        if current_bytes + fsize > batch_size and current_files:
            batches.append({
                'id': batch_id,
                'files': current_files,
                'total_bytes': current_bytes,
            })
            batch_id += 1
            current_files = []
            current_bytes = 0

        current_files.append((abs_path, rel_path, fsize, current_bytes))
        current_bytes += fsize

    if current_files:
        batches.append({
            'id': batch_id,
            'files': current_files,
            'total_bytes': current_bytes,
        })

    return batches


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

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
               verbose=False, checksum=None, xattr_flag=False,
               validate=False):
    """Create a BagIt-mode HDF5 archive with batched storage (streaming)."""
    output_h5 = os.path.expanduser(output_h5)
    t0 = time.time()
    hash_algo = checksum or 'sha256'

    # 1. Inventory (stat only, no content read)
    file_entries, empty_dirs = build_inventory(sources)
    if verbose:
        print(f"Inventory: {len(file_entries)} files, {len(empty_dirs)} empty dirs")

    # 2. Stat + sort
    sized_entries = []
    for abs_path, rel_path in sorted(file_entries, key=lambda x: x[1]):
        fsize = os.path.getsize(abs_path)
        sized_entries.append((abs_path, rel_path, fsize))
    total_size = sum(s for _, _, s in sized_entries)
    file_count = len(sized_entries)

    # 3. Batch assign (by size, no content)
    batches = _assign_batches_streaming(sized_entries, batch_size)
    if verbose:
        print(f"Batches: {len(batches)} (target {_human_size(batch_size)})")

    progress = ProgressBar(verbose)
    progress.start_phase("Archiving", total_size)
    verbose_file = verbose and not progress.is_tty

    # 4. Stream: read+hash+write one batch at a time
    index_records = []
    manifest_records = []
    user_metadata_map = {}
    total_bytes = 0

    with h5py.File(output_h5, 'w') as h5f:
        h5f.attrs[HAR_FORMAT_ATTR] = HAR_FORMAT_VALUE
        h5f.attrs[HAR_VERSION_ATTR] = HAR_VERSION_VALUE
        h5f.attrs['har_checksum_algo'] = hash_algo

        batches_grp = h5f.create_group('batches')

        for batch in batches:
            batch_buffer = bytearray(batch['total_bytes'])
            batch_file_records = []

            if parallel <= 1:
                for abs_path, rel_path, fsize, offset in batch['files']:
                    bagit_path = "data/" + rel_path
                    content, mode, hash_hex, uid, gid, mtime, owner, group_name = \
                        _read_and_hash(abs_path, hash_algo)
                    batch_buffer[offset:offset + len(content)] = content
                    batch_file_records.append({
                        'bagit_path': bagit_path, 'batch_id': batch['id'],
                        'offset': offset, 'length': len(content),
                        'mode': mode, 'sha256': hash_hex,
                        'uid': uid, 'gid': gid, 'mtime': mtime,
                        'owner': owner, 'group_name': group_name,
                    })
                    manifest_records.append((bagit_path, hash_hex))
                    if xattr_flag:
                        from har import _read_xattrs
                        xattrs = _read_xattrs(abs_path)
                        if xattrs:
                            xmap = {}
                            for name, value in xattrs.items():
                                xmap[name] = 'base64:' + base64.b64encode(value).decode('ascii')
                            user_metadata_map[rel_path] = {'xattrs': xmap}
                    progress.update(rel_path, len(content))
                    if verbose_file:
                        print(f"  {rel_path}")
            else:
                entries_in_batch = batch['files']
                with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                    futures = [executor.submit(_read_and_hash, e[0], hash_algo)
                               for e in entries_in_batch]
                    for (abs_path, rel_path, fsize, offset), future in \
                            zip(entries_in_batch, futures):
                        content, mode, hash_hex, uid, gid, mtime, owner, group_name = \
                            future.result()
                        bagit_path = "data/" + rel_path
                        batch_buffer[offset:offset + len(content)] = content
                        batch_file_records.append({
                            'bagit_path': bagit_path, 'batch_id': batch['id'],
                            'offset': offset, 'length': len(content),
                            'mode': mode, 'sha256': hash_hex,
                            'uid': uid, 'gid': gid, 'mtime': mtime,
                            'owner': owner, 'group_name': group_name,
                        })
                        manifest_records.append((bagit_path, hash_hex))
                        if xattr_flag:
                            from har import _read_xattrs
                            xattrs = _read_xattrs(abs_path)
                            if xattrs:
                                xmap = {}
                                for name, value in xattrs.items():
                                    xmap[name] = 'base64:' + base64.b64encode(value).decode('ascii')
                                user_metadata_map[rel_path] = {'xattrs': xmap}
                        progress.update(rel_path, len(content))
                        if verbose_file:
                            print(f"  {rel_path}")

            index_records.extend(batch_file_records)
            total_bytes += batch['total_bytes']

            # Write this batch to HDF5 and free buffer
            ds_kwargs = {}
            if compression:
                ds_kwargs['compression'] = compression
                ds_kwargs['compression_opts'] = compression_opts
                ds_kwargs['shuffle'] = shuffle
            batches_grp.create_dataset(
                str(batch['id']),
                data=np.frombuffer(bytes(batch_buffer), dtype=np.uint8),
                dtype=np.uint8,
                **ds_kwargs,
            )
            del batch_buffer

        # 5. Write index dataset
        index_arr = np.zeros(file_count, dtype=INDEX_DTYPE)
        for i, rec in enumerate(index_records):
            index_arr[i]['path'] = rec['bagit_path'].encode('utf-8')
            index_arr[i]['batch_id'] = rec['batch_id']
            index_arr[i]['offset'] = rec['offset']
            index_arr[i]['length'] = rec['length']
            index_arr[i]['mode'] = rec['mode']
            index_arr[i]['sha256'] = rec['sha256'].encode('ascii')
            index_arr[i]['uid'] = rec['uid']
            index_arr[i]['gid'] = rec['gid']
            index_arr[i]['mtime'] = rec['mtime']
            index_arr[i]['owner'] = rec['owner'].encode('utf-8')
            index_arr[i]['group_name'] = rec['group_name'].encode('utf-8')
        h5f.create_dataset('index', data=index_arr, compression='gzip',
                           compression_opts=9)

        # BagIt tag files
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

    progress.finish()

    t1 = time.time()
    print(f"\nOperation completed in {t1 - t0:.2f} seconds.")
    print(f"  {file_count} files in {len(batches)} batches, "
          f"{_human_size(total_bytes)} payload, "
          f"archive: {_human_size(os.path.getsize(output_h5))}")

    if validate:
        errors = []
        with h5py.File(output_h5, 'r') as h5v:
            v_index = h5v['index'][()]
            for rec in v_index:
                batch_id = int(rec['batch_id'])
                offset = int(rec['offset'])
                length = int(rec['length'])
                stored_sha = rec['sha256'].decode('ascii')
                path = rec['path'].decode('utf-8')
                batch_data = h5v[f'batches/{batch_id}'][()]
                file_bytes = batch_data[offset:offset + length].tobytes()
                actual = _checksum_bytes(file_bytes, hash_algo)
                if actual != stored_sha:
                    errors.append(path)
        if errors:
            print(f"VALIDATION FAILED on {len(errors)} file(s):", file=sys.stderr)
            for e in errors[:10]:
                print(f"  {e}", file=sys.stderr)
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Validation passed. {file_count} files verified.")


# ---------------------------------------------------------------------------
# Split pack
# ---------------------------------------------------------------------------

def _pack_single_split(split_index, file_entries, output_path, split_count,
                       base_name, compression, compression_opts, shuffle,
                       batch_size, hash_algo, xattr_flag, validate, verbose,
                       empty_dirs):
    """Pack one split file. Self-contained, thread-safe, streaming.

    file_entries: list of (abs_path, rel_path, size) — NO content.
    Reads files from disk on demand, one batch at a time.
    Returns list of (bagit_path, split_index, sha256_hex) for global manifest.
    """
    # Assign batches by size (no content needed)
    batches = _assign_batches_streaming(file_entries, batch_size)

    # Streaming: process each batch, collect index records + manifest entries
    index_records = []   # for HDF5 index dataset
    manifest_entries = []  # for global split_manifest + BagIt manifest
    user_metadata_map = {}

    with h5py.File(output_path, 'w') as h5f:
        # Root attributes
        h5f.attrs[HAR_FORMAT_ATTR] = HAR_FORMAT_VALUE
        h5f.attrs[HAR_VERSION_ATTR] = HAR_VERSION_VALUE
        h5f.attrs['har_checksum_algo'] = hash_algo
        h5f.attrs['har_split_count'] = np.uint32(split_count)
        h5f.attrs['har_split_index'] = np.uint32(split_index)
        h5f.attrs['har_split_base'] = base_name

        batches_grp = h5f.create_group('batches')

        for batch in batches:
            # Read this batch's files from disk (streaming — only one batch in RAM)
            batch_buffer = bytearray(batch['total_bytes'])
            batch_file_records = []

            for abs_path, rel_path, fsize, offset in batch['files']:
                bagit_path = "data/" + rel_path
                content, mode, hash_hex, uid, gid, mtime, owner, group_name = \
                    _read_and_hash(abs_path, hash_algo)

                batch_buffer[offset:offset + len(content)] = content

                batch_file_records.append({
                    'bagit_path': bagit_path,
                    'batch_id': batch['id'],
                    'offset': offset,
                    'length': len(content),
                    'mode': mode,
                    'sha256': hash_hex,
                    'uid': uid, 'gid': gid, 'mtime': mtime,
                    'owner': owner, 'group_name': group_name,
                })
                manifest_entries.append((bagit_path, split_index, hash_hex))

                if xattr_flag:
                    from har import _read_xattrs
                    xattrs = _read_xattrs(abs_path)
                    if xattrs:
                        xmap = {}
                        for name, value in xattrs.items():
                            xmap[name] = 'base64:' + base64.b64encode(value).decode('ascii')
                        user_metadata_map[rel_path] = {'xattrs': xmap}

            index_records.extend(batch_file_records)

            # Write batch dataset
            ds_kwargs = {}
            if compression:
                ds_kwargs['compression'] = compression
                ds_kwargs['compression_opts'] = compression_opts
                ds_kwargs['shuffle'] = shuffle
            batches_grp.create_dataset(
                str(batch['id']),
                data=np.frombuffer(bytes(batch_buffer), dtype=np.uint8),
                dtype=np.uint8,
                **ds_kwargs,
            )
            # Free batch buffer
            del batch_buffer

            if verbose:
                print(f"  [Split {split_index}] Wrote batch {batch['id']}: "
                      f"{len(batch['files'])} files, {_human_size(batch['total_bytes'])}")

        # Write index dataset
        file_count = len(index_records)
        index_arr = np.zeros(file_count, dtype=INDEX_DTYPE)
        for i, rec in enumerate(index_records):
            index_arr[i]['path'] = rec['bagit_path'].encode('utf-8')
            index_arr[i]['batch_id'] = rec['batch_id']
            index_arr[i]['offset'] = rec['offset']
            index_arr[i]['length'] = rec['length']
            index_arr[i]['mode'] = rec['mode']
            index_arr[i]['sha256'] = rec['sha256'].encode('ascii')
            index_arr[i]['uid'] = rec['uid']
            index_arr[i]['gid'] = rec['gid']
            index_arr[i]['mtime'] = rec['mtime']
            index_arr[i]['owner'] = rec['owner'].encode('utf-8')
            index_arr[i]['group_name'] = rec['group_name'].encode('utf-8')

        if file_count > 0:
            h5f.create_dataset('index', data=index_arr, compression='gzip',
                               compression_opts=9)
        else:
            h5f.create_dataset('index', data=index_arr)

        # BagIt tag files (per-split)
        total_bytes = sum(rec['length'] for rec in index_records)
        per_split_manifest_records = [(rec['bagit_path'], rec['sha256'])
                                      for rec in index_records]
        manifest_name = f'manifest-{hash_algo}.txt'
        tagmanifest_name = f'tagmanifest-{hash_algo}.txt'
        bagit_txt = _generate_bagit_txt()
        bag_info_txt = _generate_bag_info(total_bytes, file_count)
        manifest_txt = _generate_manifest(per_split_manifest_records)
        tagmanifest_txt = _generate_tagmanifest({
            'bagit.txt': bagit_txt,
            'bag-info.txt': bag_info_txt,
            manifest_name: manifest_txt,
        }, hash_algo)

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

    # Optional validation
    if validate:
        errors = []
        with h5py.File(output_path, 'r') as h5v:
            v_index = h5v['index'][()]
            for rec in v_index:
                batch_id = int(rec['batch_id'])
                offset = int(rec['offset'])
                length = int(rec['length'])
                stored_sha = rec['sha256'].decode('ascii')
                path = rec['path'].decode('utf-8')
                batch_data = h5v[f'batches/{batch_id}'][()]
                file_bytes = batch_data[offset:offset + length].tobytes()
                actual = _checksum_bytes(file_bytes, hash_algo)
                if actual != stored_sha:
                    errors.append(path)
        if errors:
            print(f"VALIDATION FAILED on split {split_index}, {len(errors)} file(s):",
                  file=sys.stderr)
            for e in errors[:10]:
                print(f"  {e}", file=sys.stderr)
            return None  # Signal failure
        elif file_count > 0:
            print(f"  [Split {split_index}] Validation passed. {file_count} files verified.")

    return manifest_entries


def _write_split_manifest(h5_path, manifest_entries, split_count):
    """Write global split_manifest group into the 0th split file."""
    with h5py.File(h5_path, 'a') as h5f:
        grp = h5f.create_group('split_manifest')
        count = len(manifest_entries)
        grp.attrs['count'] = np.uint64(count)

        paths_blob = "\n".join(e[0] for e in manifest_entries).encode('utf-8')
        grp.create_dataset('paths',
                           data=np.frombuffer(paths_blob, dtype=np.uint8),
                           dtype=np.uint8)

        split_ids = np.array([e[1] for e in manifest_entries], dtype=np.uint32)
        grp.create_dataset('split_ids', data=split_ids)

        sha256s_blob = "\n".join(e[2] for e in manifest_entries).encode('utf-8')
        grp.create_dataset('sha256s',
                           data=np.frombuffer(sha256s_blob, dtype=np.uint8),
                           dtype=np.uint8)


def pack_bagit_split(sources, output_h5, compression=None, compression_opts=None,
                     shuffle=False, batch_size=DEFAULT_BATCH_SIZE, parallel=1,
                     verbose=False, checksum=None, xattr_flag=False,
                     validate=False, split_spec=None):
    """Create a split BagIt archive. Orchestrates the full workflow."""
    output_h5 = os.path.expanduser(output_h5)
    t0 = time.time()
    hash_algo = checksum or 'sha256'
    base_name = os.path.basename(output_h5)

    # Phase 1: Inventory (paths + sizes only, no content)
    file_entries, empty_dirs = build_inventory(sources)
    if not file_entries:
        print("Error: no files to archive.", file=sys.stderr)
        sys.exit(1)

    sized_entries = [(fp, rp, os.path.getsize(fp)) for fp, rp in file_entries]
    total_size = sum(s for _, _, s in sized_entries)

    if verbose:
        print(f"Inventory: {len(sized_entries)} files, {_human_size(total_size)}, "
              f"{len(empty_dirs)} empty dirs")

    # Phase 2: Resolve split count
    if split_spec['mode'] == 'count':
        split_count = split_spec['count']
    else:
        split_count = max(1, -(-total_size // split_spec['size']))  # ceil division

    split_count = max(1, min(split_count, len(sized_entries)))
    if verbose:
        print(f"Splitting into {split_count} files")

    # Phase 3: Distribute files across splits
    split_assignments = distribute_files(sized_entries, split_count)

    # Generate split filenames
    split_paths = [split_filename(output_h5, i, split_count)
                   for i in range(split_count)]

    if verbose:
        for i, (path, entries) in enumerate(zip(split_paths, split_assignments)):
            split_size = sum(s for _, _, s in entries)
            print(f"  Split {i}: {len(entries)} files, {_human_size(split_size)} -> {os.path.basename(path)}")

    # Phase 4: Pack splits (parallel)
    effective_parallel = min(parallel, split_count)

    def _pack_split(idx):
        return _pack_single_split(
            split_index=idx,
            file_entries=split_assignments[idx],
            output_path=split_paths[idx],
            split_count=split_count,
            base_name=base_name,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            batch_size=batch_size,
            hash_algo=hash_algo,
            xattr_flag=xattr_flag,
            validate=validate,
            verbose=verbose,
            empty_dirs=empty_dirs if idx == 0 else [],
        )

    all_manifest_entries = []
    failed = False

    if effective_parallel <= 1:
        for i in range(split_count):
            result = _pack_split(i)
            if result is None:
                failed = True
            else:
                all_manifest_entries.extend(result)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_parallel) as executor:
            futures = {executor.submit(_pack_split, i): i for i in range(split_count)}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is None:
                    failed = True
                else:
                    all_manifest_entries.extend(result)

    if failed:
        print("VALIDATION FAILED on one or more split files.", file=sys.stderr)
        sys.exit(1)

    # Phase 5: Write global manifest to split 0
    _write_split_manifest(split_paths[0], all_manifest_entries, split_count)

    t1 = time.time()
    print(f"\nOperation completed in {t1 - t0:.2f} seconds.")
    print(f"  {len(all_manifest_entries)} files across {split_count} splits, "
          f"{_human_size(total_size)} payload")
    for p in split_paths:
        print(f"  {os.path.basename(p)}: {_human_size(os.path.getsize(p))}")


# ---------------------------------------------------------------------------
# Split extract
# ---------------------------------------------------------------------------

def _extract_single_split(split_path, extract_dir, validate, bagit_raw,
                          verbose, metadata_json, xattr_flag):
    """Extract all files from one split. Thread-safe. Returns error list."""
    try:
        extract_bagit(split_path, extract_dir, file_key=None,
                      validate=validate, bagit_raw=bagit_raw,
                      parallel=1, verbose=verbose,
                      metadata_json=metadata_json, xattr_flag=xattr_flag)
        return []
    except SystemExit:
        return [split_path]


def extract_bagit_split(h5_path, extract_dir, file_key=None, validate=False,
                        bagit_raw=False, parallel=1, verbose=False,
                        metadata_json=False, xattr_flag=False):
    """Extract from a split archive."""
    h5_path = os.path.expanduser(h5_path)
    extract_dir = os.path.expanduser(extract_dir)

    with h5py.File(h5_path, 'r') as h5f:
        split_count = int(h5f.attrs.get('har_split_count', 1))
        base_name = h5f.attrs.get('har_split_base', os.path.basename(h5_path))
        if isinstance(base_name, bytes):
            base_name = base_name.decode('utf-8')

    base_dir = os.path.dirname(os.path.abspath(h5_path))

    # Resolve all split file paths
    split_paths = []
    for i in range(split_count):
        sp = os.path.join(base_dir, split_filename(base_name, i, split_count))
        if not os.path.exists(sp):
            print(f"Error: split file '{sp}' not found. Archive may be incomplete.",
                  file=sys.stderr)
            sys.exit(1)
        split_paths.append(sp)

    # Single-file extraction: look up in split_manifest
    if file_key is not None:
        with h5py.File(h5_path, 'r') as h5f:
            if 'split_manifest' not in h5f:
                print("Error: split_manifest not found in primary archive.",
                      file=sys.stderr)
                sys.exit(1)
            sm = h5f['split_manifest']
            paths = sm['paths'][()].tobytes().decode('utf-8').split('\n')
            split_ids = sm['split_ids'][()]

            # Normalize file_key for lookup
            lookup = file_key
            lookup_data = "data/" + file_key

            target_split = None
            for j, p in enumerate(paths):
                # Match against both with and without data/ prefix
                p_stripped = p[5:] if p.startswith('data/') else p
                if p_stripped == lookup or p == lookup_data:
                    target_split = int(split_ids[j])
                    break

            if target_split is None:
                print(f"Error: '{file_key}' not found in split archive.",
                      file=sys.stderr)
                sys.exit(1)

        # Extract from the target split
        extract_bagit(split_paths[target_split], extract_dir,
                      file_key=file_key, validate=validate,
                      bagit_raw=bagit_raw, parallel=1, verbose=verbose,
                      metadata_json=metadata_json, xattr_flag=xattr_flag)
        return

    # Full extraction: process each split in parallel
    effective_parallel = min(parallel, split_count)
    errors = []

    if effective_parallel <= 1:
        for sp in split_paths:
            errs = _extract_single_split(sp, extract_dir, validate, bagit_raw,
                                         verbose, metadata_json, xattr_flag)
            errors.extend(errs)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_parallel) as executor:
            futures = [executor.submit(_extract_single_split, sp, extract_dir,
                                       validate, bagit_raw, verbose,
                                       metadata_json, xattr_flag)
                       for sp in split_paths]
            for future in concurrent.futures.as_completed(futures):
                errors.extend(future.result())

    if errors:
        print(f"Extraction failed on {len(errors)} split file(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)


def list_bagit_split(h5_path, bagit_raw=False):
    """List contents of a split archive from the global manifest."""
    h5_path = os.path.expanduser(h5_path)
    with h5py.File(h5_path, 'r') as h5f:
        split_count = int(h5f.attrs.get('har_split_count', 1))

        if 'split_manifest' not in h5f:
            print("Error: split_manifest not found.", file=sys.stderr)
            sys.exit(1)

        sm = h5f['split_manifest']
        paths = sm['paths'][()].tobytes().decode('utf-8').split('\n')

        for p in sorted(paths):
            if not p:
                continue
            if bagit_raw:
                print(p)
            else:
                # Strip data/ prefix
                if p.startswith('data/'):
                    print(p[5:])
                else:
                    print(p)

    print(f"\n{len(paths)} entries across {split_count} splits")


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
                else:
                    print("Validation passed. 1 file verified.")
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
        total_bytes = sum(int(rec['length']) for rec in index)
        progress = ProgressBar(verbose)
        progress.start_phase("Extracting", total_bytes)
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
                progress.update(out_path, length)
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
                for out_path, file_bytes, _, _ in work:
                    progress.update(out_path, len(file_bytes))
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
    elif validate:
        print(f"Validation passed. {file_count} files verified.")

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
