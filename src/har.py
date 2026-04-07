#!/usr/bin/env python3
"""
<keywords>
python, hdf5, h5py, write, binary, files, recursion, compression, hidden directories, extraction, restore, append, list
</keywords>
<description>
This script performs one of four operations on an HDF5 archive:
  - Create archive (-c): Archive one or more directories/files into a new HDF5 file.
  - Append (-r): Append one or more directories/files to an existing HDF5 file.
      In append mode, sources are processed recursively. Existing entries are skipped.
  - Extract (-x): Extract files from an HDF5 archive.
      * With no extra positional argument, extracts the entire archive.
      * With a positional file key, extracts only that dataset.
      * Optionally specify extraction directory with -C.
  - List (-t): List the contents (dataset keys) of an HDF5 file.

Usage Examples:

  # Create archive from multiple directories or files:
  python h5.py -c -f mydir.h5 mydir anotherdir file.txt

  # Append multiple directories or files to an existing archive:
  python h5.py -r -f mydir.h5 mydir anotherdir

  # Extract entire archive to current directory:
  python h5.py -x -f mydir.h5

  # Extract entire archive to a specified directory:
  python h5.py -x -f mydir.h5 -C foo

  # Extract a specific file (dataset) from the archive:
  python h5.py -x -f mydir.h5 path_to_file_in_the_archive

  # List contents of the archive:
  python h5.py -t -f mydir.h5
</description>
<seealso>
</seealso>
"""

import os
import sys
import argparse
import base64
import concurrent.futures
import hashlib
import json
import h5py
import numpy as np
import time


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


def _read_file(file_path):
    """Read a file's bytes and permission mode. Unit of work for ThreadPoolExecutor."""
    with open(file_path, 'rb') as fobj:
        content = fobj.read()
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
        group = grp_mod.getgrgid(gid).gr_name
    except KeyError:
        group = str(gid)
    return content, mode, uid, gid, owner, group, mtime


def _write_extracted_file(args):
    """Write a single extracted file to disk. Unit of work for ThreadPoolExecutor."""
    name, file_bytes, mode, extract_dir = args
    dest_file_path = os.path.join(extract_dir, name)
    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
    with open(dest_file_path, 'wb') as fout:
        fout.write(file_bytes)
    if mode is not None:
        os.chmod(dest_file_path, mode)


def _chunk_list(lst, n):
    """Split lst into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


INTERNAL_ATTRS = {'mode', 'empty_dir'}
XATTR_PREFIX = 'xattr.'


def _is_internal_attr(name):
    """Check if an HDF5 attribute name is internal (managed by har)."""
    return name.startswith('har_') or name in INTERNAL_ATTRS


def _attr_to_json(value):
    """Convert an HDF5 attribute value to a JSON-serializable Python object."""
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return 'base64:' + base64.b64encode(value).decode('ascii')
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _collect_user_metadata(ds):
    """Collect user-defined metadata from an HDF5 dataset.

    Returns (hdf5_attrs, xattrs_json, xattrs_raw).
    """
    hdf5_attrs = {}
    xattrs_json = {}
    xattrs_raw = {}
    for name in ds.attrs:
        if _is_internal_attr(name):
            continue
        if name.startswith(XATTR_PREFIX):
            xattr_name = name[len(XATTR_PREFIX):]
            raw = bytes(ds.attrs[name])
            xattrs_json[xattr_name] = 'base64:' + base64.b64encode(raw).decode('ascii')
            xattrs_raw[xattr_name] = raw
        else:
            hdf5_attrs[name] = _attr_to_json(ds.attrs[name])
    return hdf5_attrs, xattrs_json, xattrs_raw


def _read_xattrs(path):
    """Read filesystem extended attributes (user.* namespace only)."""
    result = {}
    try:
        names = os.listxattr(path)
    except OSError:
        return result
    for name in names:
        if name.startswith('user.'):
            try:
                result[name] = os.getxattr(path, name)
            except OSError:
                pass
    return result


def _restore_xattrs(path, xattrs):
    """Restore filesystem extended attributes from raw bytes dict."""
    for name, value in xattrs.items():
        try:
            os.setxattr(path, name, value)
        except OSError as e:
            print(f"Warning: failed to set xattr '{name}' on '{path}': {e}",
                  file=sys.stderr)


def _restore_xattrs_from_json(path, xattrs):
    """Restore xattrs from a JSON map (base64-encoded values)."""
    for name, value in xattrs.items():
        if isinstance(value, str) and value.startswith('base64:'):
            raw = base64.b64decode(value[7:])
        elif isinstance(value, str):
            raw = value.encode('utf-8')
        else:
            continue
        try:
            os.setxattr(path, name, raw)
        except OSError as e:
            print(f"Warning: failed to set xattr '{name}' on '{path}': {e}",
                  file=sys.stderr)


def _write_metadata_json(extract_dir, all_metadata):
    """Write metadata.json manifest to the extract directory."""
    files = {}
    for path, (hdf5_attrs, xattrs_json, _xattrs_raw) in all_metadata.items():
        if not hdf5_attrs and not xattrs_json:
            continue
        entry = {}
        if hdf5_attrs:
            entry['hdf5_attrs'] = hdf5_attrs
        if xattrs_json:
            entry['xattrs'] = xattrs_json
        if entry:
            files[path] = entry
    if not files:
        return
    root = {
        'har_metadata_version': 1,
        'files': dict(sorted(files.items())),
    }
    dest = os.path.join(extract_dir, 'metadata.json')
    with open(dest, 'w') as f:
        json.dump(root, f, indent=2, sort_keys=False)


def pack_or_append_to_h5(sources,
                         output_h5,
                         file_mode,
                         compression=None,
                         compression_opts=None,
                         shuffle=False,
                         parallel=1,
                         verbose=False,
                         checksum=None,
                         xattr_flag=False,
                         validate=False):
    """
    Archive or append the given sources (directories and/or files) into the HDF5 archive.

    :param sources: List of source directories and/or files.
    :param output_h5: The HDF5 archive filename.
    :param file_mode: 'w' to create a new archive (overwrite), or 'a' to append.
    :param compression: Compression strategy (e.g., 'gzip', 'lzf', etc.).
    :param compression_opts: Compression options (e.g., level for 'gzip').
    :param shuffle: Whether to use the shuffle filter.
    :param parallel: Number of parallel workers for file reads (default: 1).
    :param verbose: Print filenames as they are processed (default: False).
    :param checksum: Checksum algorithm (md5, sha256, blake3) or None.
    :param validate: Verify written data by reading back and comparing checksums.
    """
    output_h5 = os.path.expanduser(output_h5)
    t0 = time.time()

    # Phase 1: Collect file inventory and empty directories.
    file_entries = []  # list of (file_path, rel_path)
    empty_dirs = []    # list of rel_dir strings
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
            print(f"Warning: '{source}' is not a valid file or directory; skipping.", file=sys.stderr)

    progress = ProgressBar(len(file_entries) if verbose else 0)
    verbose_file = verbose and not progress.is_tty
    validation_errors = []
    validated_count = [0]

    def _write_to_h5(h5fobj, rel_path, content, mode, uid=0, gid=0,
                     owner='', group='', mtime=0.0,
                     create_groups=True, src_path=None):
        """Write a single file's data to the HDF5 archive."""
        if verbose_file:
            print("Storing:", rel_path)
        if file_mode == 'a' and rel_path in h5fobj:
            if verbose_file:
                print(f"Skipping {rel_path} (already exists)")
            return
        if create_groups:
            group_path = os.path.dirname(rel_path)
            if group_path:
                h5fobj.require_group(group_path)
        file_data = np.frombuffer(content, dtype=np.uint8)
        ds = h5fobj.create_dataset(
            rel_path,
            data=file_data,
            dtype=np.uint8,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle
        )
        ds.attrs['mode'] = mode
        ds.attrs['uid'] = np.uint32(uid)
        ds.attrs['gid'] = np.uint32(gid)
        ds.attrs['owner'] = owner
        ds.attrs['group'] = group
        ds.attrs['mtime'] = mtime
        if checksum:
            ds.attrs['har_checksum'] = compute_checksum(content, checksum)
            ds.attrs['har_checksum_algo'] = checksum
        if validate and checksum:
            readback = h5fobj[rel_path][()].tobytes()
            actual = compute_checksum(readback, checksum)
            if actual != ds.attrs['har_checksum']:
                validation_errors.append(rel_path)
            else:
                validated_count[0] += 1
        if xattr_flag and src_path:
            xattrs = _read_xattrs(src_path)
            for name, value in xattrs.items():
                ds.attrs[XATTR_PREFIX + name] = np.void(value)

    if parallel <= 1:
        # Sequential path: read and write one file at a time (no extra memory).
        with h5py.File(output_h5, file_mode) as h5fobj:
            for file_path, rel_path in file_entries:
                content, mode, uid, gid, owner, group, mtime = _read_file(file_path)
                _write_to_h5(h5fobj, rel_path, content, mode,
                             uid=uid, gid=gid, owner=owner, group=group, mtime=mtime,
                             src_path=file_path)
                progress.update(rel_path)
            for rel_dir in empty_dirs:
                grp = h5fobj.require_group(rel_dir)
                grp.attrs['empty_dir'] = True
                if verbose_file:
                    print(f"Storing empty dir: {rel_dir}")
        progress.finish()
    else:
        # Parallel path: read files in parallel, then write to HDF5.
        # Pre-collect all unique group paths to batch-create them.
        all_groups = set()
        for _, rel_path in file_entries:
            g = os.path.dirname(rel_path)
            if g:
                all_groups.add(g)

        # Phase 2: Parallel file reads (collect all results first).
        read_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(_read_file, fp) for fp, _ in file_entries]
            for (file_path, rel_path), future in zip(file_entries, futures):
                content, mode, uid, gid, owner, group, mtime = future.result()
                read_results.append((rel_path, content, mode, uid, gid, owner, group, mtime, file_path))

        # Phase 3: Serial HDF5 writes (no thread pool overhead, no require_group).
        with h5py.File(output_h5, file_mode) as h5fobj:
            for g in sorted(all_groups):
                h5fobj.require_group(g)
            for rel_path, content, mode, uid, gid, owner, group, mtime, src_path in read_results:
                _write_to_h5(h5fobj, rel_path, content, mode,
                             uid=uid, gid=gid, owner=owner, group=group, mtime=mtime,
                             create_groups=False, src_path=src_path)
                progress.update(rel_path)
            for rel_dir in empty_dirs:
                grp = h5fobj.require_group(rel_dir)
                grp.attrs['empty_dir'] = True
                if verbose_file:
                    print(f"Storing empty dir: {rel_dir}")
        progress.finish()

    t1 = time.time()
    print(f"\nOperation completed in {t1 - t0:.2f} seconds.")
    if validate and validation_errors:
        print(f"VALIDATION FAILED on {len(validation_errors)} file(s):", file=sys.stderr)
        for e in validation_errors[:10]:
            print(f"  {e}", file=sys.stderr)
        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more", file=sys.stderr)
        sys.exit(1)
    elif validate and validated_count[0] > 0:
        print(f"Validation passed. {validated_count[0]} files verified.")

def validate_roundtrip(h5_path, sources, bagit=False, verbose=False,
                       byte_for_byte=False, checksum='sha256'):
    """Extract archive to temp dir and compare against source files."""
    import tempfile
    import shutil

    # Build source_map: {rel_path: abs_source_path}
    source_map = {}
    for source in sources:
        source = os.path.expanduser(source)
        if os.path.isdir(source):
            source = os.path.normpath(source)
            base_dir = os.path.dirname(os.path.abspath(source))
            for root, dirs, files in os.walk(source):
                for fname in files:
                    file_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(file_path, base_dir)
                    source_map[rel_path] = file_path
        elif os.path.isfile(source):
            source_map[os.path.basename(source)] = source

    tmp_dir = tempfile.mkdtemp(dir='.', prefix='.har_roundtrip_')
    try:
        # Extract archive to temp dir
        if bagit:
            from har_bagit import extract_bagit
            extract_bagit(h5_path, tmp_dir)
        else:
            extract_h5_to_directory(h5_path, tmp_dir)

        # Compare
        errors = []
        verified = 0
        for rel_path, src_path in source_map.items():
            extracted = os.path.join(tmp_dir, rel_path)
            if not os.path.exists(extracted):
                errors.append(f"MISSING: {rel_path}")
                continue
            if byte_for_byte:
                with open(src_path, 'rb') as f1, open(extracted, 'rb') as f2:
                    if f1.read() != f2.read():
                        errors.append(f"MISMATCH: {rel_path}")
                    else:
                        verified += 1
            else:
                with open(src_path, 'rb') as f1, open(extracted, 'rb') as f2:
                    h1 = compute_checksum(f1.read(), checksum)
                    h2 = compute_checksum(f2.read(), checksum)
                    if h1 != h2:
                        errors.append(f"MISMATCH: {rel_path}")
                    else:
                        verified += 1

        if errors:
            print(f"ROUNDTRIP VALIDATION FAILED on {len(errors)} file(s):",
                  file=sys.stderr)
            for e in errors[:10]:
                print(f"  {e}", file=sys.stderr)
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more", file=sys.stderr)
            return False
        else:
            print(f"Roundtrip validation passed. {verified} files verified.")
            return True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def delete_source_files(sources, verbose=False):
    """Delete source files/directories after successful archival."""
    import shutil
    deleted = 0
    for source in sources:
        source = os.path.expanduser(source)
        try:
            if os.path.isdir(source):
                shutil.rmtree(source)
            elif os.path.isfile(source):
                os.remove(source)
            else:
                continue
            deleted += 1
            if verbose:
                print(f"Deleted: {source}")
        except OSError as e:
            print(f"Warning: could not delete '{source}': {e}",
                  file=sys.stderr)
    print(f"Deleted {deleted} source(s).")


def extract_h5_to_directory(h5_path, extract_dir, file_key=None, parallel=1,
                            verbose=False, validate=False, checksum=None,
                            metadata_json=False, xattr_flag=False):
    """
    Extract files from an HDF5 file.
      - If file_key is None, extract the entire archive.
      - Otherwise, extract only the dataset corresponding to file_key.
    Files are extracted into extract_dir.

    :param parallel: Number of parallel workers for extraction (default: 1).
    :param verbose: Print filenames as they are extracted (default: False).
    :param validate: Verify checksums on extraction (default: False).
    :param checksum: Ignored (algo read from archive attributes).
    """
    h5_path = os.path.expanduser(h5_path)
    extract_dir = os.path.expanduser(extract_dir)
    if not os.path.exists(h5_path):
        print(f"Error: archive '{h5_path}' not found.", file=sys.stderr)
        sys.exit(1)
    try:
        h5fobj = h5py.File(h5_path, 'r')
    except OSError as e:
        print(f"Error: cannot open '{h5_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if file_key:
        # Single file extraction — always sequential.
        with h5fobj:
            if file_key not in h5fobj:
                print(f"Error: '{file_key}' not found in archive.", file=sys.stderr)
                sys.exit(1)
            dest_file_path = os.path.join(extract_dir, file_key)
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            dataset = h5fobj[file_key]
            file_bytes = dataset[()].tobytes()
            with open(dest_file_path, 'wb') as fout:
                fout.write(file_bytes)
            if 'mode' in dataset.attrs:
                os.chmod(dest_file_path, dataset.attrs['mode'])
            if validate and 'har_checksum_algo' in dataset.attrs:
                algo = dataset.attrs['har_checksum_algo']
                stored = dataset.attrs['har_checksum']
                actual = compute_checksum(file_bytes, algo)
                if actual != stored:
                    print(f"CHECKSUM MISMATCH ({algo}): {file_key}", file=sys.stderr)
                    sys.exit(1)
                else:
                    print("Validation passed. 1 file verified.")
            hdf5_attrs, xattrs_json, xattrs_raw = _collect_user_metadata(dataset)
            if xattr_flag and xattrs_raw:
                _restore_xattrs(dest_file_path, xattrs_raw)
            if metadata_json and (hdf5_attrs or xattrs_json):
                _write_metadata_json(extract_dir, {file_key: (hdf5_attrs, xattrs_json, xattrs_raw)})
            if verbose:
                print(f"Extracted: {file_key}")
    else:
        # Count items for progress bar
        item_count = [0]
        with h5py.File(h5_path, 'r') as h5tmp:
            def counter(name, obj):
                if isinstance(obj, h5py.Dataset) or (isinstance(obj, h5py.Group) and obj.attrs.get('empty_dir')):
                    item_count[0] += 1
            h5tmp.visititems(counter)

        progress = ProgressBar(item_count[0] if verbose else 0)
        verbose_file = verbose and not progress.is_tty

        if parallel <= 1:
            # Sequential full extraction.
            checksum_errors = []
            validated_file_count = [0]
            all_metadata = {}
            with h5fobj:
                def extract_item(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        dest_file_path = os.path.join(extract_dir, name)
                        os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                        file_bytes = obj[()].tobytes()
                        with open(dest_file_path, 'wb') as fout:
                            fout.write(file_bytes)
                        if 'mode' in obj.attrs:
                            os.chmod(dest_file_path, obj.attrs['mode'])
                        if validate and 'har_checksum_algo' in obj.attrs:
                            algo = obj.attrs['har_checksum_algo']
                            stored = obj.attrs['har_checksum']
                            actual = compute_checksum(file_bytes, algo)
                            if actual != stored:
                                checksum_errors.append(name)
                            else:
                                validated_file_count[0] += 1
                        hdf5_attrs, xattrs_json, xattrs_raw = _collect_user_metadata(obj)
                        if xattr_flag and xattrs_raw:
                            _restore_xattrs(dest_file_path, xattrs_raw)
                        if hdf5_attrs or xattrs_json:
                            all_metadata[name] = (hdf5_attrs, xattrs_json, xattrs_raw)
                        progress.update(name)
                        if verbose_file:
                            print(f"Extracted: {name}")
                    elif isinstance(obj, h5py.Group) and obj.attrs.get('empty_dir'):
                        dest_dir_path = os.path.join(extract_dir, name)
                        os.makedirs(dest_dir_path, exist_ok=True)
                        progress.update(name)
                        if verbose_file:
                            print(f"Created empty dir: {name}")
                h5fobj.visititems(extract_item)
            progress.finish()
            if metadata_json and all_metadata:
                _write_metadata_json(extract_dir, all_metadata)
            if checksum_errors:
                print(f"CHECKSUM MISMATCH on {len(checksum_errors)} file(s):", file=sys.stderr)
                for e in checksum_errors[:10]:
                    print(f"  {e}", file=sys.stderr)
                if len(checksum_errors) > 10:
                    print(f"  ... and {len(checksum_errors) - 10} more", file=sys.stderr)
                sys.exit(1)
            elif validate and validated_file_count[0] > 0:
                print(f"Validation passed. {validated_file_count[0]} files verified.")
        else:
            # Parallel full extraction.
            # Phase 1: Bulk read all datasets from HDF5 sequentially into memory.
            read_items = []
            empty_dir_names = []
            all_metadata = {}
            with h5fobj:
                def collector(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        file_bytes = obj[()].tobytes()
                        mode = obj.attrs.get('mode', None)
                        read_items.append((name, file_bytes, mode))
                        hdf5_attrs, xattrs_json, xattrs_raw = _collect_user_metadata(obj)
                        if hdf5_attrs or xattrs_json:
                            all_metadata[name] = (hdf5_attrs, xattrs_json, xattrs_raw)
                        progress.update(name)
                    elif isinstance(obj, h5py.Group) and obj.attrs.get('empty_dir'):
                        empty_dir_names.append(name)
                h5fobj.visititems(collector)

            # Phase 2: Create empty directories.
            for name in empty_dir_names:
                dest_dir_path = os.path.join(extract_dir, name)
                os.makedirs(dest_dir_path, exist_ok=True)
                progress.update(name)
                if verbose_file:
                    print(f"Created empty dir: {name}")

            # Phase 3: Write files to filesystem in parallel.
            work_items = [(name, data, mode, extract_dir) for name, data, mode in read_items]
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                executor.map(_write_extracted_file, work_items)
            # Restore xattrs after files are written
            if xattr_flag:
                for name in all_metadata:
                    _, _, xattrs_raw = all_metadata[name]
                    if xattrs_raw:
                        dest = os.path.join(extract_dir, name)
                        _restore_xattrs(dest, xattrs_raw)
            progress.finish()
            if metadata_json and all_metadata:
                _write_metadata_json(extract_dir, all_metadata)
            if verbose_file:
                for name, _, _, _ in work_items:
                    print(f"Extracted: {name}")

    print("Extraction complete!")

def list_h5_contents(h5_path):
    """List all dataset keys in the given HDF5 file."""
    h5_path = os.path.expanduser(h5_path)
    if not os.path.exists(h5_path):
        print(f"Error: archive '{h5_path}' not found.", file=sys.stderr)
        sys.exit(1)
    try:
        h5fobj = h5py.File(h5_path, 'r')
    except OSError as e:
        print(f"Error: cannot open '{h5_path}': {e}", file=sys.stderr)
        sys.exit(1)
    with h5fobj:
        keys = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)
        h5fobj.visititems(visitor)
        print("Contents of", h5_path)
        for key in sorted(keys):
            print(key)

def main():
    parser = argparse.ArgumentParser(
        description="Archive, append, extract, or list an HDF5 archive."
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    # Mutually exclusive operation flags.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", action="store_true", help="Create archive from one or more directories/files.")
    group.add_argument("-r", action="store_true", help="Append one or more directories/files to an existing archive.")
    group.add_argument("-x", action="store_true", help="Extract from an archive.")
    group.add_argument("-t", action="store_true", help="List contents of an archive.")
    group.add_argument("-b", "--browse", action="store_true", help="Browse archive interactively (TUI).")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbosely list files processed.")

    # Required HDF5 file.
    parser.add_argument("-f", "--file", required=True,
                        help="HDF5 archive file (for output when archiving/appending; for input when extracting/listing).")

    # Optional extraction directory (used with -x).
    parser.add_argument("-C", "--directory", default=".",
                        help="Target extraction directory (default: current directory).")

    # Compression options.
    parser.add_argument("-z", "--gzip", action="store_true",
                        help="Use gzip compression (off by default; no compression unless -z, --lzf, or --szip is given).")
    parser.add_argument("--lzf", action="store_true",
                        help="Use hdf5 lzf compression (fast, moderate ratio; off by default).")
    parser.add_argument("--szip", action="store_true",
                        help="Use hdf5 szip compression (off by default, untested).")
    parser.add_argument("--zopt", type=str, default="9",
                        help="Compression level for gzip 1-9 (default: 9). Ignored for lzf.")
    parser.add_argument("--lzma", action="store_true",
                        help="Use LZMA compression (application-level, no HDF5 filter plugin needed).")
    parser.add_argument("--shuffle", action="store_true",
                        help="Use HDF5 shuffle filter before compression (off by default).")

    # Checksum.
    parser.add_argument("--checksum", nargs='?', const='sha256',
                        choices=["md5", "sha256", "blake3"],
                        help="Checksum algorithm for integrity verification (default: sha256 if flag given without value).")

    # Metadata extraction.
    parser.add_argument("--metadata-json", action="store_true",
                        help="Write metadata.json manifest of user HDF5 attributes on extraction.")
    parser.add_argument("--xattr", action="store_true",
                        help="Capture/restore filesystem extended attributes (user.* namespace).")

    # Parallelism.
    parser.add_argument("-p", "--parallel", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential).")

    # BagIt mode.
    parser.add_argument("--bagit", action="store_true",
                        help="Use BagIt batched storage mode (faster for many files, adds SHA-256 checksums).")
    parser.add_argument("--batch-size", type=str, default="64M",
                        help="Target batch size for --bagit mode (default: 64M). Examples: 64M, 1G.")
    parser.add_argument("--validate", action="store_true",
                        help="Verify checksums on extraction and creation.")
    parser.add_argument("--validate-roundtrip", action="store_true",
                        help="After creation, extract and compare against source files.")
    parser.add_argument("--byte-for-byte", action="store_true",
                        help="Use byte-for-byte comparison for --validate-roundtrip (default: checksum).")
    parser.add_argument("--delete-source", action="store_true",
                        help="Delete source files after successful ingestion.")
    parser.add_argument("--bagit-raw", action="store_true",
                        help="Extract as a full BagIt bag with tag files (--bagit mode only).")
    parser.add_argument("--mpi", action="store_true",
                        help="Use MPI parallel HDF5 I/O (requires mpirun, mpi4py, parallel h5py).")

    # Positional argument:
    # For -c and -r: one or more source directories or files.
    # For -x: optionally, the file key to extract (if omitted, extract entire archive).
    parser.add_argument(
        "path", nargs="*", default=None,
        help=("For -c and -r: source directories/files; "
        "for -x: file key to extract (if omitted, extract entire archive)."))

    args = parser.parse_args()

    if args.parallel < 1:
        print("Error: --parallel must be >= 1.", file=sys.stderr)
        sys.exit(1)

    h5_file = args.file
    target_dir = args.directory  # For extraction, the target directory.
    sources = args.path         # For archive/append mode, this is a list of sources.

    # If --validate during creation, implicitly enable sha256 checksums
    if (args.c or args.r) and args.validate and not args.checksum:
        args.checksum = 'sha256'

    # Determine compression settings.
    compression = None
    compression_opts = None
    if args.gzip:
        compression = 'gzip'
        if args.zopt:
            compression_opts = int(args.zopt)
        else:
            compression_opts = 4
    elif args.lzf:
        compression = 'lzf'
        if args.zopt:
            print("Warning: Ignoring compression level for lzf.")
        else:
            pass
    elif args.szip:
        compression = 'szip'
        print('warning: szip compression is not tested yet')
        if args.zopt:
            compression_opts = tuple(map(int, args.zopt.split(',')))
        else:
            compression_opts = (4, 4)
    elif args.lzma:
        compression = 'lzma'
    else:
        pass

    # --- Browse mode ---
    if args.browse:
        from har_browse import browse_archive
        browse_archive(h5_file)
        return

    # --- BagIt mode dispatch ---
    if args.bagit or args.mpi:
        from har_bagit import pack_bagit, extract_bagit, list_bagit, parse_batch_size

        if args.mpi:
            from har_mpi import mpi_pack_bagit, mpi_extract_bagit, mpi_list_bagit

            if args.r:
                print("Error: Append (-r) is not supported with --bagit.",
                      file=sys.stderr)
                sys.exit(1)
            if compression:
                print("Warning: Compression is not supported with MPI parallel "
                      "HDF5 I/O. Compression flags ignored.", file=sys.stderr)

            if args.c:
                if not sources:
                    print("Error: At least one source is required.", file=sys.stderr)
                    sys.exit(1)
                mpi_pack_bagit(sources, h5_file, batch_size=bs, verbose=args.verbose)
            elif args.x:
                file_key = sources[0] if sources else None
                mpi_extract_bagit(
                    h5_file, target_dir, file_key=file_key,
                    validate=args.validate, bagit_raw=args.bagit_raw,
                    verbose=args.verbose)
            elif args.t:
                mpi_list_bagit(h5_file, bagit_raw=args.bagit_raw)
            return

        if args.r:
            print("Error: Append (-r) is not supported with --bagit "
                  "(manifests include checksums of all files).", file=sys.stderr)
            sys.exit(1)

        bs = parse_batch_size(args.batch_size)

        if args.c:
            if not sources:
                print("Error: At least one source is required for archive creation (-c).",
                      file=sys.stderr)
                sys.exit(1)
            pack_bagit(
                sources, h5_file,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=args.shuffle,
                batch_size=bs,
                parallel=args.parallel,
                verbose=args.verbose,
                checksum=args.checksum,
                xattr_flag=args.xattr,
                validate=args.validate)
            validation_ok = True
            if args.validate_roundtrip:
                validation_ok = validate_roundtrip(
                    h5_file, sources, bagit=True, verbose=args.verbose,
                    byte_for_byte=args.byte_for_byte,
                    checksum=args.checksum or 'sha256')
                if not validation_ok:
                    sys.exit(1)
            if args.delete_source:
                if validation_ok:
                    delete_source_files(sources, verbose=args.verbose)
                else:
                    print("Skipping source deletion: validation failed.",
                          file=sys.stderr)
        elif args.x:
            file_key = sources[0] if sources else None
            extract_bagit(
                h5_file, target_dir,
                file_key=file_key,
                validate=args.validate,
                bagit_raw=args.bagit_raw,
                parallel=args.parallel,
                verbose=args.verbose,
                metadata_json=args.metadata_json,
                xattr_flag=args.xattr)
        elif args.t:
            list_bagit(h5_file, bagit_raw=args.bagit_raw)
        else:
            parser.print_help()
            sys.exit(1)

    # --- Auto-detect bagit format on extract/list ---
    elif (args.x or args.t) and os.path.exists(os.path.expanduser(h5_file)):
        from har_bagit import is_bagit_archive, extract_bagit, list_bagit
        if is_bagit_archive(h5_file):
            print("Note: detected bagit-v1 archive, using BagIt extraction.",
                  file=sys.stderr)
            if args.x:
                file_key = sources[0] if sources else None
                extract_bagit(
                    h5_file, target_dir,
                    file_key=file_key,
                    validate=args.validate,
                    bagit_raw=getattr(args, 'bagit_raw', False),
                    parallel=args.parallel,
                    verbose=args.verbose,
                    metadata_json=getattr(args, 'metadata_json', False),
                    xattr_flag=getattr(args, 'xattr', False))
            else:
                list_bagit(h5_file, bagit_raw=getattr(args, 'bagit_raw', False))
        elif args.x:
            file_key = sources[0] if sources else None
            extract_h5_to_directory(h5_file, target_dir, file_key=file_key,
                                    parallel=args.parallel,
                                    verbose=args.verbose,
                                    validate=args.validate,
                                    metadata_json=getattr(args, 'metadata_json', False),
                                    xattr_flag=getattr(args, 'xattr', False))
        else:
            list_h5_contents(h5_file)

    # --- Legacy mode ---
    elif args.c:
        # Create archive.
        if not sources:
            print("Error: At least one source (directory or file) is required for archive creation (-c).", file=sys.stderr)
            sys.exit(1)
        pack_or_append_to_h5(
            sources,
            h5_file,
            'w',
            compression=compression,
            compression_opts=compression_opts,
            shuffle=args.shuffle,
            parallel=args.parallel,
            verbose=args.verbose,
            checksum=args.checksum,
            xattr_flag=args.xattr,
            validate=args.validate)
        validation_ok = True
        if args.validate_roundtrip:
            validation_ok = validate_roundtrip(
                h5_file, sources, bagit=False, verbose=args.verbose,
                byte_for_byte=args.byte_for_byte,
                checksum=args.checksum or 'sha256')
            if not validation_ok:
                sys.exit(1)
        if args.delete_source:
            if validation_ok:
                delete_source_files(sources, verbose=args.verbose)
            else:
                print("Skipping source deletion: validation failed.",
                      file=sys.stderr)
    elif args.r:
        # Append to archive.
        if not sources:
            print("Error: At least one source (directory or file) is required for appending (-r).", file=sys.stderr)
            sys.exit(1)
        pack_or_append_to_h5(
            sources,
            h5_file,
            'a',
            compression=compression,
            compression_opts=compression_opts,
            shuffle=args.shuffle,
            parallel=args.parallel,
            verbose=args.verbose,
            checksum=args.checksum,
            xattr_flag=args.xattr,
            validate=args.validate)
        validation_ok = True
        if args.validate_roundtrip:
            validation_ok = validate_roundtrip(
                h5_file, sources, bagit=False, verbose=args.verbose,
                byte_for_byte=args.byte_for_byte,
                checksum=args.checksum or 'sha256')
            if not validation_ok:
                sys.exit(1)
        if args.delete_source:
            if validation_ok:
                delete_source_files(sources, verbose=args.verbose)
            else:
                print("Skipping source deletion: validation failed.",
                      file=sys.stderr)
    elif args.x:
        file_key = sources[0] if sources else None
        extract_h5_to_directory(h5_file, target_dir, file_key=file_key,
                                parallel=args.parallel,
                                verbose=args.verbose,
                                validate=args.validate,
                                metadata_json=args.metadata_json,
                                xattr_flag=args.xattr)
    elif args.t:
        list_h5_contents(h5_file)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
