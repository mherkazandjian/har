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
import concurrent.futures
import hashlib
import h5py
import numpy as np
import time


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
    mode = os.stat(file_path).st_mode
    return content, mode


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


def pack_or_append_to_h5(sources,
                         output_h5,
                         file_mode,
                         compression=None,
                         compression_opts=None,
                         shuffle=False,
                         parallel=1,
                         verbose=False,
                         checksum=None):
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

    def _write_to_h5(h5fobj, rel_path, content, mode, create_groups=True):
        """Write a single file's data to the HDF5 archive."""
        if verbose:
            print("Storing:", rel_path)
        if file_mode == 'a' and rel_path in h5fobj:
            if verbose:
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
        if checksum:
            ds.attrs['har_checksum'] = compute_checksum(content, checksum)
            ds.attrs['har_checksum_algo'] = checksum

    if parallel <= 1:
        # Sequential path: read and write one file at a time (no extra memory).
        with h5py.File(output_h5, file_mode) as h5fobj:
            for file_path, rel_path in file_entries:
                content, mode = _read_file(file_path)
                _write_to_h5(h5fobj, rel_path, content, mode)
            for rel_dir in empty_dirs:
                grp = h5fobj.require_group(rel_dir)
                grp.attrs['empty_dir'] = True
                if verbose:
                    print(f"Storing empty dir: {rel_dir}")
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
            for (_, rel_path), future in zip(file_entries, futures):
                content, mode = future.result()
                read_results.append((rel_path, content, mode))

        # Phase 3: Serial HDF5 writes (no thread pool overhead, no require_group).
        with h5py.File(output_h5, file_mode) as h5fobj:
            for g in sorted(all_groups):
                h5fobj.require_group(g)
            for rel_path, content, mode in read_results:
                _write_to_h5(h5fobj, rel_path, content, mode, create_groups=False)
            for rel_dir in empty_dirs:
                grp = h5fobj.require_group(rel_dir)
                grp.attrs['empty_dir'] = True
                if verbose:
                    print(f"Storing empty dir: {rel_dir}")

    t1 = time.time()
    print(f"\nOperation completed in {t1 - t0:.2f} seconds.")

def extract_h5_to_directory(h5_path, extract_dir, file_key=None, parallel=1,
                            verbose=False, validate=False, checksum=None):
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
            if verbose:
                print(f"Extracted: {file_key}")
    elif parallel <= 1:
        # Sequential full extraction.
        checksum_errors = []
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
                    if verbose:
                        print(f"Extracted: {name}")
                elif isinstance(obj, h5py.Group) and obj.attrs.get('empty_dir'):
                    dest_dir_path = os.path.join(extract_dir, name)
                    os.makedirs(dest_dir_path, exist_ok=True)
                    if verbose:
                        print(f"Created empty dir: {name}")
            h5fobj.visititems(extract_item)
        if checksum_errors:
            print(f"CHECKSUM MISMATCH on {len(checksum_errors)} file(s):", file=sys.stderr)
            for e in checksum_errors[:10]:
                print(f"  {e}", file=sys.stderr)
            if len(checksum_errors) > 10:
                print(f"  ... and {len(checksum_errors) - 10} more", file=sys.stderr)
            sys.exit(1)
    else:
        # Parallel full extraction.
        # Phase 1: Bulk read all datasets from HDF5 sequentially into memory.
        # (Single file sequential I/O is fast; filesystem writes are the bottleneck.)
        read_items = []  # list of (name, bytes, mode|None)
        empty_dir_names = []
        with h5fobj:
            def collector(name, obj):
                if isinstance(obj, h5py.Dataset):
                    file_bytes = obj[()].tobytes()
                    mode = obj.attrs.get('mode', None)
                    read_items.append((name, file_bytes, mode))
                elif isinstance(obj, h5py.Group) and obj.attrs.get('empty_dir'):
                    empty_dir_names.append(name)
            h5fobj.visititems(collector)

        # Phase 2: Create empty directories (sequential, fast).
        for name in empty_dir_names:
            dest_dir_path = os.path.join(extract_dir, name)
            os.makedirs(dest_dir_path, exist_ok=True)
            if verbose:
                print(f"Created empty dir: {name}")

        # Phase 3: Write files to filesystem in parallel.
        work_items = [(name, data, mode, extract_dir) for name, data, mode in read_items]
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            executor.map(_write_extracted_file, work_items)
        if verbose:
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
    # Mutually exclusive operation flags.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", action="store_true", help="Create archive from one or more directories/files.")
    group.add_argument("-r", action="store_true", help="Append one or more directories/files to an existing archive.")
    group.add_argument("-x", action="store_true", help="Extract from an archive.")
    group.add_argument("-t", action="store_true", help="List contents of an archive.")

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
    parser.add_argument("--checksum", choices=["md5", "sha256", "blake3"],
                        help="Checksum algorithm for integrity verification (md5, sha256, blake3).")

    # Parallelism.
    parser.add_argument("-p", "--parallel", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential).")

    # BagIt mode.
    parser.add_argument("--bagit", action="store_true",
                        help="Use BagIt batched storage mode (faster for many files, adds SHA-256 checksums).")
    parser.add_argument("--batch-size", type=str, default="64M",
                        help="Target batch size for --bagit mode (default: 64M). Examples: 64M, 1G.")
    parser.add_argument("--validate", action="store_true",
                        help="Verify SHA-256 checksums on extraction (--bagit mode only).")
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
                mpi_list_bagit(h5_file)
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
                checksum=args.checksum)
        elif args.x:
            file_key = sources[0] if sources else None
            extract_bagit(
                h5_file, target_dir,
                file_key=file_key,
                validate=args.validate,
                bagit_raw=args.bagit_raw,
                parallel=args.parallel,
                verbose=args.verbose)
        elif args.t:
            list_bagit(h5_file)
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
                    verbose=args.verbose)
            else:
                list_bagit(h5_file)
        elif args.x:
            file_key = sources[0] if sources else None
            extract_h5_to_directory(h5_file, target_dir, file_key=file_key,
                                    parallel=args.parallel,
                                    verbose=args.verbose,
                                    validate=args.validate)
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
            checksum=args.checksum)
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
            checksum=args.checksum)
    elif args.x:
        file_key = sources[0] if sources else None
        extract_h5_to_directory(h5_file, target_dir, file_key=file_key,
                                parallel=args.parallel,
                                verbose=args.verbose,
                                validate=args.validate)
    elif args.t:
        list_h5_contents(h5_file)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
