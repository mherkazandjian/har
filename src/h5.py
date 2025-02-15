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
import h5py
import numpy as np
import time

def pack_or_append_to_h5(sources, output_h5, file_mode):
    """
    Archive or append the given sources (directories and/or files) into the HDF5 archive.

    :param sources: List of source directories and/or files.
    :param output_h5: The HDF5 archive filename.
    :param file_mode: 'w' to create a new archive (overwrite), or 'a' to append.
    """
    output_h5 = os.path.expanduser(output_h5)
    t0 = time.time()

    with h5py.File(output_h5, file_mode) as h5fobj:
        def process_file(file_path, base_dir):
            """Process a single file and add it to the archive."""
            if base_dir:
                rel_path = os.path.relpath(file_path, base_dir)
            else:
                rel_path = os.path.basename(file_path)
            print("Storing:", rel_path)
            if file_mode == 'a' and rel_path in h5fobj:
                print(f"Skipping {rel_path} (already exists)")
                return
            group_path = os.path.dirname(rel_path)
            if group_path:
                h5fobj.require_group(group_path)
            with open(file_path, 'rb') as fobj:
                file_content = fobj.read()
            file_data = np.frombuffer(file_content, dtype=np.uint8)
            h5fobj.create_dataset(
                rel_path,
                data=file_data,
                dtype=np.uint8,
                compression=None
            )

        # Process each source.
        for source in sources:
            source = os.path.expanduser(source)
            if os.path.isdir(source):
                # Process the directory recursively.
                for root, dirs, files in os.walk(source):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        process_file(file_path, source)
            elif os.path.isfile(source):
                process_file(source, None)
            else:
                print(f"Warning: '{source}' is not a valid file or directory; skipping.", file=sys.stderr)
    t1 = time.time()
    print(f"\nOperation completed in {t1 - t0:.2f} seconds.")

def extract_h5_to_directory(h5_path, extract_dir, file_key=None):
    """
    Extract files from an HDF5 file.
      - If file_key is None, extract the entire archive.
      - Otherwise, extract only the dataset corresponding to file_key.
    Files are extracted into extract_dir.
    """
    h5_path = os.path.expanduser(h5_path)
    extract_dir = os.path.expanduser(extract_dir)
    with h5py.File(h5_path, 'r') as h5fobj:
        if file_key:
            if file_key not in h5fobj:
                print(f"Error: '{file_key}' not found in archive.", file=sys.stderr)
                sys.exit(1)
            dest_file_path = os.path.join(extract_dir, file_key)
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            file_bytes = h5fobj[file_key][()].tobytes()
            with open(dest_file_path, 'wb') as fout:
                fout.write(file_bytes)
            print(f"Extracted: {file_key}")
        else:
            def extract_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    dest_file_path = os.path.join(extract_dir, name)
                    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                    file_bytes = obj[()].tobytes()
                    with open(dest_file_path, 'wb') as fout:
                        fout.write(file_bytes)
                    print(f"Extracted: {name}")
            h5fobj.visititems(extract_dataset)
    print("Extraction complete!")

def list_h5_contents(h5_path):
    """List all dataset keys in the given HDF5 file."""
    h5_path = os.path.expanduser(h5_path)
    with h5py.File(h5_path, 'r') as h5fobj:
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

    # Required HDF5 file.
    parser.add_argument("-f", "--file", required=True,
                        help="HDF5 archive file (for output when archiving/appending; for input when extracting/listing).")

    # Optional extraction directory (used with -x).
    parser.add_argument("-C", "--directory", default=".",
                        help="Target extraction directory (default: current directory).")

    # Positional argument:
    # For -c and -r: one or more source directories or files.
    # For -x: optionally, the file key to extract (if omitted, extract entire archive).
    parser.add_argument("path", nargs="*", default=None,
                        help=("For -c and -r: source directories/files; "
                              "for -x: file key to extract (if omitted, extract entire archive)."))

    args = parser.parse_args()

    h5_file = args.file
    target_dir = args.directory  # For extraction, the target directory.
    sources = args.path         # For archive/append mode, this is a list of sources.

    if args.c:
        # Create archive.
        if not sources:
            print("Error: At least one source (directory or file) is required for archive creation (-c).", file=sys.stderr)
            sys.exit(1)
        pack_or_append_to_h5(sources, h5_file, 'w')
    elif args.r:
        # Append to archive.
        if not sources:
            print("Error: At least one source (directory or file) is required for appending (-r).", file=sys.stderr)
            sys.exit(1)
        pack_or_append_to_h5(sources, h5_file, 'a')
    elif args.x:
        # Extract from archive.
        # If sources is not empty, treat the first source as the file key.
        file_key = sources[0] if sources else None
        extract_h5_to_directory(h5_file, target_dir, file_key=file_key)
    elif args.t:
        # List contents.
        list_h5_contents(h5_file)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
