# HDF5 Archive Utility

## Purpose

This script provides a command-line interface for creating, appending to, extracting from, and listing the contents of HDF5 archives.

The main advantages over tar is the following:

  - ability to dynamically add files and not necessarily just append and concatenate especilly
    when compression is enabled.
  - ability to store metadata and attributes for each file.
  - adds all the functionality of the HDF5 library.

This script performs one of four operations on an HDF5 archive:

- Create archive (-c): Archive one or more directories/files into a new HDF5 file.
- Append (-r): Append one or more directories/files to an existing HDF5 file.
  In append mode, sources are processed recursively. Existing entries are skipped.
- Extract (-x): Extract files from an HDF5 archive.
  * With no extra positional argument, extracts the entire archive.
  * With a positional file key, extracts only that dataset.
  * Optionally specify extraction directory with -C.
- List (-t): List the contents (dataset keys) of an HDF5 file.

## Installation

```sh
pip install h5ar
```

## Usage Examples

```sh
# Create archive from multiple directories or files:
har -cf mydir.h5 mydir anotherdir file.txt
har -czf mydir.h5 mydir anotherdir file.txt
# gzip compression with level 9 of compression
har -czf mydir.h5 --zopt 9 mydir anotherdir file.txt
# use also hdf5 suffle
har -czf mydir.h5 --zopts 9 --shuffle mydir anotherdir file.txt

# Append multiple directories or files to an existing archive:
har -r -f mydir.h5 mydir anotherdir

# Extract entire archive to current directory:
har -x -f mydir.h5

# Extract entire archive to a specified directory:
har -x -f mydir.h5 -C foo

# Extract a specific file (dataset) from the archive:
har -x -f mydir.h5 path_to_file_in_the_archive

# List contents of the archive:
har -t -f mydir.h5


```
