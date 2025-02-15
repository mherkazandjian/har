# HDF5 Archive Utility

This script performs one of four operations on an HDF5 archive:
- Create archive (-c): Archive one or more directories/files into a new HDF5 file.
- Append (-r): Append one or more directories/files to an existing HDF5 file.
  In append mode, sources are processed recursively. Existing entries are skipped.
- Extract (-x): Extract files from an HDF5 archive.
  * With no extra positional argument, extracts the entire archive.
  * With a positional file key, extracts only that dataset.
  * Optionally specify extraction directory with -C.
- List (-t): List the contents (dataset keys) of an HDF5 file.

## Usage Examples

```sh
# Create archive from multiple directories or files:
python [h5.py](http://_vscodecontentref_/1) -c -f mydir.h5 mydir anotherdir file.txt

# Append multiple directories or files to an existing archive:
python [h5.py](http://_vscodecontentref_/2) -r -f mydir.h5 mydir anotherdir

# Extract entire archive to current directory:
python [h5.py](http://_vscodecontentref_/3) -x -f mydir.h5

# Extract entire archive to a specified directory:
python [h5.py](http://_vscodecontentref_/4) -x -f mydir.h5 -C foo

# Extract a specific file (dataset) from the archive:
python [h5.py](http://_vscodecontentref_/5) -x -f mydir.h5 path_to_file_in_the_archive

# List contents of the archive:
python [h5.py](http://_vscodecontentref_/6) -t -f mydir.h5