# har - HDF5 Archive Utility

A command-line tool for creating, appending to, extracting from, and listing
HDF5 archives. Designed as an alternative to `tar` with built-in parallel I/O,
per-file metadata, and indexed random access.

## Advantages over tar

- **Parallel I/O** — read and write files concurrently with `-p N`
- **Indexed access** — extract a single file without scanning the entire archive
- **Dynamic append** — add files to an existing archive without rewriting it
- **Per-file metadata** — file permissions are stored and restored automatically
- **Smaller archives** — HDF5 format has less overhead than tar for many small files
  (35% smaller on 100k files without compression)
- **Built-in compression** — gzip, lzf, and shuffle filter support

## Installation

```sh
pip install .
```

## Usage

### Flag comparison with tar

| Operation              | tar                        | har                        |
|------------------------|----------------------------|----------------------------|
| Create archive         | `tar -cf archive.tar dir`  | `har -cf archive.h5 dir`   |
| Create (verbose)       | `tar -cvf archive.tar dir` | `har -cvf archive.h5 dir`  |
| Extract archive        | `tar -xf archive.tar`      | `har -xf archive.h5`       |
| Extract to directory   | `tar -xf archive.tar -C d` | `har -xf archive.h5 -C d`  |
| Extract single file    | `tar -xf archive.tar path` | `har -xf archive.h5 path`  |
| List contents          | `tar -tf archive.tar`      | `har -tf archive.h5`       |
| Append to archive      | `tar -rf archive.tar dir`  | `har -rf archive.h5 dir`   |
| Gzip compression       | `tar -czf archive.tar.gz`  | `har -czf archive.h5`      |

### har-specific flags (not in tar)

| Flag                | Description                                         |
|---------------------|-----------------------------------------------------|
| `-p N, --parallel N`| Number of parallel workers (default: 1 = sequential) |
| `--lzf`             | Use HDF5 lzf compression (fast, moderate ratio)      |
| `--szip`            | Use HDF5 szip compression                            |
| `--zopt LEVEL`      | Gzip compression level 1-9 (default: 9)              |
| `--shuffle`         | HDF5 shuffle filter before compression               |

### Examples

```sh
# create an archive from a directory
har -cf archive.h5 mydir

# create with verbose output
har -cvf archive.h5 mydir

# create from multiple sources
har -cf archive.h5 dir1 dir2 file.txt

# create with gzip compression (level 9, shuffle filter)
har -czf archive.h5 --zopt 9 --shuffle mydir

# create with lzf compression (faster than gzip, less compression)
har -cf archive.h5 --lzf mydir

# create with parallel reads (8 workers)
har -cf archive.h5 -p 8 mydir

# append files to an existing archive (skips duplicates)
har -rf archive.h5 newdir morefile.txt

# list archive contents
har -tf archive.h5

# extract entire archive to current directory
har -xf archive.h5

# extract to a specific directory
har -xf archive.h5 -C /tmp/output

# extract a single file by its path in the archive
har -xf archive.h5 mydir/subdir/file.txt

# extract with parallel writes (8 workers)
har -xf archive.h5 -C output -p 8

# combine flags freely (verbose + gzip + parallel)
har -cvzf archive.h5 -p 16 --shuffle mydir
```

## Benchmarks

100,000 small files (29 MB total) on GPFS:

| Operation       | tar     | har -p 1 | har -p 8 | har -p 16 |
|-----------------|---------|----------|----------|-----------|
| Create archive  | 99s     | 244s     | 79s      | **67s**   |
| Extract archive | **38s** | 99s      | 48s      | 48s       |
| Archive size    | 99 MB   | 64 MB    | 64 MB    | **64 MB** |

## License

MIT License. See [LICENSE](LICENSE).
