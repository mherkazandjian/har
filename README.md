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

*(in progress)*

## BagIt mode (`--bagit`)

BagIt mode stores files in **batched datasets** instead of one HDF5 dataset per
file. This eliminates per-file metadata overhead and embeds
[RFC 8493](https://datatracker.ietf.org/doc/html/rfc8493) manifests (SHA-256
checksums, bag-info) inside the archive.

### Why batching matters

With the default (legacy) mode, archiving 100k files creates 100k HDF5 datasets
— each `create_dataset` call has fixed metadata overhead (~0.5 ms), totalling
~50 s before a single byte of data is written. BagIt mode concatenates files
into a small number of large batch datasets (~64 MB each by default) and stores
a compact index for random-access lookup. 100k files become ~50 batches + 1
index dataset instead of 100k datasets.

### HDF5 layout

```
/ (root)
  @har_format = "bagit-v1"
  /index          — compound dataset (path, batch_id, offset, length, mode, sha256)
  /batches/0      — uint8 dataset (concatenated file bytes, ~64 MB)
  /batches/1
  ...
  /bagit/
    bagit.txt
    bag-info.txt
    manifest-sha256.txt
    tagmanifest-sha256.txt
  /empty_dirs     — empty directory paths
```

### Usage

```sh
# create a BagIt archive
har --bagit -cf archive.h5 mydir

# create with custom batch size and parallel reads
har --bagit --batch-size 128M -cf archive.h5 -p 8 mydir

# create with gzip compression
har --bagit -czf archive.h5 mydir

# extract (auto-detects bagit format, --bagit flag optional)
har -xf archive.h5 -C output/

# extract a single file (indexed lookup, no scanning)
har --bagit -xf archive.h5 mydir/subdir/file.txt

# extract with SHA-256 validation
har --bagit --validate -xf archive.h5 -C output/

# extract as a full RFC 8493 BagIt bag (with tag files)
har --bagit --bagit-raw -xf archive.h5 -C output/

# list contents
har --bagit -tf archive.h5
```

### BagIt-specific flags

| Flag | Default | Description |
|------|---------|-------------|
| `--bagit` | off | Enable BagIt batched storage mode |
| `--batch-size` | `64M` | Target batch size (e.g. `64M`, `1G`) |
| `--validate` | off | Verify SHA-256 checksums on extraction |
| `--bagit-raw` | off | Extract as a full BagIt bag with tag files |

### Notes

- **Append (`-r`) is not supported** with `--bagit` — manifests include
  checksums of all files, so appending would require recomputing everything.
  Re-create the archive instead.
- **Auto-detection**: extracting or listing a bagit-v1 archive works without
  the `--bagit` flag — har detects the format from the `har_format` root
  attribute.
- On extraction, the `data/` BagIt prefix is stripped by default so the
  original directory tree is reproduced exactly. Use `--bagit-raw` to get
  the full BagIt directory with tag files.

## MPI parallel HDF5 I/O (`--mpi`) — experimental

For large-scale archiving on HPC systems, har supports MPI parallel HDF5 using
the `mpio` driver. Multiple MPI ranks write concurrently to a single `.h5` file.

**This feature is experimental.** The API and behavior may change.

MPI mode requires `--bagit` (batched datasets are the natural parallelism
boundary) and a parallel-enabled h5py + mpi4py environment.

### How it works

```
Phase 1  Rank 0 inventories the filesystem, broadcasts the file list
Phase 2  Each rank reads + SHA-256 checksums its assigned files
Phase 3  All ranks collectively open the HDF5 file (driver='mpio')
Phase 4  All ranks collectively create batch datasets (HDF5 requirement)
Phase 5  Each rank independently writes its owned batch data
Phase 6  Rank 0 writes the index + BagIt manifests
Phase 7  All ranks collectively close the file
```

Files are assigned to ranks by **batch ownership** (not round-robin by file),
so each rank reads all files for its batches and writes the full batch buffer.
This avoids data shuffling between ranks.

### Usage

```sh
# load MPI-enabled h5py (example for Snellius)
module load h5py/3.14.0-foss-2025a

# create archive with 8 MPI ranks
mpirun -np 8 har --bagit --mpi -cf archive.h5 mydir

# extract with 8 MPI ranks
mpirun -np 8 har --bagit --mpi -xf archive.h5 -C output/

# list contents (rank 0 only)
mpirun -np 1 har --bagit --mpi -tf archive.h5
```

### SLURM example

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks=16 --time=00:30:00

module purge && module load 2025 h5py/3.14.0-foss-2025a
srun python -m har --bagit --mpi -cf archive.h5 /path/to/data
```

### Limitations

- **Compression is not supported** with MPI parallel HDF5 (HDF5 library
  limitation). If `-z`/`--lzf`/`--szip` is combined with `--mpi`, the
  compression flags are ignored with a warning. Use `h5repack -f GZIP=9`
  for post-hoc compression.
- **Python only** — the Rust hdf5 crate v0.8 does not expose MPI bindings.
- **Single node** — currently tested with multiple ranks on one node. Multi-node
  requires a shared filesystem (GPFS/Lustre) visible to all ranks.

## License

MIT License. See [LICENSE](LICENSE).
