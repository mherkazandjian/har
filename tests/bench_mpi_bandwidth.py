#!/usr/bin/env python
"""
MPI parallel HDF5 write bandwidth benchmark.

Creates a 100 GB dataset split across N ranks via mpio driver on GPFS.
Three modes: serial (1 rank), raw mpio (shared dataset), batched (separate datasets).

Usage: srun -n 8 python tests/bench_mpi_bandwidth.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TOTAL_SIZE = 100 * 1024**3  # 100 GB
CHUNK_WRITE = 256 * 1024**2  # 256 MB per write call
OUTPUT_DIR = "/scratch-shared"


def human(nbytes):
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if nbytes < 1024:
            return f"{nbytes:.2f} {u}"
        nbytes /= 1024


def bench_serial_write():
    """Serial baseline: rank 0 writes 100 GB with standard h5py."""
    h5_file = os.path.join(OUTPUT_DIR, "har_bench_serial.h5")

    if rank == 0:
        print(f"--- Serial write (rank 0 only): {human(TOTAL_SIZE)} ---")
        # Use O_DIRECT-like behavior: sync after close
        t0 = time.time()
        h5f = h5py.File(h5_file, 'w')
        ds = h5f.create_dataset('data', shape=(TOTAL_SIZE,), dtype=np.uint8)

        written = 0
        for offset in range(0, TOTAL_SIZE, CHUNK_WRITE):
            chunk_end = min(offset + CHUNK_WRITE, TOTAL_SIZE)
            chunk_size = chunk_end - offset
            buf = np.zeros(chunk_size, dtype=np.uint8)
            ds[offset:chunk_end] = buf
            written += chunk_size
            if written % (5 * 1024**3) == 0:
                elapsed = time.time() - t0
                print(f"  [rank 0] {human(written)} written, "
                      f"{elapsed:.1f}s, {human(written/elapsed)}/s", flush=True)

        h5f.close()
        # fsync to force flush to GPFS
        os.sync()
        t1 = time.time()

        elapsed = t1 - t0
        fsize = os.path.getsize(h5_file)
        bw = fsize / elapsed
        print(f"\n  Total time: {elapsed:.2f} s")
        print(f"  File size:  {human(fsize)}")
        print(f"  Bandwidth:  {human(bw)}/s")
        os.unlink(h5_file)

    comm.Barrier()


def bench_raw_mpio():
    """MPI parallel write: all ranks write stripes into a single dataset."""
    h5_file = os.path.join(OUTPUT_DIR, "har_bench_mpio.h5")
    per_rank = TOTAL_SIZE // size

    if rank == 0:
        print(f"\n--- Raw mpio write: {human(TOTAL_SIZE)} total, "
              f"{human(per_rank)} per rank, {size} ranks ---")

    comm.Barrier()
    t0 = time.time()

    h5f = h5py.File(h5_file, 'w', driver='mpio', comm=comm)
    ds = h5f.create_dataset('data', shape=(TOTAL_SIZE,), dtype=np.uint8)

    my_start = rank * per_rank
    my_end = my_start + per_rank
    written = 0

    for offset in range(my_start, my_end, CHUNK_WRITE):
        chunk_end = min(offset + CHUNK_WRITE, my_end)
        chunk_size = chunk_end - offset
        buf = np.full(chunk_size, fill_value=(rank % 256), dtype=np.uint8)
        ds[offset:chunk_end] = buf
        written += chunk_size
        if rank == 0 and written % (2 * 1024**3) == 0:
            elapsed = time.time() - t0
            # Total written across all ranks (approx)
            total_written = written * size
            print(f"  [total ~{human(total_written)}] "
                  f"{elapsed:.1f}s, ~{human(total_written/elapsed)}/s", flush=True)

    h5f.close()
    comm.Barrier()
    t1 = time.time()

    if rank == 0:
        elapsed = t1 - t0
        fsize = os.path.getsize(h5_file)
        bw = fsize / elapsed
        print(f"\n  Total time: {elapsed:.2f} s")
        print(f"  File size:  {human(fsize)}")
        print(f"  Bandwidth:  {human(bw)}/s  (aggregate)")
        print(f"  Per-rank:   {human(bw / size)}/s")
        os.unlink(h5_file)

    comm.Barrier()


def bench_batched():
    """Batched parallel write: each rank writes its own dataset (independent I/O)."""
    h5_file = os.path.join(OUTPUT_DIR, "har_bench_batched.h5")
    per_rank = TOTAL_SIZE // size

    if rank == 0:
        print(f"\n--- Batched write (1 dataset per rank): {human(TOTAL_SIZE)} total, "
              f"{human(per_rank)} per rank, {size} ranks ---")

    comm.Barrier()
    t0 = time.time()

    h5f = h5py.File(h5_file, 'w', driver='mpio', comm=comm)
    h5f.attrs['test'] = 'batched'

    # Collective: create all datasets
    batches_grp = h5f.create_group('batches')
    datasets = []
    for i in range(size):
        ds = batches_grp.create_dataset(str(i), shape=(per_rank,), dtype=np.uint8)
        datasets.append(ds)

    # Independent: each rank writes only its own dataset
    ds = datasets[rank]
    written = 0
    for offset in range(0, per_rank, CHUNK_WRITE):
        chunk_end = min(offset + CHUNK_WRITE, per_rank)
        chunk_size = chunk_end - offset
        buf = np.full(chunk_size, fill_value=(rank % 256), dtype=np.uint8)
        ds[offset:chunk_end] = buf
        written += chunk_size
        if rank == 0 and written % (2 * 1024**3) == 0:
            elapsed = time.time() - t0
            total_written = written * size
            print(f"  [total ~{human(total_written)}] "
                  f"{elapsed:.1f}s, ~{human(total_written/elapsed)}/s", flush=True)

    h5f.close()
    comm.Barrier()
    t1 = time.time()

    if rank == 0:
        elapsed = t1 - t0
        fsize = os.path.getsize(h5_file)
        bw = fsize / elapsed
        print(f"\n  Total time: {elapsed:.2f} s")
        print(f"  File size:  {human(fsize)}")
        print(f"  Bandwidth:  {human(bw)}/s  (aggregate)")
        print(f"  Per-rank:   {human(bw / size)}/s")
        os.unlink(h5_file)

    comm.Barrier()


if __name__ == '__main__':
    if rank == 0:
        print(f"=== 100 GB HDF5 Write Bandwidth on GPFS ===")
        print(f"Output dir:  {OUTPUT_DIR}")
        print(f"MPI ranks:   {size}")
        print(f"Chunk size:  {human(CHUNK_WRITE)}")
        print()

    bench_serial_write()
    bench_raw_mpio()
    bench_batched()

    if rank == 0:
        print("\n=== Benchmark complete ===")
