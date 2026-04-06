#!/usr/bin/env python
"""
MPI parallel HDF5 write bandwidth on /dev/shm (RAM-backed tmpfs).

Two benchmarks:
  1. Raw mpio: 8 ranks stripe into a single shared dataset
  2. Batched: 8 ranks each write their own dataset (independent I/O)

This isolates HDF5/MPI overhead from GPFS by eliminating storage bottleneck.

Usage: srun -n 8 python tests/bench_mpi_shm.py /dev/shm/$USER
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

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else f"/dev/shm/{os.environ.get('USER', 'test')}"
TOTAL_SIZE = 50 * 1024**3  # 50 GB (must fit in /dev/shm alongside OS + buffers)
CHUNK_WRITE = 256 * 1024**2  # 256 MB per write


def human(nbytes):
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if nbytes < 1024:
            return f"{nbytes:.2f} {u}"
        nbytes /= 1024


def bench_raw_mpio():
    """8 ranks stripe into a single shared dataset via mpio."""
    h5_file = os.path.join(OUTPUT_DIR, "bench_mpio.h5")
    per_rank = TOTAL_SIZE // size

    if rank == 0:
        print(f"--- Raw mpio (shared dataset): {human(TOTAL_SIZE)} total, "
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
            total_w = written * size
            print(f"  [~{human(total_w)}] {elapsed:.1f}s, ~{human(total_w/elapsed)}/s",
                  flush=True)

    h5f.close()
    comm.Barrier()
    t1 = time.time()

    if rank == 0:
        elapsed = t1 - t0
        fsize = os.path.getsize(h5_file)
        bw = fsize / elapsed
        print(f"\n  Total time:       {elapsed:.2f} s")
        print(f"  File size:        {human(fsize)}")
        print(f"  Aggregate BW:     {human(bw)}/s")
        print(f"  Per-rank BW:      {human(bw / size)}/s")
        os.unlink(h5_file)

    comm.Barrier()


def bench_batched():
    """8 ranks each write their own dataset (independent I/O, no lock contention)."""
    h5_file = os.path.join(OUTPUT_DIR, "bench_batched.h5")
    per_rank = TOTAL_SIZE // size

    if rank == 0:
        print(f"\n--- Batched (1 dataset per rank): {human(TOTAL_SIZE)} total, "
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

    # Independent: each rank writes only its dataset
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
            total_w = written * size
            print(f"  [~{human(total_w)}] {elapsed:.1f}s, ~{human(total_w/elapsed)}/s",
                  flush=True)

    h5f.close()
    comm.Barrier()
    t1 = time.time()

    if rank == 0:
        elapsed = t1 - t0
        fsize = os.path.getsize(h5_file)
        bw = fsize / elapsed
        print(f"\n  Total time:       {elapsed:.2f} s")
        print(f"  File size:        {human(fsize)}")
        print(f"  Aggregate BW:     {human(bw)}/s")
        print(f"  Per-rank BW:      {human(bw / size)}/s")
        os.unlink(h5_file)

    comm.Barrier()


if __name__ == '__main__':
    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        shm_total = os.statvfs('/dev/shm')
        shm_gb = shm_total.f_blocks * shm_total.f_frsize / 1024**3
        print(f"=== 50 GB HDF5 Write Bandwidth on /dev/shm ===")
        print(f"Output dir:  {OUTPUT_DIR}")
        print(f"/dev/shm:    {shm_gb:.0f} GB")
        print(f"MPI ranks:   {size}")
        print(f"Chunk size:  {human(CHUNK_WRITE)}")
        print()

    comm.Barrier()

    bench_raw_mpio()
    bench_batched()

    if rank == 0:
        print("\n=== Benchmark complete ===")
