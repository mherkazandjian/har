"""
MPI parallel HDF5 tests for har --bagit --mpi mode.

Run with: mpirun -np 4 python -m pytest tests/test_h5_mpi.py -v

Requires: module load h5py/3.14.0-foss-2025a
"""

import os
import tempfile
import hashlib
import numpy as np
import pytest

from mpi4py import MPI
import h5py

from har_mpi import mpi_pack_bagit, mpi_extract_bagit, mpi_list_bagit
from har_bagit import (
    extract_bagit, pack_bagit, list_bagit, is_bagit_archive, INDEX_DTYPE,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def _sha256_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _count_files(root):
    count = 0
    for _, _, files in os.walk(root):
        count += len(files)
    return count


@pytest.fixture
def shared_tmp():
    """Temp directory created by rank 0, broadcast to all ranks."""
    if rank == 0:
        tmp = tempfile.mkdtemp(prefix="har_mpi_test_")
    else:
        tmp = None
    tmp = comm.bcast(tmp, root=0)
    comm.Barrier()
    yield tmp
    comm.Barrier()
    if rank == 0:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def _make_test_data(tmp, n_dirs=3, n_files=4):
    """Create test data (rank 0 only)."""
    src = os.path.join(tmp, "src")
    if rank == 0:
        os.makedirs(src, exist_ok=True)
        for i in range(n_dirs):
            d = os.path.join(src, f"dir{i}")
            os.makedirs(d, exist_ok=True)
            for j in range(n_files):
                with open(os.path.join(d, f"file{j}.txt"), 'w') as f:
                    f.write(f"content dir{i}/file{j}")
    comm.Barrier()
    return src


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mpi_archive_creation(shared_tmp):
    src = _make_test_data(shared_tmp)
    archive = os.path.join(shared_tmp, "test.h5")

    mpi_pack_bagit([src], archive, verbose=False)
    comm.Barrier()

    if rank == 0:
        assert os.path.exists(archive)
        assert is_bagit_archive(archive)
        with h5py.File(archive, 'r') as h5f:
            index = h5f['index'][()]
            assert len(index) == 12  # 3 dirs * 4 files


def test_mpi_roundtrip(shared_tmp):
    src = _make_test_data(shared_tmp)
    archive = os.path.join(shared_tmp, "rt.h5")
    out = os.path.join(shared_tmp, "out_rt")

    mpi_pack_bagit([src], archive)
    comm.Barrier()

    if rank == 0:
        os.makedirs(out, exist_ok=True)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        for i in range(3):
            for j in range(4):
                p = os.path.join(out, "src", f"dir{i}", f"file{j}.txt")
                assert os.path.exists(p), f"Missing: {p}"
                assert open(p).read() == f"content dir{i}/file{j}"


def test_mpi_serial_interop_create_mpi_extract_serial(shared_tmp):
    """Create with MPI, extract with serial har_bagit."""
    src = _make_test_data(shared_tmp)
    archive = os.path.join(shared_tmp, "interop1.h5")

    mpi_pack_bagit([src], archive)
    comm.Barrier()

    if rank == 0:
        out = os.path.join(shared_tmp, "out_interop1")
        os.makedirs(out)
        # Extract with serial (non-MPI) code
        extract_bagit(archive, out)
        for i in range(3):
            for j in range(4):
                p = os.path.join(out, "src", f"dir{i}", f"file{j}.txt")
                assert open(p).read() == f"content dir{i}/file{j}"


def test_mpi_serial_interop_create_serial_extract_mpi(shared_tmp):
    """Create with serial har_bagit, extract with MPI."""
    src = _make_test_data(shared_tmp)
    archive = os.path.join(shared_tmp, "interop2.h5")

    if rank == 0:
        pack_bagit([src], archive)
    comm.Barrier()

    out = os.path.join(shared_tmp, "out_interop2")
    if rank == 0:
        os.makedirs(out)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        for i in range(3):
            for j in range(4):
                p = os.path.join(out, "src", f"dir{i}", f"file{j}.txt")
                assert open(p).read() == f"content dir{i}/file{j}"


def test_mpi_empty_dirs(shared_tmp):
    src = os.path.join(shared_tmp, "emp")
    if rank == 0:
        os.makedirs(os.path.join(src, "empty_sub"), exist_ok=True)
        os.makedirs(os.path.join(src, "full"), exist_ok=True)
        with open(os.path.join(src, "full", "f.txt"), 'w') as f:
            f.write("data")
    comm.Barrier()

    archive = os.path.join(shared_tmp, "emp.h5")
    mpi_pack_bagit([src], archive)
    comm.Barrier()

    out = os.path.join(shared_tmp, "out_emp")
    if rank == 0:
        os.makedirs(out)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        assert os.path.isdir(os.path.join(out, "emp", "empty_sub"))
        assert os.path.isfile(os.path.join(out, "emp", "full", "f.txt"))


def test_mpi_permissions(shared_tmp):
    src = os.path.join(shared_tmp, "perms")
    exe_path = os.path.join(src, "run.sh")
    if rank == 0:
        os.makedirs(src, exist_ok=True)
        with open(exe_path, 'w') as f:
            f.write("#!/bin/bash")
        os.chmod(exe_path, 0o755)
    comm.Barrier()

    archive = os.path.join(shared_tmp, "perms.h5")
    mpi_pack_bagit([src], archive)
    comm.Barrier()

    out = os.path.join(shared_tmp, "out_perms")
    if rank == 0:
        os.makedirs(out)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        extracted = os.path.join(out, "perms", "run.sh")
        assert os.stat(extracted).st_mode == os.stat(exe_path).st_mode


def test_mpi_single_rank_works(shared_tmp):
    """Works correctly even with np=1 (or when run with more ranks)."""
    src = _make_test_data(shared_tmp, n_dirs=1, n_files=2)
    archive = os.path.join(shared_tmp, "single.h5")

    mpi_pack_bagit([src], archive)
    comm.Barrier()

    out = os.path.join(shared_tmp, "out_single")
    if rank == 0:
        os.makedirs(out)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        assert _count_files(os.path.join(out, "src")) == 2


def test_mpi_binary_roundtrip(shared_tmp):
    src = os.path.join(shared_tmp, "binsrc")
    if rank == 0:
        os.makedirs(src, exist_ok=True)
        with open(os.path.join(src, "allbytes.bin"), 'wb') as f:
            f.write(bytes(range(256)))
    comm.Barrier()

    archive = os.path.join(shared_tmp, "bin.h5")
    mpi_pack_bagit([src], archive)
    comm.Barrier()

    out = os.path.join(shared_tmp, "out_bin")
    if rank == 0:
        os.makedirs(out)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        with open(os.path.join(out, "binsrc", "allbytes.bin"), 'rb') as f:
            assert f.read() == bytes(range(256))


def test_mpi_many_files(shared_tmp):
    """Stress: 1000 files across 10 dirs with 4 ranks."""
    src = _make_test_data(shared_tmp, n_dirs=10, n_files=100)
    archive = os.path.join(shared_tmp, "many.h5")

    mpi_pack_bagit([src], archive)
    comm.Barrier()

    out = os.path.join(shared_tmp, "out_many")
    if rank == 0:
        os.makedirs(out)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        count = _count_files(os.path.join(out, "src"))
        assert count == 1000
        # Spot check
        for i in [0, 5, 9]:
            for j in [0, 50, 99]:
                p = os.path.join(out, "src", f"dir{i}", f"file{j}.txt")
                assert open(p).read() == f"content dir{i}/file{j}"


def test_mpi_multi_directory(shared_tmp):
    dir_a = os.path.join(shared_tmp, "dirA")
    dir_b = os.path.join(shared_tmp, "dirB")
    if rank == 0:
        os.makedirs(dir_a, exist_ok=True)
        os.makedirs(dir_b, exist_ok=True)
        with open(os.path.join(dir_a, "readme.txt"), 'w') as f:
            f.write("from A")
        with open(os.path.join(dir_b, "readme.txt"), 'w') as f:
            f.write("from B")
    comm.Barrier()

    archive = os.path.join(shared_tmp, "multi.h5")
    mpi_pack_bagit([dir_a, dir_b], archive)
    comm.Barrier()

    out = os.path.join(shared_tmp, "out_multi")
    if rank == 0:
        os.makedirs(out)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        assert open(os.path.join(out, "dirA", "readme.txt")).read() == "from A"
        assert open(os.path.join(out, "dirB", "readme.txt")).read() == "from B"
