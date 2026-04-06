#!/usr/bin/env python
"""
Direct MPI test runner (not pytest) — ensures all ranks stay in sync.

Usage: srun -n 4 python tests/run_mpi_tests_direct.py
"""
import os
import sys
import tempfile
import shutil
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def log(msg):
    if rank == 0:
        print(msg, flush=True)


def make_shared_tmp():
    if rank == 0:
        tmp = tempfile.mkdtemp(prefix="har_mpi_test_")
    else:
        tmp = None
    tmp = comm.bcast(tmp, root=0)
    comm.Barrier()
    return tmp


def cleanup(tmp):
    comm.Barrier()
    if rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


def make_test_data(tmp, n_dirs=3, n_files=4):
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


def count_files(root):
    count = 0
    for _, _, files in os.walk(root):
        count += len(files)
    return count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mpi_archive_creation():
    from har_mpi import mpi_pack_bagit
    from har_bagit import is_bagit_archive
    import h5py

    tmp = make_shared_tmp()
    src = make_test_data(tmp)
    archive = os.path.join(tmp, "test.h5")

    mpi_pack_bagit([src], archive)
    comm.Barrier()

    if rank == 0:
        assert os.path.exists(archive), "Archive not created"
        assert is_bagit_archive(archive), "Not a bagit archive"
        with h5py.File(archive, 'r') as h5f:
            index = h5f['index'][()]
            assert len(index) == 12, f"Expected 12 files, got {len(index)}"
    cleanup(tmp)


def test_mpi_roundtrip():
    from har_mpi import mpi_pack_bagit, mpi_extract_bagit

    tmp = make_shared_tmp()
    src = make_test_data(tmp)
    archive = os.path.join(tmp, "rt.h5")
    out = os.path.join(tmp, "out_rt")

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
                content = open(p).read()
                expected = f"content dir{i}/file{j}"
                assert content == expected, f"{p}: got {content!r}, expected {expected!r}"
    cleanup(tmp)


def test_mpi_serial_interop():
    """Create with MPI, extract with serial."""
    from har_mpi import mpi_pack_bagit
    from har_bagit import extract_bagit

    tmp = make_shared_tmp()
    src = make_test_data(tmp)
    archive = os.path.join(tmp, "interop.h5")

    mpi_pack_bagit([src], archive)
    comm.Barrier()

    if rank == 0:
        out = os.path.join(tmp, "out_interop")
        os.makedirs(out)
        extract_bagit(archive, out)
        for i in range(3):
            for j in range(4):
                p = os.path.join(out, "src", f"dir{i}", f"file{j}.txt")
                assert open(p).read() == f"content dir{i}/file{j}"
    comm.Barrier()
    cleanup(tmp)


def test_mpi_empty_dirs():
    from har_mpi import mpi_pack_bagit, mpi_extract_bagit

    tmp = make_shared_tmp()
    src = os.path.join(tmp, "emp")
    if rank == 0:
        os.makedirs(os.path.join(src, "empty_sub"), exist_ok=True)
        os.makedirs(os.path.join(src, "full"), exist_ok=True)
        with open(os.path.join(src, "full", "f.txt"), 'w') as f:
            f.write("data")
    comm.Barrier()

    archive = os.path.join(tmp, "emp.h5")
    mpi_pack_bagit([src], archive)
    comm.Barrier()

    out = os.path.join(tmp, "out_emp")
    if rank == 0:
        os.makedirs(out)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        assert os.path.isdir(os.path.join(out, "emp", "empty_sub"))
        assert os.path.isfile(os.path.join(out, "emp", "full", "f.txt"))
    cleanup(tmp)


def test_mpi_permissions():
    from har_mpi import mpi_pack_bagit, mpi_extract_bagit

    tmp = make_shared_tmp()
    src = os.path.join(tmp, "perms")
    exe_path = os.path.join(src, "run.sh")
    if rank == 0:
        os.makedirs(src, exist_ok=True)
        with open(exe_path, 'w') as f:
            f.write("#!/bin/bash")
        os.chmod(exe_path, 0o755)
    comm.Barrier()

    archive = os.path.join(tmp, "perms.h5")
    mpi_pack_bagit([src], archive)
    comm.Barrier()

    out = os.path.join(tmp, "out_perms")
    if rank == 0:
        os.makedirs(out)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        extracted = os.path.join(out, "perms", "run.sh")
        assert os.stat(extracted).st_mode == os.stat(exe_path).st_mode
    cleanup(tmp)


def test_mpi_binary():
    from har_mpi import mpi_pack_bagit, mpi_extract_bagit

    tmp = make_shared_tmp()
    src = os.path.join(tmp, "binsrc")
    if rank == 0:
        os.makedirs(src, exist_ok=True)
        with open(os.path.join(src, "allbytes.bin"), 'wb') as f:
            f.write(bytes(range(256)))
    comm.Barrier()

    archive = os.path.join(tmp, "bin.h5")
    mpi_pack_bagit([src], archive)
    comm.Barrier()

    out = os.path.join(tmp, "out_bin")
    if rank == 0:
        os.makedirs(out)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        with open(os.path.join(out, "binsrc", "allbytes.bin"), 'rb') as f:
            assert f.read() == bytes(range(256))
    cleanup(tmp)


def test_mpi_many_files():
    from har_mpi import mpi_pack_bagit, mpi_extract_bagit

    tmp = make_shared_tmp()
    src = make_test_data(tmp, n_dirs=10, n_files=100)
    archive = os.path.join(tmp, "many.h5")

    mpi_pack_bagit([src], archive)
    comm.Barrier()

    out = os.path.join(tmp, "out_many")
    if rank == 0:
        os.makedirs(out)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        count = count_files(os.path.join(out, "src"))
        assert count == 1000, f"Expected 1000 files, got {count}"
        # Spot check
        for i in [0, 5, 9]:
            for j in [0, 50, 99]:
                p = os.path.join(out, "src", f"dir{i}", f"file{j}.txt")
                assert open(p).read() == f"content dir{i}/file{j}"
    cleanup(tmp)


def test_mpi_multi_directory():
    from har_mpi import mpi_pack_bagit, mpi_extract_bagit

    tmp = make_shared_tmp()
    dir_a = os.path.join(tmp, "dirA")
    dir_b = os.path.join(tmp, "dirB")
    if rank == 0:
        os.makedirs(dir_a)
        os.makedirs(dir_b)
        with open(os.path.join(dir_a, "readme.txt"), 'w') as f:
            f.write("from A")
        with open(os.path.join(dir_b, "readme.txt"), 'w') as f:
            f.write("from B")
    comm.Barrier()

    archive = os.path.join(tmp, "multi.h5")
    mpi_pack_bagit([dir_a, dir_b], archive)
    comm.Barrier()

    out = os.path.join(tmp, "out_multi")
    if rank == 0:
        os.makedirs(out)
    comm.Barrier()

    mpi_extract_bagit(archive, out)
    comm.Barrier()

    if rank == 0:
        assert open(os.path.join(out, "dirA", "readme.txt")).read() == "from A"
        assert open(os.path.join(out, "dirB", "readme.txt")).read() == "from B"
    cleanup(tmp)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_mpi_archive_creation,
    test_mpi_roundtrip,
    test_mpi_serial_interop,
    test_mpi_empty_dirs,
    test_mpi_permissions,
    test_mpi_binary,
    test_mpi_many_files,
    test_mpi_multi_directory,
]

if __name__ == '__main__':
    passed = 0
    failed = 0
    errors = []

    log(f"Running {len(TESTS)} MPI tests with {size} ranks\n")

    for test_fn in TESTS:
        name = test_fn.__name__
        comm.Barrier()
        try:
            test_fn()
            comm.Barrier()
            log(f"  {name} ... PASSED")
            passed += 1
        except Exception as e:
            comm.Barrier()
            if rank == 0:
                print(f"  {name} ... FAILED: {e}", flush=True)
                traceback.print_exc()
            failed += 1
            errors.append(name)

    log(f"\n{'='*50}")
    log(f"Results: {passed} passed, {failed} failed out of {len(TESTS)}")
    if errors:
        log(f"Failed: {', '.join(errors)}")
    log(f"{'='*50}")

    sys.exit(1 if failed else 0)
