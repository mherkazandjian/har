import os
import stat
import tempfile
import h5py
import pytest
from io import StringIO
import contextlib

from har import pack_or_append_to_h5, extract_h5_to_directory, list_h5_contents

@pytest.fixture
def test_env():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = temp_dir

        # Create source directory with files
        src_dir = os.path.join(base_dir, "src")
        os.makedirs(src_dir)

        # Create file1.txt
        file1 = os.path.join(src_dir, "file1.txt")
        with open(file1, "w") as f:
            f.write("This is file 1.")

        # Create subdirectory and file2.txt
        sub_dir = os.path.join(src_dir, "subdir")
        os.makedirs(sub_dir)
        file2 = os.path.join(sub_dir, "file2.txt")
        with open(file2, "w") as f:
            f.write("This is file 2 in subdir.")

        # Define archive and extract paths
        archive_path = os.path.join(base_dir, "archive.h5")
        extract_dir = os.path.join(base_dir, "extract")
        os.makedirs(extract_dir, exist_ok=True)

        yield {
            'src_dir': src_dir,
            'file1': file1,
            'file2': file2,
            'sub_dir': sub_dir,
            'archive_path': archive_path,
            'extract_dir': extract_dir
        }

def test_archive_creation(test_env):
    pack_or_append_to_h5([test_env['src_dir']], test_env['archive_path'], 'w')
    assert os.path.exists(test_env['archive_path'])

    with h5py.File(test_env['archive_path'], 'r') as h5f:
        assert "src/file1.txt" in h5f
        assert "src/subdir" in h5f
        assert "file2.txt" in h5f["src/subdir"]

def test_append(test_env):
    pack_or_append_to_h5([test_env['src_dir']], test_env['archive_path'], 'w')

    file3 = os.path.join(test_env['src_dir'], "file3.txt")
    with open(file3, "w") as f:
        f.write("This is file 3.")

    pack_or_append_to_h5([file3], test_env['archive_path'], 'a')

    with h5py.File(test_env['archive_path'], 'r') as h5f:
        assert "file3.txt" in h5f

def test_extract_all(test_env):
    pack_or_append_to_h5([test_env['src_dir']], test_env['archive_path'], 'w')
    extract_h5_to_directory(test_env['archive_path'], test_env['extract_dir'])

    extracted_file1 = os.path.join(test_env['extract_dir'], "src", "file1.txt")
    assert os.path.exists(extracted_file1)
    with open(extracted_file1, "r") as f:
        assert f.read() == "This is file 1."

    extracted_file2 = os.path.join(test_env['extract_dir'], "src", "subdir", "file2.txt")
    assert os.path.exists(extracted_file2)
    with open(extracted_file2, "r") as f:
        assert f.read() == "This is file 2 in subdir."

def test_extract_single(test_env):
    pack_or_append_to_h5([test_env['src_dir']], test_env['archive_path'], 'w')
    extract_h5_to_directory(test_env['archive_path'], test_env['extract_dir'], file_key="src/file1.txt")

    extracted_file1 = os.path.join(test_env['extract_dir'], "src", "file1.txt")
    assert os.path.exists(extracted_file1)
    with open(extracted_file1, "r") as f:
        assert f.read() == "This is file 1."

    extracted_file2 = os.path.join(test_env['extract_dir'], "src", "subdir", "file2.txt")
    assert not os.path.exists(extracted_file2)

def test_list_contents(test_env):
    pack_or_append_to_h5([test_env['src_dir']], test_env['archive_path'], 'w')

    output = StringIO()
    with contextlib.redirect_stdout(output):
        list_h5_contents(test_env['archive_path'])
    out_str = output.getvalue()

    assert "src/file1.txt" in out_str
    assert "src/subdir" in out_str


def test_multi_directory_no_collision():
    """Archiving multiple dirs with same-named files must not crash."""
    with tempfile.TemporaryDirectory() as tmp:
        dir_a = os.path.join(tmp, "dirA")
        dir_b = os.path.join(tmp, "dirB")
        os.makedirs(dir_a)
        os.makedirs(dir_b)
        with open(os.path.join(dir_a, "readme.txt"), "w") as f:
            f.write("from A")
        with open(os.path.join(dir_b, "readme.txt"), "w") as f:
            f.write("from B")

        archive = os.path.join(tmp, "multi.h5")
        pack_or_append_to_h5([dir_a, dir_b], archive, 'w')

        with h5py.File(archive, 'r') as h5f:
            assert "dirA/readme.txt" in h5f
            assert "dirB/readme.txt" in h5f

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir)
        with open(os.path.join(extract_dir, "dirA", "readme.txt")) as f:
            assert f.read() == "from A"
        with open(os.path.join(extract_dir, "dirB", "readme.txt")) as f:
            assert f.read() == "from B"


def test_empty_directory_preserved():
    """Empty directories should survive a roundtrip."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "project")
        os.makedirs(os.path.join(src, "empty_sub"))
        os.makedirs(os.path.join(src, "nonempty"))
        with open(os.path.join(src, "nonempty", "f.txt"), "w") as f:
            f.write("data")

        archive = os.path.join(tmp, "empty.h5")
        pack_or_append_to_h5([src], archive, 'w')

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir)

        assert os.path.isdir(os.path.join(extract_dir, "project", "empty_sub"))
        assert os.path.isfile(os.path.join(extract_dir, "project", "nonempty", "f.txt"))


def test_file_permissions_preserved():
    """File permissions (mode bits) should survive a roundtrip."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "perms")
        os.makedirs(src)
        exec_file = os.path.join(src, "run.sh")
        with open(exec_file, "w") as f:
            f.write("#!/bin/bash\necho hi")
        os.chmod(exec_file, 0o755)

        ro_file = os.path.join(src, "readonly.txt")
        with open(ro_file, "w") as f:
            f.write("read only")
        os.chmod(ro_file, 0o444)

        archive = os.path.join(tmp, "perms.h5")
        pack_or_append_to_h5([src], archive, 'w')

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir)

        extracted_exec = os.path.join(extract_dir, "perms", "run.sh")
        extracted_ro = os.path.join(extract_dir, "perms", "readonly.txt")
        assert os.stat(extracted_exec).st_mode == os.stat(exec_file).st_mode
        assert os.stat(extracted_ro).st_mode == os.stat(ro_file).st_mode


def test_error_nonexistent_archive():
    """Extract/list on a non-existent archive should exit cleanly."""
    with pytest.raises(SystemExit):
        extract_h5_to_directory("/tmp/nonexistent_har_test.h5", "/tmp")
    with pytest.raises(SystemExit):
        list_h5_contents("/tmp/nonexistent_har_test.h5")


def test_error_corrupt_archive():
    """Extract/list on a corrupt file should exit cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "bad.h5")
        with open(bad, "w") as f:
            f.write("not an hdf5 file")
        with pytest.raises(SystemExit):
            extract_h5_to_directory(bad, tmp)
        with pytest.raises(SystemExit):
            list_h5_contents(bad)


def test_empty_file_roundtrip():
    """Empty files (0 bytes) should survive a roundtrip."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "emptysrc")
        os.makedirs(src)
        empty = os.path.join(src, "empty.txt")
        open(empty, "w").close()

        archive = os.path.join(tmp, "empty_file.h5")
        pack_or_append_to_h5([src], archive, 'w')

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir)

        extracted = os.path.join(extract_dir, "emptysrc", "empty.txt")
        assert os.path.exists(extracted)
        assert os.path.getsize(extracted) == 0


def test_binary_file_roundtrip():
    """Binary files with all byte values should survive a roundtrip."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "binsrc")
        os.makedirs(src)
        binfile = os.path.join(src, "allbytes.bin")
        with open(binfile, "wb") as f:
            f.write(bytes(range(256)))

        archive = os.path.join(tmp, "bin.h5")
        pack_or_append_to_h5([src], archive, 'w')

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir)

        extracted = os.path.join(extract_dir, "binsrc", "allbytes.bin")
        with open(extracted, "rb") as f:
            assert f.read() == bytes(range(256))


def test_compression_roundtrip():
    """Compressed archives should extract correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "comp")
        os.makedirs(src)
        with open(os.path.join(src, "data.txt"), "w") as f:
            f.write("A" * 100000)

        archive = os.path.join(tmp, "comp.h5")
        pack_or_append_to_h5([src], archive, 'w',
                             compression='gzip', compression_opts=9, shuffle=True)

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir)

        with open(os.path.join(extract_dir, "comp", "data.txt")) as f:
            assert f.read() == "A" * 100000

        # Compressed file should be much smaller than 100KB
        assert os.path.getsize(archive) < 15000


# ---- Parallel tests ----

def _make_test_tree(base, n_dirs=3, n_files_per_dir=4):
    """Helper to create a test directory tree with files."""
    for i in range(n_dirs):
        d = os.path.join(base, f"dir{i}")
        os.makedirs(d)
        for j in range(n_files_per_dir):
            with open(os.path.join(d, f"file{j}.txt"), "w") as f:
                f.write(f"content dir{i}/file{j}")


def test_parallel_ingest():
    """Parallel ingest produces correct archive contents."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        _make_test_tree(src)

        archive = os.path.join(tmp, "par.h5")
        pack_or_append_to_h5([src], archive, 'w', parallel=4)

        with h5py.File(archive, 'r') as h5f:
            keys = []
            h5f.visititems(lambda n, o: keys.append(n) if isinstance(o, h5py.Dataset) else None)
            assert len(keys) == 12  # 3 dirs * 4 files

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir)
        for i in range(3):
            for j in range(4):
                p = os.path.join(extract_dir, "src", f"dir{i}", f"file{j}.txt")
                assert os.path.exists(p)
                with open(p) as f:
                    assert f.read() == f"content dir{i}/file{j}"


def test_parallel_extract():
    """Parallel extract produces correct output from sequentially created archive."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        _make_test_tree(src)

        archive = os.path.join(tmp, "seq.h5")
        pack_or_append_to_h5([src], archive, 'w')  # sequential ingest

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir, parallel=4)

        for i in range(3):
            for j in range(4):
                p = os.path.join(extract_dir, "src", f"dir{i}", f"file{j}.txt")
                with open(p) as f:
                    assert f.read() == f"content dir{i}/file{j}"


def test_parallel_roundtrip():
    """Parallel ingest + parallel extract end-to-end with diverse content."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "project")
        os.makedirs(os.path.join(src, "data"))
        os.makedirs(os.path.join(src, "empty_sub"))

        with open(os.path.join(src, "readme.txt"), "w") as f:
            f.write("hello")
        with open(os.path.join(src, "data", "bin.dat"), "wb") as f:
            f.write(bytes(range(256)))
        exec_file = os.path.join(src, "data", "run.sh")
        with open(exec_file, "w") as f:
            f.write("#!/bin/bash")
        os.chmod(exec_file, 0o755)

        archive = os.path.join(tmp, "rt.h5")
        pack_or_append_to_h5([src], archive, 'w', parallel=3)

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir, parallel=3)

        # Verify text file
        with open(os.path.join(extract_dir, "project", "readme.txt")) as f:
            assert f.read() == "hello"
        # Verify binary file
        with open(os.path.join(extract_dir, "project", "data", "bin.dat"), "rb") as f:
            assert f.read() == bytes(range(256))
        # Verify permissions
        assert os.stat(os.path.join(extract_dir, "project", "data", "run.sh")).st_mode == os.stat(exec_file).st_mode
        # Verify empty dir
        assert os.path.isdir(os.path.join(extract_dir, "project", "empty_sub"))


def test_parallel_ingest_with_compression():
    """Parallel ingest with gzip+shuffle works correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "comp")
        os.makedirs(src)
        with open(os.path.join(src, "big.txt"), "w") as f:
            f.write("B" * 50000)

        archive = os.path.join(tmp, "comp.h5")
        pack_or_append_to_h5([src], archive, 'w',
                             compression='gzip', compression_opts=9, shuffle=True,
                             parallel=2)

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir)
        with open(os.path.join(extract_dir, "comp", "big.txt")) as f:
            assert f.read() == "B" * 50000
        assert os.path.getsize(archive) < 15000


def test_parallel_append_skips_existing():
    """Append with parallel skips existing entries."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "app")
        os.makedirs(src)
        with open(os.path.join(src, "f.txt"), "w") as f:
            f.write("v1")

        archive = os.path.join(tmp, "app.h5")
        pack_or_append_to_h5([src], archive, 'w', parallel=2)

        # Modify and append — should skip existing
        with open(os.path.join(src, "f.txt"), "w") as f:
            f.write("v2")
        with open(os.path.join(src, "new.txt"), "w") as f:
            f.write("new")
        pack_or_append_to_h5([src], archive, 'a', parallel=2)

        with h5py.File(archive, 'r') as h5f:
            assert h5f["app/f.txt"][()].tobytes() == b"v1"  # original kept
            assert h5f["app/new.txt"][()].tobytes() == b"new"


def test_parallel_extract_preserves_permissions():
    """Parallel extraction restores file permissions."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "perms")
        os.makedirs(src)
        f = os.path.join(src, "exec.sh")
        with open(f, "w") as fh:
            fh.write("#!/bin/bash")
        os.chmod(f, 0o755)

        archive = os.path.join(tmp, "perms.h5")
        pack_or_append_to_h5([src], archive, 'w')

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir, parallel=2)

        assert os.stat(os.path.join(extract_dir, "perms", "exec.sh")).st_mode == os.stat(f).st_mode


def test_parallel_extract_preserves_empty_dirs():
    """Parallel extraction creates empty directories."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "emp")
        os.makedirs(os.path.join(src, "empty"))
        os.makedirs(os.path.join(src, "full"))
        with open(os.path.join(src, "full", "f.txt"), "w") as f:
            f.write("data")

        archive = os.path.join(tmp, "emp.h5")
        pack_or_append_to_h5([src], archive, 'w')

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir, parallel=2)

        assert os.path.isdir(os.path.join(extract_dir, "emp", "empty"))
        assert os.path.isfile(os.path.join(extract_dir, "emp", "full", "f.txt"))


def test_parallel_many_small_files():
    """Stress test: 100 files across 10 dirs with parallel=8."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "many")
        os.makedirs(src)
        _make_test_tree(src, n_dirs=10, n_files_per_dir=10)

        archive = os.path.join(tmp, "many.h5")
        pack_or_append_to_h5([src], archive, 'w', parallel=8)

        extract_dir = os.path.join(tmp, "out")
        os.makedirs(extract_dir)
        extract_h5_to_directory(archive, extract_dir, parallel=8)

        count = 0
        for r, d, files in os.walk(os.path.join(extract_dir, "many")):
            count += len(files)
        assert count == 100

        # Spot check a few files
        for i in [0, 5, 9]:
            for j in [0, 5, 9]:
                p = os.path.join(extract_dir, "many", f"dir{i}", f"file{j}.txt")
                with open(p) as f:
                    assert f.read() == f"content dir{i}/file{j}"
