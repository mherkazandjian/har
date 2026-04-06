import os
import stat
import tempfile
import hashlib
import h5py
import numpy as np
import pytest

from har_bagit import (
    pack_bagit, extract_bagit, list_bagit, build_inventory,
    is_bagit_archive, parse_batch_size, HAR_FORMAT_VALUE, INDEX_DTYPE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_tree(base, n_dirs=3, n_files_per_dir=4):
    """Create a test directory tree."""
    for i in range(n_dirs):
        d = os.path.join(base, f"dir{i}")
        os.makedirs(d)
        for j in range(n_files_per_dir):
            with open(os.path.join(d, f"file{j}.txt"), "w") as f:
                f.write(f"content dir{i}/file{j}")


def _sha256(path):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _count_files(root):
    """Count files recursively."""
    count = 0
    for _, _, files in os.walk(root):
        count += len(files)
    return count


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

@pytest.fixture
def test_env():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        with open(os.path.join(src, "file1.txt"), "w") as f:
            f.write("This is file 1.")
        sub = os.path.join(src, "subdir")
        os.makedirs(sub)
        with open(os.path.join(sub, "file2.txt"), "w") as f:
            f.write("This is file 2 in subdir.")

        archive = os.path.join(tmp, "archive.h5")
        extract_dir = os.path.join(tmp, "extract")
        os.makedirs(extract_dir)

        yield {
            'src_dir': src,
            'archive_path': archive,
            'extract_dir': extract_dir,
            'tmp': tmp,
        }


def test_archive_creation(test_env):
    pack_bagit([test_env['src_dir']], test_env['archive_path'])
    assert os.path.exists(test_env['archive_path'])
    assert is_bagit_archive(test_env['archive_path'])

    with h5py.File(test_env['archive_path'], 'r') as h5f:
        assert h5f.attrs['har_format'] == HAR_FORMAT_VALUE
        assert 'index' in h5f
        assert 'batches' in h5f
        assert 'bagit' in h5f
        index = h5f['index'][()]
        paths = sorted(r['path'].decode('utf-8') for r in index)
        assert 'data/src/file1.txt' in paths
        assert 'data/src/subdir/file2.txt' in paths


def test_roundtrip(test_env):
    pack_bagit([test_env['src_dir']], test_env['archive_path'])
    extract_bagit(test_env['archive_path'], test_env['extract_dir'])

    f1 = os.path.join(test_env['extract_dir'], "src", "file1.txt")
    assert os.path.exists(f1)
    assert open(f1).read() == "This is file 1."

    f2 = os.path.join(test_env['extract_dir'], "src", "subdir", "file2.txt")
    assert os.path.exists(f2)
    assert open(f2).read() == "This is file 2 in subdir."


def test_extract_single_file(test_env):
    pack_bagit([test_env['src_dir']], test_env['archive_path'])
    extract_bagit(test_env['archive_path'], test_env['extract_dir'],
                  file_key="src/file1.txt")

    f1 = os.path.join(test_env['extract_dir'], "src", "file1.txt")
    assert os.path.exists(f1)
    assert open(f1).read() == "This is file 1."

    f2 = os.path.join(test_env['extract_dir'], "src", "subdir", "file2.txt")
    assert not os.path.exists(f2)


def test_list_contents(test_env):
    pack_bagit([test_env['src_dir']], test_env['archive_path'])
    # Just verify it runs without error
    list_bagit(test_env['archive_path'])


def test_validate_pass(test_env):
    pack_bagit([test_env['src_dir']], test_env['archive_path'])
    extract_bagit(test_env['archive_path'], test_env['extract_dir'], validate=True)

    f1 = os.path.join(test_env['extract_dir'], "src", "file1.txt")
    assert open(f1).read() == "This is file 1."


def test_validate_detects_corruption(test_env):
    pack_bagit([test_env['src_dir']], test_env['archive_path'])

    # Corrupt a batch by flipping a byte
    with h5py.File(test_env['archive_path'], 'r+') as h5f:
        batch = h5f['batches/0']
        data = batch[()]
        if len(data) > 0:
            data[0] = np.uint8((int(data[0]) + 1) % 256)
            batch[...] = data

    with pytest.raises(SystemExit):
        extract_bagit(test_env['archive_path'], test_env['extract_dir'], validate=True)


# ---------------------------------------------------------------------------
# BagIt manifest tests
# ---------------------------------------------------------------------------

def test_manifest_format(test_env):
    pack_bagit([test_env['src_dir']], test_env['archive_path'])

    with h5py.File(test_env['archive_path'], 'r') as h5f:
        manifest = h5f['bagit/manifest-sha256.txt'][()].tobytes().decode('utf-8')
        lines = [l for l in manifest.strip().split('\n') if l]
        for line in lines:
            parts = line.split('  ', 1)
            assert len(parts) == 2, f"Bad manifest line: {line}"
            sha, path = parts
            assert len(sha) == 64, f"Bad sha256 length: {sha}"
            assert path.startswith('data/'), f"Path not under data/: {path}"


def test_bagit_txt_content(test_env):
    pack_bagit([test_env['src_dir']], test_env['archive_path'])

    with h5py.File(test_env['archive_path'], 'r') as h5f:
        txt = h5f['bagit/bagit.txt'][()].tobytes().decode('utf-8')
        assert "BagIt-Version: 1.0" in txt
        assert "Tag-File-Character-Encoding: UTF-8" in txt


def test_bag_info_payload_oxum(test_env):
    pack_bagit([test_env['src_dir']], test_env['archive_path'])

    with h5py.File(test_env['archive_path'], 'r') as h5f:
        info = h5f['bagit/bag-info.txt'][()].tobytes().decode('utf-8')
        assert "Payload-Oxum:" in info
        # Oxum format: total_bytes.file_count
        for line in info.split('\n'):
            if line.startswith('Payload-Oxum:'):
                oxum = line.split(':')[1].strip()
                parts = oxum.split('.')
                assert len(parts) == 2
                total_bytes = int(parts[0])
                file_count = int(parts[1])
                assert file_count == 2  # file1.txt + file2.txt
                assert total_bytes == len("This is file 1.") + len("This is file 2 in subdir.")


def test_tagmanifest_covers_tag_files(test_env):
    pack_bagit([test_env['src_dir']], test_env['archive_path'])

    with h5py.File(test_env['archive_path'], 'r') as h5f:
        tm = h5f['bagit/tagmanifest-sha256.txt'][()].tobytes().decode('utf-8')
        assert 'bagit.txt' in tm
        assert 'bag-info.txt' in tm
        assert 'manifest-sha256.txt' in tm


# ---------------------------------------------------------------------------
# Bagit-raw extraction
# ---------------------------------------------------------------------------

def test_bagit_raw_extraction(test_env):
    pack_bagit([test_env['src_dir']], test_env['archive_path'])
    extract_bagit(test_env['archive_path'], test_env['extract_dir'], bagit_raw=True)

    # Tag files should exist at top level
    assert os.path.exists(os.path.join(test_env['extract_dir'], "bagit.txt"))
    assert os.path.exists(os.path.join(test_env['extract_dir'], "bag-info.txt"))
    assert os.path.exists(os.path.join(test_env['extract_dir'], "manifest-sha256.txt"))

    # Payload under data/
    f1 = os.path.join(test_env['extract_dir'], "data", "src", "file1.txt")
    assert os.path.exists(f1)
    assert open(f1).read() == "This is file 1."


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_directory_preserved():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "project")
        os.makedirs(os.path.join(src, "empty_sub"))
        os.makedirs(os.path.join(src, "nonempty"))
        with open(os.path.join(src, "nonempty", "f.txt"), "w") as f:
            f.write("data")

        archive = os.path.join(tmp, "empty.h5")
        pack_bagit([src], archive)

        out = os.path.join(tmp, "out")
        os.makedirs(out)
        extract_bagit(archive, out)

        assert os.path.isdir(os.path.join(out, "project", "empty_sub"))
        assert os.path.isfile(os.path.join(out, "project", "nonempty", "f.txt"))


def test_file_permissions_preserved():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "perms")
        os.makedirs(src)
        exe = os.path.join(src, "run.sh")
        with open(exe, "w") as f:
            f.write("#!/bin/bash\necho hi")
        os.chmod(exe, 0o755)

        ro = os.path.join(src, "readonly.txt")
        with open(ro, "w") as f:
            f.write("read only")
        os.chmod(ro, 0o444)

        archive = os.path.join(tmp, "perms.h5")
        pack_bagit([src], archive)

        out = os.path.join(tmp, "out")
        os.makedirs(out)
        extract_bagit(archive, out)

        assert os.stat(os.path.join(out, "perms", "run.sh")).st_mode == os.stat(exe).st_mode
        assert os.stat(os.path.join(out, "perms", "readonly.txt")).st_mode == os.stat(ro).st_mode


def test_empty_file_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "emptysrc")
        os.makedirs(src)
        open(os.path.join(src, "empty.txt"), "w").close()

        archive = os.path.join(tmp, "empty.h5")
        pack_bagit([src], archive)

        out = os.path.join(tmp, "out")
        os.makedirs(out)
        extract_bagit(archive, out)

        extracted = os.path.join(out, "emptysrc", "empty.txt")
        assert os.path.exists(extracted)
        assert os.path.getsize(extracted) == 0


def test_binary_file_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "binsrc")
        os.makedirs(src)
        with open(os.path.join(src, "allbytes.bin"), "wb") as f:
            f.write(bytes(range(256)))

        archive = os.path.join(tmp, "bin.h5")
        pack_bagit([src], archive)

        out = os.path.join(tmp, "out")
        os.makedirs(out)
        extract_bagit(archive, out)

        with open(os.path.join(out, "binsrc", "allbytes.bin"), "rb") as f:
            assert f.read() == bytes(range(256))


def test_compression_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "comp")
        os.makedirs(src)
        with open(os.path.join(src, "data.txt"), "w") as f:
            f.write("A" * 100000)

        archive = os.path.join(tmp, "comp.h5")
        pack_bagit([src], archive, compression='gzip', compression_opts=9, shuffle=True)

        out = os.path.join(tmp, "out")
        os.makedirs(out)
        extract_bagit(archive, out)

        assert open(os.path.join(out, "comp", "data.txt")).read() == "A" * 100000
        # Batched archive has index overhead; check it's much smaller than 100KB payload
        assert os.path.getsize(archive) < 20000


def test_multi_directory_no_collision():
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
        pack_bagit([dir_a, dir_b], archive)

        out = os.path.join(tmp, "out")
        os.makedirs(out)
        extract_bagit(archive, out)

        assert open(os.path.join(out, "dirA", "readme.txt")).read() == "from A"
        assert open(os.path.join(out, "dirB", "readme.txt")).read() == "from B"


# ---------------------------------------------------------------------------
# Batching tests
# ---------------------------------------------------------------------------

def test_batch_size_splitting():
    """Files should be split into multiple batches when exceeding batch_size."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "big")
        os.makedirs(src)
        # Create files totaling > 1KB
        for i in range(10):
            with open(os.path.join(src, f"file{i}.txt"), "w") as f:
                f.write("x" * 200)

        archive = os.path.join(tmp, "batched.h5")
        pack_bagit([src], archive, batch_size=500)  # 500 bytes per batch

        with h5py.File(archive, 'r') as h5f:
            n_batches = len(h5f['batches'])
            assert n_batches > 1, f"Expected multiple batches, got {n_batches}"

        out = os.path.join(tmp, "out")
        os.makedirs(out)
        extract_bagit(archive, out)
        for i in range(10):
            p = os.path.join(out, "big", f"file{i}.txt")
            assert open(p).read() == "x" * 200


def test_large_file_own_batch():
    """A single file larger than batch_size gets its own batch."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "mixed")
        os.makedirs(src)
        with open(os.path.join(src, "small.txt"), "w") as f:
            f.write("tiny")
        with open(os.path.join(src, "big.bin"), "wb") as f:
            f.write(b"X" * 10000)

        archive = os.path.join(tmp, "mixed.h5")
        pack_bagit([src], archive, batch_size=1000)

        out = os.path.join(tmp, "out")
        os.makedirs(out)
        extract_bagit(archive, out)

        assert open(os.path.join(out, "mixed", "small.txt")).read() == "tiny"
        with open(os.path.join(out, "mixed", "big.bin"), "rb") as f:
            assert f.read() == b"X" * 10000


# ---------------------------------------------------------------------------
# Parallel tests
# ---------------------------------------------------------------------------

def test_parallel_pack_and_extract():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        _make_test_tree(src, n_dirs=5, n_files_per_dir=10)

        archive = os.path.join(tmp, "par.h5")
        pack_bagit([src], archive, parallel=4)

        out = os.path.join(tmp, "out")
        os.makedirs(out)
        extract_bagit(archive, out, parallel=4)

        count = _count_files(os.path.join(out, "src"))
        assert count == 50

        for i in range(5):
            for j in range(10):
                p = os.path.join(out, "src", f"dir{i}", f"file{j}.txt")
                assert open(p).read() == f"content dir{i}/file{j}"


def test_parallel_with_validation():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        _make_test_tree(src)

        archive = os.path.join(tmp, "val.h5")
        pack_bagit([src], archive, parallel=2)

        out = os.path.join(tmp, "out")
        os.makedirs(out)
        extract_bagit(archive, out, validate=True, parallel=2)


# ---------------------------------------------------------------------------
# Many files stress test
# ---------------------------------------------------------------------------

def test_many_small_files():
    """Stress: 1000 files across 10 dirs."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "many")
        os.makedirs(src)
        _make_test_tree(src, n_dirs=10, n_files_per_dir=100)

        archive = os.path.join(tmp, "many.h5")
        pack_bagit([src], archive, parallel=4)

        out = os.path.join(tmp, "out")
        os.makedirs(out)
        extract_bagit(archive, out, parallel=4)

        count = _count_files(os.path.join(out, "many"))
        assert count == 1000

        # Spot check
        for i in [0, 5, 9]:
            for j in [0, 50, 99]:
                p = os.path.join(out, "many", f"dir{i}", f"file{j}.txt")
                assert open(p).read() == f"content dir{i}/file{j}"


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

def test_parse_batch_size():
    assert parse_batch_size("64M") == 64 * 1024 * 1024
    assert parse_batch_size("1G") == 1024 * 1024 * 1024
    assert parse_batch_size("512K") == 512 * 1024
    assert parse_batch_size("1000") == 1000


def test_is_bagit_archive():
    with tempfile.TemporaryDirectory() as tmp:
        # Create a legacy archive
        legacy = os.path.join(tmp, "legacy.h5")
        with h5py.File(legacy, 'w') as h5f:
            h5f.create_dataset("test", data=[1, 2, 3])
        assert not is_bagit_archive(legacy)

        # Create a bagit archive
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        with open(os.path.join(src, "f.txt"), "w") as f:
            f.write("test")
        bagit_arch = os.path.join(tmp, "bagit.h5")
        pack_bagit([src], bagit_arch)
        assert is_bagit_archive(bagit_arch)


def test_error_nonexistent_archive():
    with pytest.raises(SystemExit):
        extract_bagit("/tmp/nonexistent_har_bagit_test.h5", "/tmp")
    with pytest.raises(SystemExit):
        list_bagit("/tmp/nonexistent_har_bagit_test.h5")
