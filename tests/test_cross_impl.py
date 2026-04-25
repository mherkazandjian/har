"""Cross-implementation smoke tests: archives created by one tool must read
cleanly with the other. Skips gracefully when the Rust binary is absent or
cannot load its shared libraries, so this file is safe to include in CI that
only ships Python."""

import filecmp
import os
import subprocess
import sys

import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
HAR_PY = os.path.join(SRC_DIR, "har.py")

RUST_CANDIDATES = [
    os.path.join(SRC_DIR, "rust", "har-rust-static"),
    os.path.join(SRC_DIR, "rust", "har-rust"),
]


def _pick_rust_binary():
    for path in RUST_CANDIDATES:
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            continue
        # Ensure it actually runs (dynamic builds may be missing libhdf5).
        try:
            proc = subprocess.run([path, "--help"], capture_output=True,
                                  text=True, timeout=5)
            if proc.returncode == 0:
                return path
        except (OSError, subprocess.SubprocessError):
            continue
    return None


RUST_BIN = _pick_rust_binary()
pytestmark = pytest.mark.skipif(
    RUST_BIN is None,
    reason="Rust har binary not available or not runnable in this environment",
)


def _run(cmd, cwd=None):
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if proc.returncode != 0:
        raise AssertionError(
            f"{' '.join(cmd)} exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc


def _run_py(*args, cwd=None):
    return _run([sys.executable, HAR_PY, *args], cwd=cwd)


def _run_rust(*args, cwd=None):
    return _run([RUST_BIN, *args], cwd=cwd)


def _populate(src_dir):
    (src_dir / "a.txt").write_bytes(b"alpha\n" * 200)
    (src_dir / "b.bin").write_bytes(os.urandom(4096))
    sub = src_dir / "nested"
    sub.mkdir()
    (sub / "c.txt").write_bytes(b"deep\n" * 50)


def _compare_trees(a, b):
    a = str(a); b = str(b)
    diff = filecmp.dircmp(a, b)
    mismatches = diff.diff_files + diff.left_only + diff.right_only
    assert not mismatches, f"tree mismatch: {mismatches}"
    for sub in diff.common_dirs:
        _compare_trees(os.path.join(a, sub), os.path.join(b, sub))


class TestLegacyCrossImpl:
    def test_python_create_rust_extract(self, tmp_path):
        src = tmp_path / "src"; src.mkdir(); _populate(src)
        archive = tmp_path / "a.h5"
        out = tmp_path / "out"
        _run_py("-cf", str(archive), str(src))
        _run_rust("-xf", str(archive), "-C", str(out))
        _compare_trees(src, out / src.name)

    def test_rust_create_python_extract(self, tmp_path):
        src = tmp_path / "src"; src.mkdir(); _populate(src)
        archive = tmp_path / "a.h5"
        out = tmp_path / "out"
        _run_rust("-cf", str(archive), str(src))
        _run_py("-xf", str(archive), "-C", str(out))
        _compare_trees(src, out / src.name)


class TestBagitCrossImpl:
    # The two BagIt writers still produce different on-disk layouts
    # (Python: compound /index dataset, Rust: /index_data/ parallel arrays)
    # but both readers now accept either layout. These tests lock in that
    # compatibility.
    def test_python_create_rust_extract(self, tmp_path):
        src = tmp_path / "src"; src.mkdir(); _populate(src)
        archive = tmp_path / "a.h5"
        out = tmp_path / "out"
        _run_py("--bagit", "-cf", str(archive), str(src))
        _run_rust("-xf", str(archive), "-C", str(out))
        _compare_trees(src, out / src.name)

    def test_rust_create_python_extract(self, tmp_path):
        src = tmp_path / "src"; src.mkdir(); _populate(src)
        archive = tmp_path / "a.h5"
        out = tmp_path / "out"
        _run_rust("--bagit", "-cf", str(archive), str(src))
        _run_py("-xf", str(archive), "-C", str(out))
        _compare_trees(src, out / src.name)

    def test_python_listing_rust_archive(self, tmp_path):
        # Reader path exercised by `har -tf` (lists contents, no extract).
        src = tmp_path / "src"; src.mkdir(); _populate(src)
        archive = tmp_path / "a.h5"
        _run_rust("--bagit", "-cf", str(archive), str(src))
        proc = _run_py("-tf", str(archive))
        for name in ("a.txt", "b.bin", "nested/c.txt"):
            assert name in proc.stdout, f"missing {name} in:\n{proc.stdout}"

    def test_large_single_file_cross_extract(self, tmp_path):
        # A file big enough to stress the synthesized chunk_offset=0 path
        # when Rust reads a Python-written archive (Python doesn't emit
        # chunk_offset at all).
        src = tmp_path / "src"; src.mkdir()
        big = src / "big.bin"
        big.write_bytes(os.urandom(3 * 1024 * 1024))
        archive = tmp_path / "a.h5"
        out = tmp_path / "out"
        _run_py("--bagit", "-cf", str(archive), str(src))
        _run_rust("-xf", str(archive), "-C", str(out))
        assert (out / src.name / "big.bin").read_bytes() == big.read_bytes()
