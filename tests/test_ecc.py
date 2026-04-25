"""Tests for the HAR ECC (Reed-Solomon erasure coding) module."""

import os
import subprocess
import sys
import struct
import tempfile
import tracemalloc

import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, SRC_DIR)
import har_ecc


def _wrap_small(data, level="medium"):
    """Helper: wrap bytes to a temp file, return (wrapped_path, cleanup-fn)."""
    src = tempfile.NamedTemporaryFile(delete=False)
    wrapped = tempfile.NamedTemporaryFile(delete=False)
    src.write(data)
    src.close()
    wrapped.close()
    params = har_ecc.parse_ecc_level(level)
    har_ecc.ecc_wrap(src.name, wrapped.name, params)
    os.unlink(src.name)
    return wrapped.name


def _custom_params(k, p, block_size=1024):
    """Build non-fitted EccParams for precise control in tests."""
    return har_ecc.EccParams(har_ecc.LEVEL_CUSTOM, k, p, block_size=block_size)


class TestParseLevels:
    def test_named_levels(self):
        for name, (code, k, p) in har_ecc._LEVEL_PARAMS.items():
            params = har_ecc.parse_ecc_level(name)
            assert params.k == k
            assert params.p == p
            assert params.level == code

    def test_percentage(self):
        params = har_ecc.parse_ecc_level("10%")
        assert params.k + params.p <= 255
        assert params.p == round(255 * 0.10)

    def test_invalid(self):
        with pytest.raises(ValueError):
            har_ecc.parse_ecc_level("banana")

    def test_case_insensitive(self):
        params = har_ecc.parse_ecc_level("MEDIUM")
        assert params.k == 169


class TestUniformSlots:
    def test_parity_count_per_window(self):
        for k, p in [(10, 2), (179, 10), (127, 63)]:
            kp = k + p
            parity_count = sum(1 for w in range(kp)
                               if har_ecc.is_parity_slot(w, k, p))
            assert parity_count == p

    def test_data_indices_unique(self):
        k, p = 169, 20
        kp = k + p
        data_indices = set()
        parity_indices = set()
        for w in range(kp):
            if har_ecc.is_parity_slot(w, k, p):
                parity_indices.add(har_ecc.window_to_parity_idx(w, k, p))
            else:
                data_indices.add(har_ecc.window_to_data_idx(w, k, p))
        assert data_indices == set(range(k))
        assert parity_indices == set(range(p))


class TestHeaders:
    def test_global_header_roundtrip(self):
        params = har_ecc.EccParams(1, 169, 20)
        hdr = har_ecc._pack_global_header(params, 42, 1234567, b'\x07' * 32,
                                          har_ecc.GENERATOR_PYTHON_ZFEC)
        parsed = har_ecc._unpack_global_header(hdr)
        assert parsed["version"] == 1
        assert parsed["k"] == 169
        assert parsed["p"] == 20
        assert parsed["n_stripes"] == 42
        assert parsed["original_file_size"] == 1234567
        assert parsed["original_hash"] == b'\x07' * 32
        assert parsed["generator"] == har_ecc.GENERATOR_PYTHON_ZFEC

    def test_global_header_crc_mismatch(self):
        params = har_ecc.EccParams(0, 10, 2)
        hdr = bytearray(har_ecc._pack_global_header(params, 1, 100, b'\x00' * 32,
                                                     har_ecc.GENERATOR_PYTHON_ZFEC))
        hdr[20] ^= 0xFF  # corrupt p field
        with pytest.raises(ValueError, match="CRC"):
            har_ecc._unpack_global_header(hdr)

    def test_block_header_roundtrip(self):
        bh = har_ecc._pack_block_header(3, 7, 65536, 0, 0xDEADBEEF)
        parsed = har_ecc._unpack_block_header(bh)
        assert parsed is not None
        assert parsed["stripe_id"] == 3
        assert parsed["block_index"] == 7
        assert parsed["payload_size"] == 65536
        assert parsed["payload_crc"] == 0xDEADBEEF

    def test_block_header_corrupt(self):
        bh = bytearray(har_ecc._pack_block_header(0, 0, 1024, 0, 42))
        bh[8] ^= 0xFF
        assert har_ecc._unpack_block_header(bh) is None


class TestRoundtrip:
    def test_small_file(self):
        with tempfile.NamedTemporaryFile() as src, \
             tempfile.NamedTemporaryFile() as wrapped, \
             tempfile.NamedTemporaryFile() as out:
            data = os.urandom(1000)
            src.write(data)
            src.flush()
            params = har_ecc.parse_ecc_level("medium")
            har_ecc.ecc_wrap(src.name, wrapped.name, params)
            assert har_ecc.is_ecc_wrapped(wrapped.name)
            har_ecc.ecc_unwrap(wrapped.name, out.name)
            assert open(out.name, 'rb').read() == data

    def test_each_level(self):
        for level in ["low", "medium", "high", "max"]:
            with tempfile.NamedTemporaryFile() as src, \
                 tempfile.NamedTemporaryFile() as wrapped, \
                 tempfile.NamedTemporaryFile() as out:
                data = os.urandom(50000)
                src.write(data)
                src.flush()
                params = har_ecc.parse_ecc_level(level)
                har_ecc.ecc_wrap(src.name, wrapped.name, params)
                har_ecc.ecc_unwrap(wrapped.name, out.name)
                assert open(out.name, 'rb').read() == data, f"failed for level {level}"

    def test_exact_block_boundary(self):
        with tempfile.NamedTemporaryFile() as src, \
             tempfile.NamedTemporaryFile() as wrapped, \
             tempfile.NamedTemporaryFile() as out:
            data = os.urandom(65536)  # exactly 1 block
            src.write(data)
            src.flush()
            params = har_ecc.parse_ecc_level("low")
            har_ecc.ecc_wrap(src.name, wrapped.name, params)
            har_ecc.ecc_unwrap(wrapped.name, out.name)
            assert open(out.name, 'rb').read() == data


class TestRepair:
    def test_random_corruption(self):
        with tempfile.NamedTemporaryFile() as src, \
             tempfile.NamedTemporaryFile() as wrapped, \
             tempfile.NamedTemporaryFile() as out:
            data = os.urandom(200000)
            src.write(data)
            src.flush()
            params = har_ecc.parse_ecc_level("high")
            har_ecc.ecc_wrap(src.name, wrapped.name, params)
            # Corrupt a few bytes in the middle
            buf = bytearray(open(wrapped.name, 'rb').read())
            mid = len(buf) // 2
            buf[mid:mid + 500] = bytes(500)
            open(wrapped.name, 'wb').write(buf)
            # Verify detects corruption
            r = har_ecc.ecc_verify(wrapped.name)
            assert r["corrupt_blocks"] > 0
            # Repair
            r2 = har_ecc.ecc_repair(wrapped.name)
            assert r2["hash_ok"]
            assert not r2["unrepairable_stripes"]
            # Unwrap should give back original
            har_ecc.ecc_unwrap(wrapped.name, out.name)
            assert open(out.name, 'rb').read() == data

    def test_exceed_capacity(self):
        with tempfile.NamedTemporaryFile() as src, \
             tempfile.NamedTemporaryFile() as wrapped, \
             tempfile.NamedTemporaryFile() as out:
            data = os.urandom(1000)
            src.write(data)
            src.flush()
            params = har_ecc.parse_ecc_level("low")
            har_ecc.ecc_wrap(src.name, wrapped.name, params)
            # With k=1,p=1 for small file, corrupt both blocks
            buf = bytearray(open(wrapped.name, 'rb').read())
            # Corrupt all block data (after global header)
            for i in range(128, len(buf) - 256):
                buf[i] = 0
            open(wrapped.name, 'wb').write(buf)
            with pytest.raises(ValueError, match="unrecoverable"):
                har_ecc.ecc_unwrap(wrapped.name, out.name)


class TestDetection:
    def test_is_ecc_wrapped(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(har_ecc.GLOBAL_HEADER_MAGIC + b'\x00' * 120)
            f.flush()
            assert har_ecc.is_ecc_wrapped(f.name)

    def test_not_ecc_wrapped(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b'\x89HDF\r\n\x1a\n' + b'\x00' * 120)
            f.flush()
            assert not har_ecc.is_ecc_wrapped(f.name)

    def test_nonexistent(self):
        assert not har_ecc.is_ecc_wrapped("/tmp/nonexistent_file_xyz")


class TestStreamingUnwrap:
    """Exercise the streaming path added in the stripe-by-stripe refactor."""

    def test_unwrap_bounded_memory_200mb(self, tmp_path):
        # Wrap 200 MB of pseudo-random data, then assert the unwrap path never
        # allocates close to the file size. Regression guard for any future
        # change that accidentally reintroduces bytearray(out_data) accumulation.
        import numpy as np
        src = tmp_path / "src.bin"
        wrapped = tmp_path / "wrapped.ecc"
        out = tmp_path / "out.bin"
        rng = np.random.default_rng(1234)
        total_mb = 200
        with open(src, 'wb') as f:
            for _ in range(total_mb):
                f.write(rng.bytes(1024 * 1024))
        params = har_ecc.parse_ecc_level("low")
        har_ecc.ecc_wrap(str(src), str(wrapped), params)
        tracemalloc.start()
        try:
            har_ecc.ecc_unwrap(str(wrapped), str(out))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # kp*block_size ≈ 16 MB worst case; give 4x headroom for pytest overhead.
        assert peak < 64 * 1024 * 1024, f"unwrap peak {peak} exceeds 64 MB"
        assert os.path.getsize(out) == total_mb * 1024 * 1024

    def test_unwrap_exact_stripe_boundary(self, tmp_path):
        # k=10, p=2, block=1024 → one stripe = 10 KB. Use exactly 3 stripes so
        # there is zero padding; verify bytes_written truncation still correct.
        data = os.urandom(3 * 10 * 1024)
        src = tmp_path / "src"; src.write_bytes(data)
        wrapped = tmp_path / "w"; out = tmp_path / "o"
        har_ecc.ecc_wrap(str(src), str(wrapped), _custom_params(10, 2))
        har_ecc.ecc_unwrap(str(wrapped), str(out))
        assert out.read_bytes() == data

    def test_unwrap_file_smaller_than_one_block(self, tmp_path):
        # 17 bytes with default block_size forces fit_to_file to collapse k to 1.
        data = b"hello world 12345"
        src = tmp_path / "src"; src.write_bytes(data)
        wrapped = tmp_path / "w"; out = tmp_path / "o"
        params = har_ecc.parse_ecc_level("low")
        har_ecc.ecc_wrap(str(src), str(wrapped), params)
        har_ecc.ecc_unwrap(str(wrapped), str(out))
        assert out.read_bytes() == data

    def test_unwrap_zero_byte_file(self, tmp_path):
        src = tmp_path / "empty"; src.write_bytes(b"")
        wrapped = tmp_path / "w"; out = tmp_path / "o"
        har_ecc.ecc_wrap(str(src), str(wrapped), har_ecc.parse_ecc_level("low"))
        har_ecc.ecc_unwrap(str(wrapped), str(out))
        assert out.read_bytes() == b""

    def test_unwrap_generator_mismatch_raises(self, tmp_path):
        # Flip the generator byte (offset 92, outside the CRC domain) in every
        # header copy. ecc_unwrap must refuse with a clear message.
        src = tmp_path / "src"; src.write_bytes(os.urandom(5000))
        wrapped = tmp_path / "w"; out = tmp_path / "o"
        har_ecc.ecc_wrap(str(src), str(wrapped), har_ecc.parse_ecc_level("low"))
        size = os.path.getsize(wrapped)
        with open(wrapped, 'r+b') as f:
            for off in (0, size - 256, size - 128):
                f.seek(off + 92)
                f.write(bytes([har_ecc.GENERATOR_RUST_RSE]))
        with pytest.raises(ValueError, match="different generator"):
            har_ecc.ecc_unwrap(str(wrapped), str(out))

    def test_unwrap_primary_header_zeroed_uses_tail(self, tmp_path):
        # Wipe the primary header; tail copies at file_size-256 and -128 must
        # still let _try_read_global_header succeed.
        data = os.urandom(5000)
        src = tmp_path / "src"; src.write_bytes(data)
        wrapped = tmp_path / "w"; out = tmp_path / "o"
        har_ecc.ecc_wrap(str(src), str(wrapped), har_ecc.parse_ecc_level("low"))
        with open(wrapped, 'r+b') as f:
            f.seek(0)
            f.write(b'\x00' * har_ecc.GLOBAL_HEADER_SIZE)
        har_ecc.ecc_unwrap(str(wrapped), str(out))
        assert out.read_bytes() == data

    def test_unwrap_unrepairable_stripe_raises(self, tmp_path):
        # Destroy p+1 blocks' headers within a single stripe so they all fail
        # CRC and the stripe drops below k good shards.
        k, p = 10, 2
        data = os.urandom(3 * k * 1024)  # 3 stripes
        src = tmp_path / "src"; src.write_bytes(data)
        wrapped = tmp_path / "w"; out = tmp_path / "o"
        har_ecc.ecc_wrap(str(src), str(wrapped), _custom_params(k, p))
        file_size = os.path.getsize(wrapped)
        header = har_ecc._unpack_global_header(open(wrapped, 'rb').read(128))
        n_stripes = header["n_stripes"]
        block_size = header["block_size"]
        kp = header["k"] + header["p"]
        # Stripe 0's blocks live at slots w * n_stripes + 0 for w in 0..kp;
        # zero the first p+1 physical slots to kill that stripe.
        with open(wrapped, 'r+b') as f:
            for w in range(p + 1):
                slot = w * n_stripes + 0
                off = har_ecc._slot_offset(slot, block_size)
                f.seek(off)
                f.write(b'\x00' * (har_ecc.BLOCK_HEADER_SIZE + block_size))
        with pytest.raises(ValueError, match="stripe 0"):
            har_ecc.ecc_unwrap(str(wrapped), str(out))

    def test_unwrap_hash_mismatch_detected(self, tmp_path):
        # Overwrite the stored original_hash (with CRC fix-up) in all three
        # header copies; unwrap must fail at the final BLAKE3 check.
        data = os.urandom(5000)
        src = tmp_path / "src"; src.write_bytes(data)
        wrapped = tmp_path / "w"; out = tmp_path / "o"
        har_ecc.ecc_wrap(str(src), str(wrapped), har_ecc.parse_ecc_level("low"))
        size = os.path.getsize(wrapped)
        with open(wrapped, 'r+b') as f:
            f.seek(0)
            hdr = bytearray(f.read(128))
        hdr[56:88] = b'\x00' * 32
        new_crc = har_ecc._crc32c(bytes(hdr[0:88]))
        struct.pack_into("<I", hdr, 88, new_crc)
        with open(wrapped, 'r+b') as f:
            for off in (0, size - 256, size - 128):
                f.seek(off)
                f.write(bytes(hdr))
        with pytest.raises(ValueError, match="hash mismatch"):
            har_ecc.ecc_unwrap(str(wrapped), str(out))


class TestStreamingVerify:
    """Exercise ecc_verify's streaming path and the shape of its result dict."""

    def test_verify_corrupt_block_count_accurate(self, tmp_path):
        # Corrupt exactly 3 blocks' payload bytes (CRC will now fail on those 3).
        # Use k=10 p=2 so the archive has ≥12 slots.
        k, p = 10, 2
        data = os.urandom(k * 1024)  # 1 stripe
        src = tmp_path / "src"; src.write_bytes(data)
        wrapped = tmp_path / "w"
        har_ecc.ecc_wrap(str(src), str(wrapped), _custom_params(k, p))
        header = har_ecc._unpack_global_header(open(wrapped, 'rb').read(128))
        block_size = header["block_size"]
        with open(wrapped, 'r+b') as f:
            for slot in (0, 1, 2):
                off = har_ecc._slot_offset(slot, block_size) + har_ecc.BLOCK_HEADER_SIZE
                f.seek(off)
                f.write(b'\xFF' * 16)  # break payload CRC
        r = har_ecc.ecc_verify(str(wrapped))
        assert r["corrupt_blocks"] == 3

    def test_verify_unrepairable_stripes_listed(self, tmp_path):
        k, p = 10, 2
        data = os.urandom(8 * k * 1024)  # 8 stripes
        src = tmp_path / "src"; src.write_bytes(data)
        wrapped = tmp_path / "w"
        har_ecc.ecc_wrap(str(src), str(wrapped), _custom_params(k, p))
        header = har_ecc._unpack_global_header(open(wrapped, 'rb').read(128))
        n_stripes = header["n_stripes"]
        block_size = header["block_size"]
        target_stripes = [2, 5]
        with open(wrapped, 'r+b') as f:
            for sid in target_stripes:
                for w in range(p + 1):
                    slot = w * n_stripes + sid
                    off = har_ecc._slot_offset(slot, block_size)
                    f.seek(off)
                    f.write(b'\x00' * (har_ecc.BLOCK_HEADER_SIZE + block_size))
        r = har_ecc.ecc_verify(str(wrapped))
        assert sorted(r["unrepairable_stripes"]) == target_stripes
        assert r["hash_ok"] is False

    def test_verify_hash_ok_true_after_full_roundtrip(self, tmp_path):
        wrapped_path = _wrap_small(os.urandom(5000), "medium")
        r = har_ecc.ecc_verify(wrapped_path)
        assert r["hash_ok"] is True
        assert r["unrepairable_stripes"] == []
        os.unlink(wrapped_path)

    def test_verify_hash_ok_false_when_unrepairable(self, tmp_path):
        # Same destruction pattern as test_unwrap_unrepairable_stripe_raises,
        # but verify must NOT raise — it surfaces the state via hash_ok=False.
        k, p = 10, 2
        data = os.urandom(3 * k * 1024)
        src = tmp_path / "src"; src.write_bytes(data)
        wrapped = tmp_path / "w"
        har_ecc.ecc_wrap(str(src), str(wrapped), _custom_params(k, p))
        header = har_ecc._unpack_global_header(open(wrapped, 'rb').read(128))
        n_stripes = header["n_stripes"]
        block_size = header["block_size"]
        with open(wrapped, 'r+b') as f:
            for w in range(p + 1):
                slot = w * n_stripes + 0
                off = har_ecc._slot_offset(slot, block_size)
                f.seek(off)
                f.write(b'\x00' * (har_ecc.BLOCK_HEADER_SIZE + block_size))
        r = har_ecc.ecc_verify(str(wrapped))
        assert r["hash_ok"] is False
        assert 0 in r["unrepairable_stripes"]


class TestHelpers:
    """Direct unit tests on the new streaming helpers."""

    def test_read_stripe_returns_kp_shards(self, tmp_path):
        k, p = 10, 2
        data = os.urandom(2 * k * 1024)
        src = tmp_path / "src"; src.write_bytes(data)
        wrapped = tmp_path / "w"
        har_ecc.ecc_wrap(str(src), str(wrapped), _custom_params(k, p))
        header = har_ecc._unpack_global_header(open(wrapped, 'rb').read(128))
        n_stripes = header["n_stripes"]
        kp = header["k"] + header["p"]
        block_size = header["block_size"]
        with open(wrapped, 'rb') as f:
            shards, corrupt = har_ecc._read_stripe(f, 0, n_stripes, kp, block_size)
        assert len(shards) == kp
        assert corrupt == 0
        assert all(s is not None for s in shards)

    def test_read_stripe_counts_corrupt(self, tmp_path):
        k, p = 10, 2
        data = os.urandom(k * 1024)
        src = tmp_path / "src"; src.write_bytes(data)
        wrapped = tmp_path / "w"
        har_ecc.ecc_wrap(str(src), str(wrapped), _custom_params(k, p))
        header = har_ecc._unpack_global_header(open(wrapped, 'rb').read(128))
        n_stripes = header["n_stripes"]
        kp = header["k"] + header["p"]
        block_size = header["block_size"]
        # Zero the block-header magic of slot 0 so _unpack_block_header returns None.
        with open(wrapped, 'r+b') as f:
            f.seek(har_ecc._slot_offset(0, block_size))
            f.write(b'\x00\x00\x00\x00')
        with open(wrapped, 'rb') as f:
            shards, corrupt = har_ecc._read_stripe(f, 0, n_stripes, kp, block_size)
        assert corrupt == 1
        assert sum(1 for s in shards if s is None) == 1

    def test_make_hasher_matches_blake3_hash(self):
        payload = os.urandom(200_000)
        h = har_ecc._make_hasher()
        # Feed in three unequal chunks to exercise incremental-update semantics.
        h.update(payload[:1])
        h.update(payload[1:99999])
        h.update(payload[99999:])
        assert h.digest() == har_ecc._blake3_hash(payload)


class TestEccOnRealArchive:
    """ECC protects a real BagIt HDF5 archive, not just synthetic random bytes."""

    def _make_bagit(self, tmp_path):
        import har_bagit
        src_dir = tmp_path / "data"
        src_dir.mkdir()
        (src_dir / "a.txt").write_bytes(b"hello from a\n" * 50)
        (src_dir / "b.bin").write_bytes(os.urandom(4096))
        sub = src_dir / "sub"; sub.mkdir()
        (sub / "c.txt").write_bytes(b"nested content\n" * 10)
        archive = tmp_path / "archive.h5"
        har_bagit.pack_bagit([str(src_dir)], str(archive))
        return archive

    def test_wrap_unwrap_roundtrip_real_bagit(self, tmp_path):
        import h5py
        archive = self._make_bagit(tmp_path)
        original_bytes = archive.read_bytes()
        wrapped = tmp_path / "archive.h5.ecc"
        out = tmp_path / "restored.h5"
        har_ecc.ecc_wrap(str(archive), str(wrapped),
                         har_ecc.parse_ecc_level("medium"))
        assert har_ecc.is_ecc_wrapped(str(wrapped))
        har_ecc.ecc_unwrap(str(wrapped), str(out))
        assert out.read_bytes() == original_bytes
        # The unwrapped file must be a valid HDF5 file again.
        with h5py.File(str(out), 'r') as f:
            assert 'batches' in f or 'index_data' in f or 'index' in f

    def test_ecc_wrapped_is_not_hdf5_parseable(self, tmp_path):
        # Locks in the documented behavior: a wrapped archive is NOT HDF5. If
        # this ever flips (e.g. someone moves ECC inside HDF5), the fact needs
        # to be re-evaluated in docs and elsewhere.
        import h5py
        archive = self._make_bagit(tmp_path)
        wrapped = tmp_path / "archive.h5.ecc"
        har_ecc.ecc_wrap(str(archive), str(wrapped),
                         har_ecc.parse_ecc_level("low"))
        with pytest.raises((OSError, IOError)):
            with h5py.File(str(wrapped), 'r'):
                pass


HAR_SCRIPT = os.path.abspath(os.path.join(SRC_DIR, 'har.py'))


def _run_har(*args, check=True):
    """Invoke `python src/har.py <args>` so tests work without `pip install .`."""
    proc = subprocess.run(
        [sys.executable, HAR_SCRIPT, *args],
        capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"har exited {proc.returncode}\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}")
    return proc


class TestCliEcc:
    """End-to-end CLI tests for the --ecc flags (exercises har.py wiring)."""

    def _make_tree(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        (d / "file1.txt").write_bytes(b"one\n" * 100)
        (d / "file2.bin").write_bytes(os.urandom(8000))
        return d

    def test_har_extract_auto_unwraps(self, tmp_path):
        src = self._make_tree(tmp_path)
        archive = tmp_path / "a.h5"
        out_dir = tmp_path / "out"
        _run_har("--ecc", "medium", "-cf", str(archive), str(src))
        assert har_ecc.is_ecc_wrapped(str(archive))
        _run_har("-xf", str(archive), "-C", str(out_dir))
        # Extracted files live under <out_dir>/<basename of src>/...
        restored = out_dir / src.name
        assert (restored / "file1.txt").read_bytes() == (src / "file1.txt").read_bytes()
        assert (restored / "file2.bin").read_bytes() == (src / "file2.bin").read_bytes()

    def test_har_ecc_verify_clean(self, tmp_path):
        src = self._make_tree(tmp_path)
        archive = tmp_path / "a.h5"
        _run_har("--ecc", "low", "-cf", str(archive), str(src))
        proc = _run_har("--ecc-verify", "-f", str(archive), check=False)
        assert proc.returncode == 0, proc.stderr

    def test_har_ecc_verify_corrupt(self, tmp_path):
        src = self._make_tree(tmp_path)
        archive = tmp_path / "a.h5"
        _run_har("--ecc", "low", "-cf", str(archive), str(src))
        # Destroy enough blocks in stripe 0 to make it unrepairable.
        header = har_ecc._unpack_global_header(open(archive, 'rb').read(128))
        n_stripes = header["n_stripes"]
        block_size = header["block_size"]
        p = header["p"]
        with open(archive, 'r+b') as f:
            for w in range(p + 1):
                slot = w * n_stripes + 0
                off = har_ecc._slot_offset(slot, block_size)
                f.seek(off)
                f.write(b'\x00' * (har_ecc.BLOCK_HEADER_SIZE + block_size))
        proc = _run_har("--ecc-verify", "-f", str(archive), check=False)
        assert proc.returncode != 0

    def test_har_heal_round_trip(self, tmp_path):
        src = self._make_tree(tmp_path)
        archive = tmp_path / "a.h5"
        _run_har("--ecc", "medium", "-cf", str(archive), str(src))
        # Knock out p payload blocks (repairable). Use "medium": p=20.
        header = har_ecc._unpack_global_header(open(archive, 'rb').read(128))
        p = header["p"]
        block_size = header["block_size"]
        with open(archive, 'r+b') as f:
            for slot in range(p):
                off = har_ecc._slot_offset(slot, block_size) + har_ecc.BLOCK_HEADER_SIZE
                f.seek(off)
                f.write(b'\x00' * block_size)
        _run_har("--heal", "-f", str(archive))
        out_dir = tmp_path / "restored"
        _run_har("-xf", str(archive), "-C", str(out_dir))
        restored = out_dir / src.name
        assert (restored / "file1.txt").read_bytes() == (src / "file1.txt").read_bytes()
