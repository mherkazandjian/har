"""
HAR ECC module - Reed-Solomon erasure coding container for self-healing archives.

This module wraps a finished .h5 file (or any byte stream) in a self-describing
Reed-Solomon container with uniformly-distributed parity blocks and cross-stripe
interleaving. It mirrors the Rust implementation in src/rust/src/ecc.rs.

IMPORTANT: Although the Python and Rust implementations share the same container
byte layout (global header, block headers, slot placement), the parity bytes
themselves are NOT cross-compatible: zfec and reed-solomon-erasure use different
Reed-Solomon matrix conventions. A Python-wrapped archive must be unwrapped by
Python, and a Rust-wrapped archive must be unwrapped by Rust. Detection of the
mismatch is automatic via the generator identifier byte in the global header.
"""

import os
import sys
import struct
import hashlib
import zfec

try:
    import zlib
    _crc32c_fn = None
    try:
        import crc32c as _crc32c_mod
        _crc32c_fn = _crc32c_mod.crc32c
    except ImportError:
        # Fall back to zlib.crc32 (CRC-32 Ethernet, NOT CRC-32C).
        # This is a different polynomial but still a valid checksum - just not
        # interoperable with the Rust side at the CRC level. The block payload
        # hash and BLAKE3 whole-file hash remain authoritative, so integrity is
        # still enforced end-to-end.
        _crc32c_fn = zlib.crc32
except ImportError:
    _crc32c_fn = None


def _crc32c(data):
    return _crc32c_fn(data) & 0xFFFFFFFF


# BLAKE3 via blake3 package, fall back to hashlib's blake2b if missing.
try:
    import blake3 as _blake3_mod

    def _blake3_hash(data):
        return _blake3_mod.blake3(data).digest()
except ImportError:
    def _blake3_hash(data):
        # Not a true BLAKE3; only used if the blake3 package is unavailable.
        # Cross-compat with Rust will fail in this fallback path.
        return hashlib.blake2b(data, digest_size=32).digest()


# ---------------------------------------------------------------------------
# Format constants (must match src/rust/src/ecc.rs)
# ---------------------------------------------------------------------------

GLOBAL_HEADER_MAGIC = b"\x89HARECC\n"
BLOCK_MAGIC = b"\xECBK\x00"
GLOBAL_HEADER_SIZE = 128
BLOCK_HEADER_SIZE = 32
DEFAULT_BLOCK_PAYLOAD = 65536
FORMAT_VERSION = 1
MAX_SHARDS = 255

# Generator identifier: 0 = reed-solomon-erasure (Rust), 1 = zfec (Python).
# The wrapping tool stamps its value; the unwrapping tool must match.
GENERATOR_RUST_RSE = 0
GENERATOR_PYTHON_ZFEC = 1

LEVEL_LOW = 0
LEVEL_MEDIUM = 1
LEVEL_HIGH = 2
LEVEL_MAX = 3
LEVEL_CUSTOM = 255

_LEVEL_NAMES = {
    LEVEL_LOW: "low",
    LEVEL_MEDIUM: "medium",
    LEVEL_HIGH: "high",
    LEVEL_MAX: "max",
    LEVEL_CUSTOM: "custom",
}

# (level_code, k, p)
_LEVEL_PARAMS = {
    "low":    (LEVEL_LOW,    179, 10),
    "medium": (LEVEL_MEDIUM, 169, 20),
    "high":   (LEVEL_HIGH,   149, 40),
    "max":    (LEVEL_MAX,    127, 63),
}


class EccParams:
    __slots__ = ("level", "k", "p", "block_size")

    def __init__(self, level, k, p, block_size=DEFAULT_BLOCK_PAYLOAD):
        self.level = level
        self.k = k
        self.p = p
        self.block_size = block_size

    def fit_to_file(self, file_size):
        """Reduce k if the file is smaller than k blocks, preserving p/k ratio."""
        needed = max(1, (file_size + self.block_size - 1) // self.block_size)
        if needed >= self.k:
            return EccParams(self.level, self.k, self.p, self.block_size)
        new_k = needed
        new_p = max(1, round(self.p * new_k / self.k))
        new_p = min(new_p, MAX_SHARDS - new_k)
        return EccParams(self.level, new_k, new_p, self.block_size)


def parse_ecc_level(s):
    """Parse 'low'/'medium'/'high'/'max' or 'N%' into EccParams."""
    if s is None:
        raise ValueError("ecc level is required")
    s = s.strip().lower()
    if s in _LEVEL_PARAMS:
        code, k, p = _LEVEL_PARAMS[s]
        return EccParams(code, k, p)
    if s.endswith('%'):
        try:
            pct = float(s[:-1])
        except ValueError:
            raise ValueError(f"ecc: invalid percent: {s}")
        if pct <= 0 or pct >= 100:
            raise ValueError(f"ecc: percent must be in (0,100): {s}")
        p = max(1, round(MAX_SHARDS * pct / 100.0))
        p = min(p, MAX_SHARDS - 1)
        k = MAX_SHARDS - p
        return EccParams(LEVEL_CUSTOM, k, p)
    # Bare number → treat as percentage
    try:
        pct = float(s)
        if pct <= 0 or pct >= 100:
            raise ValueError(f"ecc: percent must be in (0,100): {s}")
        p = max(1, round(MAX_SHARDS * pct / 100.0))
        p = min(p, MAX_SHARDS - 1)
        k = MAX_SHARDS - p
        return EccParams(LEVEL_CUSTOM, k, p)
    except ValueError:
        pass
    raise ValueError(f"ecc: unknown level '{s}' (expected low/medium/high/max/N%/N)")


# ---------------------------------------------------------------------------
# Uniform slot distribution (Bresenham-like)
# ---------------------------------------------------------------------------

def is_parity_slot(w, k, p):
    kp = k + p
    return (w * p) // kp < ((w + 1) * p) // kp


def window_to_parity_idx(w, k, p):
    return ((w + 1) * p) // (k + p) - 1


def window_to_data_idx(w, k, p):
    return w - (w * p) // (k + p)


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

def _pack_global_header(params, n_stripes, original_file_size, original_hash,
                       generator):
    # Layout matches Rust ecc.rs GlobalHeader:
    # [0..8]   magic
    # [8..12]  version (u32 LE)
    # [12..16] level_code (u32 LE)
    # [16..20] k (u32 LE)
    # [20..24] p (u32 LE)
    # [24..32] n_stripes (u64 LE)
    # [32..40] block_payload_size (u64 LE)
    # [40..48] original_file_size (u64 LE)
    # [48..56] total_blocks (u64 LE)
    # [56..88] original_hash (32 bytes, BLAKE3)
    # [88..92] CRC32C of [0..88)
    # [92]     generator id
    # [93..128] reserved (zero)
    total_blocks = n_stripes * (params.k + params.p)
    buf = bytearray(GLOBAL_HEADER_SIZE)
    buf[0:8] = GLOBAL_HEADER_MAGIC
    struct.pack_into("<I", buf, 8, FORMAT_VERSION)
    struct.pack_into("<I", buf, 12, params.level)
    struct.pack_into("<I", buf, 16, params.k)
    struct.pack_into("<I", buf, 20, params.p)
    struct.pack_into("<Q", buf, 24, n_stripes)
    struct.pack_into("<Q", buf, 32, params.block_size)
    struct.pack_into("<Q", buf, 40, original_file_size)
    struct.pack_into("<Q", buf, 48, total_blocks)
    buf[56:88] = original_hash
    crc = _crc32c(bytes(buf[0:88]))
    struct.pack_into("<I", buf, 88, crc)
    buf[92] = generator & 0xFF
    return bytes(buf)


def _unpack_global_header(buf):
    if len(buf) < GLOBAL_HEADER_SIZE:
        raise ValueError("ECC: global header too short")
    if bytes(buf[0:8]) != GLOBAL_HEADER_MAGIC:
        raise ValueError("ECC: wrong magic")
    expected_crc = struct.unpack_from("<I", buf, 88)[0]
    actual_crc = _crc32c(bytes(buf[0:88]))
    if expected_crc != actual_crc:
        raise ValueError("ECC: global header CRC mismatch")
    return {
        "version": struct.unpack_from("<I", buf, 8)[0],
        "level": struct.unpack_from("<I", buf, 12)[0],
        "k": struct.unpack_from("<I", buf, 16)[0],
        "p": struct.unpack_from("<I", buf, 20)[0],
        "n_stripes": struct.unpack_from("<Q", buf, 24)[0],
        "block_size": struct.unpack_from("<Q", buf, 32)[0],
        "original_file_size": struct.unpack_from("<Q", buf, 40)[0],
        "total_blocks": struct.unpack_from("<Q", buf, 48)[0],
        "original_hash": bytes(buf[56:88]),
        "generator": buf[92],
    }


def _pack_block_header(stripe_id, block_index, payload_size, flags,
                       payload_crc):
    # Layout matches Rust ecc.rs BlockHeader: all u32 LE fields.
    # [0..4] magic, [4..8] stripe_id, [8..12] block_index,
    # [12..16] payload_size, [16..20] flags, [20..24] payload_crc,
    # [24..28] header_crc, [28..32] reserved
    buf = bytearray(BLOCK_HEADER_SIZE)
    buf[0:4] = BLOCK_MAGIC
    struct.pack_into("<I", buf, 4, stripe_id)
    struct.pack_into("<I", buf, 8, block_index)
    struct.pack_into("<I", buf, 12, payload_size)
    struct.pack_into("<I", buf, 16, flags)
    struct.pack_into("<I", buf, 20, payload_crc)
    hdr_crc = _crc32c(bytes(buf[0:24]))
    struct.pack_into("<I", buf, 24, hdr_crc)
    return bytes(buf)


def _unpack_block_header(buf):
    if bytes(buf[0:4]) != BLOCK_MAGIC:
        return None
    expected = struct.unpack_from("<I", buf, 24)[0]
    if _crc32c(bytes(buf[0:24])) != expected:
        return None
    return {
        "stripe_id": struct.unpack_from("<I", buf, 4)[0],
        "block_index": struct.unpack_from("<I", buf, 8)[0],
        "payload_size": struct.unpack_from("<I", buf, 12)[0],
        "flags": struct.unpack_from("<I", buf, 16)[0],
        "payload_crc": struct.unpack_from("<I", buf, 20)[0],
    }


def _slot_offset(slot_idx, block_size):
    return GLOBAL_HEADER_SIZE + slot_idx * (BLOCK_HEADER_SIZE + block_size)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_ecc_wrapped(path):
    try:
        with open(path, "rb") as f:
            return f.read(len(GLOBAL_HEADER_MAGIC)) == GLOBAL_HEADER_MAGIC
    except OSError:
        return False


def ecc_wrap(input_path, output_path, params, verbose=False):
    """Wrap a file with Reed-Solomon parity. Returns the number of stripes."""
    with open(input_path, "rb") as f:
        data = f.read()
    file_size = len(data)

    params = params.fit_to_file(file_size)
    original_hash = _blake3_hash(data)

    bytes_per_stripe = params.k * params.block_size
    n_stripes = max(1, (file_size + bytes_per_stripe - 1) // bytes_per_stripe)

    # Pad to exact stripe boundary
    padded_len = n_stripes * bytes_per_stripe
    if padded_len > file_size:
        data = data + bytes(padded_len - file_size)

    kp = params.k + params.p
    total_blocks = n_stripes * kp
    encoder = zfec.Encoder(params.k, kp)

    # Build a map: slot_idx -> (block_header_bytes, payload_bytes)
    # Pre-compute all slots and their contents. We encode stripe-by-stripe.
    slot_out = [None] * total_blocks

    for stripe_id in range(n_stripes):
        stripe_off = stripe_id * bytes_per_stripe
        data_blocks = [
            data[stripe_off + d * params.block_size:
                 stripe_off + (d + 1) * params.block_size]
            for d in range(params.k)
        ]
        # zfec.encode returns k+p blocks: first k primary (data), then m-k secondary (parity)
        encoded = encoder.encode(data_blocks)
        # Map physical slots within this stripe
        for window in range(kp):
            slot_idx = window * n_stripes + stripe_id
            if is_parity_slot(window, params.k, params.p):
                pi = window_to_parity_idx(window, params.k, params.p)
                shard_idx = params.k + pi
                block_index = shard_idx  # block_index in stripe: k..k+p-1
                block_type = 1
            else:
                di = window_to_data_idx(window, params.k, params.p)
                shard_idx = di
                block_index = di
                block_type = 0
            payload = bytes(encoded[shard_idx])
            payload_crc = _crc32c(payload)
            bh = _pack_block_header(
                stripe_id, block_index, params.block_size, 0,
                payload_crc)
            slot_out[slot_idx] = (bh, payload)

    # Write output
    with open(output_path, "wb") as out:
        # Primary global header
        hdr = _pack_global_header(params, n_stripes, file_size, original_hash,
                                 GENERATOR_PYTHON_ZFEC)
        out.write(hdr)
        for slot_idx in range(total_blocks):
            bh, payload = slot_out[slot_idx]
            out.write(bh)
            out.write(payload)
        # Tail copy 1 (primary tail)
        out.write(hdr)
        # Tail copy 2 (last 128 bytes)
        out.write(hdr)

    if verbose:
        print(f"ECC wrap: {n_stripes} stripes, k={params.k}, p={params.p}, "
              f"block_size={params.block_size}, total_blocks={total_blocks}")

    return n_stripes


def _try_read_global_header(f, file_size):
    # Primary at offset 0
    f.seek(0)
    buf = f.read(GLOBAL_HEADER_SIZE)
    try:
        return _unpack_global_header(buf)
    except ValueError:
        pass
    # Tail copy 1: file_size - 256
    if file_size >= GLOBAL_HEADER_SIZE * 2:
        f.seek(file_size - GLOBAL_HEADER_SIZE * 2)
        buf = f.read(GLOBAL_HEADER_SIZE)
        try:
            return _unpack_global_header(buf)
        except ValueError:
            pass
    # Tail copy 2: file_size - 128
    if file_size >= GLOBAL_HEADER_SIZE:
        f.seek(file_size - GLOBAL_HEADER_SIZE)
        buf = f.read(GLOBAL_HEADER_SIZE)
        try:
            return _unpack_global_header(buf)
        except ValueError:
            pass
    raise ValueError("ECC: no valid global header found")


def _make_hasher():
    """Return an incremental hasher compatible with _blake3_hash (supports .update/.digest)."""
    try:
        import blake3 as _b3
        return _b3.blake3()
    except ImportError:
        return hashlib.blake2b(digest_size=32)


def _read_stripe(f, sid, n_stripes, kp, block_size):
    """Read all kp blocks for stripe `sid` by seeking; return (shards, corrupt_count).

    Each stripe's blocks are at physical slots w*n_stripes+sid for w in 0..kp,
    so this does kp seeks rather than loading the full file. Peak memory is
    kp*block_size (≤255*64 KB ≈ 16 MB) regardless of archive size.
    """
    shards = [None] * kp
    corrupt = 0
    for w in range(kp):
        slot_idx = w * n_stripes + sid
        off = _slot_offset(slot_idx, block_size)
        f.seek(off)
        bh_bytes = f.read(BLOCK_HEADER_SIZE)
        payload = f.read(block_size)
        if len(bh_bytes) < BLOCK_HEADER_SIZE or len(payload) < block_size:
            corrupt += 1
            continue
        bh = _unpack_block_header(bh_bytes)
        if bh is None:
            corrupt += 1
            continue
        if _crc32c(payload) != bh["payload_crc"]:
            corrupt += 1
            continue
        bi = bh["block_index"]
        if 0 <= bi < kp:
            shards[bi] = payload
        else:
            corrupt += 1
    return shards, corrupt


def _scan_stripes(f, header):
    """Read all blocks; return (stripes, corrupt_slots) where each stripe is a
    list of k+p optional payload bytes indexed by block_index, and corrupt_slots
    is a list of slot indices whose header or payload CRC failed."""
    k = header["k"]
    p = header["p"]
    kp = k + p
    n_stripes = header["n_stripes"]
    block_size = header["block_size"]
    total_blocks = header["total_blocks"]
    assert total_blocks == n_stripes * kp

    stripes = [[None] * kp for _ in range(n_stripes)]
    corrupt = []
    for slot_idx in range(total_blocks):
        off = _slot_offset(slot_idx, block_size)
        f.seek(off)
        bh_bytes = f.read(BLOCK_HEADER_SIZE)
        payload = f.read(block_size)
        if len(bh_bytes) < BLOCK_HEADER_SIZE or len(payload) < block_size:
            corrupt.append(slot_idx)
            continue
        bh = _unpack_block_header(bh_bytes)
        if bh is None:
            corrupt.append(slot_idx)
            continue
        if _crc32c(payload) != bh["payload_crc"]:
            corrupt.append(slot_idx)
            continue
        sid = bh["stripe_id"]
        bi = bh["block_index"]
        if sid >= n_stripes or bi >= kp:
            corrupt.append(slot_idx)
            continue
        stripes[sid][bi] = payload
    return stripes, corrupt


def _reconstruct_stripe(shards, k, kp):
    """Reconstruct missing shards using zfec. Returns True on success."""
    good_indices = [i for i, s in enumerate(shards) if s is not None]
    if len(good_indices) < k:
        return False
    if len(good_indices) >= k and all(shards[i] is not None for i in range(kp)):
        return True  # nothing missing
    # Use any k good shards to decode the k primary blocks
    use = good_indices[:k]
    use_blocks = [bytes(shards[i]) for i in use]
    decoder = zfec.Decoder(k, kp)
    try:
        primary = decoder.decode(use_blocks, use)
    except Exception:
        return False
    # primary is now the k data blocks (indices 0..k-1)
    for i in range(k):
        shards[i] = bytes(primary[i])
    # Re-encode to rebuild missing parity shards if any were missing
    missing_parity = [i for i in range(k, kp) if shards[i] is None]
    if missing_parity:
        encoder = zfec.Encoder(k, kp)
        encoded = encoder.encode([bytes(shards[i]) for i in range(k)])
        for i in missing_parity:
            shards[i] = bytes(encoded[i])
    return True


def ecc_unwrap(input_path, output_path, verbose=False):
    """Decode an ECC-wrapped file and write the original to output_path.

    Streams one stripe at a time: peak RAM is O(kp * block_size) ≈ 16 MB
    regardless of archive size.
    """
    file_size = os.path.getsize(input_path)
    with open(input_path, "rb") as fin:
        header = _try_read_global_header(fin, file_size)
        if header["generator"] != GENERATOR_PYTHON_ZFEC:
            raise ValueError(
                "ECC: this file was wrapped by a different generator "
                f"(id={header['generator']}); only zfec (id=1) archives can be "
                "unwrapped by this Python implementation")

        k = header["k"]
        p = header["p"]
        kp = k + p
        block_size = header["block_size"]
        n_stripes = header["n_stripes"]
        original_file_size = header["original_file_size"]

        hasher = _make_hasher()
        bytes_written = 0

        with open(output_path, "wb") as fout:
            for sid in range(n_stripes):
                shards, _corrupt = _read_stripe(fin, sid, n_stripes, kp, block_size)
                good = sum(1 for s in shards if s is not None)
                if good < k:
                    raise ValueError(
                        f"ECC: stripe {sid} has {good}<{k} good shards, unrecoverable")
                if good < kp:
                    if not _reconstruct_stripe(shards, k, kp):
                        raise ValueError(f"ECC: stripe {sid} reconstruction failed")
                for d in range(k):
                    remaining = original_file_size - bytes_written
                    if remaining <= 0:
                        break
                    chunk = bytes(shards[d])
                    if len(chunk) > remaining:
                        chunk = chunk[:remaining]
                    fout.write(chunk)
                    hasher.update(chunk)
                    bytes_written += len(chunk)

    if hasher.digest() != header["original_hash"]:
        raise ValueError("ECC: reconstructed file BLAKE3 hash mismatch")

    if verbose:
        print(f"ECC unwrap: {n_stripes} stripes, {original_file_size} bytes restored")


def ecc_verify(input_path, verbose=False):
    """Verify an ECC-wrapped file. Returns a dict summary.

    Streams one stripe at a time: peak RAM is O(kp * block_size) ≈ 16 MB
    regardless of archive size.
    """
    file_size = os.path.getsize(input_path)
    with open(input_path, "rb") as f:
        header = _try_read_global_header(f, file_size)

        k = header["k"]
        p = header["p"]
        kp = k + p
        n_stripes = header["n_stripes"]
        block_size = header["block_size"]
        original_file_size = header["original_file_size"]

        unrepairable = []
        corrupt_count = 0
        hash_ok = False
        hasher = _make_hasher()
        bytes_hashed = 0

        for sid in range(n_stripes):
            shards, stripe_corrupt = _read_stripe(f, sid, n_stripes, kp, block_size)
            corrupt_count += stripe_corrupt
            good = sum(1 for s in shards if s is not None)
            if good < k:
                unrepairable.append(sid)
                continue
            if good < kp:
                tmp = list(shards)
                if not _reconstruct_stripe(tmp, k, kp):
                    unrepairable.append(sid)
                    continue
                shards = tmp
            for d in range(k):
                remaining = original_file_size - bytes_hashed
                if remaining <= 0:
                    break
                chunk = bytes(shards[d])
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
                hasher.update(chunk)
                bytes_hashed += len(chunk)

        if not unrepairable:
            hash_ok = hasher.digest() == header["original_hash"]

    result = {
        "n_stripes": n_stripes,
        "total_blocks": header["total_blocks"],
        "corrupt_blocks": corrupt_count,
        "unrepairable_stripes": unrepairable,
        "hash_ok": hash_ok,
    }
    if verbose:
        print(f"ECC verify: {result}")
    return result


def ecc_repair(input_path, verbose=False):
    """In-place repair of corrupted blocks. Returns dict summary."""
    file_size = os.path.getsize(input_path)
    with open(input_path, "r+b") as f:
        header = _try_read_global_header(f, file_size)
        stripes, corrupt = _scan_stripes(f, header)

        k = header["k"]
        p = header["p"]
        kp = k + p
        block_size = header["block_size"]
        n_stripes = header["n_stripes"]

        unrepairable = []
        repaired_count = 0

        for sid, shards in enumerate(stripes):
            missing_before = [i for i in range(kp) if shards[i] is None]
            good = kp - len(missing_before)
            if good < k:
                unrepairable.append(sid)
                continue
            if not missing_before:
                continue
            if not _reconstruct_stripe(shards, k, kp):
                unrepairable.append(sid)
                continue
            # Write reconstructed blocks back to their physical slots
            for bi in missing_before:
                # Find window: scan 0..kp to find the slot placing this block
                found_window = None
                for w in range(kp):
                    if is_parity_slot(w, k, p):
                        if window_to_parity_idx(w, k, p) + k == bi:
                            found_window = w
                            break
                    else:
                        if window_to_data_idx(w, k, p) == bi:
                            found_window = w
                            break
                if found_window is None:
                    continue
                slot_idx = found_window * n_stripes + sid
                off = _slot_offset(slot_idx, block_size)
                payload = bytes(shards[bi])
                payload_crc = _crc32c(payload)
                bh = _pack_block_header(
                    sid, bi, block_size, 0, payload_crc)
                f.seek(off)
                f.write(bh)
                f.write(payload)
                repaired_count += 1

        f.flush()

    # Now verify whole-file hash
    result = ecc_verify(input_path, verbose=False)
    result["repaired_blocks"] = repaired_count
    result["unrepairable_stripes"] = unrepairable or result["unrepairable_stripes"]
    if verbose:
        print(f"ECC repair: repaired {repaired_count} blocks, "
              f"{len(result['unrepairable_stripes'])} unrepairable stripes, "
              f"hash_ok={result['hash_ok']}")
    return result


def ecc_info(input_path):
    """Print ECC container info."""
    file_size = os.path.getsize(input_path)
    with open(input_path, "rb") as f:
        header = _try_read_global_header(f, file_size)
    level_name = _LEVEL_NAMES.get(header["level"], f"code-{header['level']}")
    overhead = 100.0 * header["p"] / header["k"]
    correctable = 100.0 * header["p"] / (header["k"] + header["p"])
    print(f"ECC container: {input_path}")
    print(f"  Format version:    {header['version']}")
    print(f"  Level:             {level_name}")
    print(f"  k (data shards):   {header['k']}")
    print(f"  p (parity shards): {header['p']}")
    print(f"  Block size:        {header['block_size']} B")
    print(f"  Stripes:           {header['n_stripes']}")
    print(f"  Total blocks:      {header['total_blocks']}")
    print(f"  Original size:     {header['original_file_size']} B")
    print(f"  Correctable:       {correctable:.1f}% per stripe")
    print(f"  Storage overhead:  {overhead:.1f}%")
    gen_name = {GENERATOR_RUST_RSE: "rust-rse", GENERATOR_PYTHON_ZFEC: "python-zfec"}.get(
        header["generator"], f"unknown-{header['generator']}")
    print(f"  Generator:         {gen_name}")


def ecc_map(input_path):
    """Print visual block map of ECC container."""
    file_size = os.path.getsize(input_path)
    with open(input_path, "rb") as f:
        header = _try_read_global_header(f, file_size)
    k = header["k"]
    p = header["p"]
    n_stripes = header["n_stripes"]
    kp = k + p
    total = n_stripes * kp
    block_size = header["block_size"]

    level_name = _LEVEL_NAMES.get(header["level"], f"code-{header['level']}")
    print(f"ECC block map: {input_path} (level={level_name}, k={k}, p={p}, "
          f"stripes={n_stripes}, blocks={total})")
    print()
    print("  Legend:  D = data block    P = parity block    (number = stripe ID)")
    print()

    show_full = total <= 400

    if show_full:
        cols = 80
        slot = 0
        while slot < total:
            row_end = min(slot + cols, total)
            line = f"  {slot:>6}: "
            for s in range(slot, row_end):
                window = s // n_stripes
                line += "P" if is_parity_slot(window, k, p) else "D"
            print(line)
            ids = "          "
            for s in range(slot, row_end):
                stripe = s % n_stripes
                if stripe < 10:
                    ids += str(stripe)
                elif stripe < 36:
                    ids += chr(ord('a') + stripe - 10)
                else:
                    ids += "."
            print(ids)
            slot = row_end
    else:
        print(f"  File too large for full map ({total} blocks). Showing window pattern.")
        print()
        display_len = min(kp, 200)
        pattern = ""
        for w in range(display_len):
            pattern += "P" if is_parity_slot(w, k, p) else "D"
        if display_len < kp:
            pattern += "..."
        print(f"  Window pattern (first {display_len} of {kp} slots): {pattern}")
        print()
        data_count = sum(1 for w in range(kp) if not is_parity_slot(w, k, p))
        parity_count = kp - data_count
        print(f"  Per window: {data_count} data + {parity_count} parity = {kp} blocks")
        print(f"  Cross-stripe interleaving: {n_stripes} stripes cycle through each window")
        print(f"  Contiguous damage of N blocks affects at most ceil(N/{n_stripes}) blocks per stripe")

    # Parity gap statistics
    print()
    gaps = []
    last_parity = None
    for w in range(kp):
        if is_parity_slot(w, k, p):
            if last_parity is not None:
                gaps.append(w - last_parity)
            last_parity = w
    if gaps:
        print(f"  Parity spacing (within window): min={min(gaps)}, max={max(gaps)}, "
              f"avg={sum(gaps)/len(gaps):.1f} blocks apart")
    phys_spacing = (kp // p) * n_stripes * (block_size + BLOCK_HEADER_SIZE)
    print(f"  Physical parity spacing (with cross-stripe): every ~{phys_spacing} bytes on disk")
