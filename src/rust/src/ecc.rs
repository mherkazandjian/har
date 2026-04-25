// Byte-level Reed-Solomon ECC container for HAR archives.
//
// Wraps an opaque byte stream (typically a finished HDF5 archive) in a
// Reed-Solomon protected container with parity blocks distributed uniformly
// throughout the file and cross-stripe interleaving for burst-error
// resilience.
//
// File layout:
//   [GlobalHeader (128 B)]
//   [Block 0 header (32 B)] [Block 0 payload (block_size B)]
//   [Block 1 header (32 B)] [Block 1 payload (block_size B)]
//   ...
//   [Block N-1 header (32 B)] [Block N-1 payload (block_size B)]
//   [GlobalHeader copy]
//   [GlobalHeader copy]  (last 128 bytes of file)
//
// Slot placement: slot_idx = window * n_stripes + stripe_idx
// where window iterates 0..(k+p) and stripe_idx cycles through stripes.
// Within each window, slots are classified as data or parity using a
// Bresenham-like uniform distribution rule so that p parity blocks are
// spread evenly among every (k+p) consecutive slots.

use crc32c::crc32c;
use reed_solomon_erasure::galois_8::ReedSolomon;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const GLOBAL_HEADER_MAGIC: [u8; 8] = [0x89, b'H', b'A', b'R', b'E', b'C', b'C', b'\n'];
pub const BLOCK_MAGIC: [u8; 4] = [0xEC, b'B', b'K', 0x00];
pub const GLOBAL_HEADER_SIZE: usize = 128;
pub const BLOCK_HEADER_SIZE: usize = 32;
pub const DEFAULT_BLOCK_PAYLOAD: usize = 65536;
pub const FORMAT_VERSION: u32 = 1;
pub const MAX_SHARDS: usize = 255;

// ---------------------------------------------------------------------------
// ECC level
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EccLevel {
    Low,
    Medium,
    High,
    Max,
    Custom,
}

impl EccLevel {
    pub fn code(self) -> u32 {
        match self {
            EccLevel::Low => 0,
            EccLevel::Medium => 1,
            EccLevel::High => 2,
            EccLevel::Max => 3,
            EccLevel::Custom => 255,
        }
    }
    pub fn from_code(c: u32) -> Self {
        match c {
            0 => EccLevel::Low,
            1 => EccLevel::Medium,
            2 => EccLevel::High,
            3 => EccLevel::Max,
            _ => EccLevel::Custom,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            EccLevel::Low => "low",
            EccLevel::Medium => "medium",
            EccLevel::High => "high",
            EccLevel::Max => "max",
            EccLevel::Custom => "custom",
        }
    }
}

// ---------------------------------------------------------------------------
// ECC parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct EccParams {
    pub level: EccLevel,
    pub k: usize,
    pub p: usize,
    pub block_size: usize,
}

impl EccParams {
    pub fn from_level(level: EccLevel) -> Self {
        let (k, p) = match level {
            EccLevel::Low => (179, 10),
            EccLevel::Medium => (169, 20),
            EccLevel::High => (149, 40),
            EccLevel::Max => (127, 63),
            EccLevel::Custom => (169, 20),
        };
        EccParams {
            level,
            k,
            p,
            block_size: DEFAULT_BLOCK_PAYLOAD,
        }
    }

    pub fn from_percent(pct: f64) -> Self {
        let pct = pct.clamp(0.5, 50.0);
        let p = ((255.0 * pct / 100.0).round() as usize).clamp(1, 127);
        let k = 255 - p;
        EccParams {
            level: EccLevel::Custom,
            k,
            p,
            block_size: DEFAULT_BLOCK_PAYLOAD,
        }
    }

    /// If the input file is small enough that not all `k` data slots would
    /// be used, reduce `k` to the actual number of data blocks needed while
    /// preserving the p/k ratio (with p >= 1).
    pub fn fit_to_file(&self, file_size: u64) -> EccParams {
        let block_size = self.block_size;
        let needed = (file_size as usize).div_ceil(block_size).max(1);
        if needed >= self.k {
            return *self;
        }
        let new_k = needed;
        let new_p = ((self.p as f64 * new_k as f64 / self.k as f64).round() as usize).max(1);
        let new_p = new_p.min(MAX_SHARDS - new_k);
        EccParams {
            level: self.level,
            k: new_k,
            p: new_p,
            block_size,
        }
    }
}

pub fn parse_ecc_level(s: &str) -> Result<EccParams, String> {
    let t = s.trim();
    if let Some(pct) = t.strip_suffix('%') {
        let v: f64 = pct
            .trim()
            .parse()
            .map_err(|_| format!("invalid percentage: {}", s))?;
        Ok(EccParams::from_percent(v))
    } else {
        match t.to_lowercase().as_str() {
            "low" => Ok(EccParams::from_level(EccLevel::Low)),
            "medium" | "med" => Ok(EccParams::from_level(EccLevel::Medium)),
            "high" => Ok(EccParams::from_level(EccLevel::High)),
            "max" => Ok(EccParams::from_level(EccLevel::Max)),
            _ => {
                // Bare number → treat as percentage
                if let Ok(v) = t.parse::<f64>() {
                    Ok(EccParams::from_percent(v))
                } else {
                    Err(format!(
                        "unknown ECC level '{}': expected low|medium|high|max or N%",
                        s
                    ))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Uniform slot placement
// ---------------------------------------------------------------------------

/// Returns true iff window position `w` (in [0, k+p)) holds a parity block
/// under the uniform-distribution rule.
#[inline]
pub fn is_parity_slot(w: usize, k: usize, p: usize) -> bool {
    let kp = k + p;
    (w * p) / kp < ((w + 1) * p) / kp
}

/// For a parity window position, returns the parity block index within the
/// stripe (0..p).
#[inline]
pub fn window_to_parity_idx(w: usize, k: usize, p: usize) -> usize {
    ((w + 1) * p) / (k + p) - 1
}

/// For a data window position, returns the data block index within the
/// stripe (0..k).
#[inline]
pub fn window_to_data_idx(w: usize, k: usize, p: usize) -> usize {
    w - (w * p) / (k + p)
}

// ---------------------------------------------------------------------------
// Global header
// ---------------------------------------------------------------------------

pub const GENERATOR_RUST_RSE: u8 = 0;
pub const GENERATOR_PYTHON_ZFEC: u8 = 1;

#[derive(Debug, Clone)]
pub struct GlobalHeader {
    pub version: u32,
    pub level_code: u32,
    pub k: u32,
    pub p: u32,
    pub n_stripes: u64,
    pub block_payload_size: u64,
    pub original_file_size: u64,
    pub total_blocks: u64,
    pub original_hash: [u8; 32],
    pub generator: u8,
}

impl GlobalHeader {
    pub fn serialize(&self) -> [u8; GLOBAL_HEADER_SIZE] {
        let mut buf = [0u8; GLOBAL_HEADER_SIZE];
        buf[0..8].copy_from_slice(&GLOBAL_HEADER_MAGIC);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.level_code.to_le_bytes());
        buf[16..20].copy_from_slice(&self.k.to_le_bytes());
        buf[20..24].copy_from_slice(&self.p.to_le_bytes());
        buf[24..32].copy_from_slice(&self.n_stripes.to_le_bytes());
        buf[32..40].copy_from_slice(&self.block_payload_size.to_le_bytes());
        buf[40..48].copy_from_slice(&self.original_file_size.to_le_bytes());
        buf[48..56].copy_from_slice(&self.total_blocks.to_le_bytes());
        buf[56..88].copy_from_slice(&self.original_hash);
        let crc = crc32c(&buf[0..88]);
        buf[88..92].copy_from_slice(&crc.to_le_bytes());
        buf[92] = self.generator;
        buf
    }

    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        if buf.len() < GLOBAL_HEADER_SIZE {
            return None;
        }
        if buf[0..8] != GLOBAL_HEADER_MAGIC {
            return None;
        }
        let stored_crc = u32::from_le_bytes(buf[88..92].try_into().ok()?);
        let calc_crc = crc32c(&buf[0..88]);
        if stored_crc != calc_crc {
            return None;
        }
        Some(GlobalHeader {
            version: u32::from_le_bytes(buf[8..12].try_into().ok()?),
            level_code: u32::from_le_bytes(buf[12..16].try_into().ok()?),
            k: u32::from_le_bytes(buf[16..20].try_into().ok()?),
            p: u32::from_le_bytes(buf[20..24].try_into().ok()?),
            n_stripes: u64::from_le_bytes(buf[24..32].try_into().ok()?),
            block_payload_size: u64::from_le_bytes(buf[32..40].try_into().ok()?),
            original_file_size: u64::from_le_bytes(buf[40..48].try_into().ok()?),
            total_blocks: u64::from_le_bytes(buf[48..56].try_into().ok()?),
            original_hash: buf[56..88].try_into().ok()?,
            generator: buf[92],
        })
    }
}

// ---------------------------------------------------------------------------
// Block header
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct BlockHeader {
    pub stripe_id: u32,
    pub block_index: u32, // within stripe: 0..k = data, k..k+p = parity
    pub payload_size: u32,
    pub flags: u32,
    pub payload_crc32: u32,
}

impl BlockHeader {
    pub fn serialize(&self) -> [u8; BLOCK_HEADER_SIZE] {
        let mut buf = [0u8; BLOCK_HEADER_SIZE];
        buf[0..4].copy_from_slice(&BLOCK_MAGIC);
        buf[4..8].copy_from_slice(&self.stripe_id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.block_index.to_le_bytes());
        buf[12..16].copy_from_slice(&self.payload_size.to_le_bytes());
        buf[16..20].copy_from_slice(&self.flags.to_le_bytes());
        buf[20..24].copy_from_slice(&self.payload_crc32.to_le_bytes());
        let crc = crc32c(&buf[0..24]);
        buf[24..28].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        if buf.len() < BLOCK_HEADER_SIZE {
            return None;
        }
        if buf[0..4] != BLOCK_MAGIC {
            return None;
        }
        let stored_crc = u32::from_le_bytes(buf[24..28].try_into().ok()?);
        let calc_crc = crc32c(&buf[0..24]);
        if stored_crc != calc_crc {
            return None;
        }
        Some(BlockHeader {
            stripe_id: u32::from_le_bytes(buf[4..8].try_into().ok()?),
            block_index: u32::from_le_bytes(buf[8..12].try_into().ok()?),
            payload_size: u32::from_le_bytes(buf[12..16].try_into().ok()?),
            flags: u32::from_le_bytes(buf[16..20].try_into().ok()?),
            payload_crc32: u32::from_le_bytes(buf[20..24].try_into().ok()?),
        })
    }
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

pub fn is_ecc_wrapped(path: &Path) -> bool {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut buf = [0u8; 8];
    if f.read_exact(&mut buf).is_err() {
        return false;
    }
    buf == GLOBAL_HEADER_MAGIC
}

// ---------------------------------------------------------------------------
// Physical layout helpers
// ---------------------------------------------------------------------------

fn container_size(params: &EccParams, n_stripes: u64) -> u64 {
    let total_blocks = n_stripes * (params.k as u64 + params.p as u64);
    GLOBAL_HEADER_SIZE as u64
        + total_blocks * (BLOCK_HEADER_SIZE as u64 + params.block_size as u64)
        + (GLOBAL_HEADER_SIZE as u64) * 2
}

fn slot_offset(slot_idx: u64, block_size: usize) -> u64 {
    GLOBAL_HEADER_SIZE as u64
        + slot_idx * (BLOCK_HEADER_SIZE as u64 + block_size as u64)
}

// ---------------------------------------------------------------------------
// Wrap (encode): plain file -> ECC container
// ---------------------------------------------------------------------------

pub fn ecc_wrap(
    input: &Path,
    output: &Path,
    params: EccParams,
    verbose: bool,
) -> std::io::Result<()> {
    let mut input_file = File::open(input)?;
    let file_size = input_file.metadata()?.len();
    let mut data = Vec::with_capacity(file_size as usize);
    input_file.read_to_end(&mut data)?;
    drop(input_file);

    let params = params.fit_to_file(file_size);
    let original_hash: [u8; 32] = *blake3::hash(&data).as_bytes();

    let bytes_per_stripe = params.k as u64 * params.block_size as u64;
    let n_stripes = file_size.div_ceil(bytes_per_stripe);
    let n_stripes = n_stripes.max(1);

    // Pad to exact stripe boundary
    let padded_len = (n_stripes * bytes_per_stripe) as usize;
    data.resize(padded_len, 0);

    if verbose {
        eprintln!(
            "ecc: wrapping {} ({} B) level={} k={} p={} block_size={} n_stripes={}",
            input.display(),
            file_size,
            params.level.name(),
            params.k,
            params.p,
            params.block_size,
            n_stripes
        );
    }

    let rs = ReedSolomon::new(params.k, params.p)
        .map_err(|e| std::io::Error::other(format!("RS init: {:?}", e)))?;

    let total_blocks = n_stripes * (params.k as u64 + params.p as u64);
    let out_size = container_size(&params, n_stripes);

    let mut out_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(true)
        .open(output)?;
    out_file.set_len(out_size)?;

    let header = GlobalHeader {
        version: FORMAT_VERSION,
        level_code: params.level.code(),
        k: params.k as u32,
        p: params.p as u32,
        n_stripes,
        block_payload_size: params.block_size as u64,
        original_file_size: file_size,
        total_blocks,
        original_hash,
        generator: GENERATOR_RUST_RSE,
    };
    let header_bytes = header.serialize();

    // Primary header
    out_file.seek(SeekFrom::Start(0))?;
    out_file.write_all(&header_bytes)?;

    // Two tail copies (the last two are the redundant copies)
    let tail_offset = GLOBAL_HEADER_SIZE as u64
        + total_blocks * (BLOCK_HEADER_SIZE as u64 + params.block_size as u64);
    out_file.seek(SeekFrom::Start(tail_offset))?;
    out_file.write_all(&header_bytes)?;
    out_file.write_all(&header_bytes)?;

    let kp = params.k + params.p;
    let mut shards: Vec<Vec<u8>> = (0..kp).map(|_| vec![0u8; params.block_size]).collect();

    for stripe_id in 0..n_stripes {
        let stripe_off = (stripe_id * bytes_per_stripe) as usize;
        for (d, shard) in shards.iter_mut().take(params.k).enumerate() {
            let start = stripe_off + d * params.block_size;
            let end = start + params.block_size;
            shard.copy_from_slice(&data[start..end]);
        }
        for pp in 0..params.p {
            shards[params.k + pp].fill(0);
        }
        rs.encode(&mut shards)
            .map_err(|e| std::io::Error::other(format!("RS encode: {:?}", e)))?;

        for window in 0..kp {
            let slot_idx = (window as u64) * n_stripes + stripe_id;
            let (shard_idx, block_index_in_stripe) =
                if is_parity_slot(window, params.k, params.p) {
                    let pi = window_to_parity_idx(window, params.k, params.p);
                    (params.k + pi, (params.k + pi) as u32)
                } else {
                    let di = window_to_data_idx(window, params.k, params.p);
                    (di, di as u32)
                };

            let payload = &shards[shard_idx];
            let crc = crc32c(payload);
            let block_header = BlockHeader {
                stripe_id: stripe_id as u32,
                block_index: block_index_in_stripe,
                payload_size: params.block_size as u32,
                flags: 0,
                payload_crc32: crc,
            };
            let bh_bytes = block_header.serialize();

            let offset = slot_offset(slot_idx, params.block_size);
            out_file.seek(SeekFrom::Start(offset))?;
            out_file.write_all(&bh_bytes)?;
            out_file.write_all(payload)?;
        }
    }

    out_file.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Global header recovery
// ---------------------------------------------------------------------------

fn read_global_header(path: &Path) -> std::io::Result<GlobalHeader> {
    let mut f = File::open(path)?;
    let file_size = f.metadata()?.len();
    let mut buf = [0u8; GLOBAL_HEADER_SIZE];

    // Primary
    f.seek(SeekFrom::Start(0))?;
    if f.read_exact(&mut buf).is_ok() {
        if let Some(h) = GlobalHeader::deserialize(&buf) {
            return Ok(h);
        }
    }
    // Tail copy 1 (offset file_size - 2 * header_size)
    if file_size >= 2 * GLOBAL_HEADER_SIZE as u64 {
        f.seek(SeekFrom::Start(file_size - 2 * GLOBAL_HEADER_SIZE as u64))?;
        if f.read_exact(&mut buf).is_ok() {
            if let Some(h) = GlobalHeader::deserialize(&buf) {
                return Ok(h);
            }
        }
    }
    // Tail copy 2 (last 128 bytes of file)
    if file_size >= GLOBAL_HEADER_SIZE as u64 {
        f.seek(SeekFrom::Start(file_size - GLOBAL_HEADER_SIZE as u64))?;
        if f.read_exact(&mut buf).is_ok() {
            if let Some(h) = GlobalHeader::deserialize(&buf) {
                return Ok(h);
            }
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "ECC: no valid global header found",
    ))
}

// ---------------------------------------------------------------------------
// Stripe scan (shared by unwrap/verify/repair)
// ---------------------------------------------------------------------------

struct StripeScan {
    stripes: Vec<Vec<Option<Vec<u8>>>>,
    corrupt_slots: Vec<u64>, // slot indices that had a corrupt header or payload
}

fn scan_stripes(f: &mut File, header: &GlobalHeader) -> std::io::Result<StripeScan> {
    let k = header.k as usize;
    let p = header.p as usize;
    let kp = k + p;
    let block_size = header.block_payload_size as usize;
    let n_stripes = header.n_stripes;

    let mut stripes: Vec<Vec<Option<Vec<u8>>>> =
        (0..n_stripes).map(|_| vec![None; kp]).collect();
    let mut corrupt_slots: Vec<u64> = Vec::new();
    let mut block_buf = vec![0u8; BLOCK_HEADER_SIZE + block_size];

    let slot_total = n_stripes * kp as u64;
    for slot_idx in 0..slot_total {
        let offset = slot_offset(slot_idx, block_size);
        f.seek(SeekFrom::Start(offset))?;
        if f.read_exact(&mut block_buf).is_err() {
            corrupt_slots.push(slot_idx);
            continue;
        }
        let bh = match BlockHeader::deserialize(&block_buf[..BLOCK_HEADER_SIZE]) {
            Some(bh) => bh,
            None => {
                corrupt_slots.push(slot_idx);
                continue;
            }
        };
        if (bh.stripe_id as u64) >= n_stripes || (bh.block_index as usize) >= kp {
            corrupt_slots.push(slot_idx);
            continue;
        }
        let payload = &block_buf[BLOCK_HEADER_SIZE..];
        let calc_crc = crc32c(payload);
        if calc_crc != bh.payload_crc32 {
            corrupt_slots.push(slot_idx);
            continue;
        }
        stripes[bh.stripe_id as usize][bh.block_index as usize] = Some(payload.to_vec());
    }

    Ok(StripeScan {
        stripes,
        corrupt_slots,
    })
}

// ---------------------------------------------------------------------------
// Unwrap (decode): ECC container -> plain file
// ---------------------------------------------------------------------------

pub fn ecc_unwrap(input: &Path, output: &Path, verbose: bool) -> std::io::Result<()> {
    let header = read_global_header(input)?;
    let k = header.k as usize;
    let p = header.p as usize;
    let block_size = header.block_payload_size as usize;
    let n_stripes = header.n_stripes;

    if verbose {
        eprintln!(
            "ecc: unwrapping {} (original {} B, k={} p={} block={} n_stripes={})",
            input.display(),
            header.original_file_size,
            k,
            p,
            block_size,
            n_stripes
        );
    }

    let rs =
        ReedSolomon::new(k, p).map_err(|e| std::io::Error::other(format!("RS init: {:?}", e)))?;

    let mut f = File::open(input)?;
    let mut scan = scan_stripes(&mut f, &header)?;

    let mut out_data: Vec<u8> =
        Vec::with_capacity((n_stripes * k as u64 * block_size as u64) as usize);

    for (sid, shards) in scan.stripes.iter_mut().enumerate() {
        let good = shards.iter().filter(|s| s.is_some()).count();
        if good < k {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "ECC: stripe {} has only {} intact shards, needs at least {} \
                     (corruption exceeds recovery capacity)",
                    sid, good, k
                ),
            ));
        }
        if good < k + p {
            rs.reconstruct(shards).map_err(|e| {
                std::io::Error::other(format!("RS reconstruct stripe {}: {:?}", sid, e))
            })?;
        }
        for shard in shards.iter().take(k) {
            out_data.extend_from_slice(shard.as_ref().unwrap());
        }
    }

    out_data.truncate(header.original_file_size as usize);

    let calc_hash: [u8; 32] = *blake3::hash(&out_data).as_bytes();
    if calc_hash != header.original_hash {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ECC: reconstructed file BLAKE3 hash mismatch",
        ));
    }

    let mut out_file = File::create(output)?;
    out_file.write_all(&out_data)?;
    out_file.flush()?;
    if verbose {
        eprintln!(
            "ecc: unwrapped {} B to {} (corrupt blocks repaired: {})",
            out_data.len(),
            output.display(),
            scan.corrupt_slots.len()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct VerifyResult {
    pub n_stripes: u64,
    pub total_blocks: u64,
    pub corrupt_blocks: u64,
    pub unrepairable_stripes: Vec<u64>,
    pub hash_ok: bool,
}

pub fn ecc_verify(input: &Path, verbose: bool) -> std::io::Result<VerifyResult> {
    let header = read_global_header(input)?;
    let k = header.k as usize;
    let p = header.p as usize;
    let n_stripes = header.n_stripes;

    let rs =
        ReedSolomon::new(k, p).map_err(|e| std::io::Error::other(format!("RS init: {:?}", e)))?;

    let mut f = File::open(input)?;
    let mut scan = scan_stripes(&mut f, &header)?;

    let mut result = VerifyResult {
        n_stripes,
        total_blocks: header.total_blocks,
        corrupt_blocks: scan.corrupt_slots.len() as u64,
        unrepairable_stripes: Vec::new(),
        hash_ok: false,
    };

    let mut out_data: Vec<u8> = Vec::with_capacity(header.original_file_size as usize);

    for (sid, shards) in scan.stripes.iter_mut().enumerate() {
        let good = shards.iter().filter(|s| s.is_some()).count();
        if good < k {
            result.unrepairable_stripes.push(sid as u64);
            continue;
        }
        if good < k + p && rs.reconstruct(shards).is_err() {
            result.unrepairable_stripes.push(sid as u64);
            continue;
        }
        for shard in shards.iter().take(k) {
            if let Some(s) = shard.as_ref() {
                out_data.extend_from_slice(s);
            }
        }
    }

    if result.unrepairable_stripes.is_empty() {
        out_data.truncate(header.original_file_size as usize);
        let calc_hash: [u8; 32] = *blake3::hash(&out_data).as_bytes();
        result.hash_ok = calc_hash == header.original_hash;
    }

    if verbose {
        eprintln!(
            "ecc: verify {}: {} stripes, {} corrupt blocks, {} unrepairable, hash_ok={}",
            input.display(),
            result.n_stripes,
            result.corrupt_blocks,
            result.unrepairable_stripes.len(),
            result.hash_ok
        );
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Repair (in-place)
// ---------------------------------------------------------------------------

pub fn ecc_repair(input: &Path, verbose: bool) -> std::io::Result<VerifyResult> {
    let header = read_global_header(input)?;
    let k = header.k as usize;
    let p = header.p as usize;
    let kp = k + p;
    let block_size = header.block_payload_size as usize;
    let n_stripes = header.n_stripes;

    let rs =
        ReedSolomon::new(k, p).map_err(|e| std::io::Error::other(format!("RS init: {:?}", e)))?;

    let mut f = OpenOptions::new().read(true).write(true).open(input)?;
    let mut scan = scan_stripes(&mut f, &header)?;

    let mut result = VerifyResult {
        n_stripes,
        total_blocks: header.total_blocks,
        corrupt_blocks: scan.corrupt_slots.len() as u64,
        unrepairable_stripes: Vec::new(),
        hash_ok: false,
    };

    let mut repaired_count: u64 = 0;
    let mut final_data: Vec<u8> = Vec::with_capacity(header.original_file_size as usize);

    for (sid, shards) in scan.stripes.iter_mut().enumerate() {
        let missing_before: Vec<usize> = (0..kp).filter(|i| shards[*i].is_none()).collect();
        let good = kp - missing_before.len();
        if good < k {
            result.unrepairable_stripes.push(sid as u64);
            // Still push zeros for output alignment
            final_data.extend(std::iter::repeat_n(0u8, k * block_size));
            continue;
        }
        if !missing_before.is_empty() {
            if rs.reconstruct(shards).is_err() {
                result.unrepairable_stripes.push(sid as u64);
                final_data.extend(std::iter::repeat_n(0u8, k * block_size));
                continue;
            }
            // Write reconstructed blocks back to the file
            for idx in &missing_before {
                // Find the slot that holds (stripe=sid, block_index=idx)
                // slot_idx = window * n_stripes + sid; we need the window that places this block.
                let window = if *idx < k {
                    // Data block *idx*: smallest window w with window_to_data_idx(w) == *idx
                    // i.e. w - floor(w*p/(k+p)) == *idx
                    // We scan (bounded by kp) — cheap since kp <= 255
                    (0..kp)
                        .find(|w| {
                            !is_parity_slot(*w, k, p) && window_to_data_idx(*w, k, p) == *idx
                        })
                        .unwrap()
                } else {
                    let parity_idx_in_stripe = *idx - k;
                    (0..kp)
                        .find(|w| {
                            is_parity_slot(*w, k, p)
                                && window_to_parity_idx(*w, k, p) == parity_idx_in_stripe
                        })
                        .unwrap()
                };
                let slot_idx = (window as u64) * n_stripes + sid as u64;
                let offset = slot_offset(slot_idx, block_size);

                let payload = shards[*idx].as_ref().unwrap();
                let crc = crc32c(payload);
                let bh = BlockHeader {
                    stripe_id: sid as u32,
                    block_index: *idx as u32,
                    payload_size: block_size as u32,
                    flags: 0,
                    payload_crc32: crc,
                };
                let bh_bytes = bh.serialize();
                f.seek(SeekFrom::Start(offset))?;
                f.write_all(&bh_bytes)?;
                f.write_all(payload)?;
                repaired_count += 1;
            }
        }

        for shard in shards.iter().take(k) {
            final_data.extend_from_slice(shard.as_ref().unwrap());
        }
    }

    f.flush()?;

    if result.unrepairable_stripes.is_empty() {
        final_data.truncate(header.original_file_size as usize);
        let calc_hash: [u8; 32] = *blake3::hash(&final_data).as_bytes();
        result.hash_ok = calc_hash == header.original_hash;
    }

    if verbose {
        eprintln!(
            "ecc: repair {}: {} corrupt blocks found, {} repaired, {} unrepairable stripes, hash_ok={}",
            input.display(),
            result.corrupt_blocks,
            repaired_count,
            result.unrepairable_stripes.len(),
            result.hash_ok
        );
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Info
// ---------------------------------------------------------------------------

pub fn ecc_info(input: &Path) -> std::io::Result<()> {
    let header = read_global_header(input)?;
    let level = EccLevel::from_code(header.level_code);
    let k = header.k as usize;
    let p = header.p as usize;
    let correctable_pct = 100.0 * p as f64 / (k + p) as f64;
    let overhead_pct = 100.0 * p as f64 / k as f64;
    println!("ECC container: {}", input.display());
    println!("  Format version:    {}", header.version);
    println!("  Level:             {}", level.name());
    println!("  k (data shards):   {}", k);
    println!("  p (parity shards): {}", p);
    println!("  Block size:        {} B", header.block_payload_size);
    println!("  Stripes:           {}", header.n_stripes);
    println!("  Total blocks:      {}", header.total_blocks);
    println!("  Original size:     {} B", header.original_file_size);
    println!("  Correctable:       {:.1}% per stripe", correctable_pct);
    println!("  Storage overhead:  {:.1}%", overhead_pct);
    let gen_name = match header.generator {
        GENERATOR_RUST_RSE => "rust-rse",
        GENERATOR_PYTHON_ZFEC => "python-zfec",
        other => { println!("  Generator:         unknown-{}", other); return Ok(()); }
    };
    println!("  Generator:         {}", gen_name);
    Ok(())
}

/// Print a visual block map of the ECC container showing data/parity
/// interleaving across stripes.
pub fn ecc_map(input: &Path) -> std::io::Result<()> {
    let header = read_global_header(input)?;
    let k = header.k as usize;
    let p = header.p as usize;
    let n_stripes = header.n_stripes as usize;
    let kp = k + p;
    let total = n_stripes * kp;

    // Print header summary
    let level = EccLevel::from_code(header.level_code);
    println!(
        "ECC block map: {} (level={}, k={}, p={}, stripes={}, blocks={})",
        input.display(), level.name(), k, p, n_stripes, total
    );
    println!();

    // Legend
    println!("  Legend:  D = data block    P = parity block    (number = stripe ID)");
    println!();

    // For small files, show every block. For large files, show a summary.
    let show_full = total <= 400;

    if show_full {
        // Full map: one character per block, rows of up to 80 blocks
        let cols = 80;
        let mut slot = 0;
        while slot < total {
            let row_end = std::cmp::min(slot + cols, total);
            // Offset line
            print!("  {:>6}: ", slot);
            for s in slot..row_end {
                let window = s / n_stripes;
                if is_parity_slot(window, k, p) {
                    print!("P");
                } else {
                    print!("D");
                }
            }
            println!();
            // Stripe ID line (single hex digit for stripe, '.' if >= 16)
            print!("          ");
            for s in slot..row_end {
                let stripe = s % n_stripes;
                if stripe < 16 {
                    print!("{:x}", stripe);
                } else if stripe < 36 {
                    print!("{}", (b'a' + (stripe - 10) as u8) as char);
                } else {
                    print!(".");
                }
            }
            println!();
            slot = row_end;
        }
    } else {
        // Summary: show distribution in windows
        let mut data_count = 0usize;
        let mut parity_count = 0usize;
        for w in 0..kp {
            if is_parity_slot(w, k, p) {
                parity_count += 1;
            } else {
                data_count += 1;
            }
        }
        println!("  File too large for full map ({} blocks). Showing window pattern.", total);
        println!();

        // Show one window pattern (k+p slots)
        let display_len = std::cmp::min(kp, 200);
        print!("  Window pattern (first {} of {} slots): ", display_len, kp);
        for w in 0..display_len {
            if is_parity_slot(w, k, p) {
                print!("P");
            } else {
                print!("D");
            }
        }
        if display_len < kp {
            print!("...");
        }
        println!();
        println!();
        println!("  Per window: {} data + {} parity = {} blocks", data_count, parity_count, kp);
        println!("  Cross-stripe interleaving: {} stripes cycle through each window", n_stripes);
        println!("  Contiguous damage of N blocks affects at most ceil(N/{}) blocks per stripe", n_stripes);
    }

    // Parity gap statistics
    println!();
    let mut gaps = Vec::new();
    let mut last_parity: Option<usize> = None;
    for w in 0..kp {
        if is_parity_slot(w, k, p) {
            if let Some(lp) = last_parity {
                gaps.push(w - lp);
            }
            last_parity = Some(w);
        }
    }
    if !gaps.is_empty() {
        let min_gap = *gaps.iter().min().unwrap();
        let max_gap = *gaps.iter().max().unwrap();
        let avg_gap = gaps.iter().sum::<usize>() as f64 / gaps.len() as f64;
        println!("  Parity spacing (within window): min={}, max={}, avg={:.1} blocks apart", min_gap, max_gap, avg_gap);
    }
    println!(
        "  Physical parity spacing (with cross-stripe): every ~{} bytes on disk",
        (kp / p) * n_stripes * (header.block_payload_size as usize + BLOCK_HEADER_SIZE)
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use tempfile::NamedTempFile;

    fn write_temp(data: &[u8]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(data).unwrap();
        f.flush().unwrap();
        f
    }

    fn make_data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 31 + 7) as u8).collect()
    }

    #[test]
    fn parse_levels() {
        assert_eq!(parse_ecc_level("low").unwrap().level, EccLevel::Low);
        assert_eq!(parse_ecc_level("MEDIUM").unwrap().level, EccLevel::Medium);
        assert_eq!(parse_ecc_level("high").unwrap().level, EccLevel::High);
        assert_eq!(parse_ecc_level("max").unwrap().level, EccLevel::Max);
        let p = parse_ecc_level("10%").unwrap();
        assert_eq!(p.level, EccLevel::Custom);
        assert!(p.p > 0);
        assert!(p.k > 0);
        assert!(p.k + p.p <= 255);
        assert!(parse_ecc_level("nonsense").is_err());
    }

    #[test]
    fn uniform_slot_distribution() {
        // Verify that across (k+p) slots, exactly p are parity slots.
        for (k, p) in [(10, 2), (179, 10), (169, 20), (149, 40), (127, 63)] {
            let mut parity_count = 0;
            for w in 0..(k + p) {
                if is_parity_slot(w, k, p) {
                    parity_count += 1;
                }
            }
            assert_eq!(parity_count, p, "k={} p={}", k, p);
        }
    }

    #[test]
    fn uniform_distribution_indices_unique() {
        // Each data and parity index appears exactly once per window.
        for (k, p) in [(10, 2), (179, 10), (127, 63)] {
            let mut data_seen = vec![false; k];
            let mut parity_seen = vec![false; p];
            for w in 0..(k + p) {
                if is_parity_slot(w, k, p) {
                    let pi = window_to_parity_idx(w, k, p);
                    assert!(!parity_seen[pi]);
                    parity_seen[pi] = true;
                } else {
                    let di = window_to_data_idx(w, k, p);
                    assert!(!data_seen[di]);
                    data_seen[di] = true;
                }
            }
            assert!(data_seen.iter().all(|x| *x));
            assert!(parity_seen.iter().all(|x| *x));
        }
    }

    #[test]
    fn header_roundtrip() {
        let h = GlobalHeader {
            version: 1,
            level_code: 1,
            k: 169,
            p: 20,
            n_stripes: 42,
            block_payload_size: 65536,
            original_file_size: 1234567,
            total_blocks: 42 * 189,
            original_hash: [7u8; 32],
            generator: GENERATOR_RUST_RSE,
        };
        let bytes = h.serialize();
        let h2 = GlobalHeader::deserialize(&bytes).unwrap();
        assert_eq!(h2.version, h.version);
        assert_eq!(h2.k, h.k);
        assert_eq!(h2.p, h.p);
        assert_eq!(h2.n_stripes, h.n_stripes);
        assert_eq!(h2.original_file_size, h.original_file_size);
        assert_eq!(h2.original_hash, h.original_hash);
    }

    #[test]
    fn header_crc_mismatch_rejected() {
        let h = GlobalHeader {
            version: 1,
            level_code: 0,
            k: 10,
            p: 2,
            n_stripes: 1,
            block_payload_size: 1024,
            original_file_size: 100,
            total_blocks: 12,
            original_hash: [0u8; 32],
            generator: GENERATOR_RUST_RSE,
        };
        let mut bytes = h.serialize();
        bytes[20] ^= 0xFF; // corrupt p
        assert!(GlobalHeader::deserialize(&bytes).is_none());
    }

    #[test]
    fn block_header_roundtrip() {
        let bh = BlockHeader {
            stripe_id: 3,
            block_index: 5,
            payload_size: 1024,
            flags: 1,
            payload_crc32: 0xDEADBEEF,
        };
        let bytes = bh.serialize();
        let bh2 = BlockHeader::deserialize(&bytes).unwrap();
        assert_eq!(bh2.stripe_id, bh.stripe_id);
        assert_eq!(bh2.block_index, bh.block_index);
        assert_eq!(bh2.payload_crc32, bh.payload_crc32);
    }

    #[test]
    fn roundtrip_small_file() {
        let data = make_data(5000);
        let input = write_temp(&data);
        let output = NamedTempFile::new().unwrap();
        let recovered = NamedTempFile::new().unwrap();

        let mut params = EccParams::from_level(EccLevel::Medium);
        params.block_size = 1024;
        ecc_wrap(input.path(), output.path(), params, false).unwrap();
        assert!(is_ecc_wrapped(output.path()));
        ecc_unwrap(output.path(), recovered.path(), false).unwrap();

        let mut recovered_data = Vec::new();
        File::open(recovered.path())
            .unwrap()
            .read_to_end(&mut recovered_data)
            .unwrap();
        assert_eq!(recovered_data, data);
    }

    #[test]
    fn roundtrip_each_level() {
        for level in [EccLevel::Low, EccLevel::Medium, EccLevel::High, EccLevel::Max] {
            let data = make_data(20000);
            let input = write_temp(&data);
            let output = NamedTempFile::new().unwrap();
            let recovered = NamedTempFile::new().unwrap();
            let mut params = EccParams::from_level(level);
            params.block_size = 512;
            ecc_wrap(input.path(), output.path(), params, false).unwrap();
            ecc_unwrap(output.path(), recovered.path(), false).unwrap();
            let mut rec = Vec::new();
            File::open(recovered.path())
                .unwrap()
                .read_to_end(&mut rec)
                .unwrap();
            assert_eq!(rec, data, "level={:?}", level);
        }
    }

    #[test]
    fn exact_block_boundary() {
        // File size is an exact multiple of k * block_size for k=10, block_size=1024 → 10240
        let data = make_data(10240);
        let input = write_temp(&data);
        let output = NamedTempFile::new().unwrap();
        let recovered = NamedTempFile::new().unwrap();

        let params = EccParams {
            level: EccLevel::Custom,
            k: 10,
            p: 2,
            block_size: 1024,
        };
        ecc_wrap(input.path(), output.path(), params, false).unwrap();
        ecc_unwrap(output.path(), recovered.path(), false).unwrap();
        let mut rec = Vec::new();
        File::open(recovered.path())
            .unwrap()
            .read_to_end(&mut rec)
            .unwrap();
        assert_eq!(rec, data);
    }

    #[test]
    fn repair_random_corruption() {
        let data = make_data(30000);
        let input = write_temp(&data);
        let output = NamedTempFile::new().unwrap();
        let recovered = NamedTempFile::new().unwrap();

        let params = EccParams {
            level: EccLevel::Custom,
            k: 10,
            p: 4,
            block_size: 512,
        };
        ecc_wrap(input.path(), output.path(), params, false).unwrap();

        // Corrupt two blocks in the middle of the container file
        {
            let mut f = OpenOptions::new().write(true).open(output.path()).unwrap();
            let mid = std::fs::metadata(output.path()).unwrap().len() / 2;
            f.seek(SeekFrom::Start(mid)).unwrap();
            f.write_all(&vec![0xFFu8; 1024]).unwrap();
            f.flush().unwrap();
        }

        // Unwrap should still succeed because RS can reconstruct
        ecc_unwrap(output.path(), recovered.path(), false).unwrap();
        let mut rec = Vec::new();
        File::open(recovered.path())
            .unwrap()
            .read_to_end(&mut rec)
            .unwrap();
        assert_eq!(rec, data);
    }

    #[test]
    fn repair_contiguous_start_corruption() {
        // Corrupt a region near the start — where HDF5 superblock would live.
        let data = make_data(50000);
        let input = write_temp(&data);
        let output = NamedTempFile::new().unwrap();
        let recovered = NamedTempFile::new().unwrap();

        let params = EccParams {
            level: EccLevel::Custom,
            k: 20,
            p: 8,
            block_size: 512,
        };
        ecc_wrap(input.path(), output.path(), params, false).unwrap();

        // Zero out a contiguous region just after the header (offset 128..2000)
        {
            let mut f = OpenOptions::new().write(true).open(output.path()).unwrap();
            f.seek(SeekFrom::Start(300)).unwrap();
            f.write_all(&vec![0u8; 1500]).unwrap();
            f.flush().unwrap();
        }
        ecc_unwrap(output.path(), recovered.path(), false).unwrap();
        let mut rec = Vec::new();
        File::open(recovered.path())
            .unwrap()
            .read_to_end(&mut rec)
            .unwrap();
        assert_eq!(rec, data);
    }

    #[test]
    fn exceed_capacity_fails() {
        // With p=1 per stripe, losing 2 data blocks in one stripe must fail.
        let data = make_data(2048);
        let input = write_temp(&data);
        let output = NamedTempFile::new().unwrap();
        let recovered = NamedTempFile::new().unwrap();

        let params = EccParams {
            level: EccLevel::Custom,
            k: 4,
            p: 1,
            block_size: 512,
        };
        ecc_wrap(input.path(), output.path(), params, false).unwrap();

        // There's only one stripe; zero out a big chunk to break multiple blocks
        {
            let mut f = OpenOptions::new().write(true).open(output.path()).unwrap();
            f.seek(SeekFrom::Start(GLOBAL_HEADER_SIZE as u64)).unwrap();
            // Overwrite 3 full blocks (header+payload each)
            let n = 3 * (BLOCK_HEADER_SIZE + 512);
            f.write_all(&vec![0x55u8; n]).unwrap();
            f.flush().unwrap();
        }
        // Unwrap should return an error (too many corruptions in the single stripe)
        assert!(ecc_unwrap(output.path(), recovered.path(), false).is_err());
    }

    #[test]
    fn repair_in_place() {
        let data = make_data(20000);
        let input = write_temp(&data);
        let output = NamedTempFile::new().unwrap();

        let params = EccParams {
            level: EccLevel::Custom,
            k: 10,
            p: 4,
            block_size: 512,
        };
        ecc_wrap(input.path(), output.path(), params, false).unwrap();

        // Corrupt the middle
        {
            let mut f = OpenOptions::new().write(true).open(output.path()).unwrap();
            let mid = std::fs::metadata(output.path()).unwrap().len() / 2;
            f.seek(SeekFrom::Start(mid)).unwrap();
            f.write_all(&vec![0xAAu8; 1024]).unwrap();
            f.flush().unwrap();
        }

        let res = ecc_repair(output.path(), false).unwrap();
        assert!(res.hash_ok);
        assert!(res.unrepairable_stripes.is_empty());

        // After repair, verify should report no corrupt blocks
        let v = ecc_verify(output.path(), false).unwrap();
        assert_eq!(v.corrupt_blocks, 0);
        assert!(v.hash_ok);

        // And unwrap should reproduce the original
        let recovered = NamedTempFile::new().unwrap();
        ecc_unwrap(output.path(), recovered.path(), false).unwrap();
        let mut rec = Vec::new();
        File::open(recovered.path())
            .unwrap()
            .read_to_end(&mut rec)
            .unwrap();
        assert_eq!(rec, data);
    }

    #[test]
    fn header_recovery_from_tail() {
        let data = make_data(5000);
        let input = write_temp(&data);
        let output = NamedTempFile::new().unwrap();
        let recovered = NamedTempFile::new().unwrap();

        let params = EccParams {
            level: EccLevel::Custom,
            k: 10,
            p: 4,
            block_size: 512,
        };
        ecc_wrap(input.path(), output.path(), params, false).unwrap();

        // Zero out the primary header
        {
            let mut f = OpenOptions::new().write(true).open(output.path()).unwrap();
            f.seek(SeekFrom::Start(0)).unwrap();
            f.write_all(&vec![0u8; GLOBAL_HEADER_SIZE]).unwrap();
            f.flush().unwrap();
        }
        // Should still unwrap using a tail header copy
        ecc_unwrap(output.path(), recovered.path(), false).unwrap();
        let mut rec = Vec::new();
        File::open(recovered.path())
            .unwrap()
            .read_to_end(&mut rec)
            .unwrap();
        assert_eq!(rec, data);
    }

    #[test]
    fn is_ecc_wrapped_detection() {
        let data = make_data(5000);
        let input = write_temp(&data);
        let output = NamedTempFile::new().unwrap();

        assert!(!is_ecc_wrapped(input.path()));

        let params = EccParams {
            level: EccLevel::Low,
            k: 10,
            p: 2,
            block_size: 512,
        };
        ecc_wrap(input.path(), output.path(), params, false).unwrap();
        assert!(is_ecc_wrapped(output.path()));
    }
}
