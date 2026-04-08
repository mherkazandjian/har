use rayon::prelude::*;
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Instant;
use walkdir::WalkDir;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const HAR_FORMAT_ATTR: &str = "har_format";
pub const HAR_FORMAT_VALUE: &str = "bagit-v1";
pub const HAR_VERSION_VALUE: &str = "1.0.0";
pub const DEFAULT_BATCH_SIZE: usize = 64 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

pub(crate) struct FileEntry {
    pub(crate) file_path: std::path::PathBuf,
    pub(crate) rel_path: String,
}

pub(crate) struct SizedEntry {
    pub(crate) file_path: std::path::PathBuf,
    pub(crate) rel_path: String,
    pub(crate) size: u64,
    /// For chunked files: byte offset within the original file (0 = whole file or first chunk)
    pub(crate) chunk_offset: u64,
    /// For chunked files: size of this chunk (None = whole file)
    pub(crate) chunk_size: Option<u64>,
}

pub(crate) struct StreamingBatchFile {
    pub(crate) file_path: std::path::PathBuf,
    pub(crate) rel_path: String,
    pub(crate) offset: usize,
    /// For chunked files: byte offset within the original file
    pub(crate) chunk_offset: u64,
    /// For chunked files: size of this chunk (None = whole file)
    pub(crate) chunk_size: Option<u64>,
}

pub(crate) struct StreamingBatch {
    pub(crate) id: u32,
    pub(crate) files: Vec<StreamingBatchFile>,
    pub(crate) total_bytes: usize,
}

pub(crate) struct IndexRecord {
    pub(crate) bagit_path: String,
    pub(crate) batch_id: u32,
    pub(crate) offset: u64,
    pub(crate) length: u64,
    pub(crate) mode: u32,
    pub(crate) sha256: String,
    pub(crate) uid: u32,
    pub(crate) gid: u32,
    pub(crate) mtime: f64,
    pub(crate) owner: String,
    pub(crate) group: String,
    /// For chunked files: byte offset within the original file (0 = not chunked)
    pub(crate) chunk_offset: u64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn parse_batch_size(s: &str) -> usize {
    let s = s.trim().to_uppercase();
    let (num, mult) = if s.ends_with('K') {
        (&s[..s.len() - 1], 1024)
    } else if s.ends_with('M') {
        (&s[..s.len() - 1], 1024 * 1024)
    } else if s.ends_with('G') {
        (&s[..s.len() - 1], 1024 * 1024 * 1024)
    } else {
        (s.as_str(), 1)
    };
    (num.parse::<f64>().unwrap_or(64.0) * mult as f64) as usize
}

pub(crate) fn human_size(bytes: usize) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    for unit in &units {
        if size < 1024.0 {
            return format!("{:.1} {}", size, unit);
        }
        size /= 1024.0;
    }
    format!("{:.1} PB", size)
}

pub(crate) struct HashResult {
    pub(crate) content: Vec<u8>,
    pub(crate) mode: u32,
    pub(crate) hash: String,
    pub(crate) uid: u32,
    pub(crate) gid: u32,
    pub(crate) mtime: f64,
    pub(crate) owner: String,
    pub(crate) group: String,
}

pub(crate) fn read_and_hash(path: &Path, algo: &str) -> std::io::Result<HashResult> {
    let mut content = Vec::new();
    fs::File::open(path)?.read_to_end(&mut content)?;
    let meta = fs::metadata(path)?;
    let mode = meta.permissions().mode();
    use std::os::unix::fs::MetadataExt;
    let uid = meta.uid();
    let gid = meta.gid();
    let mtime = meta.modified()?.duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_secs_f64();
    let owner = users::get_user_by_uid(uid)
        .map(|u| u.name().to_string_lossy().to_string())
        .unwrap_or_else(|| uid.to_string());
    let group = users::get_group_by_gid(gid)
        .map(|g| g.name().to_string_lossy().to_string())
        .unwrap_or_else(|| gid.to_string());
    let hash = crate::compute_checksum(&content, algo);
    Ok(HashResult { content, mode, hash, uid, gid, mtime, owner, group })
}

/// Split files larger than max_chunk_size into multiple SizedEntry items.
fn chunk_large_files(entries: Vec<SizedEntry>, max_chunk_size: u64) -> Vec<SizedEntry> {
    let mut result = Vec::new();
    for entry in entries {
        if entry.size > max_chunk_size && max_chunk_size > 0 {
            let n_chunks = entry.size.div_ceil(max_chunk_size);
            for i in 0..n_chunks {
                let offset = i * max_chunk_size;
                let len = (entry.size - offset).min(max_chunk_size);
                result.push(SizedEntry {
                    file_path: entry.file_path.clone(),
                    rel_path: entry.rel_path.clone(),
                    size: len,
                    chunk_offset: offset,
                    chunk_size: Some(len),
                });
            }
        } else {
            result.push(entry);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Inventory (shared with legacy mode)
// ---------------------------------------------------------------------------

pub(crate) fn build_inventory(sources: &[&str]) -> (Vec<FileEntry>, Vec<String>) {
    let mut file_entries = Vec::new();
    let mut empty_dirs = Vec::new();

    for source in sources {
        let source = shellexpand::tilde(source).to_string();
        let source_path = Path::new(&source);

        if source_path.is_dir() {
            let source_norm = source_path
                .canonicalize()
                .unwrap_or_else(|_| source_path.to_path_buf());
            let base_dir = source_norm
                .parent()
                .unwrap_or(Path::new(""))
                .to_path_buf();

            for entry in WalkDir::new(&source_norm)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let ep = entry.path().to_path_buf();
                if ep.is_file() {
                    let rel = pathdiff::diff_paths(&ep, &base_dir)
                        .unwrap_or_else(|| ep.clone());
                    file_entries.push(FileEntry {
                        file_path: ep,
                        rel_path: rel.to_string_lossy().to_string(),
                    });
                } else if ep.is_dir() && ep != source_norm {
                    let is_empty = WalkDir::new(&ep)
                        .min_depth(1)
                        .into_iter()
                        .filter_map(|e| e.ok())
                        .next()
                        .is_none();
                    if is_empty {
                        let rel = pathdiff::diff_paths(&ep, &base_dir)
                            .unwrap_or_else(|| ep.clone());
                        empty_dirs.push(rel.to_string_lossy().to_string());
                    }
                }
            }
        } else if source_path.is_file() {
            file_entries.push(FileEntry {
                file_path: source_path.to_path_buf(),
                rel_path: source_path.file_name().unwrap().to_string_lossy().to_string(),
            });
        } else {
            eprintln!("Warning: '{}' is not a valid file or directory; skipping.", source);
        }
    }
    (file_entries, empty_dirs)
}

// ---------------------------------------------------------------------------
// Batching
// ---------------------------------------------------------------------------

pub(crate) fn assign_batches_streaming(entries: &[SizedEntry], batch_size: usize) -> Vec<StreamingBatch> {
    let mut batches = Vec::new();
    let mut current_files = Vec::new();
    let mut current_bytes: usize = 0;
    let mut batch_id: u32 = 0;

    for entry in entries {
        let fsize = entry.size as usize;
        if current_bytes + fsize > batch_size && !current_files.is_empty() {
            batches.push(StreamingBatch {
                id: batch_id,
                files: current_files,
                total_bytes: current_bytes,
            });
            batch_id += 1;
            current_files = Vec::new();
            current_bytes = 0;
        }
        current_files.push(StreamingBatchFile {
            file_path: entry.file_path.clone(),
            rel_path: entry.rel_path.clone(),
            offset: current_bytes,
            chunk_offset: entry.chunk_offset,
            chunk_size: entry.chunk_size,
        });
        current_bytes += fsize;
    }
    if !current_files.is_empty() {
        batches.push(StreamingBatch {
            id: batch_id,
            files: current_files,
            total_bytes: current_bytes,
        });
    }
    batches
}

// ---------------------------------------------------------------------------
// BagIt manifest generation
// ---------------------------------------------------------------------------

pub(crate) fn generate_bagit_txt() -> Vec<u8> {
    b"BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n".to_vec()
}

pub(crate) fn generate_bag_info(total_bytes: usize, file_count: usize) -> Vec<u8> {
    let date = chrono::Local::now().format("%Y-%m-%d");
    format!(
        "Bagging-Date: {}\nBag-Software-Agent: har/{}\nPayload-Oxum: {}.{}\nBag-Size: {}\n",
        date, HAR_VERSION_VALUE, total_bytes, file_count, human_size(total_bytes)
    )
    .into_bytes()
}

pub(crate) fn generate_manifest(records: &[(String, String)]) -> Vec<u8> {
    let mut sorted = records.to_vec();
    sorted.sort();
    let mut out = String::new();
    for (path, sha) in &sorted {
        out.push_str(&format!("{}  {}\n", sha, path));
    }
    out.into_bytes()
}

pub(crate) fn generate_tagmanifest(tag_files: &[(&str, &[u8])], algo: &str) -> Vec<u8> {
    let mut entries: Vec<(String, String)> = tag_files
        .iter()
        .map(|(name, content)| (name.to_string(), crate::compute_checksum(content, algo)))
        .collect();
    entries.sort();
    let mut out = String::new();
    for (name, sha) in &entries {
        out.push_str(&format!("{}  {}\n", sha, name));
    }
    out.into_bytes()
}

// ---------------------------------------------------------------------------
// Split helpers
// ---------------------------------------------------------------------------

pub enum SplitSpec {
    Size(usize),
    Count(usize),
}

pub fn parse_split_arg(value: &str) -> SplitSpec {
    let v = value.trim();
    if let Some(rest) = v.strip_prefix("n=") {
        let n: usize = rest.parse().unwrap_or_else(|_| {
            eprintln!("Error: invalid split count: '{}'", rest);
            std::process::exit(1);
        });
        if n < 1 {
            eprintln!("Error: split count must be >= 1");
            std::process::exit(1);
        }
        SplitSpec::Count(n)
    } else {
        let s = if let Some(rest) = v.strip_prefix("size=") {
            rest
        } else {
            v
        };
        let size = parse_batch_size(s);
        if size == 0 {
            eprintln!("Error: split size must be > 0");
            std::process::exit(1);
        }
        SplitSpec::Size(size)
    }
}

pub fn split_filename(base: &str, index: usize, total: usize) -> String {
    if index == 0 {
        return base.to_string();
    }
    let path = Path::new(base);
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = path.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    let width = 3.max(format!("{}", total.saturating_sub(1)).len());
    let parent = path.parent().unwrap_or(Path::new(""));
    parent.join(format!("{}.{:0>width$}{}", stem, index, ext, width = width))
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
pub(crate) fn distribute_files(mut sized_entries: Vec<SizedEntry>, n_splits: usize) -> Vec<Vec<SizedEntry>> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut bins: Vec<Vec<SizedEntry>> = (0..n_splits).map(|_| Vec::new()).collect();
    if sized_entries.is_empty() || n_splits == 0 {
        return bins;
    }

    // Sort by size descending (first-fit-decreasing)
    sized_entries.sort_by(|a, b| b.size.cmp(&a.size));

    // Min-heap of (total_size, bin_index)
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    for i in 0..n_splits {
        heap.push(Reverse((0u64, i)));
    }

    for entry in sized_entries {
        let Reverse((current_size, bin_idx)) = heap.pop().unwrap();
        let new_size = current_size + entry.size;
        bins[bin_idx].push(entry);
        heap.push(Reverse((new_size, bin_idx)));
    }

    bins
}

fn distribute_batches(mut batches: Vec<StreamingBatch>, n_splits: usize) -> Vec<Vec<StreamingBatch>> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut bins: Vec<Vec<StreamingBatch>> = (0..n_splits).map(|_| Vec::new()).collect();
    if batches.is_empty() || n_splits == 0 {
        return bins;
    }

    // Sort by total_bytes descending (first-fit-decreasing)
    batches.sort_by(|a, b| b.total_bytes.cmp(&a.total_bytes));

    // Min-heap of (total_size, bin_index)
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    for i in 0..n_splits {
        heap.push(Reverse((0u64, i)));
    }

    for batch in batches {
        let Reverse((current_size, bin_idx)) = heap.pop().unwrap();
        let new_size = current_size + batch.total_bytes as u64;
        bins[bin_idx].push(batch);
        heap.push(Reverse((new_size, bin_idx)));
    }

    bins
}

pub fn is_split_archive(h5_path: &str) -> bool {
    let h5_path = shellexpand::tilde(h5_path).to_string();
    match hdf5::File::open(&h5_path) {
        Ok(h5f) => {
            h5f.attr("har_split_count")
                .ok()
                .and_then(|a| a.read_scalar::<u32>().ok())
                .map(|v| v > 1)
                .unwrap_or(false)
        }
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Pack
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn pack_bagit(
    sources: &[&str],
    output_h5: &str,
    compression: Option<&str>,
    compression_opts: Option<u8>,
    shuffle: bool,
    batch_size: usize,
    parallel: usize,
    verbose: bool,
    checksum: Option<&str>,
    xattr_flag: bool,
    validate: bool,
) {
    let output_h5 = shellexpand::tilde(output_h5).to_string();
    let t0 = Instant::now();
    let hash_algo = checksum.unwrap_or("sha256");

    // 1. Inventory (stat only, no content read)
    let (file_entries, empty_dirs) = build_inventory(sources);
    let mut sorted_entries = file_entries;
    sorted_entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    if verbose {
        println!("Inventory: {} files, {} empty dirs", sorted_entries.len(), empty_dirs.len());
    }

    // 2. Stat + sort → sized entries
    let sized_entries: Vec<SizedEntry> = sorted_entries
        .into_iter()
        .map(|e| {
            let size = std::fs::metadata(&e.file_path).map(|m| m.len()).unwrap_or(0);
            SizedEntry { file_path: e.file_path, rel_path: e.rel_path, size, chunk_offset: 0, chunk_size: None }
        })
        .collect();
    let total_size: u64 = sized_entries.iter().map(|e| e.size).sum();
    let file_count = sized_entries.len();

    // 3. Batch assign (by size, no content)
    let batches = assign_batches_streaming(&sized_entries, batch_size);
    if verbose {
        println!("Batches: {} (target {})", batches.len(), human_size(batch_size));
    }

    let mut progress = crate::PipelineProgress::new(verbose);
    // Discovery is synchronous (fast) — mark as instantly complete
    progress.discovery.files_found.store(file_count as u64, Ordering::Relaxed);
    progress.discovery.bytes_found.store(total_size, Ordering::Relaxed);
    progress.discovery.done.store(true, Ordering::Release);
    progress.finish_discovery();
    progress.archive_total = total_size;
    let verbose_file = verbose && !progress.is_tty;

    // 4. Stream: read+hash+write one batch at a time
    let mut index_records: Vec<IndexRecord> = Vec::new();
    let mut manifest_records: Vec<(String, String)> = Vec::new();
    let mut user_metadata_map: std::collections::BTreeMap<String, serde_json::Map<String, serde_json::Value>> =
        std::collections::BTreeMap::new();
    let mut total_bytes: usize = 0;

    let is_lzma = compression == Some("lzma");
    let h5f = hdf5::File::create(&output_h5)
        .unwrap_or_else(|e| panic!("Failed to create '{}': {}", output_h5, e));

    // Root attributes
    h5f.new_attr::<hdf5::types::VarLenUnicode>()
        .shape(())
        .create(HAR_FORMAT_ATTR)
        .unwrap()
        .write_scalar(&HAR_FORMAT_VALUE.parse::<hdf5::types::VarLenUnicode>().unwrap())
        .unwrap();
    h5f.new_attr::<hdf5::types::VarLenUnicode>()
        .shape(())
        .create("har_version")
        .unwrap()
        .write_scalar(&HAR_VERSION_VALUE.parse::<hdf5::types::VarLenUnicode>().unwrap())
        .unwrap();
    h5f.new_attr::<hdf5::types::VarLenUnicode>()
        .shape(())
        .create("har_checksum_algo")
        .unwrap()
        .write_scalar(&hash_algo.parse::<hdf5::types::VarLenUnicode>().unwrap())
        .unwrap();

    let batches_grp = h5f.create_group("batches").unwrap();

    // Build a flat list of upcoming file (name, size) for the queued display
    let all_file_info: Vec<(String, u64)> = batches.iter()
        .flat_map(|b| {
            b.files.iter().enumerate().map(move |(i, f)| {
                let size = if i + 1 < b.files.len() {
                    (b.files[i + 1].offset - f.offset) as u64
                } else {
                    (b.total_bytes - f.offset) as u64
                };
                (f.rel_path.clone(), size)
            })
        })
        .collect();
    let mut global_file_idx: usize = 0;

    for batch in &batches {
        let mut batch_records: Vec<IndexRecord> = Vec::new();

        if parallel <= 1 && !is_lzma {
            // --- Pipelined sequential path: chunked read/write per file ---
            // Create batch dataset with chunked storage for streaming writes
            let chunk_ds_size = crate::READ_CHUNK.min(batch.total_bytes.max(1));
            let mut builder = batches_grp.new_dataset::<u8>()
                .shape(batch.total_bytes)
                .chunk(chunk_ds_size);
            if shuffle {
                builder = builder.shuffle();
            }
            if let Some("gzip") = compression {
                builder = builder.deflate(compression_opts.unwrap_or(4));
            }
            let batch_name = batch.id.to_string();
            let batch_ds = builder.create(batch_name.as_str()).unwrap();

            for f in &batch.files {
                let file_size = fs::metadata(&f.file_path).map(|m| m.len()).unwrap_or(0);

                // Update queued display
                let queued_names: Vec<(String, u64)> = all_file_info.iter()
                    .skip(global_file_idx + 1)
                    .take(3)
                    .cloned()
                    .collect();
                progress.set_queued(&queued_names);
                progress.begin_file(&f.rel_path, file_size);

                let file_meta = crate::read_file_meta(&f.file_path).expect("Failed to stat file");

                // Spawn reader thread — reads file in 1MiB chunks
                let (chunk_tx, chunk_rx) = mpsc::sync_channel::<Vec<u8>>(4);
                let read_path = f.file_path.clone();
                let reader = std::thread::spawn(move || {
                    let mut file = fs::File::open(&read_path)
                        .expect("Failed to open file for reading");
                    let mut buf = vec![0u8; crate::READ_CHUNK];
                    loop {
                        let n = file.read(&mut buf).expect("Failed to read file chunk");
                        if n == 0 { break; }
                        if chunk_tx.send(buf[..n].to_vec()).is_err() { break; }
                    }
                });

                // Main thread: write chunks into batch dataset at file's offset
                let file_base_offset = f.offset;
                let mut write_offset: usize = 0;
                let mut hasher = crate::StreamHasher::new(hash_algo);
                for chunk in chunk_rx.iter() {
                    let chunk_len = chunk.len();
                    hasher.update(&chunk);
                    crate::write_dataset_chunk(&batch_ds, &chunk, file_base_offset + write_offset);
                    write_offset += chunk_len;
                    progress.update_file_progress(write_offset as u64);
                }
                reader.join().expect("Reader thread panicked");

                let hash = hasher.finalize_hex();
                let bagit_path = format!("data/{}", f.rel_path);
                batch_records.push(IndexRecord {
                    bagit_path: bagit_path.clone(),
                    batch_id: batch.id,
                    offset: f.offset as u64,
                    length: write_offset as u64,
                    mode: file_meta.mode,
                    sha256: hash.clone(),
                    uid: file_meta.uid, gid: file_meta.gid, mtime: file_meta.mtime,
                    owner: file_meta.owner, group: file_meta.group,
                    chunk_offset: 0,
                });
                manifest_records.push((bagit_path, hash));

                if xattr_flag {
                    let xattrs = crate::metadata::read_xattrs(&f.file_path);
                    if !xattrs.is_empty() {
                        use base64::Engine;
                        let mut xmap = serde_json::Map::new();
                        for (name, value) in &xattrs {
                            let encoded = base64::engine::general_purpose::STANDARD.encode(value);
                            xmap.insert(name.clone(), serde_json::json!(format!("base64:{}", encoded)));
                        }
                        let mut entry = serde_json::Map::new();
                        entry.insert("xattrs".to_string(), serde_json::Value::Object(xmap));
                        user_metadata_map.insert(f.rel_path.clone(), entry);
                    }
                }

                progress.finish_file(write_offset as u64);
                global_file_idx += 1;
                if verbose_file {
                    println!("  {}", f.rel_path);
                }
            }
        } else {
            // --- Parallel or LZMA path: buffer entire batch then write ---
            let mut batch_buffer = vec![0u8; batch.total_bytes];

            if parallel <= 1 {
                for f in &batch.files {
                    let file_size = fs::metadata(&f.file_path).map(|m| m.len()).unwrap_or(0);
                    let queued_names: Vec<(String, u64)> = all_file_info.iter()
                        .skip(global_file_idx + 1)
                        .take(3)
                        .cloned()
                        .collect();
                    progress.set_queued(&queued_names);
                    progress.begin_file(&f.rel_path, file_size);

                    let hr = read_and_hash(&f.file_path, hash_algo).expect("Failed to read file");
                    let bagit_path = format!("data/{}", f.rel_path);
                    let content_len = hr.content.len();
                    batch_buffer[f.offset..f.offset + content_len].copy_from_slice(&hr.content);
                    batch_records.push(IndexRecord {
                        bagit_path: bagit_path.clone(),
                        batch_id: batch.id,
                        offset: f.offset as u64,
                        length: content_len as u64,
                        mode: hr.mode,
                        sha256: hr.hash.clone(),
                        uid: hr.uid, gid: hr.gid, mtime: hr.mtime,
                        owner: hr.owner, group: hr.group,
                        chunk_offset: 0,
                    });
                    manifest_records.push((bagit_path, hr.hash));
                    if xattr_flag {
                        let xattrs = crate::metadata::read_xattrs(&f.file_path);
                        if !xattrs.is_empty() {
                            use base64::Engine;
                            let mut xmap = serde_json::Map::new();
                            for (name, value) in &xattrs {
                                let encoded = base64::engine::general_purpose::STANDARD.encode(value);
                                xmap.insert(name.clone(), serde_json::json!(format!("base64:{}", encoded)));
                            }
                            let mut entry = serde_json::Map::new();
                            entry.insert("xattrs".to_string(), serde_json::Value::Object(xmap));
                            user_metadata_map.insert(f.rel_path.clone(), entry);
                        }
                    }
                    progress.finish_file(content_len as u64);
                    global_file_idx += 1;
                    if verbose_file {
                        println!("  {}", f.rel_path);
                    }
                }
            } else {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(parallel)
                    .build()
                    .expect("Failed to build thread pool");
                let results: Vec<(usize, HashResult)> = pool.install(|| {
                    batch.files
                        .par_iter()
                        .enumerate()
                        .map(|(i, f)| {
                            let hr = read_and_hash(&f.file_path, hash_algo).expect("Failed to read file");
                            (i, hr)
                        })
                        .collect()
                });
                for (i, hr) in results {
                    let f = &batch.files[i];
                    let bagit_path = format!("data/{}", f.rel_path);
                    let content_len = hr.content.len();
                    batch_buffer[f.offset..f.offset + content_len].copy_from_slice(&hr.content);
                    batch_records.push(IndexRecord {
                        bagit_path: bagit_path.clone(),
                        batch_id: batch.id,
                        offset: f.offset as u64,
                        length: content_len as u64,
                        mode: hr.mode,
                        sha256: hr.hash.clone(),
                        uid: hr.uid, gid: hr.gid, mtime: hr.mtime,
                        owner: hr.owner, group: hr.group,
                        chunk_offset: 0,
                    });
                    manifest_records.push((bagit_path, hr.hash));
                    if xattr_flag {
                        let xattrs = crate::metadata::read_xattrs(&f.file_path);
                        if !xattrs.is_empty() {
                            use base64::Engine;
                            let mut xmap = serde_json::Map::new();
                            for (name, value) in &xattrs {
                                let encoded = base64::engine::general_purpose::STANDARD.encode(value);
                                xmap.insert(name.clone(), serde_json::json!(format!("base64:{}", encoded)));
                            }
                            let mut meta_entry = serde_json::Map::new();
                            meta_entry.insert("xattrs".to_string(), serde_json::Value::Object(xmap));
                            user_metadata_map.insert(f.rel_path.clone(), meta_entry);
                        }
                    }
                    progress.finish_file(content_len as u64);
                    global_file_idx += 1;
                    if verbose_file {
                        println!("  {}", f.rel_path);
                    }
                }
            }

            // Write buffered batch to HDF5
            let write_buf;
            let data = if is_lzma {
                write_buf = crate::lzma_compress(&batch_buffer);
                &write_buf[..]
            } else {
                &batch_buffer[..]
            };
            let builder = batches_grp.new_dataset::<u8>();
            let mut builder = builder.shape(data.len());
            if shuffle {
                builder = builder.shuffle();
            }
            if let Some("gzip") = compression {
                builder = builder.deflate(compression_opts.unwrap_or(4));
            }
            let batch_name = batch.id.to_string();
            let ds = builder.create(batch_name.as_str()).unwrap();
            ds.write_raw(data).unwrap();
            if is_lzma {
                ds.new_attr::<u8>().shape(()).create("har_lzma").unwrap()
                    .write_scalar(&1u8).unwrap();
            }
            drop(batch_buffer);
        }

        index_records.extend(batch_records);
        total_bytes += batch.total_bytes;
    }

    // 5. Write index, manifests, empty_dirs, user_metadata

    // Index — parallel arrays
    let idx_grp = h5f.create_group("index_data").unwrap();

    let paths_data: Vec<String> = index_records.iter().map(|r| r.bagit_path.clone()).collect();
    let batch_ids: Vec<u32> = index_records.iter().map(|r| r.batch_id).collect();
    let offsets: Vec<u64> = index_records.iter().map(|r| r.offset).collect();
    let lengths: Vec<u64> = index_records.iter().map(|r| r.length).collect();
    let modes: Vec<u32> = index_records.iter().map(|r| r.mode).collect();
    let shas: Vec<String> = index_records.iter().map(|r| r.sha256.clone()).collect();
    let uids: Vec<u32> = index_records.iter().map(|r| r.uid).collect();
    let gids: Vec<u32> = index_records.iter().map(|r| r.gid).collect();
    let mtimes: Vec<f64> = index_records.iter().map(|r| r.mtime).collect();
    let owners: Vec<String> = index_records.iter().map(|r| r.owner.clone()).collect();
    let groups: Vec<String> = index_records.iter().map(|r| r.group.clone()).collect();

    idx_grp.new_dataset::<u32>().shape(batch_ids.len()).create("batch_id").unwrap()
        .write_raw(&batch_ids).unwrap();
    idx_grp.new_dataset::<u64>().shape(offsets.len()).create("offset").unwrap()
        .write_raw(&offsets).unwrap();
    idx_grp.new_dataset::<u64>().shape(lengths.len()).create("length").unwrap()
        .write_raw(&lengths).unwrap();
    idx_grp.new_dataset::<u32>().shape(modes.len()).create("mode").unwrap()
        .write_raw(&modes).unwrap();

    let paths_blob = paths_data.join("\n").into_bytes();
    idx_grp.new_dataset::<u8>().shape(paths_blob.len()).create("paths").unwrap()
        .write_raw(&paths_blob).unwrap();
    let shas_blob = shas.join("\n").into_bytes();
    idx_grp.new_dataset::<u8>().shape(shas_blob.len()).create("sha256s").unwrap()
        .write_raw(&shas_blob).unwrap();

    idx_grp.new_dataset::<u32>().shape(uids.len()).create("uid").unwrap()
        .write_raw(&uids).unwrap();
    idx_grp.new_dataset::<u32>().shape(gids.len()).create("gid").unwrap()
        .write_raw(&gids).unwrap();
    idx_grp.new_dataset::<f64>().shape(mtimes.len()).create("mtime").unwrap()
        .write_raw(&mtimes).unwrap();
    let owners_blob = owners.join("\n").into_bytes();
    idx_grp.new_dataset::<u8>().shape(owners_blob.len()).create("owners").unwrap()
        .write_raw(&owners_blob).unwrap();
    let groups_blob = groups.join("\n").into_bytes();
    idx_grp.new_dataset::<u8>().shape(groups_blob.len()).create("groups").unwrap()
        .write_raw(&groups_blob).unwrap();

    idx_grp.new_attr::<u64>().shape(()).create("count").unwrap()
        .write_scalar(&(file_count as u64)).unwrap();

    // BagIt tag files
    let manifest_name = format!("manifest-{}.txt", hash_algo);
    let tagmanifest_name = format!("tagmanifest-{}.txt", hash_algo);
    let bagit_txt = generate_bagit_txt();
    let bag_info_txt = generate_bag_info(total_bytes, file_count);
    let manifest_txt = generate_manifest(&manifest_records);
    let tagmanifest_txt = generate_tagmanifest(&[
        ("bagit.txt", &bagit_txt),
        ("bag-info.txt", &bag_info_txt),
        (&manifest_name, &manifest_txt),
    ], hash_algo);

    let bagit_grp = h5f.create_group("bagit").unwrap();
    for (name, content) in &[
        ("bagit.txt", &bagit_txt[..]),
        ("bag-info.txt", &bag_info_txt[..]),
        (&manifest_name[..], &manifest_txt[..]),
        (&tagmanifest_name[..], &tagmanifest_txt[..]),
    ] {
        bagit_grp.new_dataset::<u8>().shape(content.len()).create(*name).unwrap()
            .write_raw(content).unwrap();
    }

    // Empty dirs
    if !empty_dirs.is_empty() {
        let bagit_empty: Vec<String> = empty_dirs.iter().map(|d| format!("data/{}", d)).collect();
        let blob = bagit_empty.join("\n").into_bytes();
        h5f.new_dataset::<u8>().shape(blob.len()).create("empty_dirs").unwrap()
            .write_raw(&blob).unwrap();
    }

    // User metadata (xattrs)
    crate::metadata::write_user_metadata_dataset(&h5f, &user_metadata_map);

    h5f.close().ok();
    progress.flush_current_file();

    // --- Phase 3: Validation (separate pass) ---
    if validate {
        progress.enable_validation(total_size);
        let h5v = hdf5::File::open(&output_h5)
            .unwrap_or_else(|e| panic!("Failed to reopen '{}' for validation: {}", output_h5, e));
        let idx = h5v.group("index_data").expect("Missing index_data group");
        let count_v = idx.attr("count").unwrap().read_scalar::<u64>().unwrap() as usize;
        let v_batch_ids: Vec<u32> = idx.dataset("batch_id").unwrap().read_raw().unwrap();
        let v_offsets: Vec<u64> = idx.dataset("offset").unwrap().read_raw().unwrap();
        let v_lengths: Vec<u64> = idx.dataset("length").unwrap().read_raw().unwrap();
        let v_paths_blob: Vec<u8> = idx.dataset("paths").unwrap().read_raw().unwrap();
        let v_shas_blob: Vec<u8> = idx.dataset("sha256s").unwrap().read_raw().unwrap();
        let v_paths_str = String::from_utf8(v_paths_blob).unwrap();
        let v_paths: Vec<&str> = v_paths_str.split('\n').collect();
        let v_shas_str = String::from_utf8(v_shas_blob).unwrap();
        let v_shas: Vec<&str> = v_shas_str.split('\n').collect();

        let mut val_errors: Vec<String> = Vec::new();
        // Cache for LZMA batches only (must decompress entire batch)
        let mut lzma_cache: std::collections::BTreeMap<u32, Vec<u8>> = std::collections::BTreeMap::new();
        for i in 0..count_v {
            let bid = v_batch_ids[i];
            let display_name = v_paths[i].strip_prefix("data/").unwrap_or(v_paths[i]);
            progress.begin_validation(display_name);

            let off = v_offsets[i] as usize;
            let len = v_lengths[i] as usize;
            let ds = h5v.dataset(&format!("batches/{}", bid)).unwrap();

            let is_batch_lzma = ds.attr("har_lzma").ok()
                .and_then(|a| a.read_scalar::<u8>().ok())
                .map(|v| v != 0)
                .unwrap_or(false);

            if is_batch_lzma {
                // LZMA: must read+decompress entire batch, then validate
                let batch_data = lzma_cache.entry(bid).or_insert_with(|| {
                    crate::read_dataset_content(&ds)
                });
                let file_bytes = &batch_data[off..off + len];
                let actual = crate::compute_checksum(file_bytes, hash_algo);
                if actual != v_shas[i] {
                    val_errors.push(v_paths[i].to_string());
                }
                progress.finish_validation_file(len as u64);
            } else {
                // Incremental: read + hash in 1MiB chunks via hyperslab
                let mut hasher = crate::StreamHasher::new(hash_algo);
                let mut remaining = len;
                let mut pos = off;
                while remaining > 0 {
                    let chunk_len = remaining.min(crate::READ_CHUNK);
                    let chunk: ndarray::Array1<u8> = ds.read_slice(pos..pos + chunk_len)
                        .expect("Failed to read validation chunk");
                    hasher.update(chunk.as_slice().unwrap());
                    pos += chunk_len;
                    remaining -= chunk_len;
                    progress.finish_validation_file(chunk_len as u64);
                }
                let actual = hasher.finalize_hex();
                if actual != v_shas[i] {
                    val_errors.push(v_paths[i].to_string());
                }
            }
        }
        h5v.close().ok();

        progress.finish();
        let elapsed = t0.elapsed().as_secs_f64();
        println!("\nOperation completed in {:.2} seconds.", elapsed);
        println!("  {} files in {} batches, {} payload, archive: {}",
                 file_count, batches.len(), human_size(total_bytes),
                 human_size(fs::metadata(&output_h5).map(|m| m.len() as usize).unwrap_or(0)));

        if !val_errors.is_empty() {
            eprintln!("VALIDATION FAILED on {} file(s):", val_errors.len());
            for e in val_errors.iter().take(10) {
                eprintln!("  {}", e);
            }
            if val_errors.len() > 10 {
                eprintln!("  ... and {} more", val_errors.len() - 10);
            }
            std::process::exit(1);
        } else {
            println!("Validation passed. {} files verified.", file_count);
        }
    } else {
        progress.finish();
        let elapsed = t0.elapsed().as_secs_f64();
        println!("\nOperation completed in {:.2} seconds.", elapsed);
        println!("  {} files in {} batches, {} payload, archive: {}",
                 file_count, batches.len(), human_size(total_bytes),
                 human_size(fs::metadata(&output_h5).map(|m| m.len() as usize).unwrap_or(0)));
    }
}

// ---------------------------------------------------------------------------
// Extract
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn extract_bagit(
    h5_path: &str,
    extract_dir: &str,
    file_key: Option<&str>,
    validate: bool,
    bagit_raw: bool,
    parallel: usize,
    verbose: bool,
    metadata_json: bool,
    xattr_flag: bool,
) {
    let h5_path = shellexpand::tilde(h5_path).to_string();
    let extract_dir = shellexpand::tilde(extract_dir).to_string();
    let extract_path = Path::new(&extract_dir);

    if !Path::new(&h5_path).exists() {
        eprintln!("Error: archive '{}' not found.", h5_path);
        std::process::exit(1);
    }
    let h5f = match hdf5::File::open(&h5_path) {
        Ok(f) => f,
        Err(e) => { eprintln!("Error: cannot open '{}': {}", h5_path, e); std::process::exit(1); }
    };

    // Read user metadata (for --xattr / --metadata-json)
    let user_metadata = if xattr_flag || metadata_json {
        crate::metadata::read_user_metadata_dataset(&h5f)
    } else {
        std::collections::BTreeMap::new()
    };

    // Read checksum algorithm from archive (defaults to sha256 for older archives)
    let hash_algo = h5f.attr("har_checksum_algo")
        .ok()
        .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
        .map(|v| v.as_str().to_string())
        .unwrap_or_else(|| "sha256".to_string());

    // Read index arrays
    let idx = h5f.group("index_data").expect("Missing index_data group");
    let count = idx.attr("count").unwrap().read_scalar::<u64>().unwrap() as usize;
    let batch_ids: Vec<u32> = idx.dataset("batch_id").unwrap().read_raw().unwrap();
    let offsets: Vec<u64> = idx.dataset("offset").unwrap().read_raw().unwrap();
    let lengths: Vec<u64> = idx.dataset("length").unwrap().read_raw().unwrap();
    let modes: Vec<u32> = idx.dataset("mode").unwrap().read_raw().unwrap();
    let paths_blob: Vec<u8> = idx.dataset("paths").unwrap().read_raw().unwrap();
    let shas_blob: Vec<u8> = idx.dataset("sha256s").unwrap().read_raw().unwrap();
    // Read chunk_offsets if present (for chunked file support)
    let chunk_offsets: Vec<u64> = idx.dataset("chunk_offset")
        .ok()
        .map(|ds| ds.read_raw().unwrap())
        .unwrap_or_else(|| vec![0u64; count]);

    let paths_str = String::from_utf8(paths_blob).unwrap();
    let paths: Vec<&str> = paths_str.split('\n').collect();
    let shas_str = String::from_utf8(shas_blob).unwrap();
    let shas: Vec<&str> = shas_str.split('\n').collect();

    // Single-file extraction
    if let Some(key) = file_key {
        let lookup = if key.starts_with("data/") {
            key.to_string()
        } else {
            format!("data/{}", key)
        };
        // Find all index entries for this path (may be multiple chunks)
        let mut matching_indices: Vec<usize> = paths.iter().enumerate()
            .filter(|(_, p)| **p == lookup)
            .map(|(i, _)| i)
            .collect();
        matching_indices.sort_by_key(|&i| chunk_offsets[i]);

        if !matching_indices.is_empty() {
            let i = matching_indices[0];
            let mode = modes[i];
            let out_path = if bagit_raw { &lookup } else { lookup.strip_prefix("data/").unwrap_or(&lookup) };
            let dest = extract_path.join(out_path);
            if let Some(p) = dest.parent() { fs::create_dir_all(p).ok(); }

            // Write all chunks (sorted by chunk_offset)
            {
                use std::io::Seek;
                let mut outfile = fs::OpenOptions::new().create(true).write(true).open(&dest).unwrap();
                for &ci in &matching_indices {
                    let bid = batch_ids[ci];
                    let off = offsets[ci] as usize;
                    let len = lengths[ci] as usize;
                    let batch_ds = h5f.dataset(&format!("batches/{}", bid)).unwrap();
                    let batch_data = crate::read_dataset_content(&batch_ds);
                    let file_bytes = &batch_data[off..off + len];
                    let co = chunk_offsets[ci];
                    if co > 0 {
                        outfile.seek(std::io::SeekFrom::Start(co)).unwrap();
                    }
                    outfile.write_all(file_bytes).unwrap();
                }
            }
            if mode != 0 { fs::set_permissions(&dest, fs::Permissions::from_mode(mode)).ok(); }

            if validate {
                // Re-read the written file and validate each chunk's hash
                let mut all_ok = true;
                for &ci in &matching_indices {
                    let bid = batch_ids[ci];
                    let off = offsets[ci] as usize;
                    let len = lengths[ci] as usize;
                    let batch_ds = h5f.dataset(&format!("batches/{}", bid)).unwrap();
                    let batch_data = crate::read_dataset_content(&batch_ds);
                    let file_bytes = &batch_data[off..off + len];
                    let actual = crate::compute_checksum(file_bytes, &hash_algo);
                    if actual != shas[ci] {
                        eprintln!("CHECKSUM MISMATCH: {} (chunk at offset {})", out_path, chunk_offsets[ci]);
                        all_ok = false;
                    }
                }
                if !all_ok {
                    std::process::exit(1);
                } else {
                    println!("Validation passed. 1 file verified ({} chunks).", matching_indices.len());
                }
            }

            // Restore xattrs for this file
            let rel_key = lookup.strip_prefix("data/").unwrap_or(&lookup);
            if xattr_flag {
                if let Some(meta) = user_metadata.get(rel_key) {
                    if let Some(xattrs) = meta.get("xattrs") {
                        if let Some(xmap) = xattrs.as_object() {
                            crate::metadata::restore_xattrs_from_json(&dest, xmap);
                        }
                    }
                }
            }

            // Write metadata.json for single-file extract
            if metadata_json {
                if let Some(meta) = user_metadata.get(rel_key) {
                    let mut all = std::collections::BTreeMap::new();
                    let mut fm = crate::metadata::FileMetadata {
                        hdf5_attrs: serde_json::Map::new(),
                        xattrs_json: serde_json::Map::new(),
                        xattrs_raw: std::collections::BTreeMap::new(),
                    };
                    if let Some(xa) = meta.get("xattrs") {
                        if let Some(xmap) = xa.as_object() {
                            fm.xattrs_json = xmap.clone();
                        }
                    }
                    if !fm.is_empty() {
                        all.insert(rel_key.to_string(), fm);
                        crate::metadata::write_metadata_json(extract_path, &all);
                    }
                }
            }

            if verbose { println!("Extracted: {}", out_path); }
        } else {
            eprintln!("Error: '{}' not found in archive.", key);
            std::process::exit(1);
        }
        println!("Extraction complete!");
        return;
    }

    // Full extraction — group by batch
    let mut by_batch: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, &bid) in batch_ids.iter().enumerate().take(count) {
        by_batch.entry(bid).or_default().push(i);
    }

    let mut errors = Vec::new();
    let total_bytes: u64 = lengths.iter().take(count).sum();
    let mut progress = crate::Progress::new(verbose);
    progress.start_phase("Extracting", total_bytes);
    let verbose_file = verbose && !progress.is_tty;

    for (bid, indices) in &by_batch {
        let batch_ds = h5f.dataset(&format!("batches/{}", bid)).unwrap();
        let batch_data = crate::read_dataset_content(&batch_ds);

        if parallel <= 1 {
            for &i in indices {
                let off = offsets[i] as usize;
                let len = lengths[i] as usize;
                let mode = modes[i];
                let file_bytes = &batch_data[off..off + len];
                let path = paths[i];
                let co = chunk_offsets[i];
                let out_path = if bagit_raw { path } else { path.strip_prefix("data/").unwrap_or(path) };

                let dest = extract_path.join(out_path);
                if let Some(p) = dest.parent() { fs::create_dir_all(p).ok(); }
                if co > 0 {
                    // Chunked file: open existing (or create) and seek to chunk offset
                    use std::io::Seek;
                    let mut outfile = fs::OpenOptions::new().create(true).write(true).open(&dest).unwrap();
                    outfile.seek(std::io::SeekFrom::Start(co)).unwrap();
                    outfile.write_all(file_bytes).unwrap();
                } else {
                    fs::File::create(&dest).unwrap().write_all(file_bytes).unwrap();
                }
                if mode != 0 { fs::set_permissions(&dest, fs::Permissions::from_mode(mode)).ok(); }

                // Restore xattrs (only for non-chunked or first chunk)
                if co == 0 {
                    let rel_key = path.strip_prefix("data/").unwrap_or(path);
                    if xattr_flag {
                        if let Some(meta) = user_metadata.get(rel_key) {
                            if let Some(xattrs) = meta.get("xattrs") {
                                if let Some(xmap) = xattrs.as_object() {
                                    crate::metadata::restore_xattrs_from_json(&dest, xmap);
                                }
                            }
                        }
                    }
                }

                if validate && crate::compute_checksum(file_bytes, &hash_algo) != shas[i] {
                    errors.push(out_path.to_string());
                }
                progress.inc(out_path, len as u64);
                if verbose_file { println!("Extracted: {}", out_path); }
            }
        } else {
            // Parallel write
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(parallel)
                .build()
                .unwrap();
            let ep = extract_path.to_path_buf();
            pool.install(|| {
                indices.par_iter().for_each(|&i| {
                    let off = offsets[i] as usize;
                    let len = lengths[i] as usize;
                    let mode = modes[i];
                    let file_bytes = &batch_data[off..off + len];
                    let path = paths[i];
                    let co = chunk_offsets[i];
                    let out_path = if bagit_raw { path } else { path.strip_prefix("data/").unwrap_or(path) };

                    let dest = ep.join(out_path);
                    if let Some(p) = dest.parent() { fs::create_dir_all(p).ok(); }
                    if co > 0 {
                        use std::io::Seek;
                        let mut outfile = fs::OpenOptions::new().create(true).write(true).open(&dest).unwrap();
                        outfile.seek(std::io::SeekFrom::Start(co)).unwrap();
                        outfile.write_all(file_bytes).unwrap();
                    } else {
                        fs::File::create(&dest).unwrap().write_all(file_bytes).unwrap();
                    }
                    if mode != 0 { fs::set_permissions(&dest, fs::Permissions::from_mode(mode)).ok(); }

                    // Restore xattrs (only for non-chunked or first chunk)
                    if co == 0 {
                        let rel_key = path.strip_prefix("data/").unwrap_or(path);
                        if xattr_flag {
                            if let Some(meta) = user_metadata.get(rel_key) {
                                if let Some(xattrs) = meta.get("xattrs") {
                                    if let Some(xmap) = xattrs.as_object() {
                                        crate::metadata::restore_xattrs_from_json(&dest, xmap);
                                    }
                                }
                            }
                        }
                    }
                });
            });
            for &i in indices {
                let path = paths[i];
                let out_path = if bagit_raw { path } else { path.strip_prefix("data/").unwrap_or(path) };
                progress.inc(out_path, lengths[i]);
            }
        }
    }
    progress.finish();

    // Empty dirs
    if let Ok(ds) = h5f.dataset("empty_dirs") {
        let blob: Vec<u8> = ds.read_raw().unwrap();
        let dirs_str = String::from_utf8(blob).unwrap();
        for d in dirs_str.split('\n') {
            if d.is_empty() { continue; }
            let out = if bagit_raw { d } else { d.strip_prefix("data/").unwrap_or(d) };
            fs::create_dir_all(extract_path.join(out)).ok();
            if verbose { println!("Created empty dir: {}", out); }
        }
    }

    // BagIt raw tag files
    if bagit_raw {
        if let Ok(grp) = h5f.group("bagit") {
            for name in grp.member_names().unwrap_or_default() {
                if let Ok(ds) = grp.dataset(&name) {
                    let data: Vec<u8> = ds.read_raw().unwrap();
                    let dest = extract_path.join(&name);
                    fs::File::create(&dest).unwrap().write_all(&data).unwrap();
                    if verbose { println!("Wrote tag file: {}", name); }
                }
            }
        }
    }

    // Write metadata.json manifest
    if metadata_json && !user_metadata.is_empty() {
        let mut all_meta = std::collections::BTreeMap::new();
        for (path, meta) in &user_metadata {
            let mut fm = crate::metadata::FileMetadata {
                hdf5_attrs: serde_json::Map::new(),
                xattrs_json: serde_json::Map::new(),
                xattrs_raw: std::collections::BTreeMap::new(),
            };
            if let Some(xa) = meta.get("xattrs") {
                if let Some(xmap) = xa.as_object() {
                    fm.xattrs_json = xmap.clone();
                }
            }
            if !fm.is_empty() {
                all_meta.insert(path.clone(), fm);
            }
        }
        if !all_meta.is_empty() {
            crate::metadata::write_metadata_json(extract_path, &all_meta);
        }
    }

    if !errors.is_empty() {
        eprintln!("CHECKSUM MISMATCH on {} file(s):", errors.len());
        for e in errors.iter().take(10) { eprintln!("  {}", e); }
        if errors.len() > 10 { eprintln!("  ... and {} more", errors.len() - 10); }
        std::process::exit(1);
    } else if validate {
        println!("Validation passed. {} files verified.", count);
    }

    println!("Extraction complete!");
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

pub fn list_bagit(h5_path: &str, bagit_raw: bool) {
    let h5_path = shellexpand::tilde(h5_path).to_string();
    if !Path::new(&h5_path).exists() {
        eprintln!("Error: archive '{}' not found.", h5_path);
        std::process::exit(1);
    }
    let h5f = match hdf5::File::open(&h5_path) {
        Ok(f) => f,
        Err(e) => { eprintln!("Error: cannot open '{}': {}", h5_path, e); std::process::exit(1); }
    };

    let mut tag_files: Vec<String> = Vec::new();
    if bagit_raw {
        if let Ok(grp) = h5f.group("bagit") {
            tag_files = grp.member_names().unwrap_or_default();
            tag_files.sort();
        }
    }

    let idx = h5f.group("index_data").expect("Missing index_data group");
    let paths_blob: Vec<u8> = idx.dataset("paths").unwrap().read_raw().unwrap();
    let paths_str = String::from_utf8(paths_blob).unwrap();
    let mut paths: Vec<String> = if bagit_raw {
        paths_str.split('\n').map(|p| format!("data/{}", p)).collect()
    } else {
        paths_str.split('\n').map(|p| p.to_string()).collect()
    };
    paths.sort();

    let total = paths.len() + tag_files.len();
    println!("Contents of {} ({} entries)", h5_path, total);
    for t in &tag_files {
        println!("{}", t);
    }
    for p in &paths {
        println!("{}", p);
    }
}

// ---------------------------------------------------------------------------
// Split pack
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn pack_single_split(
    split_index: usize,
    batches: &[StreamingBatch],
    output_path: &str,
    split_count: usize,
    base_name: &str,
    compression: Option<&str>,
    compression_opts: Option<u8>,
    shuffle: bool,
    hash_algo: &str,
    xattr_flag: bool,
    empty_dirs: &[String],
    progress: &crate::SplitProgress,
    verbose: bool,
) -> Option<Vec<(String, usize, String)>> {
    let is_lzma = compression == Some("lzma");
    let verbose_file = verbose && !std::io::IsTerminal::is_terminal(&std::io::stderr());

    let file_count: usize = batches.iter().map(|b| b.files.len()).sum();
    let total_bytes: usize = batches.iter().map(|b| b.total_bytes).sum();

    // Build flat file list for queued display
    let all_file_info: Vec<(String, u64)> = batches.iter()
        .flat_map(|b| b.files.iter().map(|f| {
            let size = f.chunk_size.unwrap_or_else(||
                fs::metadata(&f.file_path).map(|m| m.len()).unwrap_or(0)
            );
            let name = if f.chunk_size.is_some() {
                format!("{} [chunk@{}]", f.rel_path, human_size(f.chunk_offset as usize))
            } else {
                f.rel_path.clone()
            };
            (name, size)
        }))
        .collect();

    let h5f = hdf5::File::create(output_path)
        .unwrap_or_else(|e| panic!("Failed to create '{}': {}", output_path, e));

    // Root attributes
    h5f.new_attr::<hdf5::types::VarLenUnicode>().shape(()).create(HAR_FORMAT_ATTR).unwrap()
        .write_scalar(&HAR_FORMAT_VALUE.parse::<hdf5::types::VarLenUnicode>().unwrap()).unwrap();
    h5f.new_attr::<hdf5::types::VarLenUnicode>().shape(()).create("har_version").unwrap()
        .write_scalar(&HAR_VERSION_VALUE.parse::<hdf5::types::VarLenUnicode>().unwrap()).unwrap();
    h5f.new_attr::<hdf5::types::VarLenUnicode>().shape(()).create("har_checksum_algo").unwrap()
        .write_scalar(&hash_algo.parse::<hdf5::types::VarLenUnicode>().unwrap()).unwrap();
    // Split-specific attributes
    h5f.new_attr::<u32>().shape(()).create("har_split_count").unwrap()
        .write_scalar(&(split_count as u32)).unwrap();
    h5f.new_attr::<u32>().shape(()).create("har_split_index").unwrap()
        .write_scalar(&(split_index as u32)).unwrap();
    h5f.new_attr::<hdf5::types::VarLenUnicode>().shape(()).create("har_split_base").unwrap()
        .write_scalar(&base_name.parse::<hdf5::types::VarLenUnicode>().unwrap()).unwrap();

    let batches_grp = h5f.create_group("batches").unwrap();

    let mut index_records: Vec<IndexRecord> = Vec::new();
    let mut manifest_records: Vec<(String, String)> = Vec::new();
    let mut user_metadata_map: std::collections::BTreeMap<String, serde_json::Map<String, serde_json::Value>> = std::collections::BTreeMap::new();
    let mut global_file_idx: usize = 0;
    for (local_batch_id, batch) in (0_u32..).zip(batches.iter()) {
        let mut batch_buffer = vec![0u8; batch.total_bytes];

        for f in &batch.files {
            let display_size = f.chunk_size.unwrap_or_else(||
                fs::metadata(&f.file_path).map(|m| m.len()).unwrap_or(0)
            );
            let display_name = if f.chunk_size.is_some() {
                format!("{} [chunk@{}]", f.rel_path, human_size(f.chunk_offset as usize))
            } else {
                f.rel_path.clone()
            };
            let queued_names: Vec<(String, u64)> = all_file_info.iter()
                .skip(global_file_idx + 1)
                .take(3)
                .cloned()
                .collect();
            progress.set_queued(split_index, &queued_names);
            progress.begin_file(split_index, &display_name, display_size);

            // Read file in chunks for incremental progress
            let file_meta = crate::read_file_meta(&f.file_path).expect("Failed to stat file");
            let (chunk_tx, chunk_rx) = mpsc::sync_channel::<Vec<u8>>(4);
            let read_path = f.file_path.clone();
            let file_chunk_offset = f.chunk_offset;
            let file_chunk_size = f.chunk_size;
            let reader = std::thread::spawn(move || {
                use std::io::Seek;
                let mut file = fs::File::open(&read_path)
                    .expect("Failed to open file for reading");
                if file_chunk_offset > 0 {
                    file.seek(std::io::SeekFrom::Start(file_chunk_offset)).unwrap();
                }
                let max_bytes = file_chunk_size.map(|s| s as usize);
                let mut total_read = 0usize;
                let mut buf = vec![0u8; crate::READ_CHUNK];
                loop {
                    let to_read = if let Some(max) = max_bytes {
                        crate::READ_CHUNK.min(max - total_read)
                    } else {
                        crate::READ_CHUNK
                    };
                    if to_read == 0 { break; }
                    let n = file.read(&mut buf[..to_read]).expect("Failed to read file chunk");
                    if n == 0 { break; }
                    if chunk_tx.send(buf[..n].to_vec()).is_err() { break; }
                    total_read += n;
                }
            });

            let mut hasher = crate::StreamHasher::new(hash_algo);
            let mut write_offset = f.offset;
            let mut bytes_so_far = 0u64;
            for chunk in chunk_rx.iter() {
                hasher.update(&chunk);
                batch_buffer[write_offset..write_offset + chunk.len()].copy_from_slice(&chunk);
                write_offset += chunk.len();
                bytes_so_far += chunk.len() as u64;
                progress.update_file_progress(split_index, bytes_so_far);
            }
            reader.join().expect("Reader thread panicked");

            let hash = hasher.finalize_hex();
            let content_len = write_offset - f.offset;
            let bagit_path = format!("data/{}", f.rel_path);
            index_records.push(IndexRecord {
                bagit_path: bagit_path.clone(),
                batch_id: local_batch_id,
                offset: f.offset as u64,
                length: content_len as u64,
                mode: file_meta.mode,
                sha256: hash.clone(),
                uid: file_meta.uid, gid: file_meta.gid, mtime: file_meta.mtime,
                owner: file_meta.owner, group: file_meta.group,
                chunk_offset: f.chunk_offset,
            });
            manifest_records.push((bagit_path, hash));

            if xattr_flag {
                let xattrs = crate::metadata::read_xattrs(&f.file_path);
                if !xattrs.is_empty() {
                    use base64::Engine;
                    let mut xmap = serde_json::Map::new();
                    for (name, value) in &xattrs {
                        let encoded = base64::engine::general_purpose::STANDARD.encode(value);
                        xmap.insert(name.clone(), serde_json::json!(format!("base64:{}", encoded)));
                    }
                    let mut entry = serde_json::Map::new();
                    entry.insert("xattrs".to_string(), serde_json::Value::Object(xmap));
                    user_metadata_map.insert(f.rel_path.clone(), entry);
                }
            }

            progress.finish_file(split_index, content_len as u64);
            global_file_idx += 1;
            if verbose_file {
                println!("  [Split {}] {}", split_index, f.rel_path);
            }
        }

        // Write batch dataset (use local_batch_id for per-split numbering)
        if is_lzma {
            let compressed = crate::lzma_compress(&batch_buffer);
            let bid_str = local_batch_id.to_string();
            let ds = batches_grp.new_dataset::<u8>().shape(compressed.len())
                .create(bid_str.as_str()).unwrap();
            ds.write_raw(&compressed).unwrap();
            ds.new_attr::<u8>().shape(()).create("har_lzma").unwrap().write_scalar(&1u8).unwrap();
        } else {
            let ds_name = local_batch_id.to_string();
            let builder = batches_grp.new_dataset::<u8>();
            let mut builder = builder.shape(batch_buffer.len());
            if shuffle {
                builder = builder.shuffle();
            }
            match compression {
                Some("gzip") => {
                    builder = builder.deflate(compression_opts.unwrap_or(4));
                }
                Some("lzf") => {
                    builder = builder.deflate(1);
                }
                _ => {}
            }
            let ds = builder.create(ds_name.as_str()).unwrap();
            ds.write_raw(&batch_buffer).unwrap();
        }
    }

    // Write index_data group
    let idx_grp = h5f.create_group("index_data").unwrap();
    let paths_data: Vec<String> = index_records.iter().map(|r| r.bagit_path.clone()).collect();
    let batch_ids: Vec<u32> = index_records.iter().map(|r| r.batch_id).collect();
    let offsets: Vec<u64> = index_records.iter().map(|r| r.offset).collect();
    let lengths: Vec<u64> = index_records.iter().map(|r| r.length).collect();
    let modes: Vec<u32> = index_records.iter().map(|r| r.mode).collect();
    let shas: Vec<String> = index_records.iter().map(|r| r.sha256.clone()).collect();
    let uids: Vec<u32> = index_records.iter().map(|r| r.uid).collect();
    let gids: Vec<u32> = index_records.iter().map(|r| r.gid).collect();
    let mtimes: Vec<f64> = index_records.iter().map(|r| r.mtime).collect();
    let owners: Vec<String> = index_records.iter().map(|r| r.owner.clone()).collect();
    let groups: Vec<String> = index_records.iter().map(|r| r.group.clone()).collect();

    idx_grp.new_dataset::<u32>().shape(batch_ids.len()).create("batch_id").unwrap()
        .write_raw(&batch_ids).unwrap();
    idx_grp.new_dataset::<u64>().shape(offsets.len()).create("offset").unwrap()
        .write_raw(&offsets).unwrap();
    idx_grp.new_dataset::<u64>().shape(lengths.len()).create("length").unwrap()
        .write_raw(&lengths).unwrap();
    idx_grp.new_dataset::<u32>().shape(modes.len()).create("mode").unwrap()
        .write_raw(&modes).unwrap();
    let paths_blob = paths_data.join("\n").into_bytes();
    idx_grp.new_dataset::<u8>().shape(paths_blob.len()).create("paths").unwrap()
        .write_raw(&paths_blob).unwrap();
    let shas_blob = shas.join("\n").into_bytes();
    idx_grp.new_dataset::<u8>().shape(shas_blob.len()).create("sha256s").unwrap()
        .write_raw(&shas_blob).unwrap();
    idx_grp.new_dataset::<u32>().shape(uids.len()).create("uid").unwrap()
        .write_raw(&uids).unwrap();
    idx_grp.new_dataset::<u32>().shape(gids.len()).create("gid").unwrap()
        .write_raw(&gids).unwrap();
    idx_grp.new_dataset::<f64>().shape(mtimes.len()).create("mtime").unwrap()
        .write_raw(&mtimes).unwrap();
    let owners_blob = owners.join("\n").into_bytes();
    idx_grp.new_dataset::<u8>().shape(owners_blob.len()).create("owners").unwrap()
        .write_raw(&owners_blob).unwrap();
    let groups_blob = groups.join("\n").into_bytes();
    idx_grp.new_dataset::<u8>().shape(groups_blob.len()).create("groups").unwrap()
        .write_raw(&groups_blob).unwrap();
    // Write chunk_offsets (for chunked file support)
    let chunk_offsets: Vec<u64> = index_records.iter().map(|r| r.chunk_offset).collect();
    let has_chunks = chunk_offsets.iter().any(|&co| co > 0);
    if has_chunks {
        idx_grp.new_dataset::<u64>().shape(chunk_offsets.len()).create("chunk_offset").unwrap()
            .write_raw(&chunk_offsets).unwrap();
    }
    idx_grp.new_attr::<u64>().shape(()).create("count").unwrap()
        .write_scalar(&(file_count as u64)).unwrap();

    // BagIt tag files
    let manifest_name = format!("manifest-{}.txt", hash_algo);
    let tagmanifest_name = format!("tagmanifest-{}.txt", hash_algo);
    let bagit_txt = generate_bagit_txt();
    let bag_info_txt = generate_bag_info(total_bytes, file_count);
    let manifest_txt = generate_manifest(&manifest_records);
    let tagmanifest_txt = generate_tagmanifest(&[
        ("bagit.txt", &bagit_txt),
        ("bag-info.txt", &bag_info_txt),
        (&manifest_name, &manifest_txt),
    ], hash_algo);
    let bagit_grp = h5f.create_group("bagit").unwrap();
    for (name, content) in &[
        ("bagit.txt", &bagit_txt[..]),
        ("bag-info.txt", &bag_info_txt[..]),
        (&manifest_name[..], &manifest_txt[..]),
        (&tagmanifest_name[..], &tagmanifest_txt[..]),
    ] {
        bagit_grp.new_dataset::<u8>().shape(content.len()).create(*name).unwrap()
            .write_raw(content).unwrap();
    }

    // Empty dirs (split 0 only)
    if split_index == 0 && !empty_dirs.is_empty() {
        let bagit_empty: Vec<String> = empty_dirs.iter().map(|d| format!("data/{}", d)).collect();
        let blob = bagit_empty.join("\n").into_bytes();
        h5f.new_dataset::<u8>().shape(blob.len()).create("empty_dirs").unwrap()
            .write_raw(&blob).unwrap();
    }

    // User metadata (xattrs)
    crate::metadata::write_user_metadata_dataset(&h5f, &user_metadata_map);

    h5f.close().ok();
    progress.finish_split(split_index);

    // Return manifest entries for the global manifest
    let result: Vec<(String, usize, String)> = index_records.iter()
        .map(|r| (r.bagit_path.clone(), split_index, r.sha256.clone()))
        .collect();
    Some(result)
}

/// Compute checksum incrementally in 1 MB chunks, reporting cumulative progress to a slot.
fn compute_checksum_chunked(
    data: &[u8],
    algo: &str,
    progress: &crate::SplitProgress,
    slot: usize,
    cumulative: &mut u64,
) -> String {
    const CHUNK: usize = 1024 * 1024; // 1 MB
    match algo {
        "md5" => {
            use md5::{Md5, Digest};
            let mut hasher = Md5::new();
            for chunk in data.chunks(CHUNK) {
                hasher.update(chunk);
                *cumulative += chunk.len() as u64;
                progress.validate_update_progress(slot, *cumulative);
            }
            format!("{:x}", hasher.finalize())
        }
        "sha256" => {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            for chunk in data.chunks(CHUNK) {
                hasher.update(chunk);
                *cumulative += chunk.len() as u64;
                progress.validate_update_progress(slot, *cumulative);
            }
            format!("{:x}", hasher.finalize())
        }
        "blake3" => {
            let mut hasher = blake3::Hasher::new();
            for chunk in data.chunks(CHUNK) {
                hasher.update(chunk);
                *cumulative += chunk.len() as u64;
                progress.validate_update_progress(slot, *cumulative);
            }
            hasher.finalize().to_hex().to_string()
        }
        _ => panic!("Unsupported checksum algorithm: {}", algo),
    }
}

/// Info about one chunk of a file within a split.
struct ChunkInfo {
    split_index: usize,
    batch_id: u32,
    offset: u64,
    length: u64,
    expected_sha: String,
    bagit_path: String,
}

/// A logical file to validate, possibly spanning multiple chunks across splits.
struct ValidationTask {
    display_name: String,
    total_size: u64,
    chunks: Vec<ChunkInfo>,
}

/// Scan all split indexes (metadata only, no batch data). Build one ValidationTask
/// per logical file and a file_map for progress display.
fn build_validation_tasks(
    split_count: usize, split_paths: &[String],
) -> (Vec<ValidationTask>, std::collections::HashMap<String, (u64, u32)>, u64, u64) {
    // Collect all chunks grouped by display_name
    let mut tasks_map: std::collections::HashMap<String, ValidationTask> =
        std::collections::HashMap::new();

    for si in 0..split_count {
        let h5 = hdf5::File::open(&split_paths[si])
            .unwrap_or_else(|e| panic!("Failed to open '{}' for index scan: {}", split_paths[si], e));
        let idx = h5.group("index_data").expect("Missing index_data group");
        let count = idx.attr("count").unwrap().read_scalar::<u64>().unwrap() as usize;
        let batch_ids: Vec<u32> = idx.dataset("batch_id").unwrap().read_raw().unwrap();
        let offsets: Vec<u64> = idx.dataset("offset").unwrap().read_raw().unwrap();
        let lengths: Vec<u64> = idx.dataset("length").unwrap().read_raw().unwrap();
        let paths_blob: Vec<u8> = idx.dataset("paths").unwrap().read_raw().unwrap();
        let shas_blob: Vec<u8> = idx.dataset("sha256s").unwrap().read_raw().unwrap();
        let paths_str = String::from_utf8(paths_blob).unwrap();
        let paths: Vec<&str> = paths_str.split('\n').collect();
        let shas_str = String::from_utf8(shas_blob).unwrap();
        let shas: Vec<&str> = shas_str.split('\n').collect();

        for i in 0..count {
            let display_name = paths[i].strip_prefix("data/").unwrap_or(paths[i]).to_string();
            let task = tasks_map.entry(display_name.clone()).or_insert_with(|| ValidationTask {
                display_name: display_name.clone(),
                total_size: 0,
                chunks: Vec::new(),
            });
            task.total_size += lengths[i];
            task.chunks.push(ChunkInfo {
                split_index: si,
                batch_id: batch_ids[i],
                offset: offsets[i],
                length: lengths[i],
                expected_sha: shas[i].to_string(),
                bagit_path: paths[i].to_string(),
            });
        }
    }

    let logical_file_count = tasks_map.len() as u64;
    // file_map for progress: chunk_count=1 per file (each task is one work unit)
    let file_map: std::collections::HashMap<String, (u64, u32)> = tasks_map.iter()
        .map(|(name, t)| (name.clone(), (t.total_size, 1)))
        .collect();
    let total_bytes: u64 = tasks_map.values().map(|t| t.total_size).sum();
    let tasks: Vec<ValidationTask> = tasks_map.into_values().collect();
    (tasks, file_map, logical_file_count, total_bytes)
}

/// Read one chunk's data from its split HDF5 file.
fn read_chunk_from_split(split_paths: &[String], chunk: &ChunkInfo) -> Vec<u8> {
    let h5 = hdf5::File::open(&split_paths[chunk.split_index])
        .unwrap_or_else(|e| panic!("Failed to open '{}' for validation: {}", split_paths[chunk.split_index], e));
    let ds = h5.dataset(&format!("batches/{}", chunk.batch_id)).unwrap();
    let is_lzma = ds.attr("har_lzma").ok()
        .and_then(|a| a.read_scalar::<u8>().ok())
        .map(|v| v != 0)
        .unwrap_or(false);
    let batch_data = if is_lzma { crate::read_dataset_content(&ds) } else { ds.read_raw::<u8>().unwrap() };
    batch_data[chunk.offset as usize..(chunk.offset + chunk.length) as usize].to_vec()
}

/// Parallel validation: each worker thread picks a logical file, reads its chunks
/// from HDF5 directly, and hashes them. HDF5 reads are thread-safe for reading;
/// hashing runs concurrently across workers.
fn validate_splits_parallel(
    tasks: Vec<ValidationTask>,
    split_paths: &[String],
    hash_algo: &str,
    parallel: usize,
    progress: &crate::SplitProgress,
) -> bool {
    let failed = std::sync::atomic::AtomicBool::new(false);
    let (tx, rx) = mpsc::sync_channel::<ValidationTask>(parallel);
    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));

    std::thread::scope(|s| {
        // Feed tasks into the channel
        s.spawn(move || {
            for task in tasks {
                if tx.send(task).is_err() { break; }
            }
        });

        // Workers: each picks a file, reads its chunks from HDF5, hashes them
        let mut workers = Vec::new();
        for _ in 0..parallel {
            let rx = std::sync::Arc::clone(&rx);
            let failed_ref = &failed;
            workers.push(s.spawn(move || {
                loop {
                    let task = match { rx.lock().unwrap().recv() } {
                        Ok(t) => t,
                        Err(_) => break,
                    };
                    if failed_ref.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }

                    let slot = progress.validate_begin_file(&task.display_name, task.total_size);
                    let mut cumulative: u64 = 0;
                    let mut task_ok = true;

                    for chunk in &task.chunks {
                        let data = read_chunk_from_split(split_paths, chunk);
                        let actual = compute_checksum_chunked(
                            &data, hash_algo, progress, slot, &mut cumulative,
                        );
                        if actual != chunk.expected_sha {
                            eprintln!("VALIDATION FAILED: {} (expected {}, got {})",
                                     chunk.bagit_path, chunk.expected_sha, actual);
                            task_ok = false;
                        }
                    }

                    if !task_ok {
                        failed_ref.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    progress.validate_finish_file(slot, task.total_size);
                }
            }));
        }
        for w in workers {
            w.join().unwrap();
        }
    });

    !failed.load(std::sync::atomic::Ordering::Relaxed)
}

fn write_split_manifest(h5_path: &str, entries: &[(String, usize, String)], split_count: usize) {
    let h5f = hdf5::File::append(h5_path)
        .unwrap_or_else(|e| panic!("Failed to reopen '{}' for manifest: {}", h5_path, e));
    let grp = h5f.create_group("split_manifest").unwrap();

    let count = entries.len() as u64;
    grp.new_attr::<u64>().shape(()).create("count").unwrap()
        .write_scalar(&count).unwrap();
    grp.new_attr::<u32>().shape(()).create("split_count").unwrap()
        .write_scalar(&(split_count as u32)).unwrap();

    let paths_blob = entries.iter().map(|(p, _, _)| p.as_str()).collect::<Vec<_>>().join("\n").into_bytes();
    grp.new_dataset::<u8>().shape(paths_blob.len()).create("paths").unwrap()
        .write_raw(&paths_blob).unwrap();

    let split_ids: Vec<u32> = entries.iter().map(|(_, si, _)| *si as u32).collect();
    grp.new_dataset::<u32>().shape(split_ids.len()).create("split_ids").unwrap()
        .write_raw(&split_ids).unwrap();

    let shas_blob = entries.iter().map(|(_, _, s)| s.as_str()).collect::<Vec<_>>().join("\n").into_bytes();
    grp.new_dataset::<u8>().shape(shas_blob.len()).create("sha256s").unwrap()
        .write_raw(&shas_blob).unwrap();

    h5f.close().ok();
}

#[allow(clippy::too_many_arguments)]
pub fn pack_bagit_split(
    sources: &[&str],
    output_h5: &str,
    compression: Option<&str>,
    compression_opts: Option<u8>,
    shuffle: bool,
    batch_size: usize,
    parallel: usize,
    verbose: bool,
    checksum: Option<&str>,
    xattr_flag: bool,
    validate: bool,
    split_spec: SplitSpec,
) {
    let output_h5 = shellexpand::tilde(output_h5).to_string();
    let t0 = Instant::now();
    let hash_algo = checksum.unwrap_or("sha256");

    // Phase 1: Inventory
    let (file_entries, empty_dirs) = build_inventory(sources);
    let mut sorted_entries = file_entries;
    sorted_entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    let sized_entries: Vec<SizedEntry> = sorted_entries
        .into_iter()
        .map(|e| {
            let size = std::fs::metadata(&e.file_path).map(|m| m.len()).unwrap_or(0);
            SizedEntry { file_path: e.file_path, rel_path: e.rel_path, size, chunk_offset: 0, chunk_size: None }
        })
        .collect();
    let total_size: u64 = sized_entries.iter().map(|e| e.size).sum();
    let file_count = sized_entries.len();

    if verbose {
        println!("Inventory: {} files, {} empty dirs, total {}",
                 file_count, empty_dirs.len(), human_size(total_size as usize));
    }

    // Phase 2: Compute target split size and chunk large files
    let target_split_size: u64 = match &split_spec {
        SplitSpec::Size(s) => *s as u64,
        SplitSpec::Count(n) => {
            if *n <= 1 || total_size == 0 { total_size } else { total_size.div_ceil(*n as u64) }
        }
    };
    // Chunk files larger than target_split_size so they can be distributed across splits
    let chunked_entries = if target_split_size > 0 && sized_entries.iter().any(|e| e.size > target_split_size) {
        let chunked = chunk_large_files(sized_entries, target_split_size);
        if verbose {
            let n_chunked = chunked.iter().filter(|e| e.chunk_size.is_some()).count();
            let n_unique: std::collections::HashSet<&str> = chunked.iter()
                .filter(|e| e.chunk_size.is_some())
                .map(|e| e.rel_path.as_str())
                .collect();
            if !n_unique.is_empty() {
                println!("Chunked {} large files into {} pieces (target split size: {})",
                         n_unique.len(), n_chunked, human_size(target_split_size as usize));
            }
        }
        chunked
    } else {
        sized_entries
    };

    // Phase 3: Create batches globally from (possibly chunked) entries
    let batches = assign_batches_streaming(&chunked_entries, batch_size);
    let batch_count = batches.len();
    if verbose {
        println!("Batches: {} (target {})", batch_count, human_size(batch_size));
    }

    // Phase 4: Resolve split count and distribute batches
    let split_count = match &split_spec {
        SplitSpec::Count(n) => *n,
        SplitSpec::Size(s) => {
            if total_size == 0 { 1 } else { (total_size as usize).div_ceil(*s).max(1) }
        }
    };
    let split_count = split_count.max(1).min(batch_count.max(1));

    // If only 1 split, delegate to regular pack_bagit
    if split_count <= 1 {
        if verbose {
            println!("Split resolved to 1 — using regular BagIt mode.");
        }
        pack_bagit(sources, &output_h5, compression, compression_opts,
                   shuffle, batch_size, parallel, verbose, checksum, xattr_flag, validate);
        return;
    }

    // Distribute batches across splits (first-fit-decreasing by batch size)
    let split_batches = distribute_batches(batches, split_count);

    let split_sizes: Vec<u64> = split_batches.iter()
        .map(|sb| sb.iter().map(|b| b.total_bytes as u64).sum())
        .collect();
    let split_filenames: Vec<String> = (0..split_count)
        .map(|i| {
            let full = split_filename(&output_h5, i, split_count);
            Path::new(&full).file_name().unwrap_or_default().to_string_lossy().to_string()
        })
        .collect();
    let split_paths: Vec<String> = (0..split_count)
        .map(|i| split_filename(&output_h5, i, split_count))
        .collect();


    // Phase 4: Pack splits in parallel
    let progress = crate::SplitProgress::new(verbose, split_count, &split_sizes, &split_filenames);
    progress.finish_discovery(file_count as u64, total_size);

    let effective_parallel = parallel.min(split_count);
    let failed = std::sync::atomic::AtomicBool::new(false);

    // Counting semaphore to limit concurrency
    let sem_slots = std::sync::Mutex::new(effective_parallel);
    let sem_cvar = std::sync::Condvar::new();

    let all_manifest_entries: std::sync::Mutex<Vec<(String, usize, String)>> =
        std::sync::Mutex::new(Vec::new());

    let base_name = Path::new(&output_h5).file_name().unwrap_or_default()
        .to_str().unwrap_or(&output_h5).to_string();

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for i in 0..split_count {
            let split_batch_slice = &split_batches[i];
            let split_path = &split_paths[i];
            let progress = &progress;
            let empty_dirs = &empty_dirs;
            let all_manifest = &all_manifest_entries;
            let failed = &failed;
            let sem_slots = &sem_slots;
            let sem_cvar = &sem_cvar;
            let base_name = &base_name;

            handles.push(s.spawn(move || {
                // Acquire semaphore slot
                {
                    let mut slots = sem_slots.lock().unwrap();
                    while *slots == 0 {
                        slots = sem_cvar.wait(slots).unwrap();
                    }
                    *slots -= 1;
                }
                let result = pack_single_split(
                    i, split_batch_slice, split_path, split_count,
                    base_name,
                    compression, compression_opts, shuffle,
                    hash_algo, xattr_flag, empty_dirs, progress, verbose,
                );
                // Release semaphore slot
                {
                    let mut slots = sem_slots.lock().unwrap();
                    *slots += 1;
                    sem_cvar.notify_one();
                }
                match result {
                    Some(entries) => {
                        all_manifest.lock().unwrap().extend(entries);
                    }
                    None => {
                        failed.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });

    if failed.load(std::sync::atomic::Ordering::Relaxed) {
        progress.finish();
        eprintln!("Error: archiving failed in one or more splits.");
        std::process::exit(1);
    }

    // Phase 5: Write global manifest to split 0
    let manifest_entries = all_manifest_entries.into_inner().unwrap();
    write_split_manifest(&split_paths[0], &manifest_entries, split_count);

    // Phase 6: Validation pass — workers read chunks directly from HDF5 in parallel
    if validate {
        let (tasks, file_map, logical_file_count, total_file_bytes) =
            build_validation_tasks(split_count, &split_paths);
        progress.begin_validation_phase(logical_file_count, total_file_bytes, file_map);
        if !validate_splits_parallel(tasks, &split_paths, hash_algo, parallel, &progress) {
            progress.finish();
            eprintln!("Error: validation failed.");
            std::process::exit(1);
        }
    }

    progress.finish();

    let elapsed = t0.elapsed().as_secs_f64();
    println!("Created {} splits: {} files, {} in {:.1}s",
             split_count, file_count, human_size(total_size as usize), elapsed);
}

// ---------------------------------------------------------------------------
// Split extract
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn extract_bagit_split(
    h5_path: &str,
    extract_dir: &str,
    file_key: Option<&str>,
    validate: bool,
    bagit_raw: bool,
    parallel: usize,
    verbose: bool,
    metadata_json: bool,
    xattr_flag: bool,
) {
    let h5_path = shellexpand::tilde(h5_path).to_string();
    let extract_dir = shellexpand::tilde(extract_dir).to_string();

    if !Path::new(&h5_path).exists() {
        eprintln!("Error: archive '{}' not found.", h5_path);
        std::process::exit(1);
    }

    // Read split metadata from primary file
    let h5f = hdf5::File::open(&h5_path)
        .unwrap_or_else(|e| { eprintln!("Error: cannot open '{}': {}", h5_path, e); std::process::exit(1); });
    let split_count = h5f.attr("har_split_count")
        .ok().and_then(|a| a.read_scalar::<u32>().ok()).unwrap_or(1) as usize;
    let split_base = h5f.attr("har_split_base")
        .ok().and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
        .map(|v| v.as_str().to_string())
        .unwrap_or_default();

    // Resolve all split paths relative to the directory of h5_path
    let base_dir = Path::new(&h5_path).parent().unwrap_or(Path::new("."));
    let split_paths: Vec<String> = (0..split_count)
        .map(|i| {
            let fname = split_filename(&split_base, i, split_count);
            base_dir.join(&fname).to_string_lossy().to_string()
        })
        .collect();

    // Verify all splits exist
    for (i, sp) in split_paths.iter().enumerate() {
        if !Path::new(sp).exists() {
            eprintln!("Error: split {} not found: {}", i, sp);
            std::process::exit(1);
        }
    }

    if verbose {
        println!("Split archive: {} splits", split_count);
    }

    // Single-file extraction
    if let Some(key) = file_key {
        // Look up in split_manifest
        let sm = h5f.group("split_manifest")
            .unwrap_or_else(|_| { eprintln!("Error: no split_manifest in primary archive."); std::process::exit(1); });
        let paths_blob: Vec<u8> = sm.dataset("paths").unwrap().read_raw().unwrap();
        let paths_str = String::from_utf8(paths_blob).unwrap();
        let paths: Vec<&str> = paths_str.split('\n').collect();
        let split_ids: Vec<u32> = sm.dataset("split_ids").unwrap().read_raw().unwrap();

        let lookup = if key.starts_with("data/") {
            key.to_string()
        } else {
            format!("data/{}", key)
        };
        // Find ALL matching entries (chunked files may span multiple splits)
        let mut target_splits: Vec<usize> = paths.iter().enumerate()
            .filter(|(_, p)| **p == lookup)
            .map(|(idx, _)| split_ids[idx] as usize)
            .collect();
        target_splits.sort();
        target_splits.dedup();

        if target_splits.is_empty() {
            eprintln!("Error: '{}' not found in split archive.", key);
            std::process::exit(1);
        }

        drop(h5f);
        // Extract from each split that contains chunks of this file
        for &target_split in &target_splits {
            extract_bagit(
                &split_paths[target_split], &extract_dir, Some(key),
                validate, bagit_raw, parallel, verbose, metadata_json, xattr_flag,
            );
        }
        return;
    }
    drop(h5f);

    // Full extraction — extract each split
    for sp in &split_paths {
        extract_bagit(
            sp, &extract_dir, None,
            validate, bagit_raw, parallel, verbose, metadata_json, xattr_flag,
        );
    }
}

// ---------------------------------------------------------------------------
// Split list
// ---------------------------------------------------------------------------

pub fn list_bagit_split(h5_path: &str, bagit_raw: bool) {
    let h5_path = shellexpand::tilde(h5_path).to_string();
    if !Path::new(&h5_path).exists() {
        eprintln!("Error: archive '{}' not found.", h5_path);
        std::process::exit(1);
    }
    let h5f = hdf5::File::open(&h5_path)
        .unwrap_or_else(|e| { eprintln!("Error: cannot open '{}': {}", h5_path, e); std::process::exit(1); });

    let split_count = h5f.attr("har_split_count")
        .ok().and_then(|a| a.read_scalar::<u32>().ok()).unwrap_or(1);

    let sm = h5f.group("split_manifest")
        .unwrap_or_else(|_| { eprintln!("Error: no split_manifest in primary archive."); std::process::exit(1); });
    let paths_blob: Vec<u8> = sm.dataset("paths").unwrap().read_raw().unwrap();
    let paths_str = String::from_utf8(paths_blob).unwrap();
    let mut paths: Vec<String> = if bagit_raw {
        paths_str.split('\n').map(|p| p.to_string()).collect()
    } else {
        paths_str.split('\n')
            .map(|p| p.strip_prefix("data/").unwrap_or(p).to_string())
            .collect()
    };
    paths.sort();

    let total = paths.len();
    println!("Contents of {} ({} entries across {} splits)", h5_path, total, split_count);
    for p in &paths {
        if !p.is_empty() {
            println!("{}", p);
        }
    }
}

// ---------------------------------------------------------------------------
// Format detection
// ---------------------------------------------------------------------------

pub fn is_bagit_archive(h5_path: &str) -> bool {
    let h5_path = shellexpand::tilde(h5_path).to_string();
    match hdf5::File::open(&h5_path) {
        Ok(h5f) => {
            h5f.attr(HAR_FORMAT_ATTR)
                .ok()
                .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
                .map(|v| v.as_str() == HAR_FORMAT_VALUE)
                .unwrap_or(false)
        }
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
