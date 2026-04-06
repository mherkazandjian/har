use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
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

struct FileEntry {
    file_path: std::path::PathBuf,
    rel_path: String,
}

struct InventoryItem {
    bagit_path: String,
    content: Vec<u8>,
    mode: u32,
    sha256: String,
}

struct BatchFile {
    bagit_path: String,
    content: Vec<u8>,
    mode: u32,
    sha256: String,
    offset: usize,
}

struct Batch {
    id: u32,
    files: Vec<BatchFile>,
    total_bytes: usize,
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

fn human_size(bytes: usize) -> String {
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

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn read_and_hash(path: &Path) -> std::io::Result<(Vec<u8>, u32, String)> {
    let mut content = Vec::new();
    fs::File::open(path)?.read_to_end(&mut content)?;
    let mode = fs::metadata(path)?.permissions().mode();
    let sha = sha256_hex(&content);
    Ok((content, mode, sha))
}

// ---------------------------------------------------------------------------
// Inventory (shared with legacy mode)
// ---------------------------------------------------------------------------

pub fn build_inventory(sources: &[&str]) -> (Vec<FileEntry>, Vec<String>) {
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

fn assign_batches(inventory: Vec<InventoryItem>, batch_size: usize) -> Vec<Batch> {
    let mut batches = Vec::new();
    let mut current_files = Vec::new();
    let mut current_bytes: usize = 0;
    let mut batch_id: u32 = 0;

    for item in inventory {
        let fsize = item.content.len();
        if current_bytes + fsize > batch_size && !current_files.is_empty() {
            batches.push(Batch {
                id: batch_id,
                files: current_files,
                total_bytes: current_bytes,
            });
            batch_id += 1;
            current_files = Vec::new();
            current_bytes = 0;
        }
        current_files.push(BatchFile {
            bagit_path: item.bagit_path,
            content: item.content,
            mode: item.mode,
            sha256: item.sha256,
            offset: current_bytes,
        });
        current_bytes += fsize;
    }
    if !current_files.is_empty() {
        batches.push(Batch {
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

fn generate_bagit_txt() -> Vec<u8> {
    b"BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n".to_vec()
}

fn generate_bag_info(total_bytes: usize, file_count: usize) -> Vec<u8> {
    let date = chrono::Local::now().format("%Y-%m-%d");
    format!(
        "Bagging-Date: {}\nBag-Software-Agent: har/{}\nPayload-Oxum: {}.{}\nBag-Size: {}\n",
        date, HAR_VERSION_VALUE, total_bytes, file_count, human_size(total_bytes)
    )
    .into_bytes()
}

fn generate_manifest(records: &[(String, String)]) -> Vec<u8> {
    let mut sorted = records.to_vec();
    sorted.sort();
    let mut out = String::new();
    for (path, sha) in &sorted {
        out.push_str(&format!("{}  {}\n", sha, path));
    }
    out.into_bytes()
}

fn generate_tagmanifest(tag_files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut entries: Vec<(String, String)> = tag_files
        .iter()
        .map(|(name, content)| (name.to_string(), sha256_hex(content)))
        .collect();
    entries.sort();
    let mut out = String::new();
    for (name, sha) in &entries {
        out.push_str(&format!("{}  {}\n", sha, name));
    }
    out.into_bytes()
}

// ---------------------------------------------------------------------------
// Pack
// ---------------------------------------------------------------------------

pub fn pack_bagit(
    sources: &[&str],
    output_h5: &str,
    compression: Option<&str>,
    compression_opts: Option<u8>,
    shuffle: bool,
    batch_size: usize,
    parallel: usize,
    verbose: bool,
) {
    let output_h5 = shellexpand::tilde(output_h5).to_string();
    let t0 = Instant::now();

    // Phase 1: Inventory
    let (file_entries, empty_dirs) = build_inventory(sources);
    let mut sorted_entries = file_entries;
    sorted_entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    if verbose {
        println!("Inventory: {} files, {} empty dirs", sorted_entries.len(), empty_dirs.len());
    }

    // Phase 2: Read + SHA-256
    let inventory: Vec<InventoryItem> = if parallel <= 1 {
        sorted_entries
            .iter()
            .map(|e| {
                let (content, mode, sha) = read_and_hash(&e.file_path).expect("Failed to read file");
                if verbose {
                    println!("  Read: {}", e.rel_path);
                }
                InventoryItem {
                    bagit_path: format!("data/{}", e.rel_path),
                    content,
                    mode,
                    sha256: sha,
                }
            })
            .collect()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(parallel)
            .build()
            .expect("Failed to build thread pool");
        pool.install(|| {
            sorted_entries
                .par_iter()
                .map(|e| {
                    let (content, mode, sha) =
                        read_and_hash(&e.file_path).expect("Failed to read file");
                    InventoryItem {
                        bagit_path: format!("data/{}", e.rel_path),
                        content,
                        mode,
                        sha256: sha,
                    }
                })
                .collect()
        })
    };

    // Phase 3: Batch assignment
    let total_bytes: usize = inventory.iter().map(|i| i.content.len()).sum();
    let file_count = inventory.len();
    let batches = assign_batches(inventory, batch_size);

    if verbose {
        println!("Batches: {} (target {})", batches.len(), human_size(batch_size));
    }

    // Phase 4: Build BagIt manifests
    let manifest_records: Vec<(String, String)> = batches
        .iter()
        .flat_map(|b| b.files.iter())
        .map(|f| (f.bagit_path.clone(), f.sha256.clone()))
        .collect();

    let bagit_txt = generate_bagit_txt();
    let bag_info_txt = generate_bag_info(total_bytes, file_count);
    let manifest_txt = generate_manifest(&manifest_records);
    let tagmanifest_txt = generate_tagmanifest(&[
        ("bagit.txt", &bagit_txt),
        ("bag-info.txt", &bag_info_txt),
        ("manifest-sha256.txt", &manifest_txt),
    ]);

    // Phase 5: Write HDF5
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

    // Index — store as parallel arrays (simpler than compound types in hdf5-rs)
    let idx_grp = h5f.create_group("index_data").unwrap();

    let paths_data: Vec<String> = batches
        .iter()
        .flat_map(|b| b.files.iter().map(|f| f.bagit_path.clone()))
        .collect();
    let batch_ids: Vec<u32> = batches
        .iter()
        .flat_map(|b| b.files.iter().map(|_| b.id))
        .collect();
    let offsets: Vec<u64> = batches
        .iter()
        .flat_map(|b| b.files.iter().map(|f| f.offset as u64))
        .collect();
    let lengths: Vec<u64> = batches
        .iter()
        .flat_map(|b| b.files.iter().map(|f| f.content.len() as u64))
        .collect();
    let modes: Vec<u32> = batches
        .iter()
        .flat_map(|b| b.files.iter().map(|f| f.mode))
        .collect();
    let shas: Vec<String> = batches
        .iter()
        .flat_map(|b| b.files.iter().map(|f| f.sha256.clone()))
        .collect();

    // Write arrays
    idx_grp.new_dataset::<u32>().shape(batch_ids.len()).create("batch_id").unwrap()
        .write_raw(&batch_ids).unwrap();
    idx_grp.new_dataset::<u64>().shape(offsets.len()).create("offset").unwrap()
        .write_raw(&offsets).unwrap();
    idx_grp.new_dataset::<u64>().shape(lengths.len()).create("length").unwrap()
        .write_raw(&lengths).unwrap();
    idx_grp.new_dataset::<u32>().shape(modes.len()).create("mode").unwrap()
        .write_raw(&modes).unwrap();

    // Store paths and shas as byte datasets (fixed-width, newline-joined)
    let paths_blob = paths_data.join("\n").into_bytes();
    idx_grp.new_dataset::<u8>().shape(paths_blob.len()).create("paths").unwrap()
        .write_raw(&paths_blob).unwrap();
    let shas_blob = shas.join("\n").into_bytes();
    idx_grp.new_dataset::<u8>().shape(shas_blob.len()).create("sha256s").unwrap()
        .write_raw(&shas_blob).unwrap();

    // Store file count for parsing
    idx_grp.new_attr::<u64>().shape(()).create("count").unwrap()
        .write_scalar(&(file_count as u64)).unwrap();

    // Batch datasets
    let batches_grp = h5f.create_group("batches").unwrap();
    for batch in &batches {
        let mut buf = vec![0u8; batch.total_bytes];
        for f in &batch.files {
            buf[f.offset..f.offset + f.content.len()].copy_from_slice(&f.content);
        }
        let mut builder = batches_grp.new_dataset::<u8>();
        let mut builder = builder.shape(buf.len());
        if shuffle {
            builder = builder.shuffle();
        }
        if let Some("gzip") = compression {
            builder = builder.deflate(compression_opts.unwrap_or(4));
        }
        let batch_name = batch.id.to_string();
        let ds = builder.create(batch_name.as_str()).unwrap();
        ds.write_raw(&buf).unwrap();

        if verbose {
            println!("  Wrote batch {}: {} files, {}",
                     batch.id, batch.files.len(), human_size(batch.total_bytes));
        }
    }

    // BagIt tag files
    let bagit_grp = h5f.create_group("bagit").unwrap();
    for (name, content) in &[
        ("bagit.txt", &bagit_txt[..]),
        ("bag-info.txt", &bag_info_txt[..]),
        ("manifest-sha256.txt", &manifest_txt[..]),
        ("tagmanifest-sha256.txt", &tagmanifest_txt[..]),
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

    h5f.close().ok();
    let elapsed = t0.elapsed().as_secs_f64();
    println!("\nOperation completed in {:.2} seconds.", elapsed);
    println!("  {} files in {} batches, {} payload, archive: {}",
             file_count, batches.len(), human_size(total_bytes),
             human_size(fs::metadata(&output_h5).map(|m| m.len() as usize).unwrap_or(0)));
}

// ---------------------------------------------------------------------------
// Extract
// ---------------------------------------------------------------------------

pub fn extract_bagit(
    h5_path: &str,
    extract_dir: &str,
    file_key: Option<&str>,
    validate: bool,
    bagit_raw: bool,
    parallel: usize,
    verbose: bool,
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

    // Read index arrays
    let idx = h5f.group("index_data").expect("Missing index_data group");
    let count = idx.attr("count").unwrap().read_scalar::<u64>().unwrap() as usize;
    let batch_ids: Vec<u32> = idx.dataset("batch_id").unwrap().read_raw().unwrap();
    let offsets: Vec<u64> = idx.dataset("offset").unwrap().read_raw().unwrap();
    let lengths: Vec<u64> = idx.dataset("length").unwrap().read_raw().unwrap();
    let modes: Vec<u32> = idx.dataset("mode").unwrap().read_raw().unwrap();
    let paths_blob: Vec<u8> = idx.dataset("paths").unwrap().read_raw().unwrap();
    let shas_blob: Vec<u8> = idx.dataset("sha256s").unwrap().read_raw().unwrap();

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
        let pos = paths.iter().position(|p| *p == lookup);
        match pos {
            Some(i) => {
                let bid = batch_ids[i];
                let off = offsets[i] as usize;
                let len = lengths[i] as usize;
                let mode = modes[i];
                let batch_data: Vec<u8> = h5f.dataset(&format!("batches/{}", bid))
                    .unwrap().read_raw().unwrap();
                let file_bytes = &batch_data[off..off + len];

                let out_path = if bagit_raw { &lookup } else { lookup.strip_prefix("data/").unwrap_or(&lookup) };
                let dest = extract_path.join(out_path);
                if let Some(p) = dest.parent() { fs::create_dir_all(p).ok(); }
                fs::File::create(&dest).unwrap().write_all(file_bytes).unwrap();
                if mode != 0 { fs::set_permissions(&dest, fs::Permissions::from_mode(mode)).ok(); }

                if validate {
                    let actual = sha256_hex(file_bytes);
                    if actual != shas[i] {
                        eprintln!("CHECKSUM MISMATCH: {}", out_path);
                        std::process::exit(1);
                    }
                }
                if verbose { println!("Extracted: {}", out_path); }
            }
            None => {
                eprintln!("Error: '{}' not found in archive.", key);
                std::process::exit(1);
            }
        }
        println!("Extraction complete!");
        return;
    }

    // Full extraction — group by batch
    let mut by_batch: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for i in 0..count {
        by_batch.entry(batch_ids[i]).or_default().push(i);
    }

    let mut errors = Vec::new();

    for (bid, indices) in &by_batch {
        let batch_data: Vec<u8> = h5f.dataset(&format!("batches/{}", bid))
            .unwrap().read_raw().unwrap();

        if parallel <= 1 {
            for &i in indices {
                let off = offsets[i] as usize;
                let len = lengths[i] as usize;
                let mode = modes[i];
                let file_bytes = &batch_data[off..off + len];
                let path = paths[i];
                let out_path = if bagit_raw { path } else { path.strip_prefix("data/").unwrap_or(path) };

                let dest = extract_path.join(out_path);
                if let Some(p) = dest.parent() { fs::create_dir_all(p).ok(); }
                fs::File::create(&dest).unwrap().write_all(file_bytes).unwrap();
                if mode != 0 { fs::set_permissions(&dest, fs::Permissions::from_mode(mode)).ok(); }

                if validate && sha256_hex(file_bytes) != shas[i] {
                    errors.push(out_path.to_string());
                }
                if verbose { println!("Extracted: {}", out_path); }
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
                    let out_path = if bagit_raw { path } else { path.strip_prefix("data/").unwrap_or(path) };

                    let dest = ep.join(out_path);
                    if let Some(p) = dest.parent() { fs::create_dir_all(p).ok(); }
                    fs::File::create(&dest).unwrap().write_all(file_bytes).unwrap();
                    if mode != 0 { fs::set_permissions(&dest, fs::Permissions::from_mode(mode)).ok(); }
                });
            });
        }
    }

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

    if !errors.is_empty() {
        eprintln!("CHECKSUM MISMATCH on {} file(s):", errors.len());
        for e in errors.iter().take(10) { eprintln!("  {}", e); }
        if errors.len() > 10 { eprintln!("  ... and {} more", errors.len() - 10); }
        std::process::exit(1);
    }

    println!("Extraction complete!");
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

pub fn list_bagit(h5_path: &str) {
    let h5_path = shellexpand::tilde(h5_path).to_string();
    if !Path::new(&h5_path).exists() {
        eprintln!("Error: archive '{}' not found.", h5_path);
        std::process::exit(1);
    }
    let h5f = match hdf5::File::open(&h5_path) {
        Ok(f) => f,
        Err(e) => { eprintln!("Error: cannot open '{}': {}", h5_path, e); std::process::exit(1); }
    };

    let idx = h5f.group("index_data").expect("Missing index_data group");
    let paths_blob: Vec<u8> = idx.dataset("paths").unwrap().read_raw().unwrap();
    let paths_str = String::from_utf8(paths_blob).unwrap();
    let mut paths: Vec<&str> = paths_str.split('\n').collect();
    paths.sort();

    println!("Contents of {} ({} files)", h5_path, paths.len());
    for p in &paths {
        println!("{}", p);
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
