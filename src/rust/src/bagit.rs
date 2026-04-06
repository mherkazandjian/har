use rayon::prelude::*;
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
    uid: u32,
    gid: u32,
    mtime: f64,
    owner: String,
    group: String,
}

struct BatchFile {
    bagit_path: String,
    content: Vec<u8>,
    mode: u32,
    sha256: String,
    offset: usize,
    uid: u32,
    gid: u32,
    mtime: f64,
    owner: String,
    group: String,
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

struct HashResult {
    content: Vec<u8>,
    mode: u32,
    hash: String,
    uid: u32,
    gid: u32,
    mtime: f64,
    owner: String,
    group: String,
}

fn read_and_hash(path: &Path, algo: &str) -> std::io::Result<HashResult> {
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

// ---------------------------------------------------------------------------
// Inventory (shared with legacy mode)
// ---------------------------------------------------------------------------

fn build_inventory(sources: &[&str]) -> (Vec<FileEntry>, Vec<String>) {
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
            uid: item.uid,
            gid: item.gid,
            mtime: item.mtime,
            owner: item.owner,
            group: item.group,
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

fn generate_tagmanifest(tag_files: &[(&str, &[u8])], algo: &str) -> Vec<u8> {
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

    // Phase 2: Read + checksum
    let hash_algo = checksum.unwrap_or("sha256");
    let mut progress = crate::Progress::new(if verbose { sorted_entries.len() } else { 0 });
    let verbose_file = verbose && !progress.is_tty;

    let mut user_metadata_map: std::collections::BTreeMap<String, serde_json::Map<String, serde_json::Value>> =
        std::collections::BTreeMap::new();

    let inventory: Vec<InventoryItem> = if parallel <= 1 {
        sorted_entries
            .iter()
            .map(|e| {
                let hr = read_and_hash(&e.file_path, hash_algo).expect("Failed to read file");
                if xattr_flag {
                    let xattrs = crate::metadata::read_xattrs(&e.file_path);
                    if !xattrs.is_empty() {
                        use base64::Engine;
                        let mut xmap = serde_json::Map::new();
                        for (name, value) in &xattrs {
                            let encoded = base64::engine::general_purpose::STANDARD.encode(value);
                            xmap.insert(name.clone(), serde_json::json!(format!("base64:{}", encoded)));
                        }
                        let mut entry = serde_json::Map::new();
                        entry.insert("xattrs".to_string(), serde_json::Value::Object(xmap));
                        user_metadata_map.insert(e.rel_path.clone(), entry);
                    }
                }
                progress.inc(&e.rel_path);
                if verbose_file {
                    println!("  Read: {}", e.rel_path);
                }
                InventoryItem {
                    bagit_path: format!("data/{}", e.rel_path),
                    content: hr.content,
                    mode: hr.mode,
                    sha256: hr.hash,
                    uid: hr.uid, gid: hr.gid, mtime: hr.mtime,
                    owner: hr.owner, group: hr.group,
                }
            })
            .collect()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(parallel)
            .build()
            .expect("Failed to build thread pool");
        let items: Vec<InventoryItem> = pool.install(|| {
            sorted_entries
                .par_iter()
                .map(|e| {
                    let hr = read_and_hash(&e.file_path, hash_algo).expect("Failed to read file");
                    InventoryItem {
                        bagit_path: format!("data/{}", e.rel_path),
                        content: hr.content,
                        mode: hr.mode,
                        sha256: hr.hash,
                        uid: hr.uid, gid: hr.gid, mtime: hr.mtime,
                        owner: hr.owner, group: hr.group,
                    }
                })
                .collect()
        });
        if xattr_flag {
            for (item, entry) in items.iter().zip(sorted_entries.iter()) {
                let xattrs = crate::metadata::read_xattrs(&entry.file_path);
                if !xattrs.is_empty() {
                    use base64::Engine;
                    let mut xmap = serde_json::Map::new();
                    for (name, value) in &xattrs {
                        let encoded = base64::engine::general_purpose::STANDARD.encode(value);
                        xmap.insert(name.clone(), serde_json::json!(format!("base64:{}", encoded)));
                    }
                    let mut meta_entry = serde_json::Map::new();
                    meta_entry.insert("xattrs".to_string(), serde_json::Value::Object(xmap));
                    let rel = item.bagit_path.strip_prefix("data/").unwrap_or(&item.bagit_path);
                    user_metadata_map.insert(rel.to_string(), meta_entry);
                }
            }
        }
        for item in &items {
            progress.inc(&item.bagit_path);
        }
        items
    };
    progress.finish();

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
    h5f.new_attr::<hdf5::types::VarLenUnicode>()
        .shape(())
        .create("har_checksum_algo")
        .unwrap()
        .write_scalar(&hash_algo.parse::<hdf5::types::VarLenUnicode>().unwrap())
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
    let uids: Vec<u32> = batches.iter().flat_map(|b| b.files.iter().map(|f| f.uid)).collect();
    let gids: Vec<u32> = batches.iter().flat_map(|b| b.files.iter().map(|f| f.gid)).collect();
    let mtimes: Vec<f64> = batches.iter().flat_map(|b| b.files.iter().map(|f| f.mtime)).collect();
    let owners: Vec<String> = batches.iter().flat_map(|b| b.files.iter().map(|f| f.owner.clone())).collect();
    let groups: Vec<String> = batches.iter().flat_map(|b| b.files.iter().map(|f| f.group.clone())).collect();

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

    // Owner/group/mtime arrays
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

    // Store file count for parsing
    idx_grp.new_attr::<u64>().shape(()).create("count").unwrap()
        .write_scalar(&(file_count as u64)).unwrap();

    // Batch datasets
    let is_lzma = compression == Some("lzma");
    let batches_grp = h5f.create_group("batches").unwrap();
    for batch in &batches {
        let mut buf = vec![0u8; batch.total_bytes];
        for f in &batch.files {
            buf[f.offset..f.offset + f.content.len()].copy_from_slice(&f.content);
        }
        let write_buf;
        let data = if is_lzma {
            write_buf = crate::lzma_compress(&buf);
            &write_buf[..]
        } else {
            &buf[..]
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
    let elapsed = t0.elapsed().as_secs_f64();
    println!("\nOperation completed in {:.2} seconds.", elapsed);
    println!("  {} files in {} batches, {} payload, archive: {}",
             file_count, batches.len(), human_size(total_bytes),
             human_size(fs::metadata(&output_h5).map(|m| m.len() as usize).unwrap_or(0)));
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
                let batch_ds = h5f.dataset(&format!("batches/{}", bid)).unwrap();
                let batch_data = crate::read_dataset_content(&batch_ds);
                let file_bytes = &batch_data[off..off + len];

                let out_path = if bagit_raw { &lookup } else { lookup.strip_prefix("data/").unwrap_or(&lookup) };
                let dest = extract_path.join(out_path);
                if let Some(p) = dest.parent() { fs::create_dir_all(p).ok(); }
                fs::File::create(&dest).unwrap().write_all(file_bytes).unwrap();
                if mode != 0 { fs::set_permissions(&dest, fs::Permissions::from_mode(mode)).ok(); }

                if validate {
                    let actual = crate::compute_checksum(file_bytes, &hash_algo);
                    if actual != shas[i] {
                        eprintln!("CHECKSUM MISMATCH: {}", out_path);
                        std::process::exit(1);
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
    for (i, &bid) in batch_ids.iter().enumerate().take(count) {
        by_batch.entry(bid).or_default().push(i);
    }

    let mut errors = Vec::new();
    let mut progress = crate::Progress::new(if verbose { count } else { 0 });
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
                let out_path = if bagit_raw { path } else { path.strip_prefix("data/").unwrap_or(path) };

                let dest = extract_path.join(out_path);
                if let Some(p) = dest.parent() { fs::create_dir_all(p).ok(); }
                fs::File::create(&dest).unwrap().write_all(file_bytes).unwrap();
                if mode != 0 { fs::set_permissions(&dest, fs::Permissions::from_mode(mode)).ok(); }

                // Restore xattrs
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

                if validate && crate::compute_checksum(file_bytes, &hash_algo) != shas[i] {
                    errors.push(out_path.to_string());
                }
                progress.inc(out_path);
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
                    let out_path = if bagit_raw { path } else { path.strip_prefix("data/").unwrap_or(path) };

                    let dest = ep.join(out_path);
                    if let Some(p) = dest.parent() { fs::create_dir_all(p).ok(); }
                    fs::File::create(&dest).unwrap().write_all(file_bytes).unwrap();
                    if mode != 0 { fs::set_permissions(&dest, fs::Permissions::from_mode(mode)).ok(); }

                    // Restore xattrs
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
                });
            });
            for &i in indices {
                let path = paths[i];
                let out_path = if bagit_raw { path } else { path.strip_prefix("data/").unwrap_or(path) };
                progress.inc(out_path);
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
