use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------

pub struct Progress {
    total: usize,
    current: usize,
    recent: Vec<String>,
    prev_lines: usize,
    pub is_tty: bool,
}

impl Progress {
    pub fn new(total: usize) -> Self {
        Progress {
            total,
            current: 0,
            recent: Vec::new(),
            prev_lines: 0,
            is_tty: io::stderr().is_terminal() && total > 0,
        }
    }

    pub fn inc(&mut self, filename: &str) {
        self.current += 1;
        self.recent.push(filename.to_string());
        if self.recent.len() > 5 {
            self.recent.remove(0);
        }
        if self.is_tty {
            self.render();
        }
    }

    fn render(&mut self) {
        let mut err = io::stderr().lock();
        if self.prev_lines > 0 {
            write!(err, "\x1b[{}A", self.prev_lines).ok();
        }

        let pct = self.current * 100 / self.total.max(1);
        let bar_width = 40;
        let filled = bar_width * self.current / self.total.max(1);
        let empty = bar_width - filled;

        writeln!(
            err,
            "\x1b[2K[{}{}] {}% ({}/{})",
            "█".repeat(filled),
            "░".repeat(empty),
            pct,
            self.current,
            self.total
        )
        .ok();

        for f in &self.recent {
            writeln!(err, "\x1b[2K  {}", f).ok();
        }

        self.prev_lines = 1 + self.recent.len();
        err.flush().ok();
    }

    pub fn finish(&mut self) {
        if !self.is_tty || self.prev_lines == 0 {
            return;
        }
        let mut err = io::stderr().lock();
        write!(err, "\x1b[{}A", self.prev_lines).ok();
        for _ in 0..self.prev_lines {
            writeln!(err, "\x1b[2K").ok();
        }
        write!(err, "\x1b[{}A", self.prev_lines).ok();
        err.flush().ok();
        self.prev_lines = 0;
    }
}

/// Compress data with LZMA.
pub fn lzma_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    lzma_rs::lzma_compress(
        &mut io::BufReader::new(data),
        &mut out,
    )
    .expect("LZMA compression failed");
    out
}

/// Decompress LZMA-compressed data.
pub fn lzma_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    lzma_rs::lzma_decompress(
        &mut io::BufReader::new(data),
        &mut out,
    )
    .expect("LZMA decompression failed");
    out
}

/// Compute a hex digest using the specified algorithm.
pub fn compute_checksum(data: &[u8], algo: &str) -> String {
    match algo {
        "md5" => {
            use md5::Md5;
            let mut hasher = Md5::new();
            hasher.update(data);
            format!("{:x}", hasher.finalize())
        }
        "sha256" => {
            let mut hasher = Sha256::new();
            hasher.update(data);
            format!("{:x}", hasher.finalize())
        }
        "blake3" => {
            let hash = blake3::hash(data);
            hash.to_hex().to_string()
        }
        _ => panic!("Unsupported checksum algorithm: {}", algo),
    }
}

/// Entry collected during the inventory phase.
struct FileEntry {
    file_path: PathBuf,
    rel_path: String,
}

/// Data read from a file, ready to be written into HDF5.
struct ReadResult {
    file_path: PathBuf,
    rel_path: String,
    content: Vec<u8>,
    mode: u32,
    uid: u32,
    gid: u32,
    owner: String,
    group: String,
    mtime: f64,
}

/// File data read from the filesystem.
struct FileData {
    content: Vec<u8>,
    mode: u32,
    uid: u32,
    gid: u32,
    owner: String,
    group: String,
    mtime: f64,
}

/// Read a file's bytes and metadata.
fn read_file(path: &Path) -> io::Result<FileData> {
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
    Ok(FileData { content, mode, uid, gid, owner, group, mtime })
}

/// Read dataset content, automatically decompressing LZMA if marked.
pub fn read_dataset_content(ds: &hdf5::Dataset) -> Vec<u8> {
    let data: Vec<u8> = ds.read_raw().expect("Failed to read dataset");
    let is_lzma = ds
        .attr("har_lzma")
        .ok()
        .and_then(|a| a.read_scalar::<u8>().ok())
        .map(|v| v != 0)
        .unwrap_or(false);
    if is_lzma {
        lzma_decompress(&data)
    } else {
        data
    }
}

/// Write extracted file to disk.
fn write_extracted_file(name: &str, data: &[u8], mode: Option<u32>, extract_dir: &Path) {
    let dest = extract_dir.join(name);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).ok();
    }
    let mut f = fs::File::create(&dest).expect("Failed to create extracted file");
    f.write_all(data).expect("Failed to write extracted file");
    if let Some(m) = mode {
        fs::set_permissions(&dest, fs::Permissions::from_mode(m)).ok();
    }
}

/// Ensure a group path exists in the HDF5 file (creating intermediate groups as needed).
fn ensure_group(h5f: &hdf5::File, path: &str) -> hdf5::Group {
    if let Ok(g) = h5f.group(path) {
        return g;
    }
    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let mut current = String::new();
    let mut grp = h5f.group("/").expect("Cannot open root group");
    for part in parts {
        if current.is_empty() {
            current = part.to_string();
        } else {
            current = format!("{}/{}", current, part);
        }
        grp = match h5f.group(&current) {
            Ok(g) => g,
            Err(_) => h5f
                .create_group(&current)
                .unwrap_or_else(|e| panic!("Failed to create group '{}': {}", current, e)),
        };
    }
    grp
}

/// Create a dataset in the HDF5 file with optional compression.
fn create_dataset(
    h5f: &hdf5::File,
    rel_path: &str,
    content: &[u8],
    compression: Option<&str>,
    compression_opts: Option<u8>,
    shuffle: bool,
    checksum: Option<&str>,
) -> hdf5::Dataset {
    let is_lzma = compression == Some("lzma");
    let write_data;
    let data = if is_lzma {
        write_data = lzma_compress(content);
        &write_data[..]
    } else {
        content
    };

    let builder = h5f.new_dataset::<u8>();
    let mut builder = builder.shape(data.len());

    if shuffle {
        builder = builder.shuffle();
    }

    match compression {
        Some("gzip") => {
            builder = builder.deflate(compression_opts.unwrap_or(4));
        }
        Some("lzf") => {
            // lzf not available in hdf5 crate; fall back to gzip level 1 (fast)
            builder = builder.deflate(1);
        }
        _ => {}
    }

    let ds = builder
        .create(rel_path)
        .unwrap_or_else(|e| panic!("Failed to create dataset '{}': {}", rel_path, e));
    ds.write_raw(data)
        .unwrap_or_else(|e| panic!("Failed to write dataset '{}': {}", rel_path, e));

    if is_lzma {
        ds.new_attr::<u8>()
            .shape(())
            .create("har_lzma")
            .expect("Failed to create har_lzma attr")
            .write_scalar(&1u8)
            .expect("Failed to write har_lzma attr");
    }

    if let Some(algo) = checksum {
        let hash = compute_checksum(content, algo);
        ds.new_attr::<hdf5::types::VarLenUnicode>()
            .shape(())
            .create("har_checksum")
            .unwrap()
            .write_scalar(&hash.parse::<hdf5::types::VarLenUnicode>().unwrap())
            .unwrap();
        ds.new_attr::<hdf5::types::VarLenUnicode>()
            .shape(())
            .create("har_checksum_algo")
            .unwrap()
            .write_scalar(&algo.parse::<hdf5::types::VarLenUnicode>().unwrap())
            .unwrap();
    }

    ds
}

/// Archive or append the given sources into an HDF5 archive.
///
/// `file_mode` should be `"w"` to create a new archive (truncate) or `"a"` to append.
#[allow(clippy::too_many_arguments)]
pub fn pack_or_append_to_h5(
    sources: &[&str],
    output_h5: &str,
    file_mode: &str,
    compression: Option<&str>,
    compression_opts: Option<u8>,
    shuffle: bool,
    parallel: usize,
    verbose: bool,
    checksum: Option<&str>,
    xattr_flag: bool,
) {
    let output_h5 = shellexpand::tilde(output_h5).to_string();
    let t0 = Instant::now();

    // Phase 1: Collect file inventory and empty directories.
    let mut file_entries: Vec<FileEntry> = Vec::new();
    let mut empty_dirs: Vec<String> = Vec::new();

    for source in sources {
        let source = shellexpand::tilde(source).to_string();
        let source_path = Path::new(&source);

        if source_path.is_dir() {
            let source_norm = source_path
                .canonicalize()
                .unwrap_or_else(|_| source_path.to_path_buf());
            let base_dir = source_norm
                .parent()
                .unwrap_or_else(|| Path::new(""))
                .to_path_buf();

            for entry in WalkDir::new(&source_norm)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let entry_path = entry.path().to_path_buf();
                if entry_path.is_file() {
                    let rel = pathdiff::diff_paths(&entry_path, &base_dir)
                        .unwrap_or_else(|| entry_path.clone());
                    file_entries.push(FileEntry {
                        file_path: entry_path,
                        rel_path: rel.to_string_lossy().to_string(),
                    });
                } else if entry_path.is_dir() && entry_path != source_norm {
                    // Check if truly empty (no children at all)
                    let is_empty = WalkDir::new(&entry_path)
                        .min_depth(1)
                        .into_iter()
                        .filter_map(|e| e.ok())
                        .next()
                        .is_none();
                    if is_empty {
                        let rel = pathdiff::diff_paths(&entry_path, &base_dir)
                            .unwrap_or_else(|| entry_path.clone());
                        empty_dirs.push(rel.to_string_lossy().to_string());
                    }
                }
            }
        } else if source_path.is_file() {
            file_entries.push(FileEntry {
                file_path: source_path.to_path_buf(),
                rel_path: source_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
            });
        } else {
            eprintln!(
                "Warning: '{}' is not a valid file or directory; skipping.",
                source
            );
        }
    }

    let is_append = file_mode == "a";

    // Open or create the HDF5 file
    let h5f = if is_append {
        hdf5::File::append(&output_h5).unwrap_or_else(|_| {
            hdf5::File::create(&output_h5)
                .unwrap_or_else(|e| panic!("Failed to open/create '{}': {}", output_h5, e))
        })
    } else {
        hdf5::File::create(&output_h5)
            .unwrap_or_else(|e| panic!("Failed to create '{}': {}", output_h5, e))
    };

    let mut progress = Progress::new(if verbose { file_entries.len() } else { 0 });
    let verbose_file = verbose && !progress.is_tty;

    let write_to_h5 =
        |h5f: &hdf5::File, rel_path: &str, content: &[u8], mode: u32,
         uid: u32, gid: u32, owner: &str, group: &str, mtime: f64,
         create_groups: bool, file_path: Option<&Path>| {
            if verbose_file {
                println!("Storing: {}", rel_path);
            }
            if is_append && h5f.dataset(rel_path).is_ok() {
                if verbose_file {
                    println!("Skipping {} (already exists)", rel_path);
                }
                return;
            }
            if create_groups {
                if let Some(parent) = Path::new(rel_path).parent() {
                    let gp = parent.to_string_lossy().to_string();
                    if !gp.is_empty() {
                        ensure_group(h5f, &gp);
                    }
                }
            }
            let ds = create_dataset(h5f, rel_path, content, compression, compression_opts, shuffle, checksum);
            ds.new_attr::<u32>().shape(()).create("mode").expect("create mode").write_scalar(&mode).expect("write mode");
            ds.new_attr::<u32>().shape(()).create("uid").expect("create uid").write_scalar(&uid).expect("write uid");
            ds.new_attr::<u32>().shape(()).create("gid").expect("create gid").write_scalar(&gid).expect("write gid");
            ds.new_attr::<f64>().shape(()).create("mtime").expect("create mtime").write_scalar(&mtime).expect("write mtime");
            let owner_vu: hdf5::types::VarLenUnicode = owner.parse().unwrap();
            ds.new_attr::<hdf5::types::VarLenUnicode>().shape(()).create("owner").expect("create owner").write_scalar(&owner_vu).expect("write owner");
            let group_vu: hdf5::types::VarLenUnicode = group.parse().unwrap();
            ds.new_attr::<hdf5::types::VarLenUnicode>().shape(()).create("group").expect("create group").write_scalar(&group_vu).expect("write group");
            if xattr_flag {
                if let Some(fp) = file_path {
                    let xattrs = metadata::read_xattrs(fp);
                    for (name, value) in &xattrs {
                        let attr_name = format!("xattr.{}", name);
                        if let Ok(a) = ds.new_attr::<u8>().shape(value.len()).create(&*attr_name) {
                            a.write_raw(value).ok();
                        }
                    }
                }
            }
        };

    if parallel <= 1 {
        // Sequential path
        for entry in &file_entries {
            let fd = read_file(&entry.file_path).expect("Failed to read file");
            write_to_h5(&h5f, &entry.rel_path, &fd.content, fd.mode, fd.uid, fd.gid, &fd.owner, &fd.group, fd.mtime, true, Some(&entry.file_path));
            progress.inc(&entry.rel_path);
        }
    } else {
        // Parallel path: read files in parallel, then write to HDF5 sequentially
        let mut all_groups = BTreeSet::new();
        for entry in &file_entries {
            if let Some(parent) = Path::new(&entry.rel_path).parent() {
                let g = parent.to_string_lossy().to_string();
                if !g.is_empty() {
                    all_groups.insert(g);
                }
            }
        }
        for g in &all_groups {
            ensure_group(&h5f, g);
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(parallel)
            .build()
            .expect("Failed to build rayon thread pool");

        let read_results: Vec<ReadResult> = pool.install(|| {
            file_entries
                .par_iter()
                .map(|entry| {
                    let fd = read_file(&entry.file_path).expect("Failed to read file");
                    ReadResult {
                        file_path: entry.file_path.clone(),
                        rel_path: entry.rel_path.clone(),
                        content: fd.content,
                        mode: fd.mode,
                        uid: fd.uid, gid: fd.gid,
                        owner: fd.owner, group: fd.group,
                        mtime: fd.mtime,
                    }
                })
                .collect()
        });

        for r in &read_results {
            write_to_h5(&h5f, &r.rel_path, &r.content, r.mode, r.uid, r.gid, &r.owner, &r.group, r.mtime, false, Some(&r.file_path));
            progress.inc(&r.rel_path);
        }
    }
    progress.finish();

    // Store empty directories
    for rel_dir in &empty_dirs {
        let grp = ensure_group(&h5f, rel_dir);
        grp.new_attr::<u8>()
            .shape(())
            .create("empty_dir")
            .expect("Failed to create empty_dir attr")
            .write_scalar(&1u8)
            .expect("Failed to write empty_dir attr");
        if verbose_file {
            println!("Storing empty dir: {}", rel_dir);
        }
    }

    h5f.close().ok();
    let elapsed = t0.elapsed().as_secs_f64();
    println!("\nOperation completed in {:.2} seconds.", elapsed);
}

/// Types of HDF5 objects we encounter during visitation.
enum H5ObjType {
    Dataset,
    Group,
    EmptyDirGroup,
}

/// Collect all items in an HDF5 file recursively (like h5py's visititems).
/// Returns a list of (name, type) pairs.
fn collect_items(h5f: &hdf5::File) -> Vec<(String, H5ObjType)> {
    let mut items = Vec::new();

    fn visit_group(group: &hdf5::Group, prefix: &str, items: &mut Vec<(String, H5ObjType)>) {
        let member_names = group.member_names().unwrap_or_default();
        for name in member_names {
            let full_path = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{}/{}", prefix, name)
            };

            if group.dataset(&name).is_ok() {
                items.push((full_path, H5ObjType::Dataset));
            } else if let Ok(subgroup) = group.group(&name) {
                let is_empty_dir = subgroup
                    .attr("empty_dir")
                    .ok()
                    .and_then(|a| a.read_scalar::<u8>().ok())
                    .map(|v| v != 0)
                    .unwrap_or(false);
                if is_empty_dir {
                    items.push((full_path.clone(), H5ObjType::EmptyDirGroup));
                } else {
                    items.push((full_path.clone(), H5ObjType::Group));
                }
                visit_group(&subgroup, &full_path, items);
            }
        }
    }

    let root = h5f.group("/").expect("Cannot open root group");
    visit_group(&root, "", &mut items);
    items
}

/// Extract files from an HDF5 archive.
/// Verify a dataset's checksum if stored. Returns true if ok or no checksum stored.
fn verify_dataset_checksum(ds: &hdf5::Dataset, content: &[u8], name: &str) -> bool {
    let algo = ds
        .attr("har_checksum_algo")
        .ok()
        .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
        .map(|v| v.as_str().to_string());
    let stored = ds
        .attr("har_checksum")
        .ok()
        .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
        .map(|v| v.as_str().to_string());
    if let (Some(algo), Some(stored)) = (algo, stored) {
        let actual = compute_checksum(content, &algo);
        if actual != stored {
            eprintln!("CHECKSUM MISMATCH ({}): {} (expected {}, got {})", algo, name, stored, actual);
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
pub fn extract_h5_to_directory(
    h5_path: &str,
    extract_dir: &str,
    file_key: Option<&str>,
    parallel: usize,
    verbose: bool,
    validate: bool,
    _checksum: Option<&str>,
    metadata_json: bool,
    xattr_flag: bool,
) {
    let h5_path = shellexpand::tilde(h5_path).to_string();
    let extract_dir_expanded = shellexpand::tilde(extract_dir).to_string();
    let extract_path = Path::new(&extract_dir_expanded);

    if !Path::new(&h5_path).exists() {
        eprintln!("Error: archive '{}' not found.", h5_path);
        std::process::exit(1);
    }

    let h5f = match hdf5::File::open(&h5_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: cannot open '{}': {}", h5_path, e);
            std::process::exit(1);
        }
    };

    let mut checksum_errors: Vec<String> = Vec::new();
    let mut all_metadata: std::collections::BTreeMap<String, metadata::FileMetadata> =
        std::collections::BTreeMap::new();

    if let Some(key) = file_key {
        match h5f.dataset(key) {
            Ok(ds) => {
                let data = read_dataset_content(&ds);
                if validate && !verify_dataset_checksum(&ds, &data, key) {
                    checksum_errors.push(key.to_string());
                }
                let mode = ds
                    .attr("mode")
                    .ok()
                    .and_then(|a| a.read_scalar::<u32>().ok());
                write_extracted_file(key, &data, mode, extract_path);
                if metadata_json || xattr_flag {
                    let meta = metadata::collect_user_metadata(&ds);
                    if xattr_flag {
                        let dest = extract_path.join(key);
                        metadata::restore_xattrs(&dest, &meta.xattrs_raw);
                    }
                    if !meta.is_empty() {
                        all_metadata.insert(key.to_string(), meta);
                    }
                }
                if verbose {
                    println!("Extracted: {}", key);
                }
            }
            Err(_) => {
                eprintln!("Error: '{}' not found in archive.", key);
                std::process::exit(1);
            }
        }
    } else {
        let items = collect_items(&h5f);
        let total = items.iter().filter(|(_, t)| matches!(t, H5ObjType::Dataset | H5ObjType::EmptyDirGroup)).count();
        let mut progress = Progress::new(if verbose { total } else { 0 });
        let verbose_file = verbose && !progress.is_tty;

        if parallel <= 1 {
            // Sequential full extraction
            for (name, obj_type) in &items {
                match obj_type {
                    H5ObjType::Dataset => {
                        let ds = h5f.dataset(name).expect("Failed to open dataset");
                        let data = read_dataset_content(&ds);
                        if validate && !verify_dataset_checksum(&ds, &data, name) {
                            checksum_errors.push(name.clone());
                        }
                        let mode = ds
                            .attr("mode")
                            .ok()
                            .and_then(|a| a.read_scalar::<u32>().ok());
                        write_extracted_file(name, &data, mode, extract_path);
                        if metadata_json || xattr_flag {
                            let meta = metadata::collect_user_metadata(&ds);
                            if xattr_flag {
                                let dest = extract_path.join(name);
                                metadata::restore_xattrs(&dest, &meta.xattrs_raw);
                            }
                            if !meta.is_empty() {
                                all_metadata.insert(name.clone(), meta);
                            }
                        }
                        progress.inc(name);
                        if verbose_file {
                            println!("Extracted: {}", name);
                        }
                    }
                    H5ObjType::EmptyDirGroup => {
                        let dest = extract_path.join(name);
                        fs::create_dir_all(&dest).ok();
                        progress.inc(name);
                        if verbose_file {
                            println!("Created empty dir: {}", name);
                        }
                    }
                    H5ObjType::Group => {}
                }
            }
        } else {
            // Parallel full extraction
            let mut read_items: Vec<(String, Vec<u8>, Option<u32>)> = Vec::new();
            let mut empty_dir_names: Vec<String> = Vec::new();

            for (name, obj_type) in &items {
                match obj_type {
                    H5ObjType::Dataset => {
                        let ds = h5f.dataset(name).expect("Failed to open dataset");
                        let data = read_dataset_content(&ds);
                        let mode = ds
                            .attr("mode")
                            .ok()
                            .and_then(|a| a.read_scalar::<u32>().ok());
                        if metadata_json || xattr_flag {
                            let meta = metadata::collect_user_metadata(&ds);
                            if !meta.is_empty() {
                                all_metadata.insert(name.clone(), meta);
                            }
                        }
                        read_items.push((name.clone(), data, mode));
                        progress.inc(name);
                    }
                    H5ObjType::EmptyDirGroup => {
                        empty_dir_names.push(name.clone());
                    }
                    H5ObjType::Group => {}
                }
            }

            for name in &empty_dir_names {
                let dest = extract_path.join(name);
                fs::create_dir_all(&dest).ok();
                progress.inc(name);
                if verbose_file {
                    println!("Created empty dir: {}", name);
                }
            }

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(parallel)
                .build()
                .expect("Failed to build rayon thread pool");

            let ep = extract_path.to_path_buf();
            pool.install(|| {
                read_items.par_iter().for_each(|(name, data, mode)| {
                    write_extracted_file(name, data, *mode, &ep);
                });
            });

            if xattr_flag {
                for (name, _, _) in &read_items {
                    if let Some(meta) = all_metadata.get(name) {
                        let dest = ep.join(name);
                        metadata::restore_xattrs(&dest, &meta.xattrs_raw);
                    }
                }
            }

            if verbose_file {
                for (name, _, _) in &read_items {
                    println!("Extracted: {}", name);
                }
            }
        }
        progress.finish();
    }

    if metadata_json {
        metadata::write_metadata_json(extract_path, &all_metadata);
    }

    if !checksum_errors.is_empty() {
        eprintln!("CHECKSUM MISMATCH on {} file(s):", checksum_errors.len());
        for e in checksum_errors.iter().take(10) {
            eprintln!("  {}", e);
        }
        if checksum_errors.len() > 10 {
            eprintln!("  ... and {} more", checksum_errors.len() - 10);
        }
        std::process::exit(1);
    }

    println!("Extraction complete!");
}

/// List all dataset keys in the given HDF5 file.
pub fn list_h5_contents(h5_path: &str) {
    let h5_path = shellexpand::tilde(h5_path).to_string();

    if !Path::new(&h5_path).exists() {
        eprintln!("Error: archive '{}' not found.", h5_path);
        std::process::exit(1);
    }

    let h5f = match hdf5::File::open(&h5_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: cannot open '{}': {}", h5_path, e);
            std::process::exit(1);
        }
    };

    let mut keys: Vec<String> = Vec::new();
    for (name, obj_type) in collect_items(&h5f) {
        if matches!(obj_type, H5ObjType::Dataset) {
            keys.push(name);
        }
    }

    keys.sort();
    println!("Contents of {}", h5_path);
    for key in &keys {
        println!("{}", key);
    }
}

pub mod bagit;
pub mod metadata;
pub mod browse;

#[cfg(test)]
mod tests;
