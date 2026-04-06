use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

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

/// Entry collected during the inventory phase.
struct FileEntry {
    file_path: PathBuf,
    rel_path: String,
}

/// Data read from a file, ready to be written into HDF5.
struct ReadResult {
    rel_path: String,
    content: Vec<u8>,
    mode: u32,
}

/// Read a file's bytes and permission mode.
fn read_file(path: &Path) -> io::Result<(Vec<u8>, u32)> {
    let mut content = Vec::new();
    fs::File::open(path)?.read_to_end(&mut content)?;
    let mode = fs::metadata(path)?.permissions().mode();
    Ok((content, mode))
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

    ds
}

/// Archive or append the given sources into an HDF5 archive.
///
/// `file_mode` should be `"w"` to create a new archive (truncate) or `"a"` to append.
pub fn pack_or_append_to_h5(
    sources: &[&str],
    output_h5: &str,
    file_mode: &str,
    compression: Option<&str>,
    compression_opts: Option<u8>,
    shuffle: bool,
    parallel: usize,
    verbose: bool,
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

    let write_to_h5 =
        |h5f: &hdf5::File, rel_path: &str, content: &[u8], mode: u32, create_groups: bool| {
            if verbose {
                println!("Storing: {}", rel_path);
            }
            if is_append {
                if h5f.dataset(rel_path).is_ok() {
                    if verbose {
                        println!("Skipping {} (already exists)", rel_path);
                    }
                    return;
                }
            }
            if create_groups {
                if let Some(parent) = Path::new(rel_path).parent() {
                    let gp = parent.to_string_lossy().to_string();
                    if !gp.is_empty() {
                        ensure_group(h5f, &gp);
                    }
                }
            }
            let ds = create_dataset(h5f, rel_path, content, compression, compression_opts, shuffle);
            ds.new_attr::<u32>()
                .shape(())
                .create("mode")
                .expect("Failed to create mode attr")
                .write_scalar(&mode)
                .expect("Failed to write mode attr");
        };

    if parallel <= 1 {
        // Sequential path
        for entry in &file_entries {
            let (content, mode) = read_file(&entry.file_path).expect("Failed to read file");
            write_to_h5(&h5f, &entry.rel_path, &content, mode, true);
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
                    let (content, mode) =
                        read_file(&entry.file_path).expect("Failed to read file");
                    ReadResult {
                        rel_path: entry.rel_path.clone(),
                        content,
                        mode,
                    }
                })
                .collect()
        });

        for r in &read_results {
            write_to_h5(&h5f, &r.rel_path, &r.content, r.mode, false);
        }
    }

    // Store empty directories
    for rel_dir in &empty_dirs {
        let grp = ensure_group(&h5f, rel_dir);
        grp.new_attr::<u8>()
            .shape(())
            .create("empty_dir")
            .expect("Failed to create empty_dir attr")
            .write_scalar(&1u8)
            .expect("Failed to write empty_dir attr");
        if verbose {
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
pub fn extract_h5_to_directory(
    h5_path: &str,
    extract_dir: &str,
    file_key: Option<&str>,
    parallel: usize,
    verbose: bool,
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

    if let Some(key) = file_key {
        match h5f.dataset(key) {
            Ok(ds) => {
                let data = read_dataset_content(&ds);
                let mode = ds
                    .attr("mode")
                    .ok()
                    .and_then(|a| a.read_scalar::<u32>().ok());
                write_extracted_file(key, &data, mode, extract_path);
                if verbose {
                    println!("Extracted: {}", key);
                }
            }
            Err(_) => {
                eprintln!("Error: '{}' not found in archive.", key);
                std::process::exit(1);
            }
        }
    } else if parallel <= 1 {
        // Sequential full extraction
        for (name, obj_type) in collect_items(&h5f) {
            match obj_type {
                H5ObjType::Dataset => {
                    let ds = h5f.dataset(&name).expect("Failed to open dataset");
                    let data = read_dataset_content(&ds);
                    let mode = ds
                        .attr("mode")
                        .ok()
                        .and_then(|a| a.read_scalar::<u32>().ok());
                    write_extracted_file(&name, &data, mode, extract_path);
                    if verbose {
                        println!("Extracted: {}", name);
                    }
                }
                H5ObjType::EmptyDirGroup => {
                    let dest = extract_path.join(&name);
                    fs::create_dir_all(&dest).ok();
                    if verbose {
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

        for (name, obj_type) in collect_items(&h5f) {
            match obj_type {
                H5ObjType::Dataset => {
                    let ds = h5f.dataset(&name).expect("Failed to open dataset");
                    let data = read_dataset_content(&ds);
                    let mode = ds
                        .attr("mode")
                        .ok()
                        .and_then(|a| a.read_scalar::<u32>().ok());
                    read_items.push((name, data, mode));
                }
                H5ObjType::EmptyDirGroup => {
                    empty_dir_names.push(name);
                }
                H5ObjType::Group => {}
            }
        }

        for name in &empty_dir_names {
            let dest = extract_path.join(name);
            fs::create_dir_all(&dest).ok();
            if verbose {
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

        if verbose {
            for (name, _, _) in &read_items {
                println!("Extracted: {}", name);
            }
        }
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

#[cfg(test)]
mod tests;
