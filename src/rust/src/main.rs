use clap::Parser;

#[derive(Parser)]
#[command(name = "har", version = env!("HAR_VERSION"), about = "Archive, append, extract, or list an HDF5 archive.")]
struct Cli {
    /// Create archive from one or more directories/files
    #[arg(short = 'c', group = "operation")]
    create: bool,

    /// Append one or more directories/files to an existing archive
    #[arg(short = 'r', group = "operation")]
    append: bool,

    /// Extract from an archive
    #[arg(short = 'x', group = "operation")]
    extract: bool,

    /// List contents of an archive
    #[arg(short = 't', group = "operation")]
    list: bool,

    /// Browse archive interactively (TUI)
    #[arg(short = 'b', long = "browse", group = "operation")]
    browse: bool,

    /// HDF5 archive file
    #[arg(short = 'f', long = "file", required = true)]
    file: String,

    /// Target extraction directory (default: current directory)
    #[arg(short = 'C', long = "directory", default_value = ".")]
    directory: String,

    /// Use gzip compression
    #[arg(short = 'z', long = "gzip")]
    gzip: bool,

    /// Use HDF5 lzf compression
    #[arg(long = "lzf")]
    lzf: bool,

    /// Use HDF5 szip compression
    #[arg(long = "szip")]
    szip: bool,

    /// Use LZMA compression (application-level, no HDF5 filter plugin needed)
    #[arg(long = "lzma")]
    lzma: bool,

    /// Compression level for gzip 1-9 (default: 9)
    #[arg(long = "zopt", default_value = "9")]
    zopt: String,

    /// Use HDF5 shuffle filter before compression
    #[arg(long = "shuffle")]
    shuffle: bool,

    /// Number of parallel workers (default: 1 = sequential)
    #[arg(short = 'p', long = "parallel", default_value = "1")]
    parallel: usize,

    /// Verbosely list files processed
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,

    /// Use BagIt batched storage mode (faster for many files, adds SHA-256 checksums)
    #[arg(long = "bagit")]
    bagit: bool,

    /// Target batch size for --bagit mode (default: 64M)
    #[arg(long = "batch-size", default_value = "64M")]
    batch_size: String,

    /// Verify checksums on extraction and creation
    #[arg(long = "validate")]
    validate: bool,

    /// After creation, extract and compare against source files
    #[arg(long = "validate-roundtrip")]
    validate_roundtrip: bool,

    /// Use byte-for-byte comparison for --validate-roundtrip (default: checksum)
    #[arg(long = "byte-for-byte")]
    byte_for_byte: bool,

    /// Delete source files after successful ingestion
    #[arg(long = "delete-source")]
    delete_source: bool,

    /// Extract as a full BagIt bag with tag files (--bagit mode only)
    #[arg(long = "bagit-raw")]
    bagit_raw: bool,

    /// Checksum algorithm for integrity verification (default: sha256 if flag given without value)
    #[arg(long = "checksum", value_parser = ["md5", "sha256", "blake3"],
          num_args = 0..=1, default_missing_value = "sha256")]
    checksum: Option<String>,

    /// Write metadata.json manifest of user HDF5 attributes on extraction
    #[arg(long = "metadata-json")]
    metadata_json: bool,

    /// Capture/restore filesystem extended attributes (user.* namespace)
    #[arg(long = "xattr")]
    xattr: bool,

    /// Split archive into parts: SIZE (e.g. 100M) or n=COUNT (e.g. n=4). Implies --bagit.
    #[arg(long = "split")]
    split: Option<String>,

    /// Add Reed-Solomon ECC protection to archive: low, medium, high, max, or N%
    #[arg(long = "ecc")]
    ecc: Option<String>,

    /// Heal (repair in place) an ECC-wrapped archive
    #[arg(long = "heal", group = "operation")]
    heal: bool,

    /// Print ECC container info for an ECC-wrapped archive
    #[arg(long = "ecc-info", group = "operation")]
    ecc_info: bool,

    /// Print visual block map of an ECC-wrapped archive
    #[arg(long = "ecc-map", group = "operation")]
    ecc_map: bool,

    /// Verify ECC-wrapped archive integrity (no writes)
    #[arg(long = "ecc-verify", group = "operation")]
    ecc_verify: bool,

    /// Wrap an existing .h5 file with ECC protection (takes level: low/medium/high/max/N%)
    /// Can be used standalone or combined with -c to wrap after creation.
    #[arg(long = "ecc-wrap")]
    ecc_wrap: Option<String>,

    /// Unwrap an ECC-wrapped archive back to a plain .h5 file
    #[arg(long = "ecc-unwrap", group = "operation")]
    ecc_unwrap: bool,

    /// Source directories/files (for -c/-r) or file key to extract (for -x)
    #[arg(trailing_var_arg = true)]
    path: Vec<String>,
}

fn main() {
    let cli = Cli::parse();

    if cli.parallel < 1 {
        eprintln!("Error: --parallel must be >= 1.");
        std::process::exit(1);
    }

    // --- Standalone ECC operations ---
    if cli.ecc_info {
        match har::ecc::ecc_info(std::path::Path::new(&cli.file)) {
            Ok(()) => return,
            Err(e) => { eprintln!("ecc-info: {}", e); std::process::exit(1); }
        }
    }
    if cli.ecc_map {
        match har::ecc::ecc_map(std::path::Path::new(&cli.file)) {
            Ok(()) => return,
            Err(e) => { eprintln!("ecc-map: {}", e); std::process::exit(1); }
        }
    }
    if cli.ecc_verify {
        match har::ecc::ecc_verify(std::path::Path::new(&cli.file), cli.verbose) {
            Ok(r) => {
                println!(
                    "ECC verify: {} stripes, {} blocks, {} corrupt, {} unrepairable stripes, hash_ok={}",
                    r.n_stripes, r.total_blocks, r.corrupt_blocks,
                    r.unrepairable_stripes.len(), r.hash_ok
                );
                if r.corrupt_blocks == 0 && r.unrepairable_stripes.is_empty() && r.hash_ok {
                    return;
                } else {
                    std::process::exit(1);
                }
            }
            Err(e) => { eprintln!("ecc-verify: {}", e); std::process::exit(1); }
        }
    }
    if cli.ecc_wrap.is_some() && !cli.create && !cli.append && !cli.extract && !cli.list && !cli.browse {
        let level = cli.ecc_wrap.as_ref().unwrap();
        let params = match har::ecc::parse_ecc_level(level) {
            Ok(p) => p,
            Err(e) => { eprintln!("ecc-wrap: {}", e); std::process::exit(1); }
        };
        let input = std::path::PathBuf::from(&cli.file);
        let tmp = std::path::PathBuf::from(format!("{}.ecc.tmp", cli.file));
        match har::ecc::ecc_wrap(&input, &tmp, params, cli.verbose) {
            Ok(()) => {
                std::fs::rename(&tmp, &input).expect("rename");
                println!("ECC wrap complete: {}", cli.file);
                return;
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                eprintln!("ecc-wrap: {}", e);
                std::process::exit(1);
            }
        }
    }
    if cli.ecc_unwrap {
        let input = std::path::PathBuf::from(&cli.file);
        let tmp = std::path::PathBuf::from(format!("{}.unwrap.tmp", cli.file));
        match har::ecc::ecc_unwrap(&input, &tmp, cli.verbose) {
            Ok(()) => {
                std::fs::rename(&tmp, &input).expect("rename");
                println!("ECC unwrap complete: {}", cli.file);
                return;
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                eprintln!("ecc-unwrap: {}", e);
                std::process::exit(1);
            }
        }
    }
    if cli.heal {
        match har::ecc::ecc_repair(std::path::Path::new(&cli.file), cli.verbose) {
            Ok(r) => {
                println!(
                    "Heal: {} corrupt blocks found, {} unrepairable stripes, hash_ok={}",
                    r.corrupt_blocks, r.unrepairable_stripes.len(), r.hash_ok
                );
                if r.unrepairable_stripes.is_empty() { return; } else { std::process::exit(1); }
            }
            Err(e) => { eprintln!("heal: {}", e); std::process::exit(1); }
        }
    }

    // --- Auto-detect ECC wrapper on extract/list/browse: unwrap to temp file ---
    // We keep the tempfile alive for the remainder of main() so that the
    // downstream HDF5 open can read from it.
    let mut _ecc_tmp_guard: Option<tempfile::NamedTempFile> = None;
    let mut working_file = cli.file.clone();
    if (cli.extract || cli.list || cli.browse) && har::ecc::is_ecc_wrapped(std::path::Path::new(&cli.file)) {
        eprintln!("Note: detected ECC-wrapped archive, unwrapping to temp file.");
        let tmp = tempfile::Builder::new()
            .prefix("har_ecc_")
            .suffix(".h5")
            .tempfile()
            .expect("create temp file");
        if let Err(e) = har::ecc::ecc_unwrap(
            std::path::Path::new(&cli.file), tmp.path(), cli.verbose
        ) {
            eprintln!("ECC unwrap failed: {}", e);
            std::process::exit(1);
        }
        working_file = tmp.path().to_string_lossy().into_owned();
        _ecc_tmp_guard = Some(tmp);
    }
    let cli = {
        let mut c = cli;
        c.file = working_file;
        c
    };

    // --- Browse mode ---
    if cli.browse {
        har::browse::browse_archive(&cli.file);
        return;
    }

    // Determine compression settings
    let (compression, compression_opts): (Option<&str>, Option<u8>) = if cli.gzip {
        let level: u8 = cli.zopt.parse().unwrap_or(9);
        (Some("gzip"), Some(level))
    } else if cli.lzf {
        (Some("lzf"), None)
    } else if cli.szip {
        eprintln!("warning: szip compression is not tested yet");
        (Some("szip"), None)
    } else if cli.lzma {
        (Some("lzma"), None)
    } else {
        (None, None)
    };

    // If --validate during creation, implicitly enable sha256 checksums
    let mut checksum_val = cli.checksum.clone();
    if (cli.create || cli.append) && cli.validate && checksum_val.is_none() {
        checksum_val = Some("sha256".to_string());
    }
    let checksum = checksum_val.as_deref();

    // --split implies --bagit
    let use_bagit = cli.bagit || cli.split.is_some() || cli.ecc.is_some();

    // --- BagIt mode (with optional --split) ---
    if use_bagit {
        let bs = har::bagit::parse_batch_size(&cli.batch_size);

        if cli.append {
            eprintln!("Error: Append (-r) is not supported with --bagit.");
            std::process::exit(1);
        }

        if cli.create {
            if cli.path.is_empty() {
                eprintln!("Error: At least one source is required for archive creation (-c).");
                std::process::exit(1);
            }
            let sources: Vec<&str> = cli.path.iter().map(|s| s.as_str()).collect();

            if let Some(ref split_val) = cli.split {
                let split_spec = har::bagit::parse_split_arg(split_val);
                har::bagit::pack_bagit_split(
                    &sources, &cli.file, compression, compression_opts,
                    cli.shuffle, bs, cli.parallel, cli.verbose, checksum, cli.xattr,
                    cli.validate, split_spec,
                );
            } else {
                har::bagit::pack_bagit(
                    &sources, &cli.file, compression, compression_opts,
                    cli.shuffle, bs, cli.parallel, cli.verbose, checksum, cli.xattr,
                    cli.validate,
                );
            }
            let validation_ok = if cli.validate_roundtrip {
                let ok = har::validate_roundtrip(
                    &cli.file, &sources, true, cli.verbose, cli.byte_for_byte,
                    checksum.unwrap_or("sha256"),
                );
                if !ok { std::process::exit(1); }
                ok
            } else {
                true
            };
            if cli.delete_source {
                if validation_ok {
                    har::delete_source_files(&sources, cli.verbose);
                } else {
                    eprintln!("Skipping source deletion: validation failed.");
                }
            }
            let ecc_level = cli.ecc.as_ref().or(cli.ecc_wrap.as_ref());
            if let Some(level) = ecc_level {
                if cli.split.is_some() {
                    eprintln!("Error: --ecc/--ecc-wrap is not supported with --split yet.");
                    std::process::exit(1);
                }
                let params = match har::ecc::parse_ecc_level(level) {
                    Ok(p) => p,
                    Err(e) => { eprintln!("--ecc-wrap: {}", e); std::process::exit(1); }
                };
                let input = std::path::PathBuf::from(&cli.file);
                let tmp = std::path::PathBuf::from(format!("{}.ecc.tmp", cli.file));
                if let Err(e) = har::ecc::ecc_wrap(&input, &tmp, params, cli.verbose) {
                    let _ = std::fs::remove_file(&tmp);
                    eprintln!("ECC wrap: {}", e);
                    std::process::exit(1);
                }
                std::fs::rename(&tmp, &input).expect("rename");
                if cli.verbose {
                    println!("ECC wrap complete: {}", cli.file);
                }
            }
        } else if cli.extract {
            let file_key = if cli.path.is_empty() { None } else { Some(cli.path[0].as_str()) };
            if har::bagit::is_split_archive(&cli.file) {
                har::bagit::extract_bagit_split(
                    &cli.file, &cli.directory, file_key,
                    cli.validate, cli.bagit_raw, cli.parallel, cli.verbose,
                    cli.metadata_json, cli.xattr,
                );
            } else {
                har::bagit::extract_bagit(
                    &cli.file, &cli.directory, file_key,
                    cli.validate, cli.bagit_raw, cli.parallel, cli.verbose,
                    cli.metadata_json, cli.xattr,
                );
            }
        } else if cli.list {
            if har::bagit::is_split_archive(&cli.file) {
                har::bagit::list_bagit_split(&cli.file, cli.bagit_raw);
            } else {
                har::bagit::list_bagit(&cli.file, cli.bagit_raw);
            }
        }
        return;
    }

    // --- Auto-detect split/bagit on extract/list ---
    if cli.extract || cli.list {
        if har::bagit::is_split_archive(&cli.file) {
            eprintln!("Note: detected split bagit-v1 archive.");
            if cli.extract {
                let file_key = if cli.path.is_empty() { None } else { Some(cli.path[0].as_str()) };
                har::bagit::extract_bagit_split(
                    &cli.file, &cli.directory, file_key,
                    cli.validate, cli.bagit_raw, cli.parallel, cli.verbose,
                    cli.metadata_json, cli.xattr,
                );
            } else {
                har::bagit::list_bagit_split(&cli.file, cli.bagit_raw);
            }
            return;
        } else if har::bagit::is_bagit_archive(&cli.file) {
            eprintln!("Note: detected bagit-v1 archive, using BagIt extraction.");
            if cli.extract {
                let file_key = if cli.path.is_empty() { None } else { Some(cli.path[0].as_str()) };
                har::bagit::extract_bagit(
                    &cli.file, &cli.directory, file_key,
                    cli.validate, cli.bagit_raw, cli.parallel, cli.verbose,
                    cli.metadata_json, cli.xattr,
                );
            } else {
                har::bagit::list_bagit(&cli.file, cli.bagit_raw);
            }
            return;
        }
    }

    // --- Legacy mode ---
    if cli.create {
        if cli.path.is_empty() {
            eprintln!("Error: At least one source (directory or file) is required for archive creation (-c).");
            std::process::exit(1);
        }
        let sources: Vec<&str> = cli.path.iter().map(|s| s.as_str()).collect();
        har::pack_or_append_to_h5(
            &sources, &cli.file, "w", compression, compression_opts,
            cli.shuffle, cli.parallel, cli.verbose, checksum, cli.xattr,
            cli.validate,
        );
        let validation_ok = if cli.validate_roundtrip {
            let ok = har::validate_roundtrip(
                &cli.file, &sources, false, cli.verbose, cli.byte_for_byte,
                checksum.unwrap_or("sha256"),
            );
            if !ok { std::process::exit(1); }
            ok
        } else {
            true
        };
        if cli.delete_source {
            if validation_ok {
                har::delete_source_files(&sources, cli.verbose);
            } else {
                eprintln!("Skipping source deletion: validation failed.");
            }
        }
        if let Some(level) = cli.ecc_wrap.as_ref() {
            let params = match har::ecc::parse_ecc_level(level) {
                Ok(p) => p,
                Err(e) => { eprintln!("--ecc-wrap: {}", e); std::process::exit(1); }
            };
            let input = std::path::PathBuf::from(&cli.file);
            let tmp = std::path::PathBuf::from(format!("{}.ecc.tmp", cli.file));
            if let Err(e) = har::ecc::ecc_wrap(&input, &tmp, params, cli.verbose) {
                let _ = std::fs::remove_file(&tmp);
                eprintln!("ECC wrap: {}", e);
                std::process::exit(1);
            }
            std::fs::rename(&tmp, &input).expect("rename");
            if cli.verbose { println!("ECC wrap complete: {}", cli.file); }
        }
    } else if cli.append {
        if cli.path.is_empty() {
            eprintln!("Error: At least one source (directory or file) is required for appending (-r).");
            std::process::exit(1);
        }
        let sources: Vec<&str> = cli.path.iter().map(|s| s.as_str()).collect();
        har::pack_or_append_to_h5(
            &sources, &cli.file, "a", compression, compression_opts,
            cli.shuffle, cli.parallel, cli.verbose, checksum, cli.xattr,
            cli.validate,
        );
        let validation_ok = if cli.validate_roundtrip {
            let ok = har::validate_roundtrip(
                &cli.file, &sources, false, cli.verbose, cli.byte_for_byte,
                checksum.unwrap_or("sha256"),
            );
            if !ok { std::process::exit(1); }
            ok
        } else {
            true
        };
        if cli.delete_source {
            if validation_ok {
                har::delete_source_files(&sources, cli.verbose);
            } else {
                eprintln!("Skipping source deletion: validation failed.");
            }
        }
        if let Some(level) = cli.ecc_wrap.as_ref() {
            let params = match har::ecc::parse_ecc_level(level) {
                Ok(p) => p,
                Err(e) => { eprintln!("--ecc-wrap: {}", e); std::process::exit(1); }
            };
            let input = std::path::PathBuf::from(&cli.file);
            let tmp = std::path::PathBuf::from(format!("{}.ecc.tmp", cli.file));
            if let Err(e) = har::ecc::ecc_wrap(&input, &tmp, params, cli.verbose) {
                let _ = std::fs::remove_file(&tmp);
                eprintln!("ECC wrap: {}", e);
                std::process::exit(1);
            }
            std::fs::rename(&tmp, &input).expect("rename");
            if cli.verbose { println!("ECC wrap complete: {}", cli.file); }
        }
    } else if cli.extract {
        let file_key = if cli.path.is_empty() { None } else { Some(cli.path[0].as_str()) };
        har::extract_h5_to_directory(
            &cli.file, &cli.directory, file_key, cli.parallel, cli.verbose,
            cli.validate, checksum, cli.metadata_json, cli.xattr,
        );
    } else if cli.list {
        har::list_h5_contents(&cli.file);
    } else {
        eprintln!("Error: one of -c, -r, -x, -t is required.");
        std::process::exit(1);
    }
}
