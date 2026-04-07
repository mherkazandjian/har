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

    /// Verify SHA-256 checksums on extraction (--bagit mode only)
    #[arg(long = "validate")]
    validate: bool,

    /// Extract as a full BagIt bag with tag files (--bagit mode only)
    #[arg(long = "bagit-raw")]
    bagit_raw: bool,

    /// Checksum algorithm for integrity verification (md5, sha256, blake3)
    #[arg(long = "checksum", value_parser = ["md5", "sha256", "blake3"])]
    checksum: Option<String>,

    /// Write metadata.json manifest of user HDF5 attributes on extraction
    #[arg(long = "metadata-json")]
    metadata_json: bool,

    /// Capture/restore filesystem extended attributes (user.* namespace)
    #[arg(long = "xattr")]
    xattr: bool,

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

    let checksum = cli.checksum.as_deref();

    // --- BagIt mode ---
    if cli.bagit {
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
            har::bagit::pack_bagit(
                &sources, &cli.file, compression, compression_opts,
                cli.shuffle, bs, cli.parallel, cli.verbose, checksum, cli.xattr,
            );
        } else if cli.extract {
            let file_key = if cli.path.is_empty() { None } else { Some(cli.path[0].as_str()) };
            har::bagit::extract_bagit(
                &cli.file, &cli.directory, file_key,
                cli.validate, cli.bagit_raw, cli.parallel, cli.verbose,
                cli.metadata_json, cli.xattr,
            );
        } else if cli.list {
            har::bagit::list_bagit(&cli.file, cli.bagit_raw);
        }
        return;
    }

    // --- Auto-detect bagit on extract/list ---
    if (cli.extract || cli.list) && har::bagit::is_bagit_archive(&cli.file) {
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
        );
    } else if cli.append {
        if cli.path.is_empty() {
            eprintln!("Error: At least one source (directory or file) is required for appending (-r).");
            std::process::exit(1);
        }
        let sources: Vec<&str> = cli.path.iter().map(|s| s.as_str()).collect();
        har::pack_or_append_to_h5(
            &sources, &cli.file, "a", compression, compression_opts,
            cli.shuffle, cli.parallel, cli.verbose, checksum, cli.xattr,
        );
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
