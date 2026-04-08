use super::*;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use tempfile::TempDir;

fn make_test_tree(base: &str, n_dirs: usize, n_files: usize) {
    for i in 0..n_dirs {
        let d = format!("{}/dir{}", base, i);
        fs::create_dir_all(&d).unwrap();
        for j in 0..n_files {
            fs::write(format!("{}/file{}.txt", d, j), format!("content dir{}/file{}", i, j)).unwrap();
        }
    }
}

#[test]
fn test_bagit_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("file1.txt"), "Hello world").unwrap();
    fs::create_dir_all(src.join("sub")).unwrap();
    fs::write(src.join("sub/file2.txt"), "Nested file").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, false);

    assert!(is_bagit_archive(&archive));

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false, false, false);

    assert_eq!(fs::read_to_string(out.join("src/file1.txt")).unwrap(), "Hello world");
    assert_eq!(fs::read_to_string(out.join("src/sub/file2.txt")).unwrap(), "Nested file");
}

#[test]
fn test_bagit_single_file_extract() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("a.txt"), "AAA").unwrap();
    fs::write(src.join("b.txt"), "BBB").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, false);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, Some("src/a.txt"), false, false, 1, false, false, false);

    assert!(out.join("src/a.txt").exists());
    assert!(!out.join("src/b.txt").exists());
}

#[test]
fn test_bagit_empty_file() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("empty.txt"), "").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, false);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false, false, false);

    assert!(out.join("src/empty.txt").exists());
    assert_eq!(fs::metadata(out.join("src/empty.txt")).unwrap().len(), 0);
}

#[test]
fn test_bagit_binary_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    let all_bytes: Vec<u8> = (0..=255u8).collect();
    fs::write(src.join("allbytes.bin"), &all_bytes).unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, false);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false, false, false);

    assert_eq!(fs::read(out.join("src/allbytes.bin")).unwrap(), all_bytes);
}

#[test]
fn test_bagit_permissions() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    let exe = src.join("run.sh");
    fs::write(&exe, "#!/bin/bash").unwrap();
    fs::set_permissions(&exe, fs::Permissions::from_mode(0o755)).unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, false);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false, false, false);

    assert_eq!(
        fs::metadata(out.join("src/run.sh")).unwrap().permissions().mode(),
        fs::metadata(&exe).unwrap().permissions().mode()
    );
}

#[test]
fn test_bagit_empty_dir() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("proj");
    fs::create_dir_all(src.join("empty_sub")).unwrap();
    fs::create_dir_all(src.join("full")).unwrap();
    fs::write(src.join("full/f.txt"), "data").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, false);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false, false, false);

    assert!(out.join("proj/empty_sub").is_dir());
    assert!(out.join("proj/full/f.txt").is_file());
}

#[test]
fn test_bagit_batch_splitting() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    for i in 0..10 {
        fs::write(src.join(format!("file{}.txt", i)), "x".repeat(200)).unwrap();
    }

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, 500, 1, false, None, false, false);

    let h5f = hdf5::File::open(&archive).unwrap();
    let n_batches = h5f.group("batches").unwrap().member_names().unwrap().len();
    assert!(n_batches > 1, "Expected multiple batches, got {}", n_batches);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false, false, false);

    for i in 0..10 {
        assert_eq!(fs::read_to_string(out.join(format!("src/file{}.txt", i))).unwrap(), "x".repeat(200));
    }
}

#[test]
fn test_bagit_parallel_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    make_test_tree(&src.to_string_lossy(), 5, 10);

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 4, false, None, false, false);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 4, false, false, false);

    for i in 0..5 {
        for j in 0..10 {
            let p = out.join(format!("src/dir{}/file{}.txt", i, j));
            assert_eq!(fs::read_to_string(&p).unwrap(), format!("content dir{}/file{}", i, j));
        }
    }
}

#[test]
fn test_bagit_many_files() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("many");
    fs::create_dir_all(&src).unwrap();
    make_test_tree(&src.to_string_lossy(), 10, 100);

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 4, false, None, false, false);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 4, false, false, false);

    let mut count = 0;
    for entry in walkdir::WalkDir::new(out.join("many")).into_iter().filter_map(|e| e.ok()) {
        if entry.path().is_file() { count += 1; }
    }
    assert_eq!(count, 1000);
}

#[test]
fn test_bagit_compression() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("comp");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("big.txt"), "A".repeat(100000)).unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, Some("gzip"), Some(9), true, DEFAULT_BATCH_SIZE, 1, false, None, false, false);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false, false, false);

    assert_eq!(fs::read_to_string(out.join("comp/big.txt")).unwrap(), "A".repeat(100000));
    assert!(fs::metadata(&archive).unwrap().len() < 20000);
}

#[test]
fn test_parse_batch_size_fn() {
    assert_eq!(parse_batch_size("64M"), 64 * 1024 * 1024);
    assert_eq!(parse_batch_size("1G"), 1024 * 1024 * 1024);
    assert_eq!(parse_batch_size("512K"), 512 * 1024);
}

#[test]
fn test_bagit_checksum_blake3() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("file1.txt"), "Hello world").unwrap();
    fs::write(src.join("file2.txt"), "Another file").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, Some("blake3"), false, false);

    // Verify the checksum algo attribute is stored
    let h5f = hdf5::File::open(&archive).unwrap();
    let algo: hdf5::types::VarLenUnicode = h5f.attr("har_checksum_algo").unwrap().read_scalar().unwrap();
    assert_eq!(algo.as_str(), "blake3");
    drop(h5f);

    // Extract with validation
    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, true, false, 1, false, false, false);

    assert_eq!(fs::read_to_string(out.join("src/file1.txt")).unwrap(), "Hello world");
    assert_eq!(fs::read_to_string(out.join("src/file2.txt")).unwrap(), "Another file");
}

#[test]
fn test_bagit_checksum_md5() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("data.txt"), "test data").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, Some("md5"), false, false);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, true, false, 1, false, false, false);

    assert_eq!(fs::read_to_string(out.join("src/data.txt")).unwrap(), "test data");
}

// ---- Validate during creation tests ----

#[test]
fn test_bagit_validate_on_creation() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("file1.txt"), "Hello world").unwrap();
    fs::write(src.join("file2.txt"), "Another file").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, true);

    assert!(std::path::Path::new(&archive).exists());
    assert!(is_bagit_archive(&archive));
}

#[test]
fn test_bagit_validate_on_creation_with_checksum() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("data.txt"), "test data").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, Some("blake3"), false, true);

    let h5f = hdf5::File::open(&archive).unwrap();
    let algo: hdf5::types::VarLenUnicode = h5f.attr("har_checksum_algo").unwrap().read_scalar().unwrap();
    assert_eq!(algo.as_str(), "blake3");
}

// ---- Validate roundtrip tests (BagIt) ----

#[test]
fn test_bagit_validate_roundtrip_checksum() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("file1.txt"), "Hello world").unwrap();
    fs::create_dir_all(src.join("sub")).unwrap();
    fs::write(src.join("sub/file2.txt"), "Nested file").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, false);

    let result = crate::validate_roundtrip(
        &archive, &[src_str.as_str()], true, false, false, "sha256",
    );
    assert!(result);
}

#[test]
fn test_bagit_validate_roundtrip_byte_for_byte() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("file1.txt"), "Hello world").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, false);

    let result = crate::validate_roundtrip(
        &archive, &[src_str.as_str()], true, false, true, "sha256",
    );
    assert!(result);
}

#[test]
fn test_bagit_validate_roundtrip_detects_mismatch() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("data.txt"), "original").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, false);

    // Modify source after archival
    fs::write(src.join("data.txt"), "modified").unwrap();

    let result = crate::validate_roundtrip(
        &archive, &[src_str.as_str()], true, false, false, "sha256",
    );
    assert!(!result);
}

// ---- Delete source tests (BagIt) ----

#[test]
fn test_bagit_delete_source() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("todelete");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("f.txt"), "data").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, false);

    assert!(src.exists());
    crate::delete_source_files(&[src_str.as_str()], false);
    assert!(!src.exists());
}

#[test]
fn test_bagit_delete_source_after_validate() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("validated");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("f.txt"), "data").unwrap();

    let archive = tmp.path().join("test.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, true);

    let result = crate::validate_roundtrip(
        &archive, &[src_str.as_str()], true, false, false, "sha256",
    );
    assert!(result);

    crate::delete_source_files(&[src_str.as_str()], false);
    assert!(!src.exists());
}

// ---------------------------------------------------------------------------
// Split archive tests
// ---------------------------------------------------------------------------

#[test]
fn test_parse_split_arg() {
    match parse_split_arg("100M") {
        SplitSpec::Size(s) => assert_eq!(s, 100 * 1024 * 1024),
        _ => panic!("expected Size"),
    }
    match parse_split_arg("2G") {
        SplitSpec::Size(s) => assert_eq!(s, 2 * 1024 * 1024 * 1024),
        _ => panic!("expected Size"),
    }
    match parse_split_arg("512K") {
        SplitSpec::Size(s) => assert_eq!(s, 512 * 1024),
        _ => panic!("expected Size"),
    }
    match parse_split_arg("size=64M") {
        SplitSpec::Size(s) => assert_eq!(s, 64 * 1024 * 1024),
        _ => panic!("expected Size"),
    }
    match parse_split_arg("n=4") {
        SplitSpec::Count(n) => assert_eq!(n, 4),
        _ => panic!("expected Count"),
    }
    match parse_split_arg("n=100") {
        SplitSpec::Count(n) => assert_eq!(n, 100),
        _ => panic!("expected Count"),
    }
}

#[test]
fn test_split_filename() {
    assert_eq!(split_filename("archive.h5", 0, 3), "archive.h5");
    assert_eq!(split_filename("archive.h5", 1, 3), "archive.001.h5");
    assert_eq!(split_filename("archive.h5", 2, 3), "archive.002.h5");
    assert_eq!(split_filename("/tmp/test.h5", 0, 2), "/tmp/test.h5");
    assert_eq!(split_filename("/tmp/test.h5", 1, 2), "/tmp/test.001.h5");
    // 4-digit padding for 1000+ splits
    assert_eq!(split_filename("a.h5", 1, 1001), "a.0001.h5");
    assert_eq!(split_filename("a.h5", 999, 1001), "a.0999.h5");
}

#[test]
fn test_distribute_files() {
    let entries = vec![
        SizedEntry { file_path: "a".into(), rel_path: "a".into(), size: 100, chunk_offset: 0, chunk_size: None },
        SizedEntry { file_path: "b".into(), rel_path: "b".into(), size: 200, chunk_offset: 0, chunk_size: None },
        SizedEntry { file_path: "c".into(), rel_path: "c".into(), size: 50, chunk_offset: 0, chunk_size: None },
        SizedEntry { file_path: "d".into(), rel_path: "d".into(), size: 150, chunk_offset: 0, chunk_size: None },
        SizedEntry { file_path: "e".into(), rel_path: "e".into(), size: 80, chunk_offset: 0, chunk_size: None },
        SizedEntry { file_path: "f".into(), rel_path: "f".into(), size: 40, chunk_offset: 0, chunk_size: None },
    ];
    let bins = distribute_files(entries, 3);
    assert_eq!(bins.len(), 3);
    let total: usize = bins.iter().map(|b| b.len()).sum();
    assert_eq!(total, 6);
    let sizes: Vec<u64> = bins.iter().map(|b| b.iter().map(|e| e.size).sum()).collect();
    let max = *sizes.iter().max().unwrap();
    let min = *sizes.iter().filter(|&&s| s > 0).min().unwrap();
    assert!(max <= min * 3, "bins not balanced: {:?}", sizes);
}

#[test]
fn test_split_roundtrip_basic() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    make_test_tree(src.to_str().unwrap(), 4, 5);

    let archive = tmp.path().join("archive.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();

    // Small batch size so each file is its own batch (enables splitting)
    pack_bagit_split(
        &[src_str.as_str()], &archive, None, None, false, 64,
        1, false, None, false, false, SplitSpec::Count(3),
    );

    // Verify split files exist
    assert!(std::path::Path::new(&archive).exists());
    let split1 = tmp.path().join("archive.001.h5");
    let split2 = tmp.path().join("archive.002.h5");
    assert!(split1.exists(), "split 1 missing");
    assert!(split2.exists(), "split 2 missing");

    // Verify split detection
    assert!(is_split_archive(&archive));

    // Verify split_manifest in split 0
    let h5f = hdf5::File::open(&archive).unwrap();
    assert!(h5f.group("split_manifest").is_ok());
    let count = h5f.group("split_manifest").unwrap()
        .attr("count").unwrap().read_scalar::<u64>().unwrap();
    assert_eq!(count, 20); // 4 dirs * 5 files

    // Extract
    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit_split(&archive, &out_str, None, false, false, 1, false, false, false);

    // Verify all files
    for i in 0..4 {
        for j in 0..5 {
            let p = out.join(format!("src/dir{}/file{}.txt", i, j));
            assert!(p.exists(), "Missing: {:?}", p);
            assert_eq!(fs::read_to_string(&p).unwrap(), format!("content dir{}/file{}", i, j));
        }
    }
}

#[test]
fn test_split_roundtrip_by_size() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    for i in 0..10 {
        fs::write(src.join(format!("file{}.txt", i)), "x".repeat(200)).unwrap();
    }

    let archive = tmp.path().join("archive.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();

    // batch_size=200 so each file is its own batch, then split by size
    pack_bagit_split(
        &[src_str.as_str()], &archive, None, None, false, 200,
        1, false, None, false, false, SplitSpec::Size(500),
    );

    assert!(is_split_archive(&archive));

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit_split(&archive, &out_str, None, false, false, 1, false, false, false);

    for i in 0..10 {
        let p = out.join(format!("src/file{}.txt", i));
        assert!(p.exists(), "Missing: {:?}", p);
        assert_eq!(fs::read_to_string(&p).unwrap(), "x".repeat(200));
    }
}

#[test]
fn test_split_single_file_extract() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("a.txt"), "AAA").unwrap();
    fs::write(src.join("b.txt"), "BBB").unwrap();

    let archive = tmp.path().join("archive.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();

    // batch_size=1 so each file is its own batch
    pack_bagit_split(
        &[src_str.as_str()], &archive, None, None, false, 1,
        1, false, None, false, false, SplitSpec::Count(2),
    );

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit_split(&archive, &out_str, Some("src/a.txt"), false, false, 1, false, false, false);

    assert!(out.join("src/a.txt").exists());
    assert_eq!(fs::read_to_string(out.join("src/a.txt")).unwrap(), "AAA");
}

#[test]
fn test_split_one_is_normal() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("f.txt"), "data").unwrap();

    let archive = tmp.path().join("archive.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();

    pack_bagit_split(
        &[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE,
        1, false, None, false, false, SplitSpec::Count(1),
    );

    assert!(is_bagit_archive(&archive));
    assert!(!is_split_archive(&archive));

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false, false, false);
    assert_eq!(fs::read_to_string(out.join("src/f.txt")).unwrap(), "data");
}

#[test]
fn test_split_with_compression() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("comp");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("big1.txt"), "A".repeat(100_000)).unwrap();
    fs::write(src.join("big2.txt"), "B".repeat(100_000)).unwrap();

    let archive = tmp.path().join("archive.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();

    // batch_size=1 so each file is its own batch
    pack_bagit_split(
        &[src_str.as_str()], &archive, Some("gzip"), Some(9), false, 1,
        1, false, None, false, false, SplitSpec::Count(2),
    );

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit_split(&archive, &out_str, None, false, false, 1, false, false, false);

    assert_eq!(fs::read_to_string(out.join("comp/big1.txt")).unwrap(), "A".repeat(100_000));
    assert_eq!(fs::read_to_string(out.join("comp/big2.txt")).unwrap(), "B".repeat(100_000));
}

#[test]
fn test_split_with_validation() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    make_test_tree(src.to_str().unwrap(), 3, 4);

    let archive = tmp.path().join("archive.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();

    // batch_size=64 so files spread across batches
    pack_bagit_split(
        &[src_str.as_str()], &archive, None, None, false, 64,
        1, false, Some("sha256"), false, true, SplitSpec::Count(2),
    );

    assert!(is_split_archive(&archive));
}

#[test]
fn test_split_empty_dirs() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(src.join("full")).unwrap();
    fs::write(src.join("full/f1.txt"), "data1").unwrap();
    fs::write(src.join("full/f2.txt"), "data2").unwrap();
    fs::write(src.join("full/f3.txt"), "data3").unwrap();
    fs::create_dir_all(src.join("empty")).unwrap();

    let archive = tmp.path().join("archive.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();

    // batch_size=1 so each file is its own batch
    pack_bagit_split(
        &[src_str.as_str()], &archive, None, None, false, 1,
        1, false, None, false, false, SplitSpec::Count(2),
    );

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit_split(&archive, &out_str, None, false, false, 1, false, false, false);

    assert!(out.join("src/empty").is_dir());
    assert_eq!(fs::read_to_string(out.join("src/full/f1.txt")).unwrap(), "data1");
}

#[test]
fn test_is_split_archive_detection() {
    let tmp = TempDir::new().unwrap();

    // Non-existent file
    assert!(!is_split_archive("/tmp/nonexistent_12345.h5"));

    // Regular bagit archive (not split)
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("f.txt"), "data").unwrap();
    fs::write(src.join("g.txt"), "data2").unwrap();
    let archive = tmp.path().join("regular.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None, false, false);
    assert!(!is_split_archive(&archive));

    // Split archive (batch_size=1 so each file is its own batch)
    let split_archive = tmp.path().join("split.h5").to_string_lossy().to_string();
    pack_bagit_split(
        &[src_str.as_str()], &split_archive, None, None, false, 1,
        1, false, None, false, false, SplitSpec::Count(2),
    );
    assert!(is_split_archive(&split_archive));
}

#[test]
fn test_split_parallel_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    make_test_tree(src.to_str().unwrap(), 5, 10);

    let archive = tmp.path().join("archive.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();

    // batch_size=64 so files spread across many batches
    pack_bagit_split(
        &[src_str.as_str()], &archive, None, None, false, 64,
        4, false, None, false, false, SplitSpec::Count(4),
    );

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit_split(&archive, &out_str, None, false, false, 1, false, false, false);

    let mut count = 0;
    for entry in walkdir::WalkDir::new(out.join("src")) {
        let e = entry.unwrap();
        if e.path().is_file() { count += 1; }
    }
    assert_eq!(count, 50);
}

#[test]
fn test_split_list() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("a.txt"), "AAA").unwrap();
    fs::write(src.join("b.txt"), "BBB").unwrap();

    let archive = tmp.path().join("archive.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();

    // batch_size=1 so each file is its own batch
    pack_bagit_split(
        &[src_str.as_str()], &archive, None, None, false, 1,
        1, false, None, false, false, SplitSpec::Count(2),
    );

    list_bagit_split(&archive, false);
    list_bagit_split(&archive, true);
}

#[test]
fn test_split_chunking_large_file() {
    // Test that a file larger than the split target gets chunked across splits
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();

    // Create a "large" file (1000 bytes) and a small file (10 bytes)
    let large_data: Vec<u8> = (0..1000u32).map(|i| (i % 256) as u8).collect();
    fs::write(src.join("large.bin"), &large_data).unwrap();
    fs::write(src.join("small.txt"), "0123456789").unwrap();

    let archive = tmp.path().join("archive.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    let out = tmp.path().join("out");

    // Split target = 400 bytes. The 1000-byte file should be chunked into 3 pieces (400+400+200).
    // batch_size=500 so chunks don't get merged into huge batches.
    pack_bagit_split(
        &[src_str.as_str()], &archive, None, None, false, 500,
        1, false, Some("sha256"), false, false, SplitSpec::Size(400),
    );

    // Verify multiple split files were created
    let split_0 = tmp.path().join("archive.h5");
    let split_1 = tmp.path().join("archive.001.h5");
    assert!(split_0.exists(), "Split 0 should exist");
    assert!(split_1.exists(), "Split 1 should exist");

    // Extract
    extract_bagit_split(
        &archive, out.to_str().unwrap(), None,
        true, false, 1, false, false, false,
    );

    // Verify the large file was reassembled correctly
    let extracted_large = out.join("src/large.bin");
    assert!(extracted_large.exists(), "large.bin should exist after extraction");
    let extracted_data = fs::read(&extracted_large).unwrap();
    assert_eq!(extracted_data, large_data, "large.bin content should match after chunk reassembly");

    // Verify small file also extracted correctly
    let extracted_small = out.join("src/small.txt");
    assert!(extracted_small.exists(), "small.txt should exist");
    assert_eq!(fs::read_to_string(&extracted_small).unwrap(), "0123456789");
}

#[test]
fn test_chunk_large_files_fn() {
    let entries = vec![
        SizedEntry { file_path: "big".into(), rel_path: "big".into(), size: 1000, chunk_offset: 0, chunk_size: None },
        SizedEntry { file_path: "small".into(), rel_path: "small".into(), size: 100, chunk_offset: 0, chunk_size: None },
    ];
    let chunked = chunk_large_files(entries, 400);
    // big (1000) -> 3 chunks: 400, 400, 200
    // small (100) -> unchanged
    assert_eq!(chunked.len(), 4);
    assert_eq!(chunked[0].rel_path, "big");
    assert_eq!(chunked[0].chunk_offset, 0);
    assert_eq!(chunked[0].chunk_size, Some(400));
    assert_eq!(chunked[1].chunk_offset, 400);
    assert_eq!(chunked[1].chunk_size, Some(400));
    assert_eq!(chunked[2].chunk_offset, 800);
    assert_eq!(chunked[2].chunk_size, Some(200));
    assert_eq!(chunked[3].rel_path, "small");
    assert_eq!(chunked[3].chunk_offset, 0);
    assert_eq!(chunked[3].chunk_size, None);
}
