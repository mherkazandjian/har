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
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None);

    assert!(is_bagit_archive(&archive));

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false);

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
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, Some("src/a.txt"), false, false, 1, false);

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
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false);

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
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false);

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
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false);

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
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, None);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false);

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
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, 500, 1, false, None);

    let h5f = hdf5::File::open(&archive).unwrap();
    let n_batches = h5f.group("batches").unwrap().member_names().unwrap().len();
    assert!(n_batches > 1, "Expected multiple batches, got {}", n_batches);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false);

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
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 4, false, None);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 4, false);

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
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 4, false, None);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 4, false);

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
    pack_bagit(&[src_str.as_str()], &archive, Some("gzip"), Some(9), true, DEFAULT_BATCH_SIZE, 1, false, None);

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, false, false, 1, false);

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
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, Some("blake3"));

    // Verify the checksum algo attribute is stored
    let h5f = hdf5::File::open(&archive).unwrap();
    let algo: hdf5::types::VarLenUnicode = h5f.attr("har_checksum_algo").unwrap().read_scalar().unwrap();
    assert_eq!(algo.as_str(), "blake3");
    drop(h5f);

    // Extract with validation
    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, true, false, 1, false);

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
    pack_bagit(&[src_str.as_str()], &archive, None, None, false, DEFAULT_BATCH_SIZE, 1, false, Some("md5"));

    let out = tmp.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let out_str = out.to_string_lossy().to_string();
    extract_bagit(&archive, &out_str, None, true, false, 1, false);

    assert_eq!(fs::read_to_string(out.join("src/data.txt")).unwrap(), "test data");
}
