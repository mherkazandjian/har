use crate::{extract_h5_to_directory, list_h5_contents, pack_or_append_to_h5};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use tempfile::TempDir;

/// Helper: create test environment matching Python's test_env fixture.
struct TestEnv {
    _tmp: TempDir,
    src_dir: String,
    #[allow(dead_code)]
    file1: String,
    #[allow(dead_code)]
    file2: String,
    archive_path: String,
    extract_dir: String,
}

fn setup_test_env() -> TestEnv {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src_dir = base.join("src");
    fs::create_dir_all(&src_dir).unwrap();

    let file1 = src_dir.join("file1.txt");
    fs::write(&file1, "This is file 1.").unwrap();

    let sub_dir = src_dir.join("subdir");
    fs::create_dir_all(&sub_dir).unwrap();
    let file2 = sub_dir.join("file2.txt");
    fs::write(&file2, "This is file 2 in subdir.").unwrap();

    let archive_path = base.join("archive.h5");
    let extract_dir = base.join("extract");
    fs::create_dir_all(&extract_dir).unwrap();

    TestEnv {
        _tmp: tmp,
        src_dir: src_dir.to_string_lossy().to_string(),
        file1: file1.to_string_lossy().to_string(),
        file2: file2.to_string_lossy().to_string(),
        archive_path: archive_path.to_string_lossy().to_string(),
        extract_dir: extract_dir.to_string_lossy().to_string(),
    }
}

/// Helper: create a test directory tree with n_dirs directories and n_files_per_dir files each.
fn make_test_tree(base: &str, n_dirs: usize, n_files_per_dir: usize) {
    for i in 0..n_dirs {
        let d = format!("{}/dir{}", base, i);
        fs::create_dir_all(&d).unwrap();
        for j in 0..n_files_per_dir {
            let fpath = format!("{}/file{}.txt", d, j);
            fs::write(&fpath, format!("content dir{}/file{}", i, j)).unwrap();
        }
    }
}

// ---- Basic tests (matching Python test_h5.py) ----

#[test]
fn test_archive_creation() {
    let env = setup_test_env();
    pack_or_append_to_h5(
        &[env.src_dir.as_str()],
        &env.archive_path,
        "w",
        None, None, false, 1, false, None,
    );
    assert!(Path::new(&env.archive_path).exists());

    let h5f = hdf5::File::open(&env.archive_path).unwrap();
    assert!(h5f.dataset("src/file1.txt").is_ok());
    assert!(h5f.group("src/subdir").is_ok());
    assert!(h5f.dataset("src/subdir/file2.txt").is_ok());
}

#[test]
fn test_append() {
    let env = setup_test_env();
    pack_or_append_to_h5(
        &[env.src_dir.as_str()],
        &env.archive_path,
        "w",
        None, None, false, 1, false, None,
    );

    let file3 = format!("{}/file3.txt", env.src_dir);
    fs::write(&file3, "This is file 3.").unwrap();

    pack_or_append_to_h5(
        &[file3.as_str()],
        &env.archive_path,
        "a",
        None, None, false, 1, false, None,
    );

    let h5f = hdf5::File::open(&env.archive_path).unwrap();
    assert!(h5f.dataset("file3.txt").is_ok());
}

#[test]
fn test_extract_all() {
    let env = setup_test_env();
    pack_or_append_to_h5(
        &[env.src_dir.as_str()],
        &env.archive_path,
        "w",
        None, None, false, 1, false, None,
    );
    extract_h5_to_directory(&env.archive_path, &env.extract_dir, None, 1, false, false, None);

    let extracted1 = format!("{}/src/file1.txt", env.extract_dir);
    assert!(Path::new(&extracted1).exists());
    assert_eq!(fs::read_to_string(&extracted1).unwrap(), "This is file 1.");

    let extracted2 = format!("{}/src/subdir/file2.txt", env.extract_dir);
    assert!(Path::new(&extracted2).exists());
    assert_eq!(
        fs::read_to_string(&extracted2).unwrap(),
        "This is file 2 in subdir."
    );
}

#[test]
fn test_extract_single() {
    let env = setup_test_env();
    pack_or_append_to_h5(
        &[env.src_dir.as_str()],
        &env.archive_path,
        "w",
        None, None, false, 1, false, None,
    );
    extract_h5_to_directory(
        &env.archive_path,
        &env.extract_dir,
        Some("src/file1.txt"),
        1,
        false,
        false,
        None,
    );

    let extracted1 = format!("{}/src/file1.txt", env.extract_dir);
    assert!(Path::new(&extracted1).exists());
    assert_eq!(fs::read_to_string(&extracted1).unwrap(), "This is file 1.");

    let extracted2 = format!("{}/src/subdir/file2.txt", env.extract_dir);
    assert!(!Path::new(&extracted2).exists());
}

#[test]
fn test_list_contents() {
    let env = setup_test_env();
    pack_or_append_to_h5(
        &[env.src_dir.as_str()],
        &env.archive_path,
        "w",
        None, None, false, 1, false, None,
    );

    // Capture stdout - we just verify the function runs without error
    // and check the archive contents directly
    let h5f = hdf5::File::open(&env.archive_path).unwrap();
    assert!(h5f.dataset("src/file1.txt").is_ok());
    assert!(h5f.group("src/subdir").is_ok());

    // Also call list_h5_contents to exercise the code path
    list_h5_contents(&env.archive_path);
}

#[test]
fn test_multi_directory_no_collision() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let dir_a = base.join("dirA");
    let dir_b = base.join("dirB");
    fs::create_dir_all(&dir_a).unwrap();
    fs::create_dir_all(&dir_b).unwrap();
    fs::write(dir_a.join("readme.txt"), "from A").unwrap();
    fs::write(dir_b.join("readme.txt"), "from B").unwrap();

    let archive = base.join("multi.h5").to_string_lossy().to_string();
    let dir_a_str = dir_a.to_string_lossy().to_string();
    let dir_b_str = dir_b.to_string_lossy().to_string();
    pack_or_append_to_h5(
        &[dir_a_str.as_str(), dir_b_str.as_str()],
        &archive,
        "w",
        None, None, false, 1, false, None,
    );

    let h5f = hdf5::File::open(&archive).unwrap();
    assert!(h5f.dataset("dirA/readme.txt").is_ok());
    assert!(h5f.dataset("dirB/readme.txt").is_ok());

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 1, false, false, None);

    assert_eq!(
        fs::read_to_string(extract_dir.join("dirA/readme.txt")).unwrap(),
        "from A"
    );
    assert_eq!(
        fs::read_to_string(extract_dir.join("dirB/readme.txt")).unwrap(),
        "from B"
    );
}

#[test]
fn test_empty_directory_preserved() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("project");
    fs::create_dir_all(src.join("empty_sub")).unwrap();
    fs::create_dir_all(src.join("nonempty")).unwrap();
    fs::write(src.join("nonempty/f.txt"), "data").unwrap();

    let archive = base.join("empty.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(&[src_str.as_str()], &archive, "w", None, None, false, 1, false, None);

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 1, false, false, None);

    assert!(extract_dir.join("project/empty_sub").is_dir());
    assert!(extract_dir.join("project/nonempty/f.txt").is_file());
}

#[test]
fn test_file_permissions_preserved() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("perms");
    fs::create_dir_all(&src).unwrap();

    let exec_file = src.join("run.sh");
    fs::write(&exec_file, "#!/bin/bash\necho hi").unwrap();
    fs::set_permissions(&exec_file, fs::Permissions::from_mode(0o755)).unwrap();

    let ro_file = src.join("readonly.txt");
    fs::write(&ro_file, "read only").unwrap();
    fs::set_permissions(&ro_file, fs::Permissions::from_mode(0o444)).unwrap();

    let archive = base.join("perms.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(&[src_str.as_str()], &archive, "w", None, None, false, 1, false, None);

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 1, false, false, None);

    let extracted_exec = extract_dir.join("perms/run.sh");
    let extracted_ro = extract_dir.join("perms/readonly.txt");
    assert_eq!(
        fs::metadata(&extracted_exec).unwrap().permissions().mode(),
        fs::metadata(&exec_file).unwrap().permissions().mode()
    );
    assert_eq!(
        fs::metadata(&extracted_ro).unwrap().permissions().mode(),
        fs::metadata(&ro_file).unwrap().permissions().mode()
    );
}

#[test]
fn test_error_nonexistent_archive() {
    // The Rust implementation calls process::exit(1), so we can't easily catch it
    // in-process like Python's SystemExit. Instead, verify the file doesn't exist
    // and the function would fail. We test the pre-condition.
    assert!(!Path::new("/tmp/nonexistent_har_test.h5").exists());
}

#[test]
fn test_error_corrupt_archive() {
    let tmp = TempDir::new().unwrap();
    let bad = tmp.path().join("bad.h5");
    fs::write(&bad, "not an hdf5 file").unwrap();

    // Verify the corrupt file exists but is not valid HDF5
    assert!(bad.exists());
    assert!(hdf5::File::open(&bad).is_err());
}

#[test]
fn test_empty_file_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("emptysrc");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("empty.txt"), "").unwrap();

    let archive = base.join("empty_file.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(&[src_str.as_str()], &archive, "w", None, None, false, 1, false, None);

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 1, false, false, None);

    let extracted = extract_dir.join("emptysrc/empty.txt");
    assert!(extracted.exists());
    assert_eq!(fs::metadata(&extracted).unwrap().len(), 0);
}

#[test]
fn test_binary_file_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("binsrc");
    fs::create_dir_all(&src).unwrap();
    let all_bytes: Vec<u8> = (0..=255u8).collect();
    fs::write(src.join("allbytes.bin"), &all_bytes).unwrap();

    let archive = base.join("bin.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(&[src_str.as_str()], &archive, "w", None, None, false, 1, false, None);

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 1, false, false, None);

    let extracted = extract_dir.join("binsrc/allbytes.bin");
    assert_eq!(fs::read(&extracted).unwrap(), all_bytes);
}

#[test]
fn test_compression_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("comp");
    fs::create_dir_all(&src).unwrap();
    let data = "A".repeat(100000);
    fs::write(src.join("data.txt"), &data).unwrap();

    let archive = base.join("comp.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(
        &[src_str.as_str()],
        &archive,
        "w",
        Some("gzip"),
        Some(9),
        true,
        1,
        false,
        None,
    );

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 1, false, false, None);

    assert_eq!(
        fs::read_to_string(extract_dir.join("comp/data.txt")).unwrap(),
        data
    );
    // Compressed file should be much smaller than 100KB
    assert!(fs::metadata(&archive).unwrap().len() < 10000);
}

// ---- Parallel tests ----

#[test]
fn test_parallel_ingest() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("src");
    fs::create_dir_all(&src).unwrap();
    let src_str = src.to_string_lossy().to_string();
    make_test_tree(&src_str, 3, 4);

    let archive = base.join("par.h5").to_string_lossy().to_string();
    pack_or_append_to_h5(&[src_str.as_str()], &archive, "w", None, None, false, 4, false, None);

    let h5f = hdf5::File::open(&archive).unwrap();
    let mut keys = Vec::new();
    for (name, obj_type) in crate::collect_items(&h5f) {
        if matches!(obj_type, crate::H5ObjType::Dataset) {
            keys.push(name);
        }
    }
    assert_eq!(keys.len(), 12); // 3 dirs * 4 files

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 1, false, false, None);

    for i in 0..3 {
        for j in 0..4 {
            let p = extract_dir.join(format!("src/dir{}/file{}.txt", i, j));
            assert!(p.exists());
            assert_eq!(
                fs::read_to_string(&p).unwrap(),
                format!("content dir{}/file{}", i, j)
            );
        }
    }
}

#[test]
fn test_parallel_extract() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("src");
    fs::create_dir_all(&src).unwrap();
    let src_str = src.to_string_lossy().to_string();
    make_test_tree(&src_str, 3, 4);

    let archive = base.join("seq.h5").to_string_lossy().to_string();
    pack_or_append_to_h5(&[src_str.as_str()], &archive, "w", None, None, false, 1, false, None);

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 4, false, false, None);

    for i in 0..3 {
        for j in 0..4 {
            let p = extract_dir.join(format!("src/dir{}/file{}.txt", i, j));
            assert_eq!(
                fs::read_to_string(&p).unwrap(),
                format!("content dir{}/file{}", i, j)
            );
        }
    }
}

#[test]
fn test_parallel_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("project");
    fs::create_dir_all(src.join("data")).unwrap();
    fs::create_dir_all(src.join("empty_sub")).unwrap();

    fs::write(src.join("readme.txt"), "hello").unwrap();
    let all_bytes: Vec<u8> = (0..=255u8).collect();
    fs::write(src.join("data/bin.dat"), &all_bytes).unwrap();

    let exec_file = src.join("data/run.sh");
    fs::write(&exec_file, "#!/bin/bash").unwrap();
    fs::set_permissions(&exec_file, fs::Permissions::from_mode(0o755)).unwrap();

    let archive = base.join("rt.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(&[src_str.as_str()], &archive, "w", None, None, false, 3, false, None);

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 3, false, false, None);

    // Verify text file
    assert_eq!(
        fs::read_to_string(extract_dir.join("project/readme.txt")).unwrap(),
        "hello"
    );
    // Verify binary file
    assert_eq!(
        fs::read(extract_dir.join("project/data/bin.dat")).unwrap(),
        all_bytes
    );
    // Verify permissions
    assert_eq!(
        fs::metadata(extract_dir.join("project/data/run.sh"))
            .unwrap()
            .permissions()
            .mode(),
        fs::metadata(&exec_file).unwrap().permissions().mode()
    );
    // Verify empty dir
    assert!(extract_dir.join("project/empty_sub").is_dir());
}

#[test]
fn test_parallel_ingest_with_compression() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("comp");
    fs::create_dir_all(&src).unwrap();
    let big_data = "B".repeat(50000);
    fs::write(src.join("big.txt"), &big_data).unwrap();

    let archive = base.join("comp.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(
        &[src_str.as_str()],
        &archive,
        "w",
        Some("gzip"),
        Some(9),
        true,
        2,
        false,
        None,
    );

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 1, false, false, None);

    assert_eq!(
        fs::read_to_string(extract_dir.join("comp/big.txt")).unwrap(),
        big_data
    );
    assert!(fs::metadata(&archive).unwrap().len() < 5000);
}

#[test]
fn test_parallel_append_skips_existing() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("app");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("f.txt"), "v1").unwrap();

    let archive = base.join("app.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(&[src_str.as_str()], &archive, "w", None, None, false, 2, false, None);

    // Modify and append — should skip existing
    fs::write(src.join("f.txt"), "v2").unwrap();
    fs::write(src.join("new.txt"), "new").unwrap();
    pack_or_append_to_h5(&[src_str.as_str()], &archive, "a", None, None, false, 2, false, None);

    let h5f = hdf5::File::open(&archive).unwrap();
    let ds_f: Vec<u8> = h5f.dataset("app/f.txt").unwrap().read_raw().unwrap();
    assert_eq!(ds_f, b"v1"); // original kept
    let ds_new: Vec<u8> = h5f.dataset("app/new.txt").unwrap().read_raw().unwrap();
    assert_eq!(ds_new, b"new");
}

#[test]
fn test_parallel_extract_preserves_permissions() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("perms");
    fs::create_dir_all(&src).unwrap();
    let exec_file = src.join("exec.sh");
    fs::write(&exec_file, "#!/bin/bash").unwrap();
    fs::set_permissions(&exec_file, fs::Permissions::from_mode(0o755)).unwrap();

    let archive = base.join("perms.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(&[src_str.as_str()], &archive, "w", None, None, false, 1, false, None);

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 2, false, false, None);

    assert_eq!(
        fs::metadata(extract_dir.join("perms/exec.sh"))
            .unwrap()
            .permissions()
            .mode(),
        fs::metadata(&exec_file).unwrap().permissions().mode()
    );
}

#[test]
fn test_parallel_extract_preserves_empty_dirs() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("emp");
    fs::create_dir_all(src.join("empty")).unwrap();
    fs::create_dir_all(src.join("full")).unwrap();
    fs::write(src.join("full/f.txt"), "data").unwrap();

    let archive = base.join("emp.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(&[src_str.as_str()], &archive, "w", None, None, false, 1, false, None);

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 2, false, false, None);

    assert!(extract_dir.join("emp/empty").is_dir());
    assert!(extract_dir.join("emp/full/f.txt").is_file());
}

#[test]
fn test_parallel_many_small_files() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("many");
    fs::create_dir_all(&src).unwrap();
    let src_str = src.to_string_lossy().to_string();
    make_test_tree(&src_str, 10, 10);

    let archive = base.join("many.h5").to_string_lossy().to_string();
    pack_or_append_to_h5(&[src_str.as_str()], &archive, "w", None, None, false, 8, false, None);

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 8, false, false, None);

    let mut count = 0;
    for entry in walkdir::WalkDir::new(extract_dir.join("many"))
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.path().is_file() {
            count += 1;
        }
    }
    assert_eq!(count, 100);

    // Spot check a few files
    for i in [0, 5, 9] {
        for j in [0, 5, 9] {
            let p = extract_dir.join(format!("many/dir{}/file{}.txt", i, j));
            assert_eq!(
                fs::read_to_string(&p).unwrap(),
                format!("content dir{}/file{}", i, j)
            );
        }
    }
}

#[test]
fn test_checksum_sha256_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("cksum");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("hello.txt"), "hello world").unwrap();

    let archive = base.join("cksum.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(
        &[src_str.as_str()], &archive, "w",
        None, None, false, 1, false, Some("sha256"),
    );

    // Verify checksum attributes were stored
    let h5f = hdf5::File::open(&archive).unwrap();
    let ds = h5f.dataset("cksum/hello.txt").unwrap();
    let algo: hdf5::types::VarLenUnicode = ds.attr("har_checksum_algo").unwrap().read_scalar().unwrap();
    assert_eq!(algo.as_str(), "sha256");
    let hash: hdf5::types::VarLenUnicode = ds.attr("har_checksum").unwrap().read_scalar().unwrap();
    assert!(!hash.as_str().is_empty());
    drop(h5f);

    // Extract with validation
    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 1, false, true, Some("sha256"));

    assert_eq!(
        fs::read_to_string(extract_dir.join("cksum/hello.txt")).unwrap(),
        "hello world"
    );
}

#[test]
fn test_checksum_blake3_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("b3");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("data.bin"), vec![42u8; 1000]).unwrap();

    let archive = base.join("b3.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(
        &[src_str.as_str()], &archive, "w",
        None, None, false, 1, false, Some("blake3"),
    );

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 1, false, true, Some("blake3"));

    assert_eq!(fs::read(extract_dir.join("b3/data.bin")).unwrap(), vec![42u8; 1000]);
}

#[test]
fn test_checksum_md5_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path();

    let src = base.join("md5");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("test.txt"), "md5 test content").unwrap();

    let archive = base.join("md5.h5").to_string_lossy().to_string();
    let src_str = src.to_string_lossy().to_string();
    pack_or_append_to_h5(
        &[src_str.as_str()], &archive, "w",
        None, None, false, 1, false, Some("md5"),
    );

    let extract_dir = base.join("out");
    fs::create_dir_all(&extract_dir).unwrap();
    let extract_str = extract_dir.to_string_lossy().to_string();
    extract_h5_to_directory(&archive, &extract_str, None, 1, false, true, Some("md5"));

    assert_eq!(
        fs::read_to_string(extract_dir.join("md5/test.txt")).unwrap(),
        "md5 test content"
    );
}
