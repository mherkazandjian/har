use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::Path;

const INTERNAL_ATTRS: &[&str] = &["mode", "empty_dir"];
const XATTR_PREFIX: &str = "xattr.";

/// Check if an attribute name is internal (managed by har).
pub fn is_internal_attr(name: &str) -> bool {
    name.starts_with("har_") || INTERNAL_ATTRS.contains(&name)
}

/// Convert an HDF5 attribute value to a JSON value.
/// Tries string, then numeric types, falls back to base64-encoded raw bytes.
pub fn attr_to_json_value(attr: &hdf5::Attribute) -> Option<Value> {
    // Try VarLenUnicode (string) — most common for user attributes
    if let Ok(v) = attr.read_scalar::<hdf5::types::VarLenUnicode>() {
        return Some(json!(v.as_str()));
    }
    // Try f64 — covers int and float via HDF5 type conversion.
    // Check if the value is a whole number to output as integer.
    if let Ok(v) = attr.read_scalar::<f64>() {
        if v == v.trunc() && v.is_finite() && v.abs() < (i64::MAX as f64) {
            return Some(json!(v as i64));
        }
        return Some(json!(v));
    }
    // Try u64 for very large unsigned values
    if let Ok(v) = attr.read_scalar::<u64>() {
        return Some(json!(v));
    }
    // Try 1-D numeric array (read as f64 via HDF5 conversion)
    if let Ok(data) = attr.read_raw::<f64>() {
        if data.len() > 1 {
            let arr: Vec<Value> = data
                .iter()
                .map(|&x| {
                    if x == x.trunc() && x.is_finite() && x.abs() < (i64::MAX as f64) {
                        json!(x as i64)
                    } else {
                        json!(x)
                    }
                })
                .collect();
            return Some(Value::Array(arr));
        }
    }
    // Fallback: raw bytes as base64
    if let Ok(data) = attr.read_raw::<u8>() {
        let encoded = BASE64_STANDARD.encode(&data);
        return Some(json!(format!("base64:{}", encoded)));
    }
    None
}

/// Metadata collected from a single file's HDF5 attributes.
pub struct FileMetadata {
    pub hdf5_attrs: Map<String, Value>,
    pub xattrs_json: Map<String, Value>,
    pub xattrs_raw: BTreeMap<String, Vec<u8>>,
}

impl FileMetadata {
    pub fn is_empty(&self) -> bool {
        self.hdf5_attrs.is_empty() && self.xattrs_json.is_empty()
    }
}

/// Collect user-defined metadata from a dataset's HDF5 attributes.
/// Internal attributes (har_*, mode, empty_dir) are excluded.
/// Attributes with `xattr.` prefix are split into xattr maps.
pub fn collect_user_metadata(ds: &hdf5::Dataset) -> FileMetadata {
    let mut hdf5_attrs = Map::new();
    let mut xattrs_json = Map::new();
    let mut xattrs_raw = BTreeMap::new();

    for name in ds.attr_names().unwrap_or_default() {
        if is_internal_attr(&name) {
            continue;
        }
        if let Some(xattr_name) = name.strip_prefix(XATTR_PREFIX) {
            if let Ok(attr) = ds.attr(&name) {
                if let Ok(raw) = attr.read_raw::<u8>() {
                    let encoded = BASE64_STANDARD.encode(&raw);
                    xattrs_json.insert(
                        xattr_name.to_string(),
                        json!(format!("base64:{}", encoded)),
                    );
                    xattrs_raw.insert(xattr_name.to_string(), raw);
                }
            }
        } else if let Ok(attr) = ds.attr(&name) {
            if let Some(val) = attr_to_json_value(&attr) {
                hdf5_attrs.insert(name, val);
            }
        }
    }

    FileMetadata {
        hdf5_attrs,
        xattrs_json,
        xattrs_raw,
    }
}

/// Read filesystem extended attributes (user.* namespace only).
pub fn read_xattrs(path: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut result = BTreeMap::new();
    if let Ok(names) = xattr::list(path) {
        for name in names {
            let name_str = name.to_string_lossy().to_string();
            if name_str.starts_with("user.") {
                if let Ok(Some(value)) = xattr::get(path, &name) {
                    result.insert(name_str, value);
                }
            }
        }
    }
    result
}

/// Write metadata.json manifest to the given directory.
pub fn write_metadata_json(dir: &Path, all_metadata: &BTreeMap<String, FileMetadata>) {
    let mut files = Map::new();
    for (path, meta) in all_metadata {
        if meta.is_empty() {
            continue;
        }
        let mut entry = Map::new();
        if !meta.hdf5_attrs.is_empty() {
            entry.insert(
                "hdf5_attrs".to_string(),
                Value::Object(meta.hdf5_attrs.clone()),
            );
        }
        if !meta.xattrs_json.is_empty() {
            entry.insert(
                "xattrs".to_string(),
                Value::Object(meta.xattrs_json.clone()),
            );
        }
        if !entry.is_empty() {
            files.insert(path.clone(), Value::Object(entry));
        }
    }
    if files.is_empty() {
        return;
    }
    let root = json!({
        "har_metadata_version": 1,
        "files": files,
    });
    let dest = dir.join("metadata.json");
    let content = serde_json::to_string_pretty(&root).unwrap();
    std::fs::write(dest, content).unwrap();
}

/// Restore filesystem extended attributes from raw bytes.
pub fn restore_xattrs(path: &Path, xattrs: &BTreeMap<String, Vec<u8>>) {
    for (name, value) in xattrs {
        if let Err(e) = xattr::set(path, name, value) {
            eprintln!(
                "Warning: failed to set xattr '{}' on '{}': {}",
                name,
                path.display(),
                e
            );
        }
    }
}

/// Restore xattrs from a JSON map (base64-encoded values, used for BagIt mode).
pub fn restore_xattrs_from_json(path: &Path, xattrs: &Map<String, Value>) {
    for (name, value) in xattrs {
        let bytes = match value.as_str() {
            Some(s) if s.starts_with("base64:") => match BASE64_STANDARD.decode(&s[7..]) {
                Ok(b) => b,
                Err(_) => continue,
            },
            Some(s) => s.as_bytes().to_vec(),
            None => continue,
        };
        if let Err(e) = xattr::set(path, name, &bytes) {
            eprintln!(
                "Warning: failed to set xattr '{}' on '{}': {}",
                name,
                path.display(),
                e
            );
        }
    }
}

/// Write user_metadata JSON blob to an HDF5 file (for BagIt mode).
pub fn write_user_metadata_dataset(
    h5f: &hdf5::File,
    metadata: &BTreeMap<String, serde_json::Map<String, Value>>,
) {
    if metadata.is_empty() {
        return;
    }
    let json_blob = serde_json::to_string_pretty(metadata).unwrap();
    let bytes = json_blob.as_bytes();
    h5f.new_dataset::<u8>()
        .shape(bytes.len())
        .create("user_metadata")
        .unwrap()
        .write_raw(bytes)
        .unwrap();
}

/// Read user_metadata JSON blob from an HDF5 file (for BagIt mode).
pub fn read_user_metadata_dataset(
    h5f: &hdf5::File,
) -> BTreeMap<String, serde_json::Map<String, Value>> {
    if let Ok(ds) = h5f.dataset("user_metadata") {
        if let Ok(blob) = ds.read_raw::<u8>() {
            if let Ok(s) = String::from_utf8(blob) {
                if let Ok(map) = serde_json::from_str(&s) {
                    return map;
                }
            }
        }
    }
    BTreeMap::new()
}
