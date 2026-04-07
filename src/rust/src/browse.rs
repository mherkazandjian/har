//! Interactive TUI for browsing HDF5 archives (legacy and BagIt).
//!
//! Dual-pane layout: tree (left 40%) + details (right 60%).
//! vim keybindings: j/k navigate, h/l collapse/expand, / search, q quit.

use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};
use std::io;

// ---------------------------------------------------------------------------
// BagIt index — dual format support (Python compound vs Rust parallel arrays)
// ---------------------------------------------------------------------------

/// Python BagIt compound index entry — old format (without owner/mtime).
#[derive(hdf5::H5Type, Clone)]
#[repr(C)]
struct CompoundIndexEntry {
    path: hdf5::types::FixedAscii<4096>,
    batch_id: u32,
    offset: u64,
    length: u64,
    mode: u32,
    sha256: hdf5::types::FixedAscii<64>,
}

/// Python BagIt compound index entry — new format (with owner/mtime).
#[derive(hdf5::H5Type, Clone)]
#[repr(C)]
struct CompoundIndexEntryNew {
    path: hdf5::types::FixedAscii<4096>,
    batch_id: u32,
    offset: u64,
    length: u64,
    mode: u32,
    sha256: hdf5::types::FixedAscii<64>,
    uid: u32,
    gid: u32,
    mtime: f64,
    owner: hdf5::types::FixedAscii<256>,
    group_name: hdf5::types::FixedAscii<256>,
}

struct BagitIndex {
    paths: Vec<String>,
    batch_ids: Vec<u32>,
    offsets: Vec<u64>,
    lengths: Vec<u64>,
    modes: Vec<u32>,
    sha256s: Vec<String>,
    uids: Vec<u32>,
    gids: Vec<u32>,
    mtimes: Vec<f64>,
    owners: Vec<String>,
    groups: Vec<String>,
}

fn empty_bagit_index() -> BagitIndex {
    BagitIndex {
        paths: Vec::new(), batch_ids: Vec::new(), offsets: Vec::new(),
        lengths: Vec::new(), modes: Vec::new(), sha256s: Vec::new(),
        uids: Vec::new(), gids: Vec::new(), mtimes: Vec::new(),
        owners: Vec::new(), groups: Vec::new(),
    }
}

/// Read BagIt index from either Rust parallel-array or Python compound format.
fn read_bagit_index(h5f: &hdf5::File) -> BagitIndex {
    if let Ok(idx) = h5f.group("index_data") {
        // Rust format: parallel arrays
        let paths_blob: Vec<u8> = idx.dataset("paths").unwrap().read_raw().unwrap();
        let paths_str = String::from_utf8(paths_blob).unwrap();
        let paths: Vec<String> = paths_str.split('\n').map(|s| s.to_string()).collect();
        let batch_ids: Vec<u32> = idx.dataset("batch_id").unwrap().read_raw().unwrap();
        let offsets: Vec<u64> = idx.dataset("offset").unwrap().read_raw().unwrap();
        let lengths: Vec<u64> = idx.dataset("length").unwrap().read_raw().unwrap();
        let modes: Vec<u32> = idx.dataset("mode").unwrap().read_raw().unwrap();
        let shas_blob: Vec<u8> = idx.dataset("sha256s").unwrap().read_raw().unwrap();
        let shas_str = String::from_utf8(shas_blob).unwrap();
        let sha256s: Vec<String> = shas_str.split('\n').map(|s| s.to_string()).collect();
        let n = paths.len();
        // Optional owner/mtime fields
        let uids: Vec<u32> = idx.dataset("uid").ok().and_then(|d| d.read_raw().ok()).unwrap_or_else(|| vec![0; n]);
        let gids: Vec<u32> = idx.dataset("gid").ok().and_then(|d| d.read_raw().ok()).unwrap_or_else(|| vec![0; n]);
        let mtimes: Vec<f64> = idx.dataset("mtime").ok().and_then(|d| d.read_raw().ok()).unwrap_or_else(|| vec![0.0; n]);
        let owners = idx.dataset("owners").ok()
            .and_then(|d| d.read_raw::<u8>().ok())
            .map(|b| String::from_utf8(b).unwrap_or_default().split('\n').map(|s| s.to_string()).collect())
            .unwrap_or_else(|| vec![String::new(); n]);
        let groups = idx.dataset("groups").ok()
            .and_then(|d| d.read_raw::<u8>().ok())
            .map(|b| String::from_utf8(b).unwrap_or_default().split('\n').map(|s| s.to_string()).collect())
            .unwrap_or_else(|| vec![String::new(); n]);
        BagitIndex { paths, batch_ids, offsets, lengths, modes, sha256s, uids, gids, mtimes, owners, groups }
    } else if let Ok(ds) = h5f.dataset("index") {
        // Python format: compound dataset — try new format first, fall back to old
        if let Ok(entries) = ds.read_raw::<CompoundIndexEntryNew>() {
            let paths: Vec<String> = entries.iter().map(|e| e.path.as_str().to_string()).collect();
            let batch_ids = entries.iter().map(|e| e.batch_id).collect();
            let offsets = entries.iter().map(|e| e.offset).collect();
            let lengths = entries.iter().map(|e| e.length).collect();
            let modes = entries.iter().map(|e| e.mode).collect();
            let sha256s: Vec<String> = entries.iter().map(|e| e.sha256.as_str().to_string()).collect();
            let uids = entries.iter().map(|e| e.uid).collect();
            let gids = entries.iter().map(|e| e.gid).collect();
            let mtimes = entries.iter().map(|e| e.mtime).collect();
            let owners: Vec<String> = entries.iter().map(|e| e.owner.as_str().to_string()).collect();
            let groups: Vec<String> = entries.iter().map(|e| e.group_name.as_str().to_string()).collect();
            BagitIndex { paths, batch_ids, offsets, lengths, modes, sha256s, uids, gids, mtimes, owners, groups }
        } else if let Ok(entries) = ds.read_raw::<CompoundIndexEntry>() {
            let n = entries.len();
            let paths: Vec<String> = entries.iter().map(|e| e.path.as_str().to_string()).collect();
            let batch_ids = entries.iter().map(|e| e.batch_id).collect();
            let offsets = entries.iter().map(|e| e.offset).collect();
            let lengths = entries.iter().map(|e| e.length).collect();
            let modes = entries.iter().map(|e| e.mode).collect();
            let sha256s: Vec<String> = entries.iter().map(|e| e.sha256.as_str().to_string()).collect();
            BagitIndex { paths, batch_ids, offsets, lengths, modes, sha256s,
                uids: vec![0; n], gids: vec![0; n], mtimes: vec![0.0; n],
                owners: vec![String::new(); n], groups: vec![String::new(); n] }
        } else {
            empty_bagit_index()
        }
    } else {
        empty_bagit_index()
    }
}

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq)]
enum NodeType {
    Group,
    Dataset,
    EmptyDir,
}

#[derive(Clone)]
struct TreeNode {
    name: String,
    full_path: String,
    node_type: NodeType,
    children: Vec<TreeNode>,
    expanded: bool,
    depth: u16,
    size_bytes: u64,
    child_count: usize,
}

struct VisibleRow {
    name: String,
    full_path: String,
    node_type: NodeType,
    expanded: bool,
    depth: u16,
    size_bytes: u64,
    child_count: usize,
    has_children: bool,
}

struct App {
    root: TreeNode,
    visible: Vec<VisibleRow>,
    cursor: usize,
    tree_scroll: usize,
    detail_scroll: usize,
    pane: usize, // 0=left, 1=right
    search_mode: bool,
    search_query: String,
    archive_name: String,
    is_bagit: bool,
    h5_path: String,
    total_datasets: usize,
    total_groups: usize,
    cached_detail_path: String,
    cached_detail: Vec<DetailLine>,
}

#[derive(Clone)]
struct DetailLine {
    text: String,
    is_header: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn human_size(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    for unit in &units {
        if size < 1024.0 {
            if size == size.trunc() {
                return format!("{} {}", size as u64, unit);
            }
            return format!("{:.1} {}", size, unit);
        }
        size /= 1024.0;
    }
    format!("{:.1} PB", size)
}

// ---------------------------------------------------------------------------
// Tree building — legacy
// ---------------------------------------------------------------------------

fn build_tree_legacy(h5f: &hdf5::File) -> TreeNode {
    let mut root = TreeNode {
        name: String::new(),
        full_path: "/".to_string(),
        node_type: NodeType::Group,
        children: Vec::new(),
        expanded: true,
        depth: 0,
        size_bytes: 0,
        child_count: 0,
    };

    // Collect all items
    let items = crate::collect_items(h5f);
    // Build group map
    let mut group_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    // We'll build tree by inserting items
    // First pass: create groups
    for (path, obj_type) in &items {
        match obj_type {
            crate::H5ObjType::Group => {
                insert_group(&mut root, path, &mut group_map);
            }
            crate::H5ObjType::Dataset => {
                // Will handle in second pass
            }
            crate::H5ObjType::EmptyDirGroup => {
                insert_group(&mut root, path, &mut group_map);
            }
        }
    }
    // Second pass: insert datasets
    for (path, obj_type) in &items {
        if matches!(obj_type, crate::H5ObjType::Dataset) {
            insert_dataset(&mut root, h5f, path);
        }
    }

    // Check for empty_dir attribute on groups
    mark_empty_dirs(h5f, &mut root);

    sort_tree(&mut root);
    root
}

fn insert_group(root: &mut TreeNode, path: &str, _map: &mut std::collections::HashMap<String, usize>) {
    let parts: Vec<&str> = path.split('/').collect();
    let mut current = root;
    for (i, part) in parts.iter().enumerate() {
        let depth = (i + 1) as u16;
        let full = parts[..=i].join("/");
        let pos = current.children.iter().position(|c| c.name == *part);
        if pos.is_none() {
            current.children.push(TreeNode {
                name: part.to_string(),
                full_path: full,
                node_type: NodeType::Group,
                children: Vec::new(),
                expanded: false,
                depth,
                size_bytes: 0,
                child_count: 0,
            });
        }
        let idx = current.children.iter().position(|c| c.name == *part).unwrap();
        current = &mut current.children[idx];
    }
}

fn insert_dataset(root: &mut TreeNode, h5f: &hdf5::File, path: &str) {
    let parts: Vec<&str> = path.split('/').collect();
    let mut current = root;
    // Navigate to parent
    for part in &parts[..parts.len() - 1] {
        let idx = current.children.iter().position(|c| c.name == *part);
        if let Some(i) = idx {
            current = &mut current.children[i];
        } else {
            // Create missing group
            let depth = current.depth + 1;
            current.children.push(TreeNode {
                name: part.to_string(),
                full_path: String::new(),
                node_type: NodeType::Group,
                children: Vec::new(),
                expanded: false,
                depth,
                size_bytes: 0,
                child_count: 0,
            });
            let len = current.children.len();
            current = &mut current.children[len - 1];
        }
    }
    let leaf_name = parts.last().unwrap();
    let depth = parts.len() as u16;
    let storage_size = h5f.dataset(path).map(|ds| ds.storage_size()).unwrap_or(0);
    current.children.push(TreeNode {
        name: leaf_name.to_string(),
        full_path: path.to_string(),
        node_type: NodeType::Dataset,
        children: Vec::new(),
        expanded: false,
        depth,
        size_bytes: storage_size,
        child_count: 0,
    });
}

fn mark_empty_dirs(h5f: &hdf5::File, node: &mut TreeNode) {
    if node.node_type == NodeType::Group && !node.full_path.is_empty() && node.full_path != "/" {
        if let Ok(grp) = h5f.group(&node.full_path) {
            if let Ok(attr) = grp.attr("empty_dir") {
                if attr.read_scalar::<bool>().unwrap_or(false) {
                    node.node_type = NodeType::EmptyDir;
                }
            }
        }
    }
    for child in &mut node.children {
        mark_empty_dirs(h5f, child);
    }
}

// ---------------------------------------------------------------------------
// Tree building — BagIt
// ---------------------------------------------------------------------------

fn build_tree_bagit(h5f: &hdf5::File) -> TreeNode {
    let mut root = TreeNode {
        name: String::new(),
        full_path: "/".to_string(),
        node_type: NodeType::Group,
        children: Vec::new(),
        expanded: true,
        depth: 0,
        size_bytes: 0,
        child_count: 0,
    };

    let bi = read_bagit_index(h5f);

    for (i, p) in bi.paths.iter().enumerate() {
        let clean = p.strip_prefix("data/").unwrap_or(p);
        let parts: Vec<&str> = clean.split('/').collect();

        // Ensure parent groups
        let mut current = &mut root;
        for (j, part) in parts[..parts.len() - 1].iter().enumerate() {
            let depth = (j + 1) as u16;
            let full = parts[..=j].join("/");
            let pos = current.children.iter().position(|c| c.name == *part);
            if pos.is_none() {
                current.children.push(TreeNode {
                    name: part.to_string(),
                    full_path: full,
                    node_type: NodeType::Group,
                    children: Vec::new(),
                    expanded: false,
                    depth,
                    size_bytes: 0,
                    child_count: 0,
                });
            }
            let idx2 = current.children.iter().position(|c| c.name == *part).unwrap();
            current = &mut current.children[idx2];
        }

        let leaf = parts.last().unwrap();
        let depth = parts.len() as u16;
        current.children.push(TreeNode {
            name: leaf.to_string(),
            full_path: clean.to_string(),
            node_type: NodeType::Dataset,
            children: Vec::new(),
            expanded: false,
            depth,
            size_bytes: if i < bi.lengths.len() { bi.lengths[i] } else { 0 },
            child_count: 0,
        });
    }

    // Empty dirs
    if let Ok(ds) = h5f.dataset("empty_dirs") {
        if let Ok(blob) = ds.read_raw::<u8>() {
            let dirs_str = String::from_utf8(blob).unwrap_or_default();
            for d in dirs_str.split('\n') {
                if d.is_empty() {
                    continue;
                }
                let clean = d.strip_prefix("data/").unwrap_or(d);
                let parts: Vec<&str> = clean.split('/').collect();
                let mut current = &mut root;
                for (j, part) in parts[..parts.len() - 1].iter().enumerate() {
                    let depth = (j + 1) as u16;
                    let full = parts[..=j].join("/");
                    let pos = current.children.iter().position(|c| c.name == *part);
                    if pos.is_none() {
                        current.children.push(TreeNode {
                            name: part.to_string(),
                            full_path: full,
                            node_type: NodeType::Group,
                            children: Vec::new(),
                            expanded: false,
                            depth,
                            size_bytes: 0,
                            child_count: 0,
                        });
                    }
                    let idx2 = current.children.iter().position(|c| c.name == *part).unwrap();
                    current = &mut current.children[idx2];
                }
                let leaf = parts.last().unwrap();
                current.children.push(TreeNode {
                    name: leaf.to_string(),
                    full_path: clean.to_string(),
                    node_type: NodeType::EmptyDir,
                    children: Vec::new(),
                    expanded: false,
                    depth: parts.len() as u16,
                    size_bytes: 0,
                    child_count: 0,
                });
            }
        }
    }

    // BagIt tag files (show under a [bagit] virtual group)
    if let Ok(bagit_grp) = h5f.group("bagit") {
        let mut bagit_node = TreeNode {
            name: "[bagit]".to_string(),
            full_path: "[bagit]".to_string(),
            node_type: NodeType::Group,
            children: Vec::new(),
            expanded: false,
            depth: 1,
            size_bytes: 0,
            child_count: 0,
        };
        let mut names = bagit_grp.member_names().unwrap_or_default();
        names.sort();
        for name in &names {
            let size = bagit_grp.dataset(name).map(|d| d.storage_size()).unwrap_or(0);
            bagit_node.children.push(TreeNode {
                name: name.clone(),
                full_path: format!("[bagit]/{}", name),
                node_type: NodeType::Dataset,
                children: Vec::new(),
                expanded: false,
                depth: 2,
                size_bytes: size,
                child_count: 0,
            });
        }
        root.children.push(bagit_node);
    }

    sort_tree(&mut root);
    root
}

// ---------------------------------------------------------------------------
// Tree utilities
// ---------------------------------------------------------------------------

fn sort_tree(node: &mut TreeNode) {
    node.children.sort_by(|a, b| {
        let a_is_file = a.node_type == NodeType::Dataset;
        let b_is_file = b.node_type == NodeType::Dataset;
        a_is_file.cmp(&b_is_file).then(a.name.cmp(&b.name))
    });
    node.child_count = node.children.iter().map(|c| {
        if c.node_type == NodeType::Dataset { 1 } else { c.child_count }
    }).sum();
    for child in &mut node.children {
        sort_tree(child);
    }
    // Recompute after children are sorted
    node.child_count = node.children.iter().map(|c| {
        if c.node_type == NodeType::Dataset { 1 } else { c.child_count }
    }).sum();
    // Accumulate size from children for groups
    if node.node_type != NodeType::Dataset {
        node.size_bytes = node.children.iter().map(|c| c.size_bytes).sum();
    }
}

fn flatten(node: &TreeNode, out: &mut Vec<VisibleRow>, skip_root: bool) {
    if !skip_root {
        out.push(VisibleRow {
            name: node.name.clone(),
            full_path: node.full_path.clone(),
            node_type: node.node_type.clone(),
            expanded: node.expanded,
            depth: node.depth,
            size_bytes: node.size_bytes,
            child_count: node.child_count,
            has_children: !node.children.is_empty(),
        });
    }
    if node.expanded {
        for child in &node.children {
            flatten(child, out, false);
        }
    }
}

fn count_items(node: &TreeNode) -> (usize, usize) {
    let mut ds = 0usize;
    let mut grp = 0usize;
    match node.node_type {
        NodeType::Dataset => ds += 1,
        NodeType::Group | NodeType::EmptyDir => grp += 1,
    }
    for c in &node.children {
        let (d, g) = count_items(c);
        ds += d;
        grp += g;
    }
    (ds, grp)
}

fn find_node_mut<'a>(root: &'a mut TreeNode, path: &str) -> Option<&'a mut TreeNode> {
    if root.full_path == path {
        return Some(root);
    }
    for child in &mut root.children {
        if let Some(found) = find_node_mut(child, path) {
            return Some(found);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Detail info collection
// ---------------------------------------------------------------------------

fn collect_detail_legacy(h5_path: &str, path: &str, node_type: &NodeType, size_bytes: u64, _child_count: usize) -> Vec<DetailLine> {
    let mut lines = Vec::new();

    let h5f = match hdf5::File::open(h5_path) {
        Ok(f) => f,
        Err(_) => return lines,
    };

    match node_type {
        NodeType::Dataset => {
            if let Ok(ds) = h5f.dataset(path) {
                // Storage section
                lines.push(DetailLine { text: "STORAGE".into(), is_header: true});
                lines.push(DetailLine {
                    text: format!("  Shape     {:?}", ds.shape()),
                    is_header: false,
                });
                let raw_size = ds.size() * std::mem::size_of::<u8>();
                let stored = ds.storage_size();
                lines.push(DetailLine {
                    text: format!("  Type      {:?}", ds.dtype()),
                    is_header: false,
                });
                lines.push(DetailLine {
                    text: format!("  Raw       {}", human_size(raw_size as u64)),
                    is_header: false,
                });
                let filters = format!("{:?}", ds.filters());
                lines.push(DetailLine {
                    text: format!("  Stored    {} ({})", human_size(stored), filters),
                    is_header: false,
                });
                if raw_size > 0 && stored > 0 {
                    let ratio = raw_size as f64 / stored as f64;
                    lines.push(DetailLine {
                        text: format!("  Ratio     {:.1}x", ratio),
                        is_header: false,
                    });
                }
                if let Ok(mode_attr) = ds.attr("mode") {
                    if let Ok(m) = mode_attr.read_scalar::<u32>() {
                        lines.push(DetailLine { text: format!("  Mode      0o{:o}", m), is_header: false });
                    }
                }
                if let Ok(owner_attr) = ds.attr("owner") {
                    if let Ok(o) = owner_attr.read_scalar::<hdf5::types::VarLenUnicode>() {
                        let uid_str = ds.attr("uid").ok().and_then(|a| a.read_scalar::<u32>().ok())
                            .map(|u| format!(" ({})", u)).unwrap_or_default();
                        lines.push(DetailLine { text: format!("  Owner     {}{}", o.as_str(), uid_str), is_header: false });
                    }
                }
                if let Ok(group_attr) = ds.attr("group") {
                    if let Ok(g) = group_attr.read_scalar::<hdf5::types::VarLenUnicode>() {
                        let gid_str = ds.attr("gid").ok().and_then(|a| a.read_scalar::<u32>().ok())
                            .map(|g| format!(" ({})", g)).unwrap_or_default();
                        lines.push(DetailLine { text: format!("  Group     {}{}", g.as_str(), gid_str), is_header: false });
                    }
                }
                if let Ok(mtime_attr) = ds.attr("mtime") {
                    if let Ok(mt) = mtime_attr.read_scalar::<f64>() {
                        if mt > 0.0 {
                            let dt = chrono::DateTime::from_timestamp(mt as i64, ((mt.fract()) * 1e9) as u32)
                                .map(|d| d.format("%Y-%m-%d %H:%M:%S").to_string())
                                .unwrap_or_else(|| format!("{:.0}", mt));
                            lines.push(DetailLine { text: format!("  Modified  {}", dt), is_header: false });
                        }
                    }
                }
                lines.push(DetailLine { text: String::new(), is_header: false });

                // Attributes
                let attr_names = ds.attr_names().unwrap_or_default();
                if !attr_names.is_empty() {
                    lines.push(DetailLine { text: "ATTRIBUTES".into(), is_header: true});
                    let max_key = attr_names.iter().map(|n| n.len()).max().unwrap_or(0);
                    for name in &attr_names {
                        let val = if name == "mode" {
                            ds.attr(name).ok()
                                .and_then(|a| a.read_scalar::<u32>().ok())
                                .map(|m| format!("0o{:o}", m))
                                .unwrap_or_else(|| "?".into())
                        } else if let Ok(attr) = ds.attr(name) {
                            crate::metadata::attr_to_json_value(&attr)
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "?".into())
                        } else {
                            "?".into()
                        };
                        lines.push(DetailLine {
                            text: format!("  {:<width$}  {}", name, val, width = max_key),
                            is_header: false,
                        });
                    }
                    lines.push(DetailLine { text: String::new(), is_header: false });
                }

                // Hex preview (skip for large files >10 MB)
                if raw_size <= 10 * 1024 * 1024 {
                    if let Ok(data) = ds.read_raw::<u8>() {
                        let preview = &data[..data.len().min(96)];
                        if !preview.is_empty() {
                            lines.push(DetailLine { text: "DATA PREVIEW".into(), is_header: true});
                            for chunk_start in (0..preview.len()).step_by(6) {
                                let end = (chunk_start + 6).min(preview.len());
                                let chunk = &preview[chunk_start..end];
                                let hex: String = chunk.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
                                let ascii: String = chunk.iter().map(|&b| {
                                    if (32..127).contains(&b) { b as char } else { '.' }
                                }).collect();
                                lines.push(DetailLine {
                                    text: format!("  {:05x}  {:<18}  {}", chunk_start, hex, ascii),
                                    is_header: false,
                                });
                            }
                        }
                    }
                } else {
                    lines.push(DetailLine { text: "DATA PREVIEW".into(), is_header: true });
                    lines.push(DetailLine { text: "  (skipped, file > 10 MB)".into(), is_header: false });
                }
            }
        }
        NodeType::Group => {
            if let Ok(grp) = h5f.group(path) {
                let members = grp.member_names().unwrap_or_default();
                lines.push(DetailLine { text: "GROUP INFO".into(), is_header: true});
                lines.push(DetailLine {
                    text: format!("  Children    {}", members.len()),
                    is_header: false,
                });
                if size_bytes > 0 {
                    lines.push(DetailLine {
                        text: format!("  Total size  {}", human_size(size_bytes)),
                        is_header: false,
                    });
                }
                let attrs = grp.attr_names().unwrap_or_default();
                if !attrs.is_empty() {
                    lines.push(DetailLine { text: String::new(), is_header: false });
                    lines.push(DetailLine { text: "ATTRIBUTES".into(), is_header: true});
                    for name in &attrs {
                        lines.push(DetailLine {
                            text: format!("  {}", name),
                            is_header: false,
                        });
                    }
                }
            }
        }
        NodeType::EmptyDir => {
            lines.push(DetailLine { text: "EMPTY DIRECTORY".into(), is_header: true});
        }
    }

    lines
}

fn collect_detail_bagit(h5_path: &str, path: &str, node_type: &NodeType, size_bytes: u64, child_count: usize) -> Vec<DetailLine> {
    let mut lines = Vec::new();

    let h5f = match hdf5::File::open(h5_path) {
        Ok(f) => f,
        Err(_) => return lines,
    };

    match node_type {
        NodeType::Dataset => {
            // BagIt tag files (virtual [bagit]/ group)
            if let Some(tag_name) = path.strip_prefix("[bagit]/") {
                if let Ok(grp) = h5f.group("bagit") {
                    if let Ok(ds) = grp.dataset(tag_name) {
                        if let Ok(data) = ds.read_raw::<u8>() {
                            let text = String::from_utf8_lossy(&data);
                            lines.push(DetailLine { text: "CONTENT".into(), is_header: true });
                            for line in text.lines() {
                                lines.push(DetailLine { text: format!(" {}", line), is_header: false });
                            }
                        }
                    }
                }
                return lines;
            }

            let bi = read_bagit_index(&h5f);
            let lookup = format!("data/{}", path);

            let pos = bi.paths.iter().position(|p| *p == lookup);
            if let Some(i) = pos {
                let bid = bi.batch_ids[i];
                let off = bi.offsets[i];
                let length = bi.lengths[i];
                let mode = bi.modes[i];
                let sha = if i < bi.sha256s.len() { bi.sha256s[i].clone() } else { String::new() };
                let uid = bi.uids[i];
                let gid = bi.gids[i];
                let mtime = bi.mtimes[i];
                let owner = if i < bi.owners.len() { bi.owners[i].clone() } else { String::new() };
                let group = if i < bi.groups.len() { bi.groups[i].clone() } else { String::new() };

                let hash_algo = h5f.attr("har_checksum_algo")
                    .ok()
                    .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
                    .map(|v| v.as_str().to_string())
                    .unwrap_or_else(|| "sha256".to_string());

                lines.push(DetailLine { text: "STORAGE".into(), is_header: true});
                lines.push(DetailLine {
                    text: format!("  Size      {}", human_size(length)),
                    is_header: false,
                });
                lines.push(DetailLine {
                    text: format!("  Batch     {} (offset {})", bid, off),
                    is_header: false,
                });
                if mode != 0 {
                    lines.push(DetailLine {
                        text: format!("  Mode      0o{:o}", mode),
                        is_header: false,
                    });
                }
                if !owner.is_empty() {
                    lines.push(DetailLine {
                        text: format!("  Owner     {} ({})", owner, uid),
                        is_header: false,
                    });
                }
                if !group.is_empty() {
                    lines.push(DetailLine {
                        text: format!("  Group     {} ({})", group, gid),
                        is_header: false,
                    });
                }
                if mtime > 0.0 {
                    let dt = chrono::DateTime::from_timestamp(mtime as i64, ((mtime.fract()) * 1e9) as u32)
                        .map(|d| d.format("%Y-%m-%d %H:%M:%S").to_string())
                        .unwrap_or_else(|| format!("{:.0}", mtime));
                    lines.push(DetailLine {
                        text: format!("  Modified  {}", dt),
                        is_header: false,
                    });
                }
                lines.push(DetailLine { text: String::new(), is_header: false });

                if !sha.is_empty() {
                    lines.push(DetailLine { text: "CHECKSUM".into(), is_header: true});
                    lines.push(DetailLine {
                        text: format!("  {}  {}", hash_algo, sha),
                        is_header: false,
                    });
                    lines.push(DetailLine { text: String::new(), is_header: false });
                }

                // User metadata (xattrs)
                if let Ok(ds) = h5f.dataset("user_metadata") {
                    if let Ok(blob) = ds.read_raw::<u8>() {
                        if let Ok(s) = String::from_utf8(blob) {
                            if let Ok(meta) = serde_json::from_str::<std::collections::BTreeMap<String, serde_json::Map<String, serde_json::Value>>>(&s) {
                                if let Some(entry) = meta.get(path) {
                                    if let Some(xattrs) = entry.get("xattrs") {
                                        if let Some(xmap) = xattrs.as_object() {
                                            lines.push(DetailLine { text: "EXTENDED ATTRIBUTES".into(), is_header: true});
                                            let max_key = xmap.keys().map(|k| k.len()).max().unwrap_or(0);
                                            for (k, v) in xmap {
                                                lines.push(DetailLine {
                                                    text: format!("  {:<width$}  {}", k, v, width = max_key),
                                                    is_header: false,
                                                });
                                            }
                                            lines.push(DetailLine { text: String::new(), is_header: false });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Hex preview (skip for large files >10 MB)
                if length <= 10 * 1024 * 1024 {
                    let batch_ds_path = format!("batches/{}", bid);
                    if let Ok(batch_ds) = h5f.dataset(&batch_ds_path) {
                        let batch_data = crate::read_dataset_content(&batch_ds);
                        let start = off as usize;
                        let end = (off + length.min(96)) as usize;
                        if end <= batch_data.len() {
                            let preview = &batch_data[start..end];
                            if !preview.is_empty() {
                                lines.push(DetailLine { text: "DATA PREVIEW".into(), is_header: true});
                                for chunk_start in (0..preview.len()).step_by(6) {
                                    let chunk_end = (chunk_start + 6).min(preview.len());
                                    let chunk = &preview[chunk_start..chunk_end];
                                    let hex: String = chunk.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
                                    let ascii: String = chunk.iter().map(|&b| {
                                        if (32..127).contains(&b) { b as char } else { '.' }
                                    }).collect();
                                    lines.push(DetailLine {
                                        text: format!("  {:05x}  {:<18}  {}", chunk_start, hex, ascii),
                                        is_header: false,
                                    });
                                }
                            }
                        }
                    }
                } else {
                    lines.push(DetailLine { text: "DATA PREVIEW".into(), is_header: true });
                    lines.push(DetailLine { text: "  (skipped, file > 10 MB)".into(), is_header: false });
                }
            }
        }
        NodeType::Group => {
            lines.push(DetailLine { text: "GROUP".into(), is_header: true});
            lines.push(DetailLine {
                text: format!("  Children    {}", child_count),
                is_header: false,
            });
            if size_bytes > 0 {
                lines.push(DetailLine {
                    text: format!("  Total size  {}", human_size(size_bytes)),
                    is_header: false,
                });
            }
        }
        NodeType::EmptyDir => {
            lines.push(DetailLine { text: "EMPTY DIRECTORY".into(), is_header: true});
        }
    }

    lines
}

// ---------------------------------------------------------------------------
// App implementation
// ---------------------------------------------------------------------------

impl App {
    fn new(root: TreeNode, archive_name: String, is_bagit: bool, h5_path: String) -> Self {
        let (ds, grp) = count_items(&root);
        let mut app = App {
            root,
            visible: Vec::new(),
            cursor: 0,
            tree_scroll: 0,
            detail_scroll: 0,
            pane: 0,
            search_mode: false,
            search_query: String::new(),
            archive_name,
            is_bagit,
            h5_path,
            total_datasets: ds,
            total_groups: grp,
            cached_detail_path: String::new(),
            cached_detail: Vec::new(),
        };
        app.rebuild_visible();
        app
    }

    fn rebuild_visible(&mut self) {
        self.visible.clear();
        flatten(&self.root, &mut self.visible, true);
        if self.cursor >= self.visible.len() {
            self.cursor = self.visible.len().saturating_sub(1);
        }
    }

    fn selected_path(&self) -> &str {
        if self.cursor < self.visible.len() {
            &self.visible[self.cursor].full_path
        } else {
            ""
        }
    }

    fn get_detail(&mut self) -> Vec<DetailLine> {
        let path = self.selected_path().to_string();
        if path == self.cached_detail_path {
            return self.cached_detail.clone();
        }
        let node_type = if self.cursor < self.visible.len() {
            self.visible[self.cursor].node_type.clone()
        } else {
            return Vec::new();
        };

        let (size_bytes, child_count) = if self.cursor < self.visible.len() {
            (self.visible[self.cursor].size_bytes, self.visible[self.cursor].child_count)
        } else {
            (0, 0)
        };
        let detail = if self.is_bagit {
            collect_detail_bagit(&self.h5_path, &path, &node_type, size_bytes, child_count)
        } else {
            collect_detail_legacy(&self.h5_path, &path, &node_type, size_bytes, child_count)
        };
        self.cached_detail_path = path;
        self.cached_detail = detail.clone();
        self.detail_scroll = 0;
        detail
    }

    fn move_down(&mut self) {
        if self.pane == 0 {
            if self.cursor < self.visible.len().saturating_sub(1) {
                self.cursor += 1;
            }
        } else {
            self.detail_scroll += 1;
        }
    }

    fn move_up(&mut self) {
        if self.pane == 0 {
            self.cursor = self.cursor.saturating_sub(1);
        } else {
            self.detail_scroll = self.detail_scroll.saturating_sub(1);
        }
    }

    fn page_down(&mut self, n: usize) {
        if self.pane == 0 {
            self.cursor = (self.cursor + n).min(self.visible.len().saturating_sub(1));
        } else {
            self.detail_scroll += n;
        }
    }

    fn page_up(&mut self, n: usize) {
        if self.pane == 0 {
            self.cursor = self.cursor.saturating_sub(n);
        } else {
            self.detail_scroll = self.detail_scroll.saturating_sub(n);
        }
    }

    fn expand(&mut self) {
        if self.cursor >= self.visible.len() { return; }
        let path = self.visible[self.cursor].full_path.clone();
        let has_children = self.visible[self.cursor].has_children;
        let expanded = self.visible[self.cursor].expanded;
        if has_children {
            if !expanded {
                if let Some(node) = find_node_mut(&mut self.root, &path) {
                    node.expanded = true;
                }
                self.rebuild_visible();
            } else if self.cursor < self.visible.len().saturating_sub(1) {
                self.cursor += 1;
            }
        }
    }

    fn collapse(&mut self) {
        if self.cursor >= self.visible.len() { return; }
        let path = self.visible[self.cursor].full_path.clone();
        let node_type = self.visible[self.cursor].node_type.clone();
        let expanded = self.visible[self.cursor].expanded;

        if node_type != NodeType::Dataset && expanded {
            if let Some(node) = find_node_mut(&mut self.root, &path) {
                node.expanded = false;
            }
            self.rebuild_visible();
        } else {
            // Go to parent
            let parts: Vec<&str> = path.split('/').collect();
            if parts.len() > 1 {
                let parent = parts[..parts.len() - 1].join("/");
                if let Some(pos) = self.visible.iter().position(|r| r.full_path == parent) {
                    self.cursor = pos;
                }
            }
        }
    }

    fn toggle_expand(&mut self) {
        if self.cursor >= self.visible.len() { return; }
        let path = self.visible[self.cursor].full_path.clone();
        let has_children = self.visible[self.cursor].has_children;
        if has_children {
            if let Some(node) = find_node_mut(&mut self.root, &path) {
                node.expanded = !node.expanded;
            }
            self.rebuild_visible();
        }
    }

    fn search_next(&mut self) {
        if self.search_query.is_empty() { return; }
        let q = self.search_query.to_lowercase();
        let start = self.cursor + 1;
        for i in start..self.visible.len() {
            if self.visible[i].name.to_lowercase().contains(&q)
                || self.visible[i].full_path.to_lowercase().contains(&q)
            {
                self.cursor = i;
                return;
            }
        }
        for i in 0..start.min(self.visible.len()) {
            if self.visible[i].name.to_lowercase().contains(&q)
                || self.visible[i].full_path.to_lowercase().contains(&q)
            {
                self.cursor = i;
                return;
            }
        }
    }

    fn search_prev(&mut self) {
        if self.search_query.is_empty() { return; }
        let q = self.search_query.to_lowercase();
        if self.cursor > 0 {
            for i in (0..self.cursor).rev() {
                if self.visible[i].name.to_lowercase().contains(&q)
                    || self.visible[i].full_path.to_lowercase().contains(&q)
                {
                    self.cursor = i;
                    return;
                }
            }
        }
        for i in (self.cursor..self.visible.len()).rev() {
            if self.visible[i].name.to_lowercase().contains(&q)
                || self.visible[i].full_path.to_lowercase().contains(&q)
            {
                self.cursor = i;
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();

    let outer = Layout::vertical([
        Constraint::Length(2), // title + breadcrumb
        Constraint::Min(1),   // body
        Constraint::Length(1), // status
    ])
    .split(area);

    // Title + breadcrumb
    let title = format!(" har browse: {} ", app.archive_name);
    let bc = if app.cursor < app.visible.len() {
        app.visible[app.cursor].full_path.replace('/', " > ")
    } else {
        String::new()
    };
    let header_lines = vec![
        Line::from(Span::styled(title, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled(format!(" {}", bc), Style::default().fg(Color::DarkGray))),
    ];
    frame.render_widget(Paragraph::new(header_lines), outer[0]);

    // Body: 40/60 split
    let body = Layout::horizontal([
        Constraint::Percentage(40),
        Constraint::Percentage(60),
    ])
    .split(outer[1]);

    draw_tree(frame, app, body[0]);
    draw_detail(frame, app, body[1]);

    // Status bar
    let stats = format!(
        " {} datasets  {} groups",
        app.total_datasets, app.total_groups
    );
    let keys = if app.search_mode {
        format!(" /{}_  Enter:find  Esc:cancel", app.search_query)
    } else {
        " hjkl:nav  Tab:pane  /:search  q:quit".to_string()
    };
    let status_text = format!("{}{:>width$}", stats, keys, width = area.width as usize - stats.len());
    frame.render_widget(
        Paragraph::new(status_text)
            .style(Style::default().bg(Color::DarkGray).fg(Color::White)),
        outer[2],
    );
}

fn draw_tree(frame: &mut Frame, app: &mut App, area: Rect) {
    let border_style = if app.pane == 0 {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" Archive ");

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let height = inner.height as usize;
    if height == 0 { return; }

    // Scroll
    if app.cursor < app.tree_scroll {
        app.tree_scroll = app.cursor;
    }
    if app.cursor >= app.tree_scroll + height {
        app.tree_scroll = app.cursor - height + 1;
    }

    for row in 0..height {
        let idx = app.tree_scroll + row;
        if idx >= app.visible.len() { break; }
        let v = &app.visible[idx];
        let y = inner.y + row as u16;

        let indent = "  ".repeat(v.depth as usize);
        let icon = match v.node_type {
            NodeType::Group if v.has_children && v.expanded => "[-] ",
            NodeType::Group if v.has_children => "[+] ",
            NodeType::Group => "    ",
            NodeType::EmptyDir => "    ",
            NodeType::Dataset => "    ",
        };

        let suffix = match v.node_type {
            NodeType::Group if v.has_children && !v.expanded => {
                format!("/  [{}]", v.child_count)
            }
            NodeType::Group => "/".to_string(),
            NodeType::EmptyDir => "/  [empty]".to_string(),
            NodeType::Dataset => String::new(),
        };

        let label = format!("{}{}{}{}", indent, icon, v.name, suffix);

        // Size for datasets and groups
        let size_str = if v.size_bytes > 0 {
            human_size(v.size_bytes)
        } else {
            String::new()
        };

        let avail = inner.width as usize;
        let line = if !size_str.is_empty() && label.len() + size_str.len() + 1 < avail {
            let pad = avail - label.len() - size_str.len();
            format!("{}{:>pad$}{}", label, "", size_str, pad = pad)
        } else {
            label
        };

        let is_selected = idx == app.cursor;
        let style = if is_selected {
            Style::default().bg(Color::White).fg(Color::Black).add_modifier(Modifier::BOLD)
        } else {
            match v.node_type {
                NodeType::Group | NodeType::EmptyDir => Style::default().fg(Color::Blue),
                NodeType::Dataset => {
                    if !app.search_query.is_empty()
                        && (v.name.to_lowercase().contains(&app.search_query.to_lowercase())
                            || v.full_path.to_lowercase().contains(&app.search_query.to_lowercase()))
                    {
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::UNDERLINED)
                    } else {
                        Style::default().fg(Color::White)
                    }
                }
            }
        };

        let truncated: String = line.chars().take(avail).collect();
        let padded = format!("{:<width$}", truncated, width = avail);
        frame.render_widget(
            Paragraph::new(padded).style(style),
            Rect::new(inner.x, y, inner.width, 1),
        );
    }
}

fn draw_detail(frame: &mut Frame, app: &mut App, area: Rect) {
    let border_style = if app.pane == 1 {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let selected_path = if app.cursor < app.visible.len() {
        app.visible[app.cursor].full_path.clone()
    } else {
        String::new()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(format!(" {} ", selected_path));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let detail = app.get_detail();
    let height = inner.height as usize;
    if height == 0 { return; }

    for row in 0..height {
        let idx = app.detail_scroll + row;
        if idx >= detail.len() { break; }
        let dl = &detail[idx];
        let y = inner.y + row as u16;

        let style = if dl.is_header {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        let text: String = dl.text.chars().take(inner.width as usize).collect();
        frame.render_widget(
            Paragraph::new(text).style(style),
            Rect::new(inner.x, y, inner.width, 1),
        );
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn browse_archive(h5_path: &str) {
    let h5_path_expanded = shellexpand::tilde(h5_path).to_string();
    if !std::path::Path::new(&h5_path_expanded).exists() {
        eprintln!("Error: archive '{}' not found.", h5_path);
        std::process::exit(1);
    }
    let h5f = match hdf5::File::open(&h5_path_expanded) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: cannot open '{}': {}", h5_path, e);
            std::process::exit(1);
        }
    };

    let is_bagit = crate::bagit::is_bagit_archive(&h5_path_expanded);
    let mut tree = if is_bagit {
        build_tree_bagit(&h5f)
    } else {
        build_tree_legacy(&h5f)
    };

    // Expand first level
    for child in &mut tree.children {
        child.expanded = true;
    }

    drop(h5f); // Close; we'll reopen per-detail-query to avoid borrow issues

    let archive_name = std::path::Path::new(&h5_path_expanded)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let mut app = App::new(tree, archive_name, is_bagit, h5_path_expanded);

    // Terminal setup
    enable_raw_mode().expect("Failed to enable raw mode");
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).expect("Failed to enter alternate screen");
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).expect("Failed to create terminal");

    // Event loop
    loop {
        terminal.draw(|f| draw(f, &mut app)).ok();

        if event::poll(std::time::Duration::from_millis(100)).unwrap_or(false) {
            if let Ok(ev) = event::read() {
                if app.search_mode {
                    match ev {
                        Event::Key(KeyEvent { code: KeyCode::Esc, .. }) => {
                            app.search_mode = false;
                            app.search_query.clear();
                        }
                        Event::Key(KeyEvent { code: KeyCode::Enter, .. }) => {
                            app.search_mode = false;
                            app.search_next();
                        }
                        Event::Key(KeyEvent { code: KeyCode::Backspace, .. }) => {
                            app.search_query.pop();
                        }
                        Event::Key(KeyEvent { code: KeyCode::Char(c), .. }) => {
                            app.search_query.push(c);
                        }
                        _ => {}
                    }
                    continue;
                }

                match ev {
                    Event::Key(KeyEvent { code: KeyCode::Char('q'), .. }) => break,
                    Event::Key(KeyEvent { code: KeyCode::Char('c'), modifiers: KeyModifiers::CONTROL, .. }) => break,
                    Event::Key(KeyEvent { code: KeyCode::Char('j'), .. })
                    | Event::Key(KeyEvent { code: KeyCode::Down, .. }) => app.move_down(),
                    Event::Key(KeyEvent { code: KeyCode::Char('k'), .. })
                    | Event::Key(KeyEvent { code: KeyCode::Up, .. }) => app.move_up(),
                    Event::Key(KeyEvent { code: KeyCode::Char('l'), .. })
                    | Event::Key(KeyEvent { code: KeyCode::Right, .. }) => app.expand(),
                    Event::Key(KeyEvent { code: KeyCode::Char('h'), .. })
                    | Event::Key(KeyEvent { code: KeyCode::Left, .. }) => app.collapse(),
                    Event::Key(KeyEvent { code: KeyCode::Enter | KeyCode::Char(' '), .. }) => {
                        app.toggle_expand();
                    }
                    Event::Key(KeyEvent { code: KeyCode::Tab, .. }) => {
                        app.pane = 1 - app.pane;
                    }
                    Event::Key(KeyEvent { code: KeyCode::Char('/'), .. }) => {
                        app.search_mode = true;
                        app.search_query.clear();
                    }
                    Event::Key(KeyEvent { code: KeyCode::Char('n'), .. }) => app.search_next(),
                    Event::Key(KeyEvent { code: KeyCode::Char('N'), .. }) => app.search_prev(),
                    Event::Key(KeyEvent { code: KeyCode::Char('g'), .. }) => {
                        if app.pane == 0 { app.cursor = 0; } else { app.detail_scroll = 0; }
                    }
                    Event::Key(KeyEvent { code: KeyCode::Char('G'), .. }) => {
                        if app.pane == 0 {
                            app.cursor = app.visible.len().saturating_sub(1);
                        } else {
                            app.detail_scroll = 999;
                        }
                    }
                    Event::Key(KeyEvent { code: KeyCode::PageDown, .. }) => {
                        let h = terminal.size().map(|s| s.height as usize).unwrap_or(20);
                        app.page_down(h.saturating_sub(5));
                    }
                    Event::Key(KeyEvent { code: KeyCode::PageUp, .. }) => {
                        let h = terminal.size().map(|s| s.height as usize).unwrap_or(20);
                        app.page_up(h.saturating_sub(5));
                    }
                    _ => {}
                }
            }
        }
    }

    // Cleanup
    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
}
