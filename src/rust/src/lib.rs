use ndarray::ArrayView1;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, VecDeque};
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use walkdir::WalkDir;

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------

fn human_size_bytes(nbytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut size = nbytes as f64;
    for unit in &units {
        if size < 1024.0 {
            return format!("{:.1} {}", size, unit);
        }
        size /= 1024.0;
    }
    format!("{:.1} PB", size)
}

fn term_width() -> usize {
    crossterm::terminal::size()
        .map(|(w, _)| (w as usize).max(80))
        .unwrap_or(80)
}

fn truncate_name(name: &str, max_len: usize) -> String {
    if max_len < 4 {
        return name.chars().take(max_len).collect();
    }
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("...{}", &name[name.len() - (max_len - 3)..])
    }
}

/// Shared counters updated by the discovery thread, read by the render loop.
pub struct DiscoveryCounters {
    pub files_found: AtomicU64,
    pub bytes_found: AtomicU64,
    pub done: AtomicBool,
}

/// Multi-phase progress display with discovery/archiving/validating bars.
pub struct PipelineProgress {
    pub(crate) is_tty: bool,
    // Discovery
    pub(crate) discovery: Arc<DiscoveryCounters>,
    pub(crate) discovery_finished: bool,
    // Archiving
    pub(crate) archive_total: u64,
    archive_current: u64,
    archive_label: String,
    // Current file (chunked read progress)
    current_file_name: Option<String>,
    current_file_size: u64,
    current_file_read: u64,
    current_file_writing: bool,
    // File status window (name, size)
    recent_completed: VecDeque<(String, u64)>,
    queued_files: VecDeque<(String, u64)>,
    // Validation
    validate_enabled: bool,
    validate_total: u64,
    validate_current: u64,
    validate_recent: VecDeque<String>,
    validate_current_name: Option<String>,
    // Rendering
    prev_lines: usize,
    last_render: Instant,
    start_time: Instant,
}

const SPINNER: [char; 4] = ['|', '/', '-', '\\'];

impl PipelineProgress {
    pub fn new(verbose: bool) -> Self {
        let now = Instant::now();
        PipelineProgress {
            is_tty: io::stderr().is_terminal() && verbose,
            discovery: Arc::new(DiscoveryCounters {
                files_found: AtomicU64::new(0),
                bytes_found: AtomicU64::new(0),
                done: AtomicBool::new(false),
            }),
            discovery_finished: false,
            archive_total: 0,
            archive_current: 0,
            archive_label: "Archiving".to_string(),
            current_file_name: None,
            current_file_size: 0,
            current_file_read: 0,
            current_file_writing: false,
            recent_completed: VecDeque::new(),
            queued_files: VecDeque::new(),
            validate_enabled: false,
            validate_total: 0,
            validate_current: 0,
            validate_recent: VecDeque::new(),
            validate_current_name: None,
            prev_lines: 0,
            last_render: now,
            start_time: now,
        }
    }

    pub fn discovery_counters(&self) -> Arc<DiscoveryCounters> {
        Arc::clone(&self.discovery)
    }

    pub fn begin_file(&mut self, rel_path: &str, file_size: u64) {
        if let Some(prev) = self.current_file_name.take() {
            self.recent_completed.push_back((prev, self.current_file_size));
            if self.recent_completed.len() > 2 {
                self.recent_completed.pop_front();
            }
        }
        self.current_file_name = Some(rel_path.to_string());
        self.current_file_size = file_size;
        self.current_file_read = 0;
        self.current_file_writing = false;
        if self.is_tty {
            self.render();
        }
    }

    pub fn update_file_progress(&mut self, bytes_read: u64) {
        self.current_file_read = bytes_read;
        self.maybe_render();
    }

    pub fn mark_file_writing(&mut self) {
        self.current_file_writing = true;
        if self.is_tty {
            self.render();
        }
    }

    pub fn finish_file(&mut self, nbytes: u64) {
        self.archive_current += nbytes;
        // current_file_name stays until next begin_file moves it to completed
        if self.is_tty {
            self.render();
        }
    }

    pub fn set_queued(&mut self, files: &[(String, u64)]) {
        self.queued_files.clear();
        for f in files.iter().take(3) {
            self.queued_files.push_back(f.clone());
        }
    }

    /// Move the current in-progress file to the completed list and clear queued.
    pub fn flush_current_file(&mut self) {
        if let Some(name) = self.current_file_name.take() {
            self.recent_completed.push_back((name, self.current_file_size));
            if self.recent_completed.len() > 2 {
                self.recent_completed.pop_front();
            }
        }
        self.queued_files.clear();
        self.current_file_size = 0;
        self.current_file_read = 0;
        self.current_file_writing = false;
        if self.is_tty {
            self.render();
        }
    }

    pub fn finish_discovery(&mut self) {
        self.discovery_finished = true;
        if self.is_tty {
            self.render();
        }
    }

    pub fn enable_validation(&mut self, total: u64) {
        self.validate_enabled = true;
        self.validate_total = total;
    }

    pub fn begin_validation(&mut self, name: &str) {
        if let Some(prev) = self.validate_current_name.take() {
            self.validate_recent.push_back(prev);
            if self.validate_recent.len() > 2 {
                self.validate_recent.pop_front();
            }
        }
        self.validate_current_name = Some(name.to_string());
        self.maybe_render();
    }

    pub fn finish_validation_file(&mut self, nbytes: u64) {
        self.validate_current += nbytes;
        self.maybe_render();
    }

    fn maybe_render(&mut self) {
        if !self.is_tty {
            return;
        }
        let now = Instant::now();
        if now.duration_since(self.last_render) >= Duration::from_millis(50) {
            self.render();
            self.last_render = now;
        }
    }

    fn render(&mut self) {
        let mut err = io::stderr().lock();
        if self.prev_lines > 0 {
            write!(err, "\x1b[{}A", self.prev_lines).ok();
        }

        let tw = term_width();
        let bar_width: usize = if tw >= 100 { 40 } else if tw >= 60 { tw - 60 + 20 } else { 20 };
        let name_max = tw.saturating_sub(10);
        let mut lines: usize = 0;

        // Auto-detect discovery completion from the atomic flag
        if !self.discovery_finished && self.discovery.done.load(Ordering::Acquire) {
            self.discovery_finished = true;
        }

        // --- Discovery bar ---
        if self.discovery_finished {
            let total = self.discovery.bytes_found.load(Ordering::Relaxed);
            let count = self.discovery.files_found.load(Ordering::Relaxed);
            let bar = "\u{2588}".repeat(bar_width);
            writeln!(err, "\x1b[2K\x1b[32m\u{2713}\x1b[0m {:12} [\x1b[32m{}\x1b[0m] 100.0% ({} files / {})",
                     "Discovering", bar, count, human_size_bytes(total)).ok();
        } else {
            let count = self.discovery.files_found.load(Ordering::Relaxed);
            let bytes = self.discovery.bytes_found.load(Ordering::Relaxed);
            let frame = (self.start_time.elapsed().as_millis() / 100) as usize % 4;
            writeln!(err, "\x1b[2K{} {:12} ... ({} files / {} found)",
                     SPINNER[frame], "Discovering", count, human_size_bytes(bytes)).ok();
        }
        lines += 1;

        // --- Archiving bar ---
        let total = self.archive_total;
        if total > 0 {
            let effective = (self.archive_current + self.current_file_read).min(total);
            let pct = (effective as f64 * 100.0 / total as f64).min(100.0);
            let filled = ((bar_width as u64).saturating_mul(effective) / total) as usize;
            let filled = filled.min(bar_width);
            let empty = bar_width - filled;
            writeln!(err, "\x1b[2K  {:12} [{}{}] {:.1}% ({} / {})",
                     self.archive_label,
                     "\u{2588}".repeat(filled),
                     "\u{2591}".repeat(empty),
                     pct,
                     human_size_bytes(effective),
                     human_size_bytes(total)).ok();
        } else {
            writeln!(err, "\x1b[2K  {:12} [{}] waiting...",
                     self.archive_label, "\u{2591}".repeat(bar_width)).ok();
        }
        lines += 1;

        // --- File status window ---
        // Completed files (green) with size
        for (name, size) in &self.recent_completed {
            let size_str = human_size_bytes(*size);
            let suffix = format!("  ({})", size_str);
            writeln!(err, "\x1b[2K    \x1b[32m\u{2713} {}{}\x1b[0m",
                     truncate_name(name, name_max.saturating_sub(suffix.len())), suffix).ok();
            lines += 1;
        }
        // Current file (magenta with progress + size)
        if let Some(ref name) = self.current_file_name {
            let size_str = human_size_bytes(self.current_file_size);
            if self.current_file_writing {
                let suffix = format!("  writing... ({})", size_str);
                writeln!(err, "\x1b[2K    \x1b[35m\u{25b8} {}{}\x1b[0m",
                         truncate_name(name, name_max.saturating_sub(suffix.len())), suffix).ok();
            } else if self.current_file_size > 0 && self.current_file_read < self.current_file_size {
                let pct = self.current_file_read as f64 * 100.0 / self.current_file_size as f64;
                let suffix = format!("  {:.1}% ({})", pct, size_str);
                writeln!(err, "\x1b[2K    \x1b[35m\u{25b8} {}{}\x1b[0m",
                         truncate_name(name, name_max.saturating_sub(suffix.len())), suffix).ok();
            } else {
                let suffix = format!("  ({})", size_str);
                writeln!(err, "\x1b[2K    \x1b[35m\u{25b8} {}{}\x1b[0m",
                         truncate_name(name, name_max.saturating_sub(suffix.len())), suffix).ok();
            }
            lines += 1;
        }
        // Queued files (dim gray) with size
        for (name, size) in &self.queued_files {
            let size_str = human_size_bytes(*size);
            let suffix = format!("  ({})", size_str);
            writeln!(err, "\x1b[2K    \x1b[90m\u{25cb} {}{}\x1b[0m",
                     truncate_name(name, name_max.saturating_sub(suffix.len())), suffix).ok();
            lines += 1;
        }

        // --- Validation bar (only if enabled) ---
        if self.validate_enabled && self.validate_total > 0 {
            let pct = (self.validate_current as f64 * 100.0 / self.validate_total as f64).min(100.0);
            let filled = ((bar_width as u64).saturating_mul(self.validate_current) / self.validate_total) as usize;
            let filled = filled.min(bar_width);
            let empty = bar_width - filled;
            writeln!(err, "\x1b[2K  {:12} [{}{}] {:.1}% ({} / {})",
                     "Validating",
                     "\u{2588}".repeat(filled),
                     "\u{2591}".repeat(empty),
                     pct,
                     human_size_bytes(self.validate_current),
                     human_size_bytes(self.validate_total)).ok();
            lines += 1;

            // Validation file status
            for name in &self.validate_recent {
                writeln!(err, "\x1b[2K    \x1b[32m\u{2713} {}\x1b[0m",
                         truncate_name(name, name_max)).ok();
                lines += 1;
            }
            if let Some(ref name) = self.validate_current_name {
                writeln!(err, "\x1b[2K    \x1b[35m\u{25b8} {}\x1b[0m",
                         truncate_name(name, name_max)).ok();
                lines += 1;
            }
        }

        // Clear leftover lines from previous render
        let prev = self.prev_lines;
        for _ in 0..prev.saturating_sub(lines) {
            writeln!(err, "\x1b[2K").ok();
        }
        let extra = prev.saturating_sub(lines);
        if extra > 0 {
            write!(err, "\x1b[{}A", extra).ok();
        }

        self.prev_lines = lines;
        err.flush().ok();
    }

    pub fn finish(&mut self) {
        // Move current file to completed
        if let Some(name) = self.current_file_name.take() {
            self.recent_completed.push_back((name, self.current_file_size));
        }
        if let Some(name) = self.validate_current_name.take() {
            self.validate_recent.push_back(name);
        }
        if !self.is_tty || self.prev_lines == 0 {
            return;
        }
        let mut err = io::stderr().lock();
        write!(err, "\x1b[{}A", self.prev_lines).ok();

        let tw = term_width();
        let bar_width: usize = if tw >= 100 { 40 } else if tw >= 60 { tw - 60 + 20 } else { 20 };
        let bar = "\u{2588}".repeat(bar_width);
        let mut lines: usize = 0;

        // Discovery complete
        let dtotal = self.discovery.bytes_found.load(Ordering::Relaxed);
        let dcount = self.discovery.files_found.load(Ordering::Relaxed);
        writeln!(err, "\x1b[2K\x1b[32m\u{2713}\x1b[0m {:12} [\x1b[32m{}\x1b[0m] 100.0% ({} files / {})",
                 "Discovering", bar, dcount, human_size_bytes(dtotal)).ok();
        lines += 1;

        // Archiving complete
        let atotal_h = human_size_bytes(self.archive_total);
        writeln!(err, "\x1b[2K\x1b[32m\u{2713}\x1b[0m {:12} [\x1b[32m{}\x1b[0m] 100.0% ({} / {})",
                 self.archive_label, bar, atotal_h, atotal_h).ok();
        lines += 1;

        // Validation complete
        if self.validate_enabled && self.validate_total > 0 {
            let vtotal_h = human_size_bytes(self.validate_total);
            writeln!(err, "\x1b[2K\x1b[32m\u{2713}\x1b[0m {:12} [\x1b[32m{}\x1b[0m] 100.0% ({} / {})",
                     "Validating", bar, vtotal_h, vtotal_h).ok();
            lines += 1;
        }

        // Clear leftover lines
        for _ in 0..self.prev_lines.saturating_sub(lines) {
            writeln!(err, "\x1b[2K").ok();
        }
        let extra = self.prev_lines.saturating_sub(lines);
        if extra > 0 {
            write!(err, "\x1b[{}A", extra).ok();
        }
        err.flush().ok();
        self.prev_lines = 0;
    }
}

/// Backward-compatible wrapper around PipelineProgress for use by bagit.rs,
/// extraction, and the parallel pack path.
pub struct Progress {
    pub is_tty: bool,
    inner: PipelineProgress,
    completed_phases: Vec<(String, u64)>,
    label: Option<String>,
}

impl Progress {
    pub fn new(verbose: bool) -> Self {
        let mut inner = PipelineProgress::new(verbose);
        // Mark discovery as already finished (simple mode)
        inner.discovery_finished = true;
        inner.discovery.done.store(true, Ordering::Relaxed);
        let is_tty = inner.is_tty;
        Progress {
            is_tty,
            inner,
            completed_phases: Vec::new(),
            label: None,
        }
    }

    pub fn start_phase(&mut self, label: &str, total_bytes: u64) {
        if let Some(prev_label) = self.label.take() {
            self.completed_phases.push((prev_label, self.inner.archive_total));
        }
        self.label = Some(label.to_string());
        self.inner.archive_label = label.to_string();
        self.inner.archive_total = total_bytes;
        self.inner.archive_current = 0;
        self.inner.recent_completed.clear();
        self.inner.current_file_name = None;
        self.inner.queued_files.clear();
        self.inner.discovery.bytes_found.store(total_bytes, Ordering::Relaxed);
        if self.is_tty {
            self.inner.render();
        }
    }

    pub fn inc(&mut self, filename: &str, nbytes: u64) {
        self.inner.archive_current += nbytes;
        // Move current to completed, set this as current briefly then move it
        if let Some(prev) = self.inner.current_file_name.take() {
            self.inner.recent_completed.push_back((prev, self.inner.current_file_size));
            if self.inner.recent_completed.len() > 5 {
                self.inner.recent_completed.pop_front();
            }
        }
        self.inner.current_file_name = Some(filename.to_string());
        self.inner.current_file_size = nbytes;
        if self.is_tty {
            self.inner.render();
        }
    }

    pub fn finish(&mut self) {
        if let Some(prev_label) = self.label.take() {
            self.completed_phases.push((prev_label, self.inner.archive_total));
        }
        self.inner.finish();
    }
}

// ---------------------------------------------------------------------------
// SplitProgress — thread-safe multi-split progress display
// ---------------------------------------------------------------------------

use std::sync::Mutex;

struct SplitState {
    label: String,
    filename: String,
    total_bytes: u64,
    archived_bytes: u64,
    recent_completed: VecDeque<(String, u64)>,
    current_file: Option<(String, u64, u64)>, // (name, size, bytes_read)
    queued_files: VecDeque<(String, u64)>,
    finished: bool,
}

struct FileAggState {
    total_size: u64,
    chunk_count: u32,
    chunks_done: u32,
    bytes_hashed: u64,
    active: bool,
}

struct SplitProgressInner {
    is_tty: bool,
    discovery_total_files: u64,
    discovery_total_bytes: u64,
    split_count: usize,
    total_bytes: u64,
    splits: Vec<SplitState>,
    prev_lines: usize,
    last_render: Instant,
    validation_phase: bool,
    // Global validation tracking (file-count + bytes)
    validate_total_files: u64,
    validate_done_files: u64,
    validate_total_bytes: u64,
    validate_done_bytes: u64,
    validate_recent: VecDeque<(String, u64)>,  // (name, size)
    // Multiple concurrent active files: slot -> (name, size, bytes_hashed) — fallback mode
    validate_active: std::collections::BTreeMap<usize, (String, u64, u64)>,
    validate_slot_counter: usize,
    validate_loading: Option<(String, u64, u64)>,  // (split basename, total_bytes, bytes_read)
    // File-level aggregation (for chunked files across splits)
    validate_file_map: std::collections::HashMap<String, FileAggState>,
    validate_slot_to_file: std::collections::HashMap<usize, String>,
    validate_slot_progress: std::collections::HashMap<usize, u64>,
}

pub struct SplitProgress {
    inner: Mutex<SplitProgressInner>,
}

impl SplitProgress {
    pub fn new(verbose: bool, split_count: usize, split_sizes: &[u64], split_filenames: &[String]) -> Self {
        let is_tty = io::stderr().is_terminal() && verbose;
        let total_bytes: u64 = split_sizes.iter().sum();
        let now = Instant::now();
        let splits = (0..split_count)
            .map(|i| SplitState {
                label: format!("Split {}", i),
                filename: split_filenames[i].clone(),
                total_bytes: split_sizes[i],
                archived_bytes: 0,
                recent_completed: VecDeque::new(),
                current_file: None,
                queued_files: VecDeque::new(),
                finished: false,
            })
            .collect();
        SplitProgress {
            inner: Mutex::new(SplitProgressInner {
                is_tty,
                discovery_total_files: 0,
                discovery_total_bytes: total_bytes,
                split_count,
                total_bytes,
                splits,
                prev_lines: 0,
                last_render: now,
                validation_phase: false,
                validate_total_files: 0,
                validate_done_files: 0,
                validate_total_bytes: 0,
                validate_done_bytes: 0,
                validate_recent: VecDeque::new(),
                validate_active: std::collections::BTreeMap::new(),
                validate_slot_counter: 0,
                validate_loading: None,
                validate_file_map: std::collections::HashMap::new(),
                validate_slot_to_file: std::collections::HashMap::new(),
                validate_slot_progress: std::collections::HashMap::new(),
            }),
        }
    }

    pub fn finish_discovery(&self, total_files: u64, total_bytes: u64) {
        let mut inner = self.inner.lock().unwrap();
        inner.discovery_total_files = total_files;
        inner.discovery_total_bytes = total_bytes;
        if inner.is_tty {
            Self::render(&mut inner);
        }
    }

    pub fn begin_file(&self, split_index: usize, rel_path: &str, file_size: u64) {
        let mut inner = self.inner.lock().unwrap();
        let split = &mut inner.splits[split_index];
        if let Some((prev_name, prev_size, _)) = split.current_file.take() {
            split.recent_completed.push_back((prev_name, prev_size));
            if split.recent_completed.len() > 2 {
                split.recent_completed.pop_front();
            }
        }
        split.current_file = Some((rel_path.to_string(), file_size, 0));
        Self::maybe_render(&mut inner);
    }

    pub fn update_file_progress(&self, split_index: usize, bytes_read: u64) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(ref mut cur) = inner.splits[split_index].current_file {
            cur.2 = bytes_read;
        }
        Self::maybe_render(&mut inner);
    }

    pub fn finish_file(&self, split_index: usize, nbytes: u64) {
        let mut inner = self.inner.lock().unwrap();
        let split = &mut inner.splits[split_index];
        split.archived_bytes += nbytes;
        // Move current file to recent_completed to avoid double-counting
        // (archived_bytes already includes this file; keeping current_file
        // would add its bytes_read again in the global progress sum)
        if let Some((name, size, _)) = split.current_file.take() {
            split.recent_completed.push_back((name, size));
            if split.recent_completed.len() > 2 {
                split.recent_completed.pop_front();
            }
        }
        if inner.is_tty {
            Self::render(&mut inner);
        }
    }

    pub fn set_queued(&self, split_index: usize, files: &[(String, u64)]) {
        let mut inner = self.inner.lock().unwrap();
        let split = &mut inner.splits[split_index];
        split.queued_files.clear();
        for f in files.iter().take(3) {
            split.queued_files.push_back(f.clone());
        }
    }

    pub fn begin_validation_phase(
        &self, total_files: u64, total_bytes: u64,
        file_map: std::collections::HashMap<String, (u64, u32)>,
    ) {
        let mut inner = self.inner.lock().unwrap();
        inner.validation_phase = true;
        inner.validate_total_files = total_files;
        inner.validate_done_files = 0;
        inner.validate_total_bytes = total_bytes;
        inner.validate_done_bytes = 0;
        inner.validate_recent.clear();
        inner.validate_active.clear();
        inner.validate_slot_counter = 0;
        inner.validate_loading = None;
        inner.validate_file_map = file_map.into_iter().map(|(name, (total_size, chunk_count))| {
            (name, FileAggState { total_size, chunk_count, chunks_done: 0, bytes_hashed: 0, active: false })
        }).collect();
        inner.validate_slot_to_file.clear();
        inner.validate_slot_progress.clear();
        if inner.is_tty {
            Self::render(&mut inner);
        }
    }

    /// Start validating a chunk. Returns a slot ID for progress updates.
    pub fn validate_begin_file(&self, name: &str, size: u64) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let slot = inner.validate_slot_counter;
        inner.validate_slot_counter += 1;
        if inner.validate_file_map.contains_key(name) {
            inner.validate_file_map.get_mut(name).unwrap().active = true;
            inner.validate_slot_to_file.insert(slot, name.to_string());
            inner.validate_slot_progress.insert(slot, 0);
        } else {
            inner.validate_active.insert(slot, (name.to_string(), size, 0));
        }
        Self::maybe_render(&mut inner);
        slot
    }

    pub fn validate_update_progress(&self, slot: usize, bytes_hashed: u64) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(file_name) = inner.validate_slot_to_file.get(&slot).cloned() {
            let prev = inner.validate_slot_progress.get(&slot).copied().unwrap_or(0);
            let delta = bytes_hashed.saturating_sub(prev);
            inner.validate_slot_progress.insert(slot, bytes_hashed);
            if let Some(state) = inner.validate_file_map.get_mut(&file_name) {
                state.bytes_hashed += delta;
            }
        } else if let Some(entry) = inner.validate_active.get_mut(&slot) {
            entry.2 = bytes_hashed;
        }
        Self::maybe_render(&mut inner);
    }

    pub fn validate_set_loading(&self, filename: &str, total_bytes: u64) {
        let basename = std::path::Path::new(filename)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| filename.to_string());
        let mut inner = self.inner.lock().unwrap();
        inner.validate_loading = Some((basename, total_bytes, 0));
        Self::maybe_render(&mut inner);
    }

    pub fn validate_update_loading(&self, bytes_read: u64) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(ref mut loading) = inner.validate_loading {
            loading.2 = bytes_read;
        }
        Self::maybe_render(&mut inner);
    }

    pub fn validate_clear_loading(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.validate_loading = None;
        Self::maybe_render(&mut inner);
    }

    pub fn validate_finish_file(&self, slot: usize, nbytes: u64) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(file_name) = inner.validate_slot_to_file.remove(&slot) {
            let prev = inner.validate_slot_progress.remove(&slot).unwrap_or(0);
            // Extract values before borrowing other fields
            let mut file_done = false;
            let mut file_total_size = 0u64;
            if let Some(state) = inner.validate_file_map.get_mut(&file_name) {
                state.bytes_hashed = state.bytes_hashed.saturating_sub(prev) + nbytes;
                state.chunks_done += 1;
                if state.chunks_done >= state.chunk_count {
                    state.active = false;
                    file_done = true;
                    file_total_size = state.total_size;
                }
            }
            if file_done {
                inner.validate_done_files += 1;
                inner.validate_done_bytes += file_total_size;
                inner.validate_recent.push_back((file_name, file_total_size));
                while inner.validate_recent.len() > 7 {
                    inner.validate_recent.pop_front();
                }
            }
        } else if let Some((name, size, _)) = inner.validate_active.remove(&slot) {
            inner.validate_recent.push_back((name, size));
            while inner.validate_recent.len() > 7 {
                inner.validate_recent.pop_front();
            }
            inner.validate_done_files += 1;
            inner.validate_done_bytes += nbytes;
        }
        if inner.is_tty {
            Self::render(&mut inner);
        }
    }

    pub fn finish_split(&self, split_index: usize) {
        let mut inner = self.inner.lock().unwrap();
        let split = &mut inner.splits[split_index];
        if let Some((name, size, _)) = split.current_file.take() {
            split.recent_completed.push_back((name, size));
        }
        split.recent_completed.clear();
        split.queued_files.clear();
        split.finished = true;
        if inner.is_tty {
            Self::render(&mut inner);
        }
    }

    pub fn finish(&self) {
        let mut inner = self.inner.lock().unwrap();
        for split in &mut inner.splits {
            split.finished = true;
            split.current_file = None;
            split.recent_completed.clear();
            split.queued_files.clear();
        }
        if !inner.is_tty || inner.prev_lines == 0 {
            return;
        }
        Self::render_final(&mut inner);
    }

    fn maybe_render(inner: &mut SplitProgressInner) {
        if !inner.is_tty {
            return;
        }
        let now = Instant::now();
        if now.duration_since(inner.last_render) >= Duration::from_millis(50) {
            Self::render(inner);
            inner.last_render = now;
        }
    }

    fn render(inner: &mut SplitProgressInner) {
        let mut err = io::stderr().lock();
        if inner.prev_lines > 0 {
            write!(err, "\x1b[{}A", inner.prev_lines).ok();
        }

        let tw = term_width();
        let bar_width: usize = if tw >= 100 { 40 } else if tw >= 60 { tw - 60 + 20 } else { 20 };
        let name_max = tw.saturating_sub(10);
        let mut lines: usize = 0;
        let bar_full = "\u{2588}".repeat(bar_width);

        // --- Discovery bar (always complete for split mode) ---
        writeln!(err, "\x1b[2K\x1b[32m\u{2713}\x1b[0m {:12} [\x1b[32m{}\x1b[0m] 100.0% ({} files / {})",
                 "Discovering", bar_full, inner.discovery_total_files,
                 human_size_bytes(inner.discovery_total_bytes)).ok();
        lines += 1;

        const MAX_DETAIL_LINES: usize = 8;

        if inner.validation_phase {
            // === VALIDATION PHASE ===
            // Archiving collapsed
            writeln!(err, "\x1b[2K\x1b[32m\u{2713}\x1b[0m {:12} [\x1b[32m{}\x1b[0m] 100.0% ({}) — {} splits",
                     "Archiving", bar_full, human_size_bytes(inner.total_bytes),
                     inner.split_count).ok();
            lines += 1;

            // Global bytes-based validation bar (include partial hashing from active files)
            let use_file_map = !inner.validate_file_map.is_empty();
            let partial_hashed: u64 = if use_file_map {
                inner.validate_file_map.values().filter(|s| s.active).map(|s| s.bytes_hashed).sum()
            } else {
                inner.validate_active.values().map(|(_, _, h)| *h).sum()
            };
            let v_bytes = (inner.validate_done_bytes + partial_hashed)
                .min(inner.validate_total_bytes);
            let v_total_bytes = inner.validate_total_bytes.max(1);
            let v_pct = (v_bytes as f64 * 100.0 / v_total_bytes as f64).min(100.0);
            let v_filled = ((bar_width as u64).saturating_mul(v_bytes) / v_total_bytes) as usize;
            let v_filled = v_filled.min(bar_width);
            let v_empty = bar_width - v_filled;
            writeln!(err, "\x1b[2K  {:12} [{}{}] {:.1}% ({} / {}) — {}/{} files",
                     "Validating",
                     "\u{2588}".repeat(v_filled),
                     "\u{2591}".repeat(v_empty),
                     v_pct,
                     human_size_bytes(v_bytes),
                     human_size_bytes(inner.validate_total_bytes),
                     inner.validate_done_files,
                     inner.validate_total_files).ok();
            lines += 1;

            // Collect detail lines: recent completed + active files with per-file progress bars
            let file_bar_width: usize = 20;
            let mut detail: Vec<String> = Vec::new();
            for (name, size) in &inner.validate_recent {
                let bar_str = "\u{2588}".repeat(file_bar_width);
                let suffix = format!("  ({})", human_size_bytes(*size));
                let name_budget = name_max.saturating_sub(file_bar_width + 18 + suffix.len());
                detail.push(format!("\x1b[2K    \x1b[32m\u{2713}\x1b[0m [\x1b[32m{}\x1b[0m] 100.0%  {}{}",
                         bar_str,
                         truncate_name(name, name_budget), suffix));
            }
            if use_file_map {
                // File-level aggregated display
                let mut active_files: Vec<(&String, &FileAggState)> = inner.validate_file_map.iter()
                    .filter(|(_, state)| state.active)
                    .collect();
                active_files.sort_by_key(|(name, _)| (*name).clone());
                for (name, state) in &active_files {
                    let (file_pct, f_filled) = if state.total_size > 0 {
                        let pct = (state.bytes_hashed as f64 * 100.0 / state.total_size as f64).min(100.0);
                        let filled = ((file_bar_width as u64).saturating_mul(state.bytes_hashed) / state.total_size.max(1)) as usize;
                        (pct, filled.min(file_bar_width))
                    } else {
                        (0.0, 0)
                    };
                    let f_empty = file_bar_width - f_filled;
                    let suffix = format!("  ({})", human_size_bytes(state.total_size));
                    let name_budget = name_max.saturating_sub(file_bar_width + 18 + suffix.len());
                    detail.push(format!("\x1b[2K    \x1b[35m\u{25b8}\x1b[0m [{}{}] {:.1}%  {}{}",
                             "\u{2588}".repeat(f_filled),
                             "\u{2591}".repeat(f_empty),
                             file_pct,
                             truncate_name(name, name_budget), suffix));
                }
            } else {
                // Fallback: per-slot display (non-split archives)
                for (_, (name, size, hashed)) in &inner.validate_active {
                    let (file_pct, f_filled) = if *size > 0 {
                        let pct = (*hashed as f64 * 100.0 / *size as f64).min(100.0);
                        let filled = ((file_bar_width as u64).saturating_mul(*hashed) / (*size).max(1)) as usize;
                        (pct, filled.min(file_bar_width))
                    } else {
                        (0.0, 0)
                    };
                    let f_empty = file_bar_width - f_filled;
                    let suffix = format!("  ({})", human_size_bytes(*size));
                    let name_budget = name_max.saturating_sub(file_bar_width + 18 + suffix.len());
                    detail.push(format!("\x1b[2K    \x1b[35m\u{25b8}\x1b[0m [{}{}] {:.1}%  {}{}",
                             "\u{2588}".repeat(f_filled),
                             "\u{2591}".repeat(f_empty),
                             file_pct,
                             truncate_name(name, name_budget), suffix));
                }
            }
            if let Some((ref name, total, read)) = inner.validate_loading {
                let (load_pct, l_filled) = if total > 0 {
                    let pct = (read as f64 * 100.0 / total as f64).min(100.0);
                    let filled = ((file_bar_width as u64).saturating_mul(read) / total.max(1)) as usize;
                    (pct, filled.min(file_bar_width))
                } else {
                    (0.0, 0)
                };
                let l_empty = file_bar_width - l_filled;
                let suffix = format!("  ({})", human_size_bytes(total));
                let name_budget = name_max.saturating_sub(file_bar_width + 18 + suffix.len());
                detail.push(format!("\x1b[2K    \x1b[2m\u{25cb}\x1b[0m \x1b[2m[{}{}] {:.1}%  {}{}  reading\x1b[0m",
                         "\u{2588}".repeat(l_filled),
                         "\u{2591}".repeat(l_empty),
                         load_pct,
                         truncate_name(name, name_budget), suffix));
            }
            let skip = detail.len().saturating_sub(MAX_DETAIL_LINES);
            for line in detail.iter().skip(skip) {
                writeln!(err, "{}", line).ok();
                lines += 1;
            }
        } else {
            // === ARCHIVING PHASE ===
            // Global archiving bar
            let global_archived: u64 = inner.splits.iter()
                .map(|s| s.archived_bytes + s.current_file.as_ref().map(|c| c.2).unwrap_or(0))
                .sum();
            let global_total = inner.total_bytes.max(1);
            let global_pct = (global_archived as f64 * 100.0 / global_total as f64).min(100.0);
            let global_filled = ((bar_width as u64).saturating_mul(global_archived) / global_total) as usize;
            let global_filled = global_filled.min(bar_width);
            let global_empty = bar_width - global_filled;
            if global_archived >= global_total {
                writeln!(err, "\x1b[2K\x1b[32m\u{2713}\x1b[0m {:12} [\x1b[32m{}\x1b[0m] 100.0% ({}) — {} splits",
                         "Archiving", bar_full, human_size_bytes(inner.total_bytes),
                         inner.split_count).ok();
            } else {
                writeln!(err, "\x1b[2K  {:14} [{}{}] {:.1}% ({} / {}) — {} splits",
                         "Archiving",
                         "\u{2588}".repeat(global_filled),
                         "\u{2591}".repeat(global_empty),
                         global_pct,
                         human_size_bytes(global_archived),
                         human_size_bytes(inner.total_bytes),
                         inner.split_count).ok();
            }
            lines += 1;

            // Build detail lines per category: finished, active (has progress), pending (0%)
            let mut finished_lines: Vec<String> = Vec::new();
            let mut active_lines: Vec<String> = Vec::new();
            let mut pending_lines: Vec<String> = Vec::new();

            for split in &inner.splits {
                if split.finished {
                    let total_h = human_size_bytes(split.total_bytes);
                    finished_lines.push(format!("\x1b[2K  \x1b[32m\u{2713}\x1b[0m {:10} [\x1b[32m{}\x1b[0m] 100.0% ({})  {}",
                             split.label, bar_full, total_h,
                             truncate_name(&split.filename, name_max)));
                } else {
                    let effective = (split.archived_bytes
                        + split.current_file.as_ref().map(|c| c.2).unwrap_or(0))
                        .min(split.total_bytes);
                    let total = split.total_bytes.max(1);
                    let pct = (effective as f64 * 100.0 / total as f64).min(100.0);
                    let filled = ((bar_width as u64).saturating_mul(effective) / total) as usize;
                    let filled = filled.min(bar_width);
                    let empty = bar_width - filled;

                    let is_active = effective > 0 || split.current_file.is_some();
                    let target = if is_active { &mut active_lines } else { &mut pending_lines };

                    target.push(format!("\x1b[2K    {:10} [{}{}] {:.1}% ({} / {})  {}",
                             split.label,
                             "\u{2588}".repeat(filled),
                             "\u{2591}".repeat(empty),
                             pct,
                             human_size_bytes(effective),
                             human_size_bytes(split.total_bytes),
                             truncate_name(&split.filename, name_max)));

                    if is_active {
                        for (name, size) in &split.recent_completed {
                            let size_str = human_size_bytes(*size);
                            let suffix = format!("  ({})", size_str);
                            target.push(format!("\x1b[2K      \x1b[32m\u{2713} {}{}\x1b[0m",
                                     truncate_name(name, name_max.saturating_sub(suffix.len() + 6)), suffix));
                        }
                        if let Some((ref name, size, bytes_read)) = split.current_file {
                            let size_str = human_size_bytes(size);
                            if size > 0 && bytes_read < size {
                                let pct = bytes_read as f64 * 100.0 / size as f64;
                                let suffix = format!("  {:.1}% ({})", pct, size_str);
                                target.push(format!("\x1b[2K      \x1b[35m\u{25b8} {}{}\x1b[0m",
                                         truncate_name(name, name_max.saturating_sub(suffix.len() + 6)), suffix));
                            } else {
                                let suffix = format!("  ({})", size_str);
                                target.push(format!("\x1b[2K      \x1b[35m\u{25b8} {}{}\x1b[0m",
                                         truncate_name(name, name_max.saturating_sub(suffix.len() + 6)), suffix));
                            }
                        }
                        for (name, size) in &split.queued_files {
                            let size_str = human_size_bytes(*size);
                            let suffix = format!("  ({})", size_str);
                            target.push(format!("\x1b[2K      \x1b[90m\u{25cb} {}{}\x1b[0m",
                                     truncate_name(name, name_max.saturating_sub(suffix.len() + 6)), suffix));
                        }
                    }
                }
            }

            // Priority: active lines always shown, then fill remaining with
            // finished (tail) and pending (head) if space allows
            let active_count = active_lines.len();
            let remaining = MAX_DETAIL_LINES.saturating_sub(active_count);
            // Give finished lines half the remaining (tail), pending the other half (head)
            let finished_budget = remaining / 2;
            let pending_budget = remaining.saturating_sub(finished_budget);
            let f_skip = finished_lines.len().saturating_sub(finished_budget);
            let p_take = pending_budget.min(pending_lines.len());

            for line in finished_lines.iter().skip(f_skip) {
                writeln!(err, "{}", line).ok();
                lines += 1;
            }
            for line in &active_lines {
                writeln!(err, "{}", line).ok();
                lines += 1;
            }
            for line in pending_lines.iter().take(p_take) {
                writeln!(err, "{}", line).ok();
                lines += 1;
            }
        }

        // Clear leftover lines from previous render
        let prev = inner.prev_lines;
        for _ in 0..prev.saturating_sub(lines) {
            writeln!(err, "\x1b[2K").ok();
        }
        let extra = prev.saturating_sub(lines);
        if extra > 0 {
            write!(err, "\x1b[{}A", extra).ok();
        }

        inner.prev_lines = lines;
        err.flush().ok();
    }

    fn render_final(inner: &mut SplitProgressInner) {
        let mut err = io::stderr().lock();
        if inner.prev_lines > 0 {
            write!(err, "\x1b[{}A", inner.prev_lines).ok();
        }

        let tw = term_width();
        let bar_width: usize = if tw >= 100 { 40 } else if tw >= 60 { tw - 60 + 20 } else { 20 };
        let bar_full = "\u{2588}".repeat(bar_width);
        let mut lines: usize = 0;

        // Discovery complete
        writeln!(err, "\x1b[2K\x1b[32m\u{2713}\x1b[0m {:12} [\x1b[32m{}\x1b[0m] 100.0% ({} files / {})",
                 "Discovering", bar_full, inner.discovery_total_files,
                 human_size_bytes(inner.discovery_total_bytes)).ok();
        lines += 1;

        // Archiving complete
        writeln!(err, "\x1b[2K\x1b[32m\u{2713}\x1b[0m {:12} [\x1b[32m{}\x1b[0m] 100.0% ({}) — {} splits",
                 "Archiving", bar_full, human_size_bytes(inner.total_bytes),
                 inner.split_count).ok();
        lines += 1;

        // Validating complete (if validation was done)
        if inner.validation_phase {
            writeln!(err, "\x1b[2K\x1b[32m\u{2713}\x1b[0m {:12} [\x1b[32m{}\x1b[0m] 100.0% ({} / {}) — {} files",
                     "Validating", bar_full,
                     human_size_bytes(inner.validate_total_bytes),
                     human_size_bytes(inner.validate_total_bytes),
                     inner.validate_total_files).ok();
            lines += 1;
        }

        // Clear leftover
        for _ in 0..inner.prev_lines.saturating_sub(lines) {
            writeln!(err, "\x1b[2K").ok();
        }
        let extra = inner.prev_lines.saturating_sub(lines);
        if extra > 0 {
            write!(err, "\x1b[{}A", extra).ok();
        }

        err.flush().ok();
        inner.prev_lines = 0;
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

pub(crate) const READ_CHUNK: usize = 1 << 20; // 1 MiB

/// Read a file in chunks, calling `on_progress(total_bytes_read_so_far)` after each chunk.
fn read_file_chunked<F>(path: &Path, mut on_progress: F) -> io::Result<FileData>
where
    F: FnMut(u64),
{
    let meta = fs::metadata(path)?;
    let file_size = meta.len() as usize;
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

    let mut content = Vec::with_capacity(file_size);
    let mut file = fs::File::open(path)?;
    let mut buf = vec![0u8; READ_CHUNK];
    let mut total_read: u64 = 0;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        content.extend_from_slice(&buf[..n]);
        total_read += n as u64;
        on_progress(total_read);
    }

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

/// Create a pre-allocated dataset for chunked writing (no data written yet).
/// Returns the dataset handle. Caller writes data via `write_slice`.
/// Falls back to None for LZMA (needs full buffer for application-level compression).
fn create_dataset_shell(
    h5f: &hdf5::File,
    rel_path: &str,
    file_size: usize,
    compression: Option<&str>,
    compression_opts: Option<u8>,
    shuffle: bool,
) -> Option<hdf5::Dataset> {
    if compression == Some("lzma") {
        return None; // LZMA needs full buffer
    }

    let chunk_size = READ_CHUNK.min(file_size.max(1)); // at least 1 byte chunk
    let mut builder = h5f.new_dataset::<u8>().shape(file_size).chunk(chunk_size);

    if shuffle {
        builder = builder.shuffle();
    }

    match compression {
        Some("gzip") => {
            builder = builder.deflate(compression_opts.unwrap_or(4));
        }
        Some("lzf") => {
            builder = builder.deflate(1);
        }
        _ => {}
    }

    Some(
        builder
            .create(rel_path)
            .unwrap_or_else(|e| panic!("Failed to create dataset '{}': {}", rel_path, e)),
    )
}

/// Write a chunk to a pre-allocated dataset via hyperslab selection.
pub(crate) fn write_dataset_chunk(ds: &hdf5::Dataset, chunk: &[u8], offset: usize) {
    let view = ArrayView1::from(chunk);
    ds.write_slice(&view, offset..offset + chunk.len())
        .unwrap_or_else(|e| panic!("Failed to write chunk at offset {}: {}", offset, e));
}

/// Streaming hasher that supports all checksum algorithms.
pub(crate) enum StreamHasher {
    Md5(md5::Md5),
    Sha256(Sha256),
    Blake3(Box<blake3::Hasher>),
}

impl StreamHasher {
    pub(crate) fn new(algo: &str) -> Self {
        match algo {
            "md5" => StreamHasher::Md5(<md5::Md5 as Digest>::new()),
            "sha256" => StreamHasher::Sha256(Sha256::new()),
            "blake3" => StreamHasher::Blake3(Box::new(blake3::Hasher::new())),
            _ => panic!("Unsupported checksum algorithm: {}", algo),
        }
    }
    pub(crate) fn update(&mut self, data: &[u8]) {
        match self {
            StreamHasher::Md5(h) => { h.update(data); }
            StreamHasher::Sha256(h) => { h.update(data); }
            StreamHasher::Blake3(h) => { h.update(data); }
        }
    }
    pub(crate) fn finalize_hex(self) -> String {
        match self {
            StreamHasher::Md5(h) => format!("{:x}", h.finalize()),
            StreamHasher::Sha256(h) => format!("{:x}", h.finalize()),
            StreamHasher::Blake3(h) => h.finalize().to_hex().to_string(),
        }
    }
}

/// Finalize a chunked dataset: write checksum attributes.
fn finalize_dataset_checksum(
    ds: &hdf5::Dataset,
    algo: Option<&str>,
    hasher: Option<StreamHasher>,
) {
    if let (Some(algo), Some(h)) = (algo, hasher) {
        let hash = h.finalize_hex();
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
}

/// Read file metadata without content.
pub(crate) struct FileMeta {
    pub(crate) mode: u32,
    pub(crate) uid: u32,
    pub(crate) gid: u32,
    pub(crate) owner: String,
    pub(crate) group: String,
    pub(crate) mtime: f64,
}

pub(crate) fn read_file_meta(path: &Path) -> io::Result<FileMeta> {
    let meta = fs::metadata(path)?;
    let mode = meta.permissions().mode();
    use std::os::unix::fs::MetadataExt;
    let uid = meta.uid();
    let gid = meta.gid();
    let mtime = meta
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    let owner = users::get_user_by_uid(uid)
        .map(|u| u.name().to_string_lossy().to_string())
        .unwrap_or_else(|| uid.to_string());
    let group = users::get_group_by_gid(gid)
        .map(|g| g.name().to_string_lossy().to_string())
        .unwrap_or_else(|| gid.to_string());
    Ok(FileMeta { mode, uid, gid, owner, group, mtime })
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
    validate: bool,
) {
    let output_h5 = shellexpand::tilde(output_h5).to_string();
    let t0 = Instant::now();
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

    let mut progress = PipelineProgress::new(verbose);
    let verbose_file = verbose && !progress.is_tty;
    let mut validation_errors: Vec<String> = Vec::new();
    let mut validated_count: usize = 0;

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
        // ---- Sequential pipelined path ----
        // Spawn discovery thread to walk directories asynchronously.
        let discovery = progress.discovery_counters();
        let sources_owned: Vec<String> = sources.iter().map(|s| s.to_string()).collect();

        let (tx, rx) = mpsc::sync_channel::<(FileEntry, u64)>(64);
        let discovery_handle = std::thread::spawn(move || {
            let mut empty_dirs_local: Vec<String> = Vec::new();
            for source in &sources_owned {
                let source = shellexpand::tilde(source).to_string();
                let source_path = Path::new(&source).to_path_buf();

                if source_path.is_dir() {
                    let source_norm = source_path
                        .canonicalize()
                        .unwrap_or_else(|_| source_path.clone());
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
                            let size = fs::metadata(&entry_path).map(|m| m.len()).unwrap_or(0);
                            let rel = pathdiff::diff_paths(&entry_path, &base_dir)
                                .unwrap_or_else(|| entry_path.clone());
                            discovery.files_found.fetch_add(1, Ordering::Relaxed);
                            discovery.bytes_found.fetch_add(size, Ordering::Relaxed);
                            let fe = FileEntry {
                                file_path: entry_path,
                                rel_path: rel.to_string_lossy().to_string(),
                            };
                            if tx.send((fe, size)).is_err() {
                                return empty_dirs_local;
                            }
                        } else if entry_path.is_dir() && entry_path != source_norm {
                            let is_empty = WalkDir::new(&entry_path)
                                .min_depth(1)
                                .into_iter()
                                .filter_map(|e| e.ok())
                                .next()
                                .is_none();
                            if is_empty {
                                let rel = pathdiff::diff_paths(&entry_path, &base_dir)
                                    .unwrap_or_else(|| entry_path.clone());
                                empty_dirs_local.push(rel.to_string_lossy().to_string());
                            }
                        }
                    }
                } else if source_path.is_file() {
                    let size = fs::metadata(&source_path).map(|m| m.len()).unwrap_or(0);
                    discovery.files_found.fetch_add(1, Ordering::Relaxed);
                    discovery.bytes_found.fetch_add(size, Ordering::Relaxed);
                    let fe = FileEntry {
                        file_path: source_path.clone(),
                        rel_path: source_path
                            .file_name()
                            .unwrap()
                            .to_string_lossy()
                            .to_string(),
                    };
                    if tx.send((fe, size)).is_err() {
                        return empty_dirs_local;
                    }
                } else {
                    eprintln!(
                        "Warning: '{}' is not a valid file or directory; skipping.",
                        source
                    );
                }
            }
            discovery.done.store(true, Ordering::Release);
            empty_dirs_local
        });

        // Main thread: consume from channel and archive files as they arrive.
        let mut queued: VecDeque<(FileEntry, u64)> = VecDeque::new();
        let discovery_ref = progress.discovery_counters();
        let mut archived_files: Vec<(String, u64)> = Vec::new();

        // Render initial state
        if progress.is_tty {
            progress.render();
        }

        // --- Phase 2: Archiving ---
        loop {
            // Drain available entries from channel into queue
            loop {
                match rx.try_recv() {
                    Ok((entry, size)) => {
                        progress.archive_total += size;
                        queued.push_back((entry, size));
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => break,
                }
            }

            if let Some((entry, _size)) = queued.pop_front() {
                // Update queued display (after popping current file)
                let queued_names: Vec<(String, u64)> = queued.iter().take(3)
                    .map(|(e, s)| (e.rel_path.clone(), *s)).collect();
                progress.set_queued(&queued_names);

                let file_size = fs::metadata(&entry.file_path).map(|m| m.len()).unwrap_or(0);
                progress.begin_file(&entry.rel_path, file_size);

                // Ensure parent groups exist
                if let Some(parent) = Path::new(&entry.rel_path).parent() {
                    let gp = parent.to_string_lossy().to_string();
                    if !gp.is_empty() {
                        ensure_group(&h5f, &gp);
                    }
                }

                // Skip if already exists in append mode
                if is_append && h5f.dataset(&entry.rel_path).is_ok() {
                    if verbose_file {
                        println!("Skipping {} (already exists)", &entry.rel_path);
                    }
                    progress.finish_file(file_size);
                    continue;
                }

                let file_meta = read_file_meta(&entry.file_path).expect("Failed to stat file");

                // Try pipelined read/write (not available for LZMA)
                if let Some(ds) = create_dataset_shell(
                    &h5f, &entry.rel_path, file_size as usize,
                    compression, compression_opts, shuffle,
                ) {
                    // Spawn reader thread — reads file in chunks, sends via channel
                    let (chunk_tx, chunk_rx) = mpsc::sync_channel::<Vec<u8>>(4);
                    let read_path = entry.file_path.clone();
                    let reader = std::thread::spawn(move || {
                        let mut file = fs::File::open(&read_path)
                            .expect("Failed to open file for reading");
                        let mut buf = vec![0u8; READ_CHUNK];
                        loop {
                            let n = file.read(&mut buf).expect("Failed to read file chunk");
                            if n == 0 { break; }
                            if chunk_tx.send(buf[..n].to_vec()).is_err() { break; }
                        }
                    });

                    // Main thread: write chunks to HDF5 as they arrive
                    let mut offset: usize = 0;
                    let mut hasher = checksum.map(StreamHasher::new);
                    for chunk in chunk_rx.iter() {
                        let chunk_len = chunk.len();
                        if let Some(ref mut h) = hasher {
                            h.update(&chunk);
                        }
                        write_dataset_chunk(&ds, &chunk, offset);
                        offset += chunk_len;
                        progress.update_file_progress(offset as u64);
                    }
                    reader.join().expect("Reader thread panicked");

                    // Store checksum
                    finalize_dataset_checksum(&ds, checksum, hasher);

                    // Write metadata attributes
                    ds.new_attr::<u32>().shape(()).create("mode").unwrap().write_scalar(&file_meta.mode).unwrap();
                    ds.new_attr::<u32>().shape(()).create("uid").unwrap().write_scalar(&file_meta.uid).unwrap();
                    ds.new_attr::<u32>().shape(()).create("gid").unwrap().write_scalar(&file_meta.gid).unwrap();
                    ds.new_attr::<f64>().shape(()).create("mtime").unwrap().write_scalar(&file_meta.mtime).unwrap();
                    let owner_vu: hdf5::types::VarLenUnicode = file_meta.owner.parse().unwrap();
                    ds.new_attr::<hdf5::types::VarLenUnicode>().shape(()).create("owner").unwrap().write_scalar(&owner_vu).unwrap();
                    let group_vu: hdf5::types::VarLenUnicode = file_meta.group.parse().unwrap();
                    ds.new_attr::<hdf5::types::VarLenUnicode>().shape(()).create("group").unwrap().write_scalar(&group_vu).unwrap();

                    if xattr_flag {
                        let xattrs = metadata::read_xattrs(&entry.file_path);
                        for (name, value) in &xattrs {
                            let attr_name = format!("xattr.{}", name);
                            if let Ok(a) = ds.new_attr::<u8>().shape(value.len()).create(&*attr_name) {
                                a.write_raw(value).ok();
                            }
                        }
                    }

                    if verbose_file {
                        println!("Storing: {}", &entry.rel_path);
                    }
                } else {
                    // LZMA fallback: read all then write
                    let fd = read_file_chunked(&entry.file_path, |bytes_read| {
                        progress.update_file_progress(bytes_read);
                    }).expect("Failed to read file");
                    progress.mark_file_writing();
                    write_to_h5(&h5f, &entry.rel_path, &fd.content, fd.mode, fd.uid, fd.gid,
                                &fd.owner, &fd.group, fd.mtime, true, Some(&entry.file_path));
                }

                archived_files.push((entry.rel_path.clone(), file_size));
                progress.finish_file(file_size);
            } else if discovery_ref.done.load(Ordering::Acquire) {
                // Discovery done — drain any remaining items
                let mut found_more = false;
                for item in rx.try_iter() {
                    progress.archive_total += item.1;
                    queued.push_back(item);
                    found_more = true;
                }
                if !found_more && queued.is_empty() {
                    break;
                }
            } else {
                // Queue empty, discovery still running — wait briefly
                match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok((entry, size)) => {
                        progress.archive_total += size;
                        queued.push_back((entry, size));
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        progress.maybe_render();
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {}
                }
            }
        }

        progress.finish_discovery();
        progress.flush_current_file();

        // --- Phase 3: Validation (separate pass) ---
        if validate {
            if let Some(algo) = checksum {
                progress.enable_validation(progress.archive_total);
                for (rel_path, content_len) in &archived_files {
                    progress.begin_validation(rel_path);
                    let ds = h5f.dataset(rel_path).unwrap();

                    let is_lzma_ds = ds.attr("har_lzma").ok()
                        .and_then(|a| a.read_scalar::<u8>().ok())
                        .map(|v| v != 0)
                        .unwrap_or(false);

                    let stored_hash = ds
                        .attr("har_checksum")
                        .ok()
                        .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
                        .map(|v| v.as_str().to_string());

                    if is_lzma_ds {
                        // LZMA: must read+decompress entire dataset
                        let readback = read_dataset_content(&ds);
                        let actual = compute_checksum(&readback, algo);
                        if let Some(expected) = stored_hash {
                            if actual != expected {
                                validation_errors.push(rel_path.clone());
                            } else {
                                validated_count += 1;
                            }
                        } else {
                            validated_count += 1;
                        }
                        progress.finish_validation_file(*content_len);
                    } else {
                        // Incremental: read + hash in 1MiB chunks via hyperslab
                        let ds_size = *content_len as usize;
                        let mut hasher = StreamHasher::new(algo);
                        let mut remaining = ds_size;
                        let mut pos: usize = 0;
                        while remaining > 0 {
                            let chunk_len = remaining.min(READ_CHUNK);
                            let chunk: ndarray::Array1<u8> = ds.read_slice(pos..pos + chunk_len)
                                .expect("Failed to read validation chunk");
                            hasher.update(chunk.as_slice().unwrap());
                            pos += chunk_len;
                            remaining -= chunk_len;
                            progress.finish_validation_file(chunk_len as u64);
                        }
                        let actual = hasher.finalize_hex();
                        if let Some(expected) = stored_hash {
                            if actual != expected {
                                validation_errors.push(rel_path.clone());
                            } else {
                                validated_count += 1;
                            }
                        } else {
                            validated_count += 1;
                        }
                    }
                }
            }
        }

        progress.finish();

        // Collect empty dirs from the discovery thread
        let empty_dirs = discovery_handle.join().expect("Discovery thread panicked");

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
    } else {
        // ---- Parallel path: synchronous walk, parallel read, sequential write ----
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

        let total_size: u64 = file_entries.iter().map(|e| {
            fs::metadata(&e.file_path).map(|m| m.len()).unwrap_or(0)
        }).sum();

        let mut compat_progress = Progress::new(verbose);

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

        compat_progress.start_phase("Reading", total_size);
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
            compat_progress.inc(&r.rel_path, r.content.len() as u64);
        }

        compat_progress.start_phase("Writing", total_size);
        for r in &read_results {
            write_to_h5(&h5f, &r.rel_path, &r.content, r.mode, r.uid, r.gid, &r.owner, &r.group, r.mtime, false, Some(&r.file_path));
            compat_progress.inc(&r.rel_path, r.content.len() as u64);
        }

        if validate {
            if let Some(algo) = checksum {
                compat_progress.start_phase("Validating", total_size);
                for r in &read_results {
                    let readback = read_dataset_content(&h5f.dataset(&r.rel_path).unwrap());
                    let actual = compute_checksum(&readback, algo);
                    let expected = compute_checksum(&r.content, algo);
                    if actual != expected {
                        validation_errors.push(r.rel_path.clone());
                    } else {
                        validated_count += 1;
                    }
                    compat_progress.inc(&r.rel_path, r.content.len() as u64);
                }
            }
        }
        compat_progress.finish();

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
    }

    h5f.close().ok();
    let elapsed = t0.elapsed().as_secs_f64();
    println!("\nOperation completed in {:.2} seconds.", elapsed);
    if validate && !validation_errors.is_empty() {
        eprintln!("VALIDATION FAILED on {} file(s):", validation_errors.len());
        for e in validation_errors.iter().take(10) {
            eprintln!("  {}", e);
        }
        if validation_errors.len() > 10 {
            eprintln!("  ... and {} more", validation_errors.len() - 10);
        }
        std::process::exit(1);
    } else if validate && validated_count > 0 {
        println!("Validation passed. {} files verified.", validated_count);
    }
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
        let total_bytes: u64 = items.iter().filter_map(|(name, t)| {
            if matches!(t, H5ObjType::Dataset) {
                h5f.dataset(name).ok().map(|ds| ds.size() as u64)
            } else {
                None
            }
        }).sum();
        let mut progress = Progress::new(verbose);
        progress.start_phase("Extracting", total_bytes);
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
                        progress.inc(name, data.len() as u64);
                        if verbose_file {
                            println!("Extracted: {}", name);
                        }
                    }
                    H5ObjType::EmptyDirGroup => {
                        let dest = extract_path.join(name);
                        fs::create_dir_all(&dest).ok();
                        progress.inc(name, 0);
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
                        let data_len = data.len() as u64;
                        read_items.push((name.clone(), data, mode));
                        progress.inc(name, data_len);
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
                progress.inc(name, 0);
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
    } else if validate {
        let count = if file_key.is_some() { 1 } else {
            collect_items(&h5f).iter().filter(|(_, t)| matches!(t, H5ObjType::Dataset)).count()
        };
        println!("Validation passed. {} files verified.", count);
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

/// Extract archive to temp dir and compare against source files.
pub fn validate_roundtrip(
    h5_path: &str,
    sources: &[&str],
    bagit: bool,
    _verbose: bool,
    byte_for_byte: bool,
    checksum_algo: &str,
) -> bool {
    // Build source_map: rel_path -> abs_source_path
    let mut source_map: std::collections::BTreeMap<String, PathBuf> = std::collections::BTreeMap::new();
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
                if entry.path().is_file() {
                    let rel = pathdiff::diff_paths(entry.path(), &base_dir)
                        .unwrap_or_else(|| entry.path().to_path_buf());
                    source_map.insert(
                        rel.to_string_lossy().to_string(),
                        entry.path().to_path_buf(),
                    );
                }
            }
        } else if source_path.is_file() {
            source_map.insert(
                source_path.file_name().unwrap().to_string_lossy().to_string(),
                source_path.to_path_buf(),
            );
        }
    }

    // Create temp dir in CWD
    let tmp_dir = {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let name = format!(".har_roundtrip_{}", ts);
        let p = PathBuf::from(&name);
        fs::create_dir_all(&p).expect("Failed to create roundtrip temp dir");
        p
    };

    // Extract
    let tmp_str = tmp_dir.to_string_lossy().to_string();
    if bagit {
        bagit::extract_bagit(h5_path, &tmp_str, None, false, false, 1, false, false, false);
    } else {
        extract_h5_to_directory(h5_path, &tmp_str, None, 1, false, false, None, false, false);
    }

    // Compare
    let mut errors: Vec<String> = Vec::new();
    let mut verified: usize = 0;

    for (rel_path, src_path) in &source_map {
        let extracted = tmp_dir.join(rel_path);
        if !extracted.exists() {
            errors.push(format!("MISSING: {}", rel_path));
            continue;
        }
        let src_data = fs::read(src_path).expect("Failed to read source file");
        let ext_data = fs::read(&extracted).expect("Failed to read extracted file");

        if byte_for_byte {
            if src_data != ext_data {
                errors.push(format!("MISMATCH: {}", rel_path));
            } else {
                verified += 1;
            }
        } else {
            let h1 = compute_checksum(&src_data, checksum_algo);
            let h2 = compute_checksum(&ext_data, checksum_algo);
            if h1 != h2 {
                errors.push(format!("MISMATCH: {}", rel_path));
            } else {
                verified += 1;
            }
        }
    }

    // Cleanup
    fs::remove_dir_all(&tmp_dir).ok();

    if errors.is_empty() {
        println!("Roundtrip validation passed. {} files verified.", verified);
        true
    } else {
        eprintln!("ROUNDTRIP VALIDATION FAILED on {} file(s):", errors.len());
        for e in errors.iter().take(10) {
            eprintln!("  {}", e);
        }
        if errors.len() > 10 {
            eprintln!("  ... and {} more", errors.len() - 10);
        }
        false
    }
}

/// Delete source files/directories after successful archival.
pub fn delete_source_files(sources: &[&str], verbose: bool) {
    let mut deleted = 0;
    for source in sources {
        let source = shellexpand::tilde(source).to_string();
        let path = Path::new(&source);
        let result = if path.is_dir() {
            fs::remove_dir_all(path)
        } else if path.is_file() {
            fs::remove_file(path)
        } else {
            continue;
        };
        match result {
            Ok(_) => {
                deleted += 1;
                if verbose {
                    println!("Deleted: {}", source);
                }
            }
            Err(e) => {
                eprintln!("Warning: could not delete '{}': {}", source, e);
            }
        }
    }
    println!("Deleted {} source(s).", deleted);
}

pub mod bagit;
pub mod metadata;
pub mod browse;

#[cfg(test)]
mod tests;
