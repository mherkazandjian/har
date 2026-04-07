"""
Interactive TUI for browsing HDF5 archives (legacy and BagIt).

Dual-pane layout:
  Left  (40%) — collapsible tree of groups/datasets
  Right (60%) — details of selected item (storage, attrs, hex preview)
"""

import curses
import os
import sys

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class TreeNode:
    __slots__ = ("name", "full_path", "node_type", "children",
                 "expanded", "depth", "size_bytes", "child_count")

    def __init__(self, name, full_path, node_type, depth):
        self.name = name
        self.full_path = full_path
        self.node_type = node_type  # "group" | "dataset" | "empty_dir"
        self.children = []
        self.expanded = depth == 0  # root expanded by default
        self.depth = depth
        self.size_bytes = 0
        self.child_count = 0


def _human_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            if n == int(n):
                return f"{int(n)} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Tree builders
# ---------------------------------------------------------------------------

def _build_tree_legacy(h5f):
    """Build tree from a legacy (one-dataset-per-file) archive."""
    root = TreeNode("", "/", "group", 0)
    group_map = {"": root}

    def _ensure_group(path, depth):
        if path in group_map:
            return group_map[path]
        parent_path = "/".join(path.split("/")[:-1])
        parent = _ensure_group(parent_path, depth - 1)
        name = path.split("/")[-1]
        node = TreeNode(name, path, "group", depth)
        parent.children.append(node)
        group_map[path] = node
        return node

    def visitor(name, obj):
        parts = name.split("/")
        depth = len(parts)
        if isinstance(obj, h5py.Dataset):
            parent_path = "/".join(parts[:-1])
            parent = _ensure_group(parent_path, depth - 1)
            ds_node = TreeNode(parts[-1], name, "dataset", depth)
            ds_node.size_bytes = obj.id.get_storage_size()
            parent.children.append(ds_node)
        elif isinstance(obj, h5py.Group):
            g = _ensure_group(name, depth)
            if obj.attrs.get("empty_dir"):
                g.node_type = "empty_dir"

    h5f.visititems(visitor)

    # Sort children, compute child_count
    def _sort(node):
        node.children.sort(key=lambda c: (c.node_type == "dataset", c.name))
        node.child_count = sum(
            1 if c.node_type == "dataset" else c.child_count
            for c in node.children
        )
        for c in node.children:
            _sort(c)

    _sort(root)
    return root


def _read_bagit_index(h5f):
    """Read BagIt index, supporting both compound (Python) and parallel-array (Rust) formats.
    Returns dict with keys: paths, batch_ids, offsets, lengths, modes, sha256s, uids, gids, mtimes, owners, groups."""
    result = {}
    if "index_data" in h5f:
        # Rust format: parallel arrays in index_data/ group
        idx = h5f["index_data"]
        paths_blob = idx["paths"][()].tobytes().decode("utf-8")
        result["paths"] = paths_blob.split("\n")
        result["batch_ids"] = idx["batch_id"][()].tolist()
        result["offsets"] = idx["offset"][()].tolist()
        result["lengths"] = idx["length"][()].tolist()
        result["modes"] = idx["mode"][()].tolist()
        shas_blob = idx["sha256s"][()].tobytes().decode("utf-8")
        result["sha256s"] = shas_blob.split("\n")
        # Optional owner/mtime fields
        if "uid" in idx:
            result["uids"] = idx["uid"][()].tolist()
            result["gids"] = idx["gid"][()].tolist()
            result["mtimes"] = idx["mtime"][()].tolist()
            owners_blob = idx["owners"][()].tobytes().decode("utf-8")
            result["owners"] = owners_blob.split("\n")
            groups_blob = idx["groups"][()].tobytes().decode("utf-8")
            result["groups"] = groups_blob.split("\n")
    elif "index" in h5f:
        # Python format: compound dataset
        data = h5f["index"][()]
        result["paths"] = [row["path"].decode("utf-8") for row in data]
        result["batch_ids"] = [int(row["batch_id"]) for row in data]
        result["offsets"] = [int(row["offset"]) for row in data]
        result["lengths"] = [int(row["length"]) for row in data]
        result["modes"] = [int(row["mode"]) for row in data]
        result["sha256s"] = [row["sha256"].decode("utf-8") for row in data]
        # Optional owner/mtime fields
        if "uid" in data.dtype.names:
            result["uids"] = [int(row["uid"]) for row in data]
            result["gids"] = [int(row["gid"]) for row in data]
            result["mtimes"] = [float(row["mtime"]) for row in data]
            result["owners"] = [row["owner"].decode("utf-8") for row in data]
            result["groups"] = [row["group_name"].decode("utf-8") for row in data]
    n = len(result.get("paths", []))
    result.setdefault("paths", [])
    result.setdefault("batch_ids", [])
    result.setdefault("offsets", [])
    result.setdefault("lengths", [])
    result.setdefault("modes", [])
    result.setdefault("sha256s", [])
    result.setdefault("uids", [0] * n)
    result.setdefault("gids", [0] * n)
    result.setdefault("mtimes", [0.0] * n)
    result.setdefault("owners", [""] * n)
    result.setdefault("groups", [""] * n)
    return result


def _build_tree_bagit(h5f):
    """Build tree from a BagIt archive's index."""
    root = TreeNode("", "/", "group", 0)
    group_map = {"": root}

    # Read index (supports both Python compound and Rust parallel-array formats)
    idx = _read_bagit_index(h5f)
    paths = idx["paths"]
    lengths = idx["lengths"]

    def _ensure_group(path, depth):
        if path in group_map:
            return group_map[path]
        parent_path = "/".join(path.split("/")[:-1])
        parent = _ensure_group(parent_path, depth - 1)
        name = path.split("/")[-1]
        node = TreeNode(name, path, "group", depth)
        parent.children.append(node)
        group_map[path] = node
        return node

    for i, p in enumerate(paths):
        clean = p.removeprefix("data/") if p.startswith("data/") else p
        parts = clean.split("/")
        depth = len(parts)
        parent_path = "/".join(parts[:-1])
        parent = _ensure_group(parent_path, depth - 1)
        ds_node = TreeNode(parts[-1], clean, "dataset", depth)
        ds_node.size_bytes = int(lengths[i])
        parent.children.append(ds_node)

    # Empty dirs (Python writes S-dtype array, Rust writes u8 blob with newlines)
    if "empty_dirs" in h5f:
        ed_data = h5f["empty_dirs"][()]
        if ed_data.dtype.kind == 'S':
            dir_list = [d.decode("utf-8") for d in ed_data]
        else:
            dir_list = ed_data.tobytes().decode("utf-8").split("\n")
        for d_str in dir_list:
            if not d_str:
                continue
            clean = d_str.removeprefix("data/") if d_str.startswith("data/") else d_str
            parts = clean.split("/")
            depth = len(parts)
            parent_path = "/".join(parts[:-1])
            parent = _ensure_group(parent_path, depth - 1)
            em = TreeNode(parts[-1], clean, "empty_dir", depth)
            parent.children.append(em)

    def _sort(node):
        node.children.sort(key=lambda c: (c.node_type == "dataset", c.name))
        node.child_count = sum(
            1 if c.node_type == "dataset" else c.child_count
            for c in node.children
        )
        for c in node.children:
            _sort(c)

    # BagIt tag files (show under a [bagit] virtual group)
    if "bagit" in h5f:
        bagit_grp_node = TreeNode("[bagit]", "[bagit]", "group", 1)
        for name in sorted(h5f["bagit"].keys()):
            ds = h5f["bagit"][name]
            tag_node = TreeNode(name, f"[bagit]/{name}", "dataset", 2)
            tag_node.size_bytes = ds.id.get_storage_size()
            bagit_grp_node.children.append(tag_node)
        root.children.append(bagit_grp_node)

    _sort(root)
    return root


# ---------------------------------------------------------------------------
# Detail info
# ---------------------------------------------------------------------------

def _collect_detail_legacy(h5f, node):
    """Gather detail info for a legacy dataset."""
    sections = []
    if node.node_type == "dataset":
        ds = h5f[node.full_path]
        raw_size = ds.size * ds.dtype.itemsize if ds.size else 0
        stored = ds.id.get_storage_size()

        # Storage section
        rows = []
        rows.append(("Shape", str(ds.shape)))
        rows.append(("Type", str(ds.dtype)))
        rows.append(("Raw", _human_size(raw_size)))
        comp = ds.compression or "none"
        if ds.compression_opts:
            comp += f"-{ds.compression_opts}"
        if ds.shuffle:
            comp += "+shuffle"
        rows.append(("Stored", f"{_human_size(stored)} ({comp})"))
        if raw_size > 0 and stored > 0:
            rows.append(("Ratio", f"{raw_size / stored:.1f}x"))
        if 'mode' in ds.attrs:
            rows.append(("Mode", f"0o{int(ds.attrs['mode']):o}"))
        if 'owner' in ds.attrs:
            uid_str = str(int(ds.attrs['uid'])) if 'uid' in ds.attrs else "?"
            rows.append(("Owner", f"{ds.attrs['owner']} ({uid_str})"))
        if 'group' in ds.attrs:
            gid_str = str(int(ds.attrs['gid'])) if 'gid' in ds.attrs else "?"
            rows.append(("Group", f"{ds.attrs['group']} ({gid_str})"))
        if 'mtime' in ds.attrs:
            import datetime
            ts = datetime.datetime.fromtimestamp(float(ds.attrs['mtime'])).strftime("%Y-%m-%d %H:%M:%S")
            rows.append(("Modified", ts))
        sections.append(("STORAGE", rows))

        # Attributes section
        attr_rows = []
        for k in sorted(ds.attrs.keys()):
            val = ds.attrs[k]
            if k == "mode":
                attr_rows.append((k, f"0o{int(val):o}"))
            elif isinstance(val, bytes):
                try:
                    attr_rows.append((k, val.decode("utf-8")))
                except UnicodeDecodeError:
                    attr_rows.append((k, repr(val)))
            elif isinstance(val, np.ndarray):
                attr_rows.append((k, str(val.tolist())))
            elif isinstance(val, (np.integer,)):
                attr_rows.append((k, str(int(val))))
            elif isinstance(val, (np.floating,)):
                attr_rows.append((k, str(float(val))))
            else:
                attr_rows.append((k, str(val)))
        if attr_rows:
            sections.append(("ATTRIBUTES", attr_rows))

        # Hex preview (skip for large files >10 MB)
        if raw_size <= 10 * 1024 * 1024:
            try:
                raw = ds[()].tobytes()[:256]
                hex_lines = _format_hex(raw)
                if hex_lines:
                    sections.append(("DATA PREVIEW", [(l, "") for l in hex_lines]))
            except Exception:
                pass
        else:
            sections.append(("DATA PREVIEW", [("(skipped, file > 10 MB)", "")]))

    elif node.node_type == "group":
        try:
            grp = h5f[node.full_path]
            rows = [("Children", str(len(grp.keys())))]
            attrs = [(k, str(grp.attrs[k])) for k in grp.attrs]
            if attrs:
                sections.append(("GROUP INFO", rows))
                sections.append(("ATTRIBUTES", attrs))
            else:
                sections.append(("GROUP INFO", rows))
        except Exception:
            sections.append(("GROUP", [("Path", node.full_path)]))

    elif node.node_type == "empty_dir":
        sections.append(("EMPTY DIRECTORY", [("Path", node.full_path)]))

    return sections


def _collect_detail_bagit(h5f, node, is_bagit):
    """Gather detail info for a BagIt dataset."""
    sections = []
    if node.node_type != "dataset":
        if node.node_type == "empty_dir":
            sections.append(("EMPTY DIRECTORY", [("Path", node.full_path)]))
        else:
            sections.append(("GROUP", [("Children", str(node.child_count))]))
        return sections

    # BagIt tag files (under virtual [bagit]/ group)
    if node.full_path.startswith("[bagit]/"):
        tag_name = node.full_path.split("/", 1)[1]
        try:
            raw = h5f["bagit"][tag_name][()].tobytes()
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            sections.append(("CONTENT", [(l, "") for l in lines]))
        except Exception:
            sections.append(("ERROR", [("Cannot read", tag_name)]))
        return sections

    idx = _read_bagit_index(h5f)
    lookup = "data/" + node.full_path

    pos = None
    for i, p in enumerate(idx["paths"]):
        if p == lookup:
            pos = i
            break
    if pos is None:
        sections.append(("ERROR", [("Not found", node.full_path)]))
        return sections

    bid = int(idx["batch_ids"][pos])
    off = int(idx["offsets"][pos])
    length = int(idx["lengths"][pos])
    mode = int(idx["modes"][pos])
    sha = idx["sha256s"][pos] if pos < len(idx["sha256s"]) else ""
    uid = int(idx["uids"][pos])
    gid = int(idx["gids"][pos])
    mtime = float(idx["mtimes"][pos])
    owner = idx["owners"][pos] if pos < len(idx["owners"]) else ""
    group = idx["groups"][pos] if pos < len(idx["groups"]) else ""

    # Read checksum algo
    hash_algo = h5f.attrs.get("har_checksum_algo", "sha256")

    # Storage
    rows = []
    rows.append(("Size", _human_size(length)))
    rows.append(("Batch", f"{bid} (offset {off})"))
    if mode:
        rows.append(("Mode", f"0o{mode:o}"))
    if owner:
        rows.append(("Owner", f"{owner} ({uid})"))
    if group:
        rows.append(("Group", f"{group} ({gid})"))
    if mtime > 0:
        import datetime
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(("Modified", ts))
    sections.append(("STORAGE", rows))

    # Checksum
    if sha:
        sections.append(("CHECKSUM", [(hash_algo, sha)]))

    # User metadata
    if "user_metadata" in h5f:
        import json
        try:
            blob = h5f["user_metadata"][()].tobytes()
            meta = json.loads(blob)
            if node.full_path in meta:
                entry = meta[node.full_path]
                if "xattrs" in entry:
                    xattr_rows = [(k, str(v)) for k, v in sorted(entry["xattrs"].items())]
                    sections.append(("EXTENDED ATTRIBUTES", xattr_rows))
        except Exception:
            pass

    # Hex preview (skip for large files >10 MB)
    if length <= 10 * 1024 * 1024:
        try:
            batch_data = h5f[f"batches/{bid}"][()].tobytes()
            raw = batch_data[off:off + min(length, 256)]
            hex_lines = _format_hex(raw)
            if hex_lines:
                sections.append(("DATA PREVIEW", [(l, "") for l in hex_lines]))
        except Exception:
            pass
    else:
        sections.append(("DATA PREVIEW", [("(skipped, file > 10 MB)", "")]))

    return sections


def _format_hex(data, width=6):
    """Format bytes as hex + ASCII lines."""
    lines = []
    for offset in range(0, len(data), width):
        chunk = data[offset:offset + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"  {offset:05x}  {hex_part:<{width * 3}}  {ascii_part}")
    return lines


# ---------------------------------------------------------------------------
# Flatten tree for rendering
# ---------------------------------------------------------------------------

def _flatten(node, out=None, skip_root=True):
    if out is None:
        out = []
    if not skip_root:
        out.append(node)
    if node.expanded:
        for child in node.children:
            _flatten(child, out, skip_root=False)
    return out


# ---------------------------------------------------------------------------
# Breadcrumb
# ---------------------------------------------------------------------------

def _breadcrumb(node):
    if not node or not node.full_path or node.full_path == "/":
        return ""
    return node.full_path.replace("/", " > ")


# ---------------------------------------------------------------------------
# Main TUI
# ---------------------------------------------------------------------------

class BrowseApp:
    def __init__(self, h5f, tree_root, is_bagit):
        self.h5f = h5f
        self.tree_root = tree_root
        self.is_bagit = is_bagit
        self.visible = _flatten(tree_root)
        self.cursor = 0
        self.tree_scroll = 0
        self.detail_scroll = 0
        self.pane = 0  # 0=left, 1=right
        self.search_mode = False
        self.search_query = ""
        self.cached_path = None
        self.cached_detail = []
        self.archive_name = ""
        self.total_datasets = 0
        self.total_groups = 0

        # Count items
        def _count(n):
            if n.node_type == "dataset":
                self.total_datasets += 1
            elif n.node_type in ("group", "empty_dir"):
                self.total_groups += 1
            for c in n.children:
                _count(c)
        _count(tree_root)

    def _rebuild_visible(self):
        self.visible = _flatten(self.tree_root)
        if self.cursor >= len(self.visible):
            self.cursor = max(0, len(self.visible) - 1)

    def _selected(self):
        if 0 <= self.cursor < len(self.visible):
            return self.visible[self.cursor]
        return None

    def _get_detail(self, node):
        if node is None:
            return []
        if self.cached_path == node.full_path:
            return self.cached_detail
        if self.is_bagit:
            d = _collect_detail_bagit(self.h5f, node, self.is_bagit)
        else:
            d = _collect_detail_legacy(self.h5f, node)
        self.cached_path = node.full_path
        self.cached_detail = d
        self.detail_scroll = 0
        return d

    # -- Navigation --

    def move_down(self):
        if self.pane == 0:
            if self.cursor < len(self.visible) - 1:
                self.cursor += 1
        else:
            self.detail_scroll += 1

    def move_up(self):
        if self.pane == 0:
            if self.cursor > 0:
                self.cursor -= 1
        else:
            if self.detail_scroll > 0:
                self.detail_scroll -= 1

    def page_down(self, height):
        if self.pane == 0:
            self.cursor = min(len(self.visible) - 1, self.cursor + height)
        else:
            self.detail_scroll += height

    def page_up(self, height):
        if self.pane == 0:
            self.cursor = max(0, self.cursor - height)
        else:
            self.detail_scroll = max(0, self.detail_scroll - height)

    def expand(self):
        node = self._selected()
        if node and node.node_type in ("group",) and node.children:
            if not node.expanded:
                node.expanded = True
                self._rebuild_visible()
            elif self.cursor < len(self.visible) - 1:
                self.cursor += 1

    def collapse(self):
        node = self._selected()
        if node is None:
            return
        if node.node_type in ("group",) and node.expanded:
            node.expanded = False
            self._rebuild_visible()
        else:
            # Go to parent
            parts = node.full_path.split("/")
            if len(parts) > 1:
                parent_path = "/".join(parts[:-1])
                for i, v in enumerate(self.visible):
                    if v.full_path == parent_path:
                        self.cursor = i
                        break

    def toggle_expand(self):
        node = self._selected()
        if node and node.node_type in ("group",) and node.children:
            node.expanded = not node.expanded
            self._rebuild_visible()

    def search_next(self):
        if not self.search_query:
            return
        q = self.search_query.lower()
        start = self.cursor + 1
        for i in range(start, len(self.visible)):
            if q in self.visible[i].name.lower() or q in self.visible[i].full_path.lower():
                self.cursor = i
                return
        # Wrap around
        for i in range(0, start):
            if q in self.visible[i].name.lower() or q in self.visible[i].full_path.lower():
                self.cursor = i
                return

    def search_prev(self):
        if not self.search_query:
            return
        q = self.search_query.lower()
        start = self.cursor - 1
        for i in range(start, -1, -1):
            if q in self.visible[i].name.lower() or q in self.visible[i].full_path.lower():
                self.cursor = i
                return
        for i in range(len(self.visible) - 1, start, -1):
            if q in self.visible[i].name.lower() or q in self.visible[i].full_path.lower():
                self.cursor = i
                return

    # -- Drawing --

    def draw(self, stdscr):
        h, w = stdscr.getmaxyx()
        if h < 5 or w < 40:
            stdscr.addstr(0, 0, "Terminal too small")
            return

        left_w = max(20, w * 40 // 100)
        right_w = w - left_w - 1  # 1 for divider
        body_h = h - 3  # breadcrumb + status + search bar
        if body_h < 1:
            body_h = 1

        # Colors
        curses.init_pair(1, curses.COLOR_CYAN, -1)     # breadcrumb / borders
        curses.init_pair(2, curses.COLOR_BLUE, -1)      # groups
        curses.init_pair(3, curses.COLOR_WHITE, -1)     # files
        curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_WHITE)  # selected
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)   # status bar
        curses.init_pair(6, curses.COLOR_YELLOW, -1)    # section headers
        curses.init_pair(7, curses.COLOR_GREEN, -1)     # values
        curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_BLUE)   # search

        stdscr.erase()

        # -- Title bar --
        title = f" har browse: {self.archive_name} "
        bar = "─" * max(0, w - 2 - len(title))
        stdscr.addnstr(0, 0, "┌─" + title + bar + "┐", w, curses.color_pair(1))

        # -- Breadcrumb --
        node = self._selected()
        bc = _breadcrumb(node) if node else ""
        bc_line = "│ " + bc
        bc_line = bc_line[:w - 2].ljust(w - 2) + " │"
        stdscr.addnstr(1, 0, bc_line, w, curses.color_pair(1))

        # -- Divider --
        div = "├" + "─" * (left_w - 1) + "┬" + "─" * (right_w) + "┤"
        stdscr.addnstr(2, 0, div[:w], w, curses.color_pair(1))

        # -- Tree pane --
        # Scroll if needed
        if self.cursor < self.tree_scroll:
            self.tree_scroll = self.cursor
        if self.cursor >= self.tree_scroll + body_h:
            self.tree_scroll = self.cursor - body_h + 1

        for row in range(body_h):
            y = 3 + row
            if y >= h - 1:
                break
            idx = self.tree_scroll + row
            if idx < len(self.visible):
                n = self.visible[idx]
                indent = "  " * n.depth

                if n.node_type == "group" and n.children:
                    icon = "[-] " if n.expanded else "[+] "
                elif n.node_type == "empty_dir":
                    icon = "    "
                else:
                    # Tree lines
                    icon = " "

                label = indent + icon + n.name
                if n.node_type == "group" and n.children:
                    if n.expanded:
                        label += "/"
                    else:
                        cnt = n.child_count
                        if cnt:
                            label += f"/  [{cnt}]"
                        else:
                            label += "/"
                elif n.node_type == "empty_dir":
                    label += "/  [empty]"

                # Size for datasets
                size_str = ""
                if n.node_type == "dataset" and n.size_bytes > 0:
                    size_str = _human_size(n.size_bytes)

                # Fit into left pane
                avail = left_w - 2  # borders
                if size_str:
                    name_w = avail - len(size_str) - 1
                    if name_w < 4:
                        name_w = avail
                        size_str = ""
                    line = label[:name_w].ljust(name_w) + " " + size_str.rjust(len(size_str))
                    line = line[:avail]
                else:
                    line = label[:avail]

                line = line.ljust(avail)

                is_selected = idx == self.cursor
                if is_selected:
                    attr = curses.color_pair(4) | curses.A_BOLD
                elif n.node_type in ("group", "empty_dir"):
                    attr = curses.color_pair(2)
                else:
                    attr = curses.color_pair(3)

                # Highlight search matches
                if self.search_query and not is_selected:
                    q = self.search_query.lower()
                    if q in n.name.lower() or q in n.full_path.lower():
                        attr |= curses.A_UNDERLINE

                try:
                    stdscr.addnstr(y, 0, "│ ", 2, curses.color_pair(1))
                    stdscr.addnstr(y, 2, line, avail, attr)
                except curses.error:
                    pass
            else:
                try:
                    stdscr.addnstr(y, 0, "│" + " " * (left_w - 1), left_w, curses.color_pair(1))
                except curses.error:
                    pass

            # Divider column
            try:
                stdscr.addnstr(y, left_w, "│", 1, curses.color_pair(1))
            except curses.error:
                pass

            # -- Right pane --
            self._draw_detail_row(stdscr, y, left_w + 1, right_w, row, body_h, node)

            # Right border
            try:
                stdscr.addnstr(y, w - 1, "│", 1, curses.color_pair(1))
            except curses.error:
                pass

        # -- Bottom divider --
        bot_y = 3 + body_h
        if bot_y < h:
            bot = "├" + "─" * (left_w - 1) + "┴" + "─" * (right_w) + "┤"
            stdscr.addnstr(bot_y, 0, bot[:w], w, curses.color_pair(1))

        # -- Status bar --
        status_y = bot_y + 1
        if status_y < h:
            stats = f" {self.total_datasets} datasets  {self.total_groups} groups"
            keys = "hjkl:nav  Tab:pane  /:search  q:quit"
            if self.search_mode:
                keys = f"/{self.search_query}_  Enter:find  Esc:cancel"
            pad = w - 2 - len(stats) - len(keys)
            if pad < 1:
                pad = 1
            status = "│" + stats + " " * pad + keys
            status = status[:w - 1] + "│"
            stdscr.addnstr(status_y, 0, status, w, curses.color_pair(5))

        # -- Bottom border --
        close_y = status_y + 1
        if close_y < h:
            close = "└" + "─" * (w - 2) + "┘"
            stdscr.addnstr(close_y, 0, close[:w], w, curses.color_pair(1))

        stdscr.refresh()

    def _draw_detail_row(self, stdscr, y, x, width, row_idx, body_h, node):
        """Draw one row of the detail pane."""
        detail = self._get_detail(node)
        # Flatten detail sections into lines
        lines = self._detail_lines(detail, width - 2)

        idx = self.detail_scroll + row_idx
        if idx < len(lines):
            text, attr = lines[idx]
        else:
            text, attr = "", curses.color_pair(3)

        text = text[:width - 1].ljust(width - 1)
        try:
            stdscr.addnstr(y, x, text, width - 1, attr)
        except curses.error:
            pass

    def _detail_lines(self, sections, width):
        """Flatten sections into (text, attr) tuples."""
        lines = []
        node = self._selected()
        if node:
            lines.append((f" {node.full_path}", curses.A_BOLD | curses.color_pair(3)))
            lines.append((" " + "─" * min(len(node.full_path) + 2, width - 2), curses.color_pair(1)))
            lines.append(("", 0))

        for section_name, rows in sections:
            lines.append((f" {section_name}", curses.A_BOLD | curses.color_pair(6)))
            # Compute key width for alignment
            key_w = 0
            for k, v in rows:
                if v:  # key-value pair
                    key_w = max(key_w, len(k))
            key_w = min(key_w, width // 2)

            for k, v in rows:
                if v:
                    line = f"   {k:<{key_w}}  {v}"
                else:
                    line = f" {k}"  # raw line (hex preview)
                lines.append((line, curses.color_pair(3)))
            lines.append(("", 0))

        if not lines:
            lines.append((" Select an item to view details", curses.color_pair(1)))

        return lines

    # -- Event loop --

    def run(self, stdscr):
        curses.curs_set(0)
        curses.use_default_colors()
        stdscr.timeout(100)

        while True:
            self.draw(stdscr)
            try:
                ch = stdscr.getch()
            except KeyboardInterrupt:
                break
            if ch == -1:
                continue

            if self.search_mode:
                if ch == 27:  # Esc
                    self.search_mode = False
                    self.search_query = ""
                elif ch in (10, 13):  # Enter
                    self.search_mode = False
                    self.search_next()
                elif ch in (curses.KEY_BACKSPACE, 127, 8):
                    self.search_query = self.search_query[:-1]
                elif 32 <= ch < 127:
                    self.search_query += chr(ch)
                continue

            if ch == ord("q"):
                break
            elif ch == ord("j") or ch == curses.KEY_DOWN:
                self.move_down()
            elif ch == ord("k") or ch == curses.KEY_UP:
                self.move_up()
            elif ch == ord("l") or ch == curses.KEY_RIGHT:
                self.expand()
            elif ch == ord("h") or ch == curses.KEY_LEFT:
                self.collapse()
            elif ch in (10, 13, ord(" ")):  # Enter / Space
                self.toggle_expand()
            elif ch == 9:  # Tab
                self.pane = 1 - self.pane
            elif ch == ord("/"):
                self.search_mode = True
                self.search_query = ""
            elif ch == ord("n"):
                self.search_next()
            elif ch == ord("N"):
                self.search_prev()
            elif ch == ord("g"):
                if self.pane == 0:
                    self.cursor = 0
                else:
                    self.detail_scroll = 0
            elif ch == ord("G"):
                if self.pane == 0:
                    self.cursor = max(0, len(self.visible) - 1)
                else:
                    self.detail_scroll = 999
            elif ch == curses.KEY_NPAGE:
                h, _ = stdscr.getmaxyx()
                self.page_down(h - 5)
            elif ch == curses.KEY_PPAGE:
                h, _ = stdscr.getmaxyx()
                self.page_up(h - 5)
            elif ch == curses.KEY_RESIZE:
                stdscr.clear()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def browse_archive(h5_path):
    h5_path = os.path.expanduser(h5_path)
    if not os.path.exists(h5_path):
        print(f"Error: archive '{h5_path}' not found.", file=sys.stderr)
        sys.exit(1)
    try:
        h5f = h5py.File(h5_path, "r")
    except OSError as e:
        print(f"Error: cannot open '{h5_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Detect format
    is_bagit = h5f.attrs.get("har_format", None) == "bagit-v1"
    if is_bagit:
        tree = _build_tree_bagit(h5f)
    else:
        tree = _build_tree_legacy(h5f)

    app = BrowseApp(h5f, tree, is_bagit)
    app.archive_name = os.path.basename(h5_path)

    # Expand first level
    for child in tree.children:
        child.expanded = True
    app._rebuild_visible()
    if app.visible:
        app.cursor = 0

    def _run(stdscr):
        app.run(stdscr)

    try:
        curses.wrapper(_run)
    finally:
        h5f.close()
