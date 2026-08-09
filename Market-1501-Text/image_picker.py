"""
Market-1501 Image Picker GUI
Hand-pick the best images (max 4) for each person ID from gt_bbox.
Progress is auto-saved to a JSON file after every change.
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from collections import OrderedDict
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────
GT_BBOX_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Market-1501-v15.09.15", "gt_bbox"
)
PROGRESS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "picker_progress.json"
)
MAX_SELECTED = 4
THUMB_SIZE = (128, 256)       # width × height for thumbnails
COLS = 8                       # thumbnails per row

# ─── Colours / Style ─────────────────────────────────────────────────────────
BG           = "#1e1e2e"
FG           = "#cdd6f4"
ACCENT       = "#89b4fa"
SELECTED_BDR = "#a6e3a1"
UNSELECTED_BDR = "#45475a"
BTN_BG       = "#313244"
BTN_FG       = "#cdd6f4"
BTN_HOVER    = "#585b70"


# ═══════════════════════════════════════════════════════════════════════════════
class ImagePickerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Market-1501 Image Picker")
        self.root.configure(bg=BG)
        self.root.geometry("1200x850")
        self.root.minsize(900, 600)

        # ── data ──────────────────────────────────────────────────────────────
        self.id_images: OrderedDict[str, list[str]] = OrderedDict()
        self._scan_images()
        self.all_ids = list(self.id_images.keys())
        self.current_idx = 0

        # ── persisted state: {id: [filename, ...]} ───────────────────────────
        self.selections: dict[str, list[str]] = {}
        self._load_progress()

        # ── runtime state ─────────────────────────────────────────────────────
        self.photo_refs: list[ImageTk.PhotoImage] = []  # prevent GC

        # ── build UI ──────────────────────────────────────────────────────────
        self._build_top_bar()
        self._build_image_area()
        self._build_bottom_bar()

        # ── keyboard shortcuts ────────────────────────────────────────────────
        self.root.bind("<Left>",  lambda e: self._go_prev())
        self.root.bind("<Right>", lambda e: self._go_next())

        # ── show first ID ─────────────────────────────────────────────────────
        self._jump_to_first_unfinished()
        self._show_current_id()

    # ──────────────────────────────────────────────────────────────────────────
    # Data helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _scan_images(self):
        """Group gt_bbox filenames by person ID."""
        for fname in sorted(os.listdir(GT_BBOX_DIR)):
            if not fname.lower().endswith(".jpg"):
                continue
            pid = fname.split("_")[0]     # e.g. "0001"
            self.id_images.setdefault(pid, []).append(fname)
        if not self.id_images:
            messagebox.showerror("Error", f"No images found in:\n{GT_BBOX_DIR}")
            sys.exit(1)

    def _load_progress(self):
        if os.path.isfile(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, "r") as f:
                    self.selections = json.load(f)
                # find the first ID that has not been handled yet
            except (json.JSONDecodeError, IOError):
                self.selections = {}

    def _save_progress(self):
        with open(PROGRESS_FILE, "w") as f:
            json.dump(self.selections, f, indent=2)

    def _jump_to_first_unfinished(self):
        """Start from the first ID that has no selections yet."""
        for i, pid in enumerate(self.all_ids):
            if pid not in self.selections:
                self.current_idx = i
                return

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────────
    def _build_top_bar(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=16, pady=(12, 4))

        self.lbl_id = tk.Label(
            bar, text="", font=("Inter", 22, "bold"),
            bg=BG, fg=ACCENT
        )
        self.lbl_id.pack(side="left")

        self.lbl_progress = tk.Label(
            bar, text="", font=("Inter", 13),
            bg=BG, fg=FG
        )
        self.lbl_progress.pack(side="left", padx=20)

        self.lbl_sel_count = tk.Label(
            bar, text="", font=("Inter", 13),
            bg=BG, fg=SELECTED_BDR
        )
        self.lbl_sel_count.pack(side="right")

        # ── search / jump ─────────────────────────────────────────────────
        tk.Label(bar, text="Go to ID:", font=("Inter", 12), bg=BG, fg=FG
                 ).pack(side="right", padx=(20, 4))
        self.entry_jump = tk.Entry(bar, width=8, font=("Inter", 12),
                                   bg=BTN_BG, fg=FG, insertbackground=FG,
                                   relief="flat", bd=2)
        self.entry_jump.pack(side="right")
        self.entry_jump.bind("<Return>", lambda e: self._jump_to_id())

    def _build_image_area(self):
        container = tk.Frame(self.root, bg=BG)
        container.pack(fill="both", expand=True, padx=16, pady=4)

        self.canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical",
                            command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner_frame = tk.Frame(self.canvas, bg=BG)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.inner_frame, anchor="nw"
        )

        self.inner_frame.bind("<Configure>",
                              lambda e: self.canvas.configure(
                                  scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # mouse‑wheel scrolling
        self.canvas.bind_all("<Button-4>",
                             lambda e: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>",
                             lambda e: self.canvas.yview_scroll(3, "units"))

    def _on_canvas_resize(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _build_bottom_bar(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=16, pady=(4, 12))

        style_kw = dict(font=("Inter", 13, "bold"), bg=BTN_BG, fg=BTN_FG,
                        activebackground=BTN_HOVER, activeforeground=FG,
                        relief="flat", bd=0, padx=18, pady=8, cursor="hand2")

        self.btn_prev = tk.Button(bar, text="◀  Prev", command=self._go_prev,
                                  **style_kw)
        self.btn_prev.pack(side="left")

        self.btn_next = tk.Button(bar, text="Next  ▶", command=self._go_next,
                                  **style_kw)
        self.btn_next.pack(side="left", padx=12)

        self.btn_clear = tk.Button(bar, text="Clear selection",
                                   command=self._clear_selection, **style_kw)
        self.btn_clear.pack(side="left", padx=12)

        # ── overall progress ──────────────────────────────────────────────
        self.lbl_overall = tk.Label(
            bar, text="", font=("Inter", 12), bg=BG, fg=FG
        )
        self.lbl_overall.pack(side="right")

    # ──────────────────────────────────────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────────────────────────────────────
    def _show_current_id(self):
        pid = self.all_ids[self.current_idx]
        images = self.id_images[pid]
        selected = set(self.selections.get(pid, []))

        # update labels
        self.lbl_id.config(text=f"ID: {pid}")
        self.lbl_progress.config(
            text=f"{self.current_idx + 1} / {len(self.all_ids)} IDs"
        )
        finished = sum(1 for v in self.selections.values() if v)
        self.lbl_overall.config(
            text=f"Completed: {finished} / {len(self.all_ids)}"
        )
        self._update_sel_count(pid)

        # clear previous thumbnails
        for w in self.inner_frame.winfo_children():
            w.destroy()
        self.photo_refs.clear()

        # render thumbnails
        for i, fname in enumerate(images):
            row, col = divmod(i, COLS)
            self._make_thumb(pid, fname, row, col, fname in selected)

        # scroll to top
        self.canvas.yview_moveto(0)

    def _make_thumb(self, pid: str, fname: str, row: int, col: int,
                    is_selected: bool):
        path = os.path.join(GT_BBOX_DIR, fname)
        img = Image.open(path).resize(THUMB_SIZE, Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.photo_refs.append(photo)

        border_col = SELECTED_BDR if is_selected else UNSELECTED_BDR
        border_width = 4 if is_selected else 2

        frame = tk.Frame(self.inner_frame, bg=border_col,
                         bd=border_width, relief="solid")
        frame.grid(row=row, column=col, padx=6, pady=6)

        lbl = tk.Label(frame, image=photo, bg=BG, cursor="hand2")
        lbl.pack()

        # filename label below image
        name_lbl = tk.Label(frame, text=fname, font=("Inter", 8),
                            bg=border_col, fg=BG if is_selected else FG)
        name_lbl.pack()

        # click handler
        def on_click(event, _fname=fname, _frame=frame, _name_lbl=name_lbl):
            self._toggle_image(pid, _fname, _frame, _name_lbl)

        lbl.bind("<Button-1>", on_click)
        name_lbl.bind("<Button-1>", on_click)

    def _toggle_image(self, pid, fname, frame, name_lbl):
        sel = self.selections.setdefault(pid, [])

        if fname in sel:
            sel.remove(fname)
            frame.config(bg=UNSELECTED_BDR, bd=2)
            name_lbl.config(bg=UNSELECTED_BDR, fg=FG)
        else:
            if len(sel) >= MAX_SELECTED:
                messagebox.showwarning(
                    "Limit reached",
                    f"You can select at most {MAX_SELECTED} images per ID."
                )
                return
            sel.append(fname)
            frame.config(bg=SELECTED_BDR, bd=4)
            name_lbl.config(bg=SELECTED_BDR, fg=BG)

        self._update_sel_count(pid)
        self._save_progress()

    def _update_sel_count(self, pid):
        n = len(self.selections.get(pid, []))
        self.lbl_sel_count.config(
            text=f"Selected: {n} / {MAX_SELECTED}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────────────────────────────────────
    def _go_prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_current_id()

    def _go_next(self):
        if self.current_idx < len(self.all_ids) - 1:
            self.current_idx += 1
            self._show_current_id()

    def _jump_to_id(self):
        target = self.entry_jump.get().strip().zfill(4)
        if target in self.id_images:
            self.current_idx = self.all_ids.index(target)
            self._show_current_id()
            self.entry_jump.delete(0, "end")
        else:
            messagebox.showwarning("Not found", f"ID '{target}' not found.")

    def _clear_selection(self):
        pid = self.all_ids[self.current_idx]
        self.selections[pid] = []
        self._save_progress()
        self._show_current_id()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()

    # try to use a nicer ttk theme
    style = ttk.Style()
    available = style.theme_names()
    for theme in ("clam", "alt", "default"):
        if theme in available:
            style.theme_use(theme)
            break

    app = ImagePickerApp(root)
    root.mainloop()
