"""
Market-1501 Collage Generator for Qwen3-VL
Reads picker_progress.json and creates dynamically sized collages per person ID.

Layout strategy (Base image 2x scaled to 128 W x 256 H, Separators = 32px):
  1 Image  -> 128x256 canvas
  2 Images -> 1x2 side-by-side -> 288x256 canvas
  3 Images -> 2x2 grid (1 empty) -> 288x544 canvas
  4 Images -> 2x2 grid -> 288x544 canvas
"""

import os
import sys
import json
from PIL import Image

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GT_BBOX_DIR = os.path.join(BASE_DIR, "Market-1501-v15.09.15", "gt_bbox")
PROGRESS_FILE = os.path.join(BASE_DIR, "picker_progress.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "collages_qwen")

# Qwen3-VL Specific Sizing
SCALE_FACTOR = 2
BASE_W = 64 * SCALE_FACTOR   # 128 pixels
BASE_H = 128 * SCALE_FACTOR  # 256 pixels
SEP = 32                     # 32-pixel separator to prevent patch blending
BG_COLOR = (40, 40, 40)      # Dark gray background to act as a clear border


def load_progress() -> dict[str, list[str]]:
    if not os.path.isfile(PROGRESS_FILE):
        print(f"[ERROR] Progress file not found: {PROGRESS_FILE}")
        sys.exit(1)
    with open(PROGRESS_FILE, "r") as f:
        data = json.load(f)
    return data


def make_collage(image_paths: list[str]) -> Image.Image:
    """
    Create a dynamically sized collage ensuring all dimensions are multiples of 32.
    Uses exact 2x scaling and 32px borders.
    """
    n = len(image_paths)
    
    # Determine canvas size and paste positions based on image count
    if n == 1:
        canvas_w = BASE_W
        canvas_h = BASE_H
        positions = [(0, 0)]
        
    elif n == 2:
        canvas_w = (BASE_W * 2) + SEP
        canvas_h = BASE_H
        positions = [
            (0, 0),                 # Left image
            (BASE_W + SEP, 0)       # Right image
        ]
        
    else: # n == 3 or n == 4
        canvas_w = (BASE_W * 2) + SEP
        canvas_h = (BASE_H * 2) + SEP
        positions = [
            (0, 0),                             # Top-left
            (BASE_W + SEP, 0),                  # Top-right
            (0, BASE_H + SEP),                  # Bottom-left
            (BASE_W + SEP, BASE_H + SEP)        # Bottom-right
        ]

    # Create the canvas using the dark gray background
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)

    # Resize and paste images
    for pos, path in zip(positions, image_paths):
        img = Image.open(path).convert("RGB")
        # Straight 2x upscale using LANCZOS for high-quality resampling
        img = img.resize((BASE_W, BASE_H), Image.LANCZOS)
        canvas.paste(img, pos)

    return canvas


def main():
    selections = load_progress()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    skipped = 0

    for pid, filenames in sorted(selections.items()):
        if not filenames:
            skipped += 1
            continue

        paths = [os.path.join(GT_BBOX_DIR, fn) for fn in filenames]

        # Verify files exist
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            print(f"[WARN] ID {pid}: missing files {missing}, skipping.")
            skipped += 1
            continue

        # Cap at 4 images just in case the JSON has more
        if len(paths) > 4:
            paths = paths[:4]

        collage = make_collage(paths)
        out_path = os.path.join(OUTPUT_DIR, f"{pid}.jpg")
        
        # Save at 100 quality to prevent JPEG compression artifacts before inference
        collage.save(out_path, quality=100)
        total += 1

    print(f"[DONE] Generated {total} Qwen3-ready collages in: {OUTPUT_DIR}")
    if skipped:
        print(f"       Skipped {skipped} IDs (no selection or missing files).")


if __name__ == "__main__":
    main()