"""
Per-image SigLIP2 text embedding generation.

Reads a CSV with per-image captions, encodes each caption with the
SigLIP2 text encoder, L2-normalises, and saves a single .pt dict:

    { "0002_c1s1_000451_03.jpg": tensor(768,), ... }

Usage:
    python scripts/save_text_embeddings.py \
        --csv   path/to/market1501_annotations_master.csv \
        --output data/market1501_text_embeddings.pt \
        --batch-size 64 \
        --model-id google/siglip2-base-patch16-224
"""

import argparse
import csv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-image SigLIP2 text embeddings from CSV captions"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to CSV with 'filename' and 'summary_description' columns"
    )
    parser.add_argument(
        "--output", default="data/market1501_text_embeddings.pt",
        help="Output .pt path"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for text encoder inference"
    )
    parser.add_argument(
        "--model-id", default="google/siglip2-base-patch16-224",
        help="HuggingFace model ID for SigLIP2"
    )
    args = parser.parse_args()

    # ── Load CSV ──────────────────────────────────────────────────────
    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    filenames = []
    captions = []
    skipped = []

    for row in rows:
        fname = row.get("filename", "").strip()
        caption = row.get("summary_description", "").strip()
        if not fname:
            continue
        if not caption:
            skipped.append(fname)
            continue
        filenames.append(fname)
        captions.append(caption)

    print(f"CSV loaded: {len(filenames)} valid rows, {len(skipped)} skipped (missing caption)")

    # ── Load model ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id).to(device)
    model.eval()

    # ── Batched inference ─────────────────────────────────────────────
    embeddings = {}
    bs = args.batch_size

    for start in range(0, len(filenames), bs):
        end = min(start + bs, len(filenames))
        batch_fnames = filenames[start:end]
        batch_captions = captions[start:end]

        inputs = tokenizer(
            batch_captions, padding=True, truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)       # (B, 768)
            outputs = F.normalize(outputs, dim=-1)            # L2-normalise

        for fname, emb in zip(batch_fnames, outputs.cpu()):
            embeddings[fname] = emb  # shape: (768,)

        if (start // bs) % 10 == 0:
            print(f"  Processed {end}/{len(filenames)} ...")

    # ── Save ──────────────────────────────────────────────────────────
    torch.save(embeddings, args.output)

    print(f"\n{'='*50}")
    print(f"Saved {len(embeddings)} per-image text embeddings → {args.output}")
    print(f"Sample keys: {list(embeddings.keys())[:5]}")
    if skipped:
        print(f"Skipped {len(skipped)} filenames (missing caption):")
        for s in skipped[:10]:
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")


if __name__ == "__main__":
    main()
