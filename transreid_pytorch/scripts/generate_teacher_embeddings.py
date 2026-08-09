#!/usr/bin/env python3
"""
Generate teacher embeddings for DeiT-style distillation.

Runs a frozen PersonViT-B/16 teacher once over the Market-1501 training set
and caches its CLS token embeddings (pre-classifier) to a .pt file, keyed by
the full image path — the same key used by the training dataloader.

This script is run ONCE before training, separately from the training loop.

Usage:
    python scripts/generate_teacher_embeddings.py \
        --config_file configs/market/distill_person_vit.yml \
        --output scripts/teacher_embeddings_b16.pt \
        --view deterministic \
        --batch_size 128
"""
import argparse
import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image

# ── Ensure project root is on sys.path ──────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..')
sys.path.insert(0, _PROJECT_ROOT)

from config import cfg
from model.backbones.vit_pytorch import vit_base_patch16_224_TransReID

# Conditional tqdm import (graceful fallback)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        total = kwargs.get('total', '?')
        for i, item in enumerate(iterable):
            if i % 100 == 0:
                print(f'{desc}: {i}/{total}')
            yield item


def _build_dataset(cfg):
    """Instantiate the Market-1501 dataset and return the training list."""
    from datasets.market1501 import Market1501
    from datasets.msmt17 import MSMT17
    from datasets.dukemtmcreid import DukeMTMCreID

    __factory = {
        'market1501': Market1501,
        'msmt17': MSMT17,
        'dukemtmc': DukeMTMCreID,
    }
    dataset_name = cfg.DATASETS.NAMES
    if dataset_name not in __factory:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         f"Available: {list(__factory.keys())}")
    dataset = __factory[dataset_name](root=cfg.DATASETS.ROOT_DIR)
    return dataset.train  # list of (img_path, pid, camid, trackid)


def _build_transform(cfg, view):
    """Build the image transform for the teacher."""
    if view == 'deterministic':
        return T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])
    elif view == 'augmented':
        from timm.data.random_erasing import RandomErasing
        return T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel',
                          max_count=1, device='cpu'),
        ])
    else:
        raise ValueError(f"Unknown view mode '{view}'. Use 'deterministic' or 'augmented'.")


def main():
    parser = argparse.ArgumentParser(
        description='Generate teacher embeddings for DeiT-style distillation')
    parser.add_argument('--config_file', required=True,
                        help='Path to config YAML (to read dataset/teacher paths)')
    parser.add_argument('--output', default='',
                        help='Output .pt path (default: scripts/teacher_embeddings_b16.pt)')
    parser.add_argument('--view', default='deterministic',
                        choices=['deterministic', 'augmented'],
                        help='Transform mode for teacher inference')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    parser.add_argument("opts", help="Modify config options", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    # ── Resolve teacher checkpoint path ──────────────────────────────
    teacher_ckpt = cfg.DISTILL.TEACHER_CHECKPOINT_PATH
    if not teacher_ckpt:
        raise ValueError("DISTILL.TEACHER_CHECKPOINT_PATH is empty in config.")
    print(f"Teacher checkpoint : {teacher_ckpt}")

    # ── Load dataset ─────────────────────────────────────────────────
    train_data = _build_dataset(cfg)
    img_paths = [item[0] for item in train_data]
    print(f"Training images    : {len(img_paths)}")

    # ── Build teacher model (B/16) ───────────────────────────────────
    teacher = vit_base_patch16_224_TransReID(
        img_size=cfg.INPUT.SIZE_TRAIN,
        stride_size=cfg.MODEL.STRIDE_SIZE,
        drop_path_rate=0.0,
        camera=0,
        view=0,
        local_feature=False,
    )
    teacher.load_param(teacher_ckpt, hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
    teacher = teacher.to(args.device).eval()
    print("Teacher model loaded and set to eval mode.")

    # ── Build transform ──────────────────────────────────────────────
    transform = _build_transform(cfg, args.view)

    # ── Extract embeddings ───────────────────────────────────────────
    embeddings = {}
    num_batches = (len(img_paths) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches),
                              desc='Extracting teacher embeddings',
                              total=num_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(img_paths))
            batch_paths = img_paths[start:end]

            batch_imgs = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                batch_imgs.append(transform(img))
            batch_tensor = torch.stack(batch_imgs).to(args.device)

            # Forward through teacher — get CLS output (pre-classifier)
            # TransReID.forward_features returns (cls_feat, patch_tokens)
            cls_out, _ = teacher(batch_tensor)
            cls_out = cls_out.cpu()

            for j, p in enumerate(batch_paths):
                embeddings[p] = cls_out[j]

    # ── Save ─────────────────────────────────────────────────────────
    output_path = args.output or os.path.join(_SCRIPT_DIR,
                                               'teacher_embeddings_b16.pt')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(embeddings, output_path)

    # ── Summary ──────────────────────────────────────────────────────
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    sample_key = next(iter(embeddings))
    print(f"\n{'=' * 60}")
    print(f"Teacher embeddings generated successfully!")
    print(f"  Images embedded : {len(embeddings)}")
    print(f"  Embedding dim   : {embeddings[sample_key].shape[0]}")
    print(f"  Output path     : {os.path.abspath(output_path)}")
    print(f"  File size       : {file_size_mb:.2f} MB")
    print(f"  View mode       : {args.view}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
