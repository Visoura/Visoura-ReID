# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .text_align_loss import InfoNCETextAlignLoss, CosineAlignLoss, TextCenterLoss
from .dist_loss import build_dist_loss, build_distill_loss
from .koleo_loss import KoLeoLoss
from .supcon_loss import SupervisedContrastiveLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    elif 'supcon' in cfg.MODEL.METRIC_LOSS_TYPE:
        supcon = SupervisedContrastiveLoss(temperature=cfg.MODEL.SUPCON_TEMPERATURE)
        print("using Supervised Contrastive Loss")
    else:
        print('expected METRIC_LOSS_TYPE should be triplet or supcon '
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.USE_KOLEO_LOSS:
        koleo = KoLeoLoss()
        print("using KoLeo Loss")

    use_gram_anchor = cfg.MODEL.USE_GRAM_ANCHOR_LOSS
    if use_gram_anchor:
        from .gram_anchor_loss import GramAnchorLoss
        gram_criterion = GramAnchorLoss(
            student_dim=cfg.MODEL.GRAM_ANCHOR_STUDENT_DIM,
            teacher_dim=cfg.MODEL.GRAM_ANCHOR_TEACHER_DIM,
        )
        gram_weight = cfg.MODEL.GRAM_ANCHOR_LOSS_WEIGHT
        print(f"Gram Anchor Loss is enabled with weight: {gram_weight}")

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    # ── Text alignment loss ─────────────────────────────────────────
    use_proj_head = cfg.MODEL.TEXT_PROJ_HEAD

    if use_proj_head:
        # Projector-based losses (InfoNCE / TextCenter / Cosine)
        if cfg.MODEL.TEXT_ALIGN_LOSS == "infonce":
            text_loss_fn = InfoNCETextAlignLoss(temperature=cfg.MODEL.TEXT_ALIGN_TEMPERATURE)
            print("using InfoNCE text alignment loss (projector mode)")
        elif cfg.MODEL.TEXT_ALIGN_LOSS == "text_center":
            text_loss_fn = TextCenterLoss(normalize=True)
            print("using TextCenter text alignment loss (projector mode)")
        else:
            text_loss_fn = CosineAlignLoss()
            print("using Cosine text alignment loss (projector mode)")
        text_loss_weight = cfg.MODEL.TEXT_ALIGN_WEIGHT
    else:
        # Similarity-matrix dist_loss family (no projector needed)
        dist_loss_fn = build_dist_loss(cfg)
        text_loss_weight = cfg.MODEL.TEXT_LOSS_WEIGHT
        print(f"using dist_loss: {cfg.MODEL.TEXT_LOSS_TYPE} "
              f"(weight={text_loss_weight}, hypr={cfg.MODEL.TEXT_LOSS_HYPR})")

    # ── Distillation loss (DeiT-style dual-token) ──────────────────
    distill_enabled = cfg.DISTILL.ENABLED
    distill_loss_fn = None
    teacher_emb_cache = None
    lambda_distill = 0.0
    if distill_enabled:
        import os
        emb_path = cfg.DISTILL.TEACHER_EMBEDDINGS_PATH
        if not os.path.isfile(emb_path):
            raise FileNotFoundError(
                f"Teacher embeddings not found at '{emb_path}'. "
                f"Run scripts/generate_teacher_embeddings.py first."
            )
        teacher_emb_cache = torch.load(emb_path, map_location='cpu', weights_only=False)
        distill_loss_fn = build_distill_loss(cfg)
        lambda_distill = cfg.DISTILL.LAMBDA
        print(f"Distillation enabled: loss={cfg.DISTILL.MATRIX_LOSS_TYPE}, "
              f"lambda={lambda_distill}, teacher embeddings={len(teacher_emb_cache)} images")

    def _compute_text_loss(text_proj, text_embs, target, vision_feat=None):
        """Compute text alignment loss.
        When TEXT_PROJ_HEAD=True:  uses text_proj (projected visual feats).
        When TEXT_PROJ_HEAD=False: uses vision_feat (raw CLS) + dist_loss.
        Returns (weighted_text_loss_tensor, raw_text_loss_float) or (0, 0.0).
        """
        if text_embs is None:
            return 0, 0.0

        # Check we have valid (non-zero) text embeddings
        valid = text_embs.abs().sum(dim=-1) > 1e-6
        if not valid.any():
            return 0, 0.0

        if use_proj_head:
            # Projector path: need text_proj from the model
            if text_proj is None:
                return 0, 0.0
            t_loss = text_loss_fn(
                text_proj[valid],
                text_embs[valid],
                target[valid]
            )
        else:
            # Dist-loss path: use raw vision features
            if vision_feat is None:
                return 0, 0.0
            t_loss = dist_loss_fn(
                vision_feat[valid],
                text_embs[valid],
            )

        return text_loss_weight * t_loss, t_loss.detach().item()

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target, target_cam,
                      text_proj=None, text_embs=None, student_tokens=None, teacher_tokens=None,
                      dist_output=None, img_paths=None):
            id_loss = F.cross_entropy(score, target)
            text_weighted, text_raw = _compute_text_loss(
                text_proj, text_embs, target, vision_feat=feat)
            total = id_loss + text_weighted
            loss_dict = {
                "id_loss": id_loss.detach().item(),
                "text_align_loss": text_raw,
            }
            if cfg.MODEL.USE_KOLEO_LOSS:
                vf = feat[0] if isinstance(feat, list) else feat
                k_loss = koleo(vf, target)
                total += cfg.MODEL.KOLEO_LOSS_WEIGHT * k_loss
                loss_dict["koleo_loss"] = k_loss.detach().item()

            if use_gram_anchor and student_tokens is not None and teacher_tokens is not None:
                gram_loss = gram_criterion(student_tokens, teacher_tokens)
                total += gram_weight * gram_loss
                loss_dict["gram_loss"] = gram_loss.detach().item()

            # ── Distillation loss ─────────────────────────────────
            if distill_enabled and dist_output is not None and img_paths is not None:
                teacher_batch = torch.stack(
                    [teacher_emb_cache[p] for p in img_paths]
                ).to(dist_output.device)
                d_loss = distill_loss_fn(dist_output, teacher_batch)
                total = total + lambda_distill * d_loss
                loss_dict["distill_matrix_loss"] = d_loss.detach().item()

            return total, loss_dict

    #  elif cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'id_triplet', 'img_triplet']:
    elif 'triplet' in sampler:
        def loss_func(score, feat, target, target_cam,
                      text_proj=None, text_embs=None, student_tokens=None, teacher_tokens=None,
                      dist_output=None, img_paths=None):
            # Compute ID_LOSS first
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                if isinstance(score, list):
                    ID_LOSS = [xent(scor, target) for scor in score[1:]]
                    ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                else:
                    ID_LOSS = xent(score, target)
            else:
                if isinstance(score, list):
                    ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                    ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

            # Compute Metric Loss
            vf = feat[0] if isinstance(feat, list) else feat
            
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if isinstance(feat, list):
                    TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                    TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                    TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                else:
                    TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]
                base_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                metric_loss_dict = {"triplet_loss": TRI_LOSS.detach().item()}
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'supcon':
                SUP_LOSS = supcon(vf, target)
                base_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.SUPCON_LOSS_WEIGHT * SUP_LOSS
                metric_loss_dict = {"supcon_loss": SUP_LOSS.detach().item()}
            else:
                print('expected METRIC_LOSS_TYPE should be triplet or supcon '
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

            text_weighted, text_raw = _compute_text_loss(
                text_proj, text_embs, target, vision_feat=vf)
            total = base_loss + text_weighted
            
            if cfg.MODEL.USE_KOLEO_LOSS:
                k_loss = koleo(vf, target)
                total += cfg.MODEL.KOLEO_LOSS_WEIGHT * k_loss
                metric_loss_dict["koleo_loss"] = k_loss.detach().item()

            if use_gram_anchor and student_tokens is not None and teacher_tokens is not None:
                gram_loss = gram_criterion(student_tokens, teacher_tokens)
                total += gram_weight * gram_loss
                metric_loss_dict["gram_loss"] = gram_loss.detach().item()

            # ── Distillation loss ─────────────────────────────────
            if distill_enabled and dist_output is not None and img_paths is not None:
                teacher_batch = torch.stack(
                    [teacher_emb_cache[p] for p in img_paths]
                ).to(dist_output.device)
                d_loss = distill_loss_fn(dist_output, teacher_batch)
                total = total + lambda_distill * d_loss
                metric_loss_dict["distill_matrix_loss"] = d_loss.detach().item()

            loss_dict = {
                "id_loss": ID_LOSS.detach().item(),
                "text_align_loss": text_raw,
                **metric_loss_dict
            }
            return total, loss_dict

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
