# loss/dist_loss.py
"""
Relational knowledge distillation losses operating on similarity matrices.

Given vision CLS embeddings V (B×D_v) and text embeddings T (B×D_t):
  1. Compute vision self-similarity:  S_v = V_norm @ V_norm^T   (B×B)
  2. Compute text  self-similarity:  S_t = T_norm @ T_norm^T   (B×B)
  3. Penalise divergence between S_v and S_t.

No projection layer is needed — the loss compares relational structure
within each modality independently, so dimension mismatch is fine.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ──────────────────────────────────────────────────────────────────

def _similarity_matrices(vision_emb: torch.Tensor, text_emb: torch.Tensor):
    """
    L2-normalise both inputs, compute self-similarity matrices,
    and return them with a mask that excludes the diagonal.
    """
    V = F.normalize(vision_emb, dim=-1)          # (B, D_v)
    T = F.normalize(text_emb.float(), dim=-1)    # (B, D_t)

    S_v = torch.mm(V, V.T)                       # (B, B)
    S_t = torch.mm(T, T.T)                       # (B, B)

    B = S_v.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=V.device)  # exclude diagonal

    return S_v, S_t, mask


# ── Loss classes ─────────────────────────────────────────────────────────────

class SimilarityMatrixL1Loss(nn.Module):
    """L = mean(|S_v[i,j] - S_t[i,j]|)  over off-diagonal cells."""

    def forward(self, vision_embeddings: torch.Tensor,
                text_embeddings: torch.Tensor) -> torch.Tensor:
        S_v, S_t, mask = _similarity_matrices(vision_embeddings, text_embeddings)
        return (S_v[mask] - S_t[mask]).abs().mean()


class SimilarityMatrixMSELoss(nn.Module):
    """L = mean((S_v[i,j] - S_t[i,j])²)  over off-diagonal cells."""

    def forward(self, vision_embeddings: torch.Tensor,
                text_embeddings: torch.Tensor) -> torch.Tensor:
        S_v, S_t, mask = _similarity_matrices(vision_embeddings, text_embeddings)
        return (S_v[mask] - S_t[mask]).pow(2).mean()


class SimilarityMatrixHuberLoss(nn.Module):
    """
    Huber loss on similarity matrices.

    Quadratic for |S_v - S_t| ≤ delta, linear beyond.
    Provides MSE precision for small errors with robustness for outliers.

    Args:
        delta (float): transition threshold.  Default: 0.3
    """

    def __init__(self, delta: float = 0.3):
        super().__init__()
        self.delta = delta

    def forward(self, vision_embeddings: torch.Tensor,
                text_embeddings: torch.Tensor) -> torch.Tensor:
        S_v, S_t, mask = _similarity_matrices(vision_embeddings, text_embeddings)
        diff = (S_v[mask] - S_t[mask]).abs()
        delta = self.delta
        loss = torch.where(
            diff <= delta,
            0.5 * diff.pow(2),
            delta * (diff - 0.5 * delta),
        )
        return loss.mean()


class SimilarityMatrixKLLoss(nn.Module):
    """
    Row-wise KL divergence between softmax'd similarity rows.

    P_i = softmax(S_t[i, :] / tau)   ← text  (target distribution)
    Q_i = softmax(S_v[i, :] / tau)   ← vision (predicted distribution)
    L   = mean_i  KL(P_i || Q_i)

    Args:
        tau (float): temperature.  Default: 0.07
    """

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(self, vision_embeddings: torch.Tensor,
                text_embeddings: torch.Tensor) -> torch.Tensor:
        S_v, S_t, mask = _similarity_matrices(vision_embeddings, text_embeddings)

        # Mask out diagonal by setting it to -inf before softmax
        B = S_v.size(0)
        neg_inf = torch.finfo(S_v.dtype).min
        diag_mask = torch.eye(B, dtype=torch.bool, device=S_v.device)

        S_v_masked = S_v.masked_fill(diag_mask, neg_inf) / self.tau
        S_t_masked = S_t.masked_fill(diag_mask, neg_inf) / self.tau

        log_q = F.log_softmax(S_v_masked, dim=1)   # vision  (log-input)
        p     = F.softmax(S_t_masked, dim=1)        # text    (target)

        # F.kl_div expects (log-input, target)
        loss = F.kl_div(log_q, p, reduction='batchmean')
        return loss


# ── Factory ──────────────────────────────────────────────────────────────────

def build_dist_loss(cfg):
    """
    Instantiate the appropriate similarity-matrix loss from config.

    Reads:
        cfg.MODEL.TEXT_LOSS_TYPE  — "l1", "mse", "huber", or "kl"
        cfg.MODEL.TEXT_LOSS_HYPR  — tau (KL) or delta (Huber); ignored for L1/MSE
    """
    loss_type = cfg.MODEL.TEXT_LOSS_TYPE.lower()
    hypr = cfg.MODEL.TEXT_LOSS_HYPR

    if loss_type == "l1":
        return SimilarityMatrixL1Loss()
    elif loss_type == "mse":
        return SimilarityMatrixMSELoss()
    elif loss_type == "huber":
        return SimilarityMatrixHuberLoss(delta=hypr)
    elif loss_type == "kl":
        return SimilarityMatrixKLLoss(tau=hypr)
    else:
        raise ValueError(
            f"Unknown dist_loss type '{loss_type}'. "
            f"Expected one of: l1, mse, huber, kl"
        )


def build_distill_loss(cfg):
    """
    Instantiate the similarity-matrix loss for DeiT-style distillation.

    Reads:
        cfg.DISTILL.MATRIX_LOSS_TYPE  — "l1", "l2"/"mse", or "huber"
        cfg.DISTILL.MATRIX_LOSS_HYPR  — delta (Huber); ignored for L1/L2
    """
    loss_type = cfg.DISTILL.MATRIX_LOSS_TYPE.lower()
    hypr = cfg.DISTILL.MATRIX_LOSS_HYPR

    if loss_type == "l1":
        return SimilarityMatrixL1Loss()
    elif loss_type in ("l2", "mse"):
        return SimilarityMatrixMSELoss()
    elif loss_type == "huber":
        return SimilarityMatrixHuberLoss(delta=hypr)
    else:
        raise ValueError(
            f"Unknown distill matrix loss type '{loss_type}'. "
            f"Expected one of: l1, l2, mse, huber"
        )
