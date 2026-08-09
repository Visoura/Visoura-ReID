# loss/text_align_loss.py
"""
Cross-modal alignment losses for VLM text-embedding integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCETextAlignLoss(nn.Module):
    """
    Cross-modal InfoNCE loss.

    For each image in the batch, treats all samples sharing the same PID
    as positives against the full batch of text embeddings as negatives.

    Args:
        temperature (float): softmax temperature. Default: 0.07
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        visual_proj: torch.Tensor,   # (B, D) — projected visual feats
        text_embs:   torch.Tensor,   # (B, D) — SigLIP2 text embs
        pids:        torch.Tensor,   # (B,)   — integer person IDs
    ) -> torch.Tensor:

        # Normalize both modalities onto the unit hypersphere
        visual_proj = F.normalize(visual_proj, dim=-1)
        text_embs   = F.normalize(text_embs.float(), dim=-1)

        # Pairwise cosine similarity matrix: (B, B)
        logits = torch.matmul(visual_proj, text_embs.T) / self.temperature

        # Boolean positive mask: True where PIDs match
        pid_eq = pids.unsqueeze(1) == pids.unsqueeze(0)   # (B, B)

        # Log-softmax over all columns (full row), averaged over positives
        log_softmax = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        num_positives = pid_eq.float().sum(dim=1).clamp(min=1.0)  # (B,)
        loss = -(log_softmax * pid_eq.float()).sum(dim=1) / num_positives

        return loss.mean()


class CosineAlignLoss(nn.Module):
    """
    Simple cosine alignment loss — useful as a warm-up alternative or
    for debugging. Minimises 1 - cos(visual_proj, text_emb) per sample.
    """
    def forward(
        self,
        visual_proj: torch.Tensor,
        text_embs:   torch.Tensor,
        pids:        torch.Tensor,    # unused here, kept for API consistency
    ) -> torch.Tensor:
        visual_proj = F.normalize(visual_proj, dim=-1)
        text_embs   = F.normalize(text_embs.float(), dim=-1)
        cos_sim = (visual_proj * text_embs).sum(dim=-1)   # (B,)
        return (1.0 - cos_sim).mean()


class TextCenterLoss(nn.Module):
    """
    Modified center loss where each identity's center is its fixed,
    frozen SigLIP2 text embedding rather than a learned or running-mean center.

    No center update rule is needed — text embeddings are fixed anchors.
    Uses cosine (angular) distance via L2 normalization before L2 loss,
    which is more robust to the modality gap between visual and text spaces.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        projected_visual: torch.Tensor,  # (B, D) — output of TextAlignProjector
        text_embeddings: torch.Tensor,   # (B, D) — SigLIP2 anchor for each sample's identity
        pids: torch.Tensor = None,       # unused, kept for API consistency
    ) -> torch.Tensor:
        if self.normalize:
            projected_visual = F.normalize(projected_visual, dim=-1)
            text_embeddings  = F.normalize(text_embeddings.float(), dim=-1)

        loss = 0.5 * torch.mean(
            torch.sum((projected_visual - text_embeddings) ** 2, dim=-1)
        )
        return loss
