import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NegUniformityLoss(nn.Module):
    """Uniformity (Wang & Isola) restricted to different-identity pairs.

    Same-ID pairs (including self) are masked out, so positives are never
    pushed apart. Every negative gets gradient, weighted by exp(-t * d^2).
    """

    def __init__(self, t=2.0):
        super().__init__()
        self.t = t

    def forward(self, x, target):
        x = F.normalize(x, p=2, dim=-1)
        d2 = (2.0 - 2.0 * x @ x.t()).clamp_min(0)
        neg = target[:, None].ne(target[None, :])
        n_neg = neg.sum(1)
        valid = n_neg > 0                      # anchors that have at least one negative
        if not valid.any():
            return x.sum() * 0.0
        logits = (-self.t * d2).masked_fill(~neg, float('-inf'))
        lme = torch.logsumexp(logits[valid], dim=1) - torch.log(n_neg[valid].float())
        return lme.mean()


class VarianceLoss(nn.Module):
    """VICReg variance hinge: keeps every feature dimension from collapsing.

    Per-dimension std over the batch, so it never pulls or pushes individual
    pairs (same-ID samples are unaffected). Features are L2-normalised and
    scaled by sqrt(D) so an isotropic spread has std ~= 1 (matches gamma=1).
    """

    def __init__(self, gamma=1.0, eps=1e-4):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, x, target=None):
        x = F.normalize(x, p=2, dim=-1) * math.sqrt(x.size(1))
        std = torch.sqrt(x.var(dim=0) + self.eps)
        return F.relu(self.gamma - std).mean()
