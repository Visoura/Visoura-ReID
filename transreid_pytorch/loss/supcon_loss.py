import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (B, D) tensor of normalized or unnormalized embeddings
        labels: (B,) tensor of identity labels
        """
        device = features.device
        batch_size = features.shape[0]

        # 1. Normalize features to unit sphere
        features = F.normalize(features, p=2, dim=1)

        # 2. Compute all-to-all cosine similarity matrix: (B, B)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 3. Create masks
        # Mask for all positives (B, B) where mask[i, j] = 1 if label[i] == label[j]
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-similarity (we don't want an image to be its own positive)
        logits_mask = torch.scatter(
            torch.ones_like(pos_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        pos_mask = pos_mask * logits_mask # True positives excluding self

        # 4. Numerical stability for log-sum-exp
        row_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - row_max.detach()

        # 5. Compute Log-Sum-Exp for the denominator (all negatives + positives)
        # We only sum over elements where logits_mask == 1 (excluding self)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 6. Compute the mean log-probability for the positives
        # Average over the number of positives per anchor
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-12)

        # 7. Final InfoNCE/SupCon loss
        loss = -mean_log_prob_pos.mean()

        return loss
