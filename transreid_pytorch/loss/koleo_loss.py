import torch
import torch.nn as nn
import torch.nn.functional as F

class KoLeoLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, target):
        """
        x: shape (B, D) where B is batch size, D is feature dimension.
        target: shape (B,) the ID labels of the batch.
        """
        # Normalize the features to the unit hypersphere
        x = F.normalize(x, p=2, dim=-1)
        
        # Calculate pairwise cosine distances (1 - cosine_similarity)
        sim_matrix = torch.mm(x, x.t())
        
        # MASKING POSITIVES: Find all images that belong to the SAME person
        # (This also inherently masks the diagonal self-similarity)
        is_pos = target.expand(x.size(0), x.size(0)).eq(target.expand(x.size(0), x.size(0)).t())
        
        # Fill positive pairs with -inf so they are ignored in the max() operation
        # Now the model will only push away the nearest neighbor of a DIFFERENT person
        sim_matrix.masked_fill_(is_pos, float('-inf'))
        
        # Find the maximum similarity (nearest neighbor of a DIFFERENT class)
        max_sim, _ = torch.max(sim_matrix, dim=1)
        
        # Euclidean distance squared between normalized vectors = 2 - 2 * cos_sim
        distances_sq = 2.0 - 2.0 * max_sim
        distances = torch.sqrt(distances_sq + self.eps)
        
        # KoLeo loss is the negative log of the nearest neighbor distances
        loss = -torch.mean(torch.log(distances))
        
        return loss
