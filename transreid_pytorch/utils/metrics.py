import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    # argsort in query chunks with int32 indices to keep peak RAM low on large
    # galleries (e.g. MSMT17: full int64 argsort + matches matrix -> OOM)
    indices = np.empty((num_q, num_g), dtype=np.int32)
    chunk = 1000
    for s in range(0, num_q, chunk):
        indices[s:s + chunk] = np.argsort(distmat[s:s + chunk], axis=1)
    #  0 2 1 3
    #  1 2 3 0
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = (g_pids[order] == q_pid).astype(np.int32)[keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP, all_AP, indices


def eval_func_chunked(qf, gf, q_pids, g_pids, q_camids, g_camids, max_rank=50, chunk=256):
    """Same metric as eval_func, but computed per query chunk (on GPU if available)
    without ever materializing the full query x gallery distance matrix in host RAM.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qf, gf = qf.to(device).float(), gf.to(device).float()
    q_pids, g_pids = torch.as_tensor(q_pids, device=device), torch.as_tensor(g_pids, device=device)
    q_camids, g_camids = torch.as_tensor(q_camids, device=device), torch.as_tensor(g_camids, device=device)
    num_g = gf.shape[0]
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    g_sq = torch.pow(gf, 2).sum(dim=1)
    ranks = torch.arange(1, max_rank + 1, device=device)

    cmc_sum = torch.zeros(max_rank, device=device, dtype=torch.float64)
    all_AP = []
    for s in range(0, qf.shape[0], chunk):
        q = qf[s:s + chunk]
        dist = torch.pow(q, 2).sum(dim=1, keepdim=True) + g_sq.unsqueeze(0)
        dist.addmm_(q, gf.t(), beta=1, alpha=-2)
        order = dist.argsort(dim=1)
        del dist

        qp, qc = q_pids[s:s + chunk, None], q_camids[s:s + chunk, None]
        gp = g_pids[order]
        match = gp == qp
        # remove gallery samples that have the same pid and camid with query
        keep = ~(match & (g_camids[order] == qc))
        del order, gp
        hits = (match & keep).float()
        # 1-based rank among kept gallery samples (clamped: removed leading entries would give 0/0)
        pos = keep.float().cumsum(dim=1).clamp(min=1)
        num_rel = hits.sum(dim=1)
        valid = num_rel > 0

        # AP = mean over correct matches of precision at that rank
        ap = (hits * hits.cumsum(dim=1) / pos).sum(dim=1) / num_rel.clamp(min=1)
        # rank of first correct match -> cmc[k] = 1 for k >= that rank
        first = pos.gather(1, hits.argmax(dim=1, keepdim=True))
        cmc = (ranks.unsqueeze(0) >= first).double()

        cmc_sum += cmc[valid].sum(dim=0)
        all_AP.extend(ap[valid].tolist())

    num_valid_q = len(all_AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = (cmc_sum / num_valid_q).float().cpu().numpy()
    mAP = np.mean(all_AP)
    return all_cmc, mAP, all_AP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        if self.reranking:
            cmc, mAP, distmat, _, _, qf, gf, all_AP, indices = self._compute()
            return cmc, mAP, distmat, self.pids, self.camids, qf, gf
        # memory-light path: no full distmat (returned as None)
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        qf = feats[:self.num_query]
        gf = feats[self.num_query:]
        print('=> Computing CMC/mAP with chunked euclidean_distance')
        cmc, mAP, _ = eval_func_chunked(
            qf, gf,
            np.asarray(self.pids[:self.num_query]), np.asarray(self.pids[self.num_query:]),
            np.asarray(self.camids[:self.num_query]), np.asarray(self.camids[self.num_query:]),
            max_rank=self.max_rank)
        return cmc, mAP, None, self.pids, self.camids, qf, gf

    def _compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP, all_AP, indices = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf, all_AP, indices



