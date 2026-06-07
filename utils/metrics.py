"""
Evaluation Metrics
==================
Implements the metrics used in the paper:

  - Rotation-Offset Entropy (RoE) -- Section 5.1, novel metric
  - NMI  (Normalised Mutual Information)  -- Table 3
  - ARI  (Adjusted Rand Index)            -- Table 3
  - SSIM (Structural Similarity)          -- Table 3
  - LEE  (Lie-derivative Equivariance Error) -- Table 1
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Rotation-Offset Entropy (RoE)
# ---------------------------------------------------------------------------

def rotation_offset_entropy(pred_angles: torch.Tensor,
                             true_angles: torch.Tensor,
                             n_bins: int = 36) -> float:
    """
    Compute Rotation-Offset Entropy (RoE) as defined in Section 5.1.

    RoE = H(p) / log2(n_bins)

    where H(p) is the Shannon entropy of the discretised distribution
    of angle differences (pred - true) mod 2pi.

    A low RoE (~0) means the predicted angles consistently differ from
    ground truth by the same offset (good equivariance).
    A high RoE (~1) means the differences are uniformly distributed
    (poor / random predictions).

    Parameters
    ----------
    pred_angles : (N,) tensor of predicted angles in radians
    true_angles : (N,) tensor of ground truth angles in radians
    n_bins      : number of histogram bins over [0, 2pi)

    Returns
    -------
    RoE in [0, 1]
    """
    diff = (pred_angles - true_angles) % (2 * math.pi)
    diff_np = diff.detach().cpu().numpy()

    counts, _ = np.histogram(diff_np, bins=n_bins,
                              range=(0, 2 * math.pi))
    p = counts / (counts.sum() + 1e-12)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    roe = entropy / math.log2(n_bins)
    return float(np.clip(roe, 0, 1))


# ---------------------------------------------------------------------------
# Clustering metrics (NMI, ARI)
# ---------------------------------------------------------------------------

def cluster_metrics(z_what: torch.Tensor,
                    labels: torch.Tensor,
                    num_classes: int) -> dict:
    """
    Cluster z_what embeddings with k-means then compute NMI and ARI
    against ground-truth labels.

    Parameters
    ----------
    z_what      : (N, D) latent appearance vectors
    labels      : (N,)   integer class labels
    num_classes : K for k-means

    Returns
    -------
    dict with 'NMI' and 'ARI' floats
    """
    z_np = z_what.detach().cpu().float().numpy()
    y_np = labels.detach().cpu().numpy()

    km = KMeans(n_clusters=num_classes, n_init=10, random_state=42)
    pred = km.fit_predict(z_np)

    nmi = normalized_mutual_info_score(y_np, pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_np, pred)
    return {'NMI': float(nmi), 'ARI': float(ari)}


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def ssim(x: torch.Tensor, y: torch.Tensor,
         window_size: int = 11, C1: float = 0.01**2,
         C2: float = 0.03**2) -> float:
    """
    Compute mean SSIM between two batches of images.

    x, y : (B, C, H, W) in [0, 1]
    """
    assert x.shape == y.shape
    B, C, H, W = x.shape

    # Gaussian window
    sigma = 1.5
    gauss = torch.tensor(
        [math.exp(-(i - window_size // 2) ** 2 / (2 * sigma ** 2))
         for i in range(window_size)],
        dtype=x.dtype, device=x.device
    )
    gauss /= gauss.sum()
    window_2d = gauss.outer(gauss).unsqueeze(0).unsqueeze(0)
    window_2d = window_2d.expand(C, 1, -1, -1)

    pad = window_size // 2
    mu_x  = F.conv2d(x, window_2d, padding=pad, groups=C)
    mu_y  = F.conv2d(y, window_2d, padding=pad, groups=C)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_xx = F.conv2d(x * x, window_2d, padding=pad, groups=C) - mu_xx
    sig_yy = F.conv2d(y * y, window_2d, padding=pad, groups=C) - mu_yy
    sig_xy = F.conv2d(x * y, window_2d, padding=pad, groups=C) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
               ((mu_xx + mu_yy + C1) * (sig_xx + sig_yy + C2))

    return float(ssim_map.mean().item())


# ---------------------------------------------------------------------------
# LEE: Lie-derivative Equivariance Error (Table 1)
# ---------------------------------------------------------------------------

def lie_equivariance_error(model: torch.nn.Module,
                           x: torch.Tensor,
                           delta_angle: float = 0.05) -> float:
    """
    Estimates the Lie-derivative equivariance error:

        LEE = E_x [ || f(R_eps * x) - R_eps * f(x) ||_F /
                    || f(x) ||_F ]

    where R_eps is a small rotation by delta_angle radians.
    Lower is better.

    Reference: Gruver et al., "The Lie Derivative for Measuring Learned
    Equivariance", arXiv 2210.02984, 2022. [Ref 44 in paper]

    Parameters
    ----------
    model       : any module with a forward that takes (B, C, H, W)
    x           : (B, C, H, W) batch of images
    delta_angle : small rotation in radians
    """
    model.eval()
    with torch.no_grad():
        # Rotate input
        cos_a = math.cos(delta_angle)
        sin_a = math.sin(delta_angle)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
            dtype=x.dtype, device=x.device
        ).unsqueeze(0).expand(x.shape[0], -1, -1)
        grid  = F.affine_grid(theta, x.shape, align_corners=False)
        x_rot = F.grid_sample(x, grid, align_corners=False)

        # f(R_eps * x)
        fx_rot = model(x_rot)
        if isinstance(fx_rot, tuple):
            fx_rot = fx_rot[0]

        # f(x)
        fx = model(x)
        if isinstance(fx, tuple):
            fx = fx[0]

        # R_eps * f(x)
        B2, C2, H2, W2 = fx.shape
        theta2 = theta[:B2] if theta.shape[0] >= B2 else \
                 theta.expand(B2, -1, -1)
        grid2   = F.affine_grid(theta2, fx.shape, align_corners=False)
        rfx     = F.grid_sample(fx, grid2, align_corners=False)

        num   = (fx_rot - rfx).norm(dim=[1, 2, 3])
        denom = fx.norm(dim=[1, 2, 3]).clamp(min=1e-8)
        return float((num / denom).mean().item())
