"""
Composite Spatial Transformer Networks (CompSTN)
=================================================
Implements the two-step sequential pose estimator described in
Section 4.2 of the paper.

Step 1:  Estimate translation (t) and scale (s) using rotation-
         INVARIANT features (from GroupMaxPool).  The STN
         resamples the rotation-EQUIVARIANT feature map at the
         estimated (t, s) to produce a scale-translation-aligned glimpse.

Step 2:  Estimate rotation angle (alpha) using the aligned glimpse.
         A second STN corrects for rotation, producing a fully
         aligned crop that is equivariant to (t, s, alpha).

The rotation angle is predicted PROBABILISTICALLY via a Von Mises
distribution (circular normal), consistent with the latent z_alpha in
Section 4.4 of the paper.

Spatial transformations use PyTorch's affine_grid / grid_sample for
differentiable resampling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Helpers: affine matrices
# ---------------------------------------------------------------------------

def _translation_scale_matrix(t: torch.Tensor,
                               s: torch.Tensor) -> torch.Tensor:
    """
    Build a 2x3 affine matrix for translation + isotropic scale.

    t : (B, 2)   translation (tx, ty) in [-1, 1]
    s : (B, 1)   log-scale, exponentiated to get scale in (0, inf)
    returns: (B, 2, 3)
    """
    B = t.shape[0]
    scale = torch.exp(s).clamp(0.2, 5.0)          # (B, 1)
    zeros = torch.zeros(B, 1, device=t.device, dtype=t.dtype)
    # Row 0: [s, 0, tx], Row 1: [0, s, ty]
    row0 = torch.cat([scale, zeros, t[:, :1]], dim=1)   # (B, 3)
    row1 = torch.cat([zeros, scale, t[:, 1:]], dim=1)   # (B, 3)
    theta = torch.stack([row0, row1], dim=1)             # (B, 2, 3)
    return theta


def _rotation_matrix(alpha: torch.Tensor) -> torch.Tensor:
    """
    Build a 2x3 affine matrix for pure rotation.

    alpha : (B,)  angle in radians
    returns: (B, 2, 3)
    """
    B = alpha.shape[0]
    cos_a = torch.cos(alpha)
    sin_a = torch.sin(alpha)
    zeros = torch.zeros(B, device=alpha.device, dtype=alpha.dtype)
    row0 = torch.stack([ cos_a, -sin_a, zeros], dim=1)
    row1 = torch.stack([ sin_a,  cos_a, zeros], dim=1)
    theta = torch.stack([row0, row1], dim=1)   # (B, 2, 3)
    return theta


# ---------------------------------------------------------------------------
# Localisation nets
# ---------------------------------------------------------------------------

class TransScaleLocaliser(nn.Module):
    """
    Predicts (tx, ty, log_s) from rotation-invariant features.
    Input: (B, C, H, W) invariant features.
    Output: tx (B,1), ty (B,1), log_s (B,1)
    """

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(in_channels * 16, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),   # (tx, ty, log_s)
        )
        # Initialise last layer near identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)                  # (B, 3)
        t   = out[:, :2].tanh()            # translation in [-1, 1]
        log_s = out[:, 2:3].clamp(-1.6, 1.6)  # log scale
        return t, log_s


class RotationLocaliser(nn.Module):
    """
    Predicts rotation angle alpha from the scale-translation-aligned glimpse.
    Uses group-max-pooled equivariant features.
    Returns (mu_alpha, kappa) of a Von Mises distribution.
    """

    def __init__(self, in_channels: int, N: int, hidden: int = 128):
        super().__init__()
        self.N = N
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(in_channels * 16, hidden),
            nn.ReLU(inplace=True),
        )
        # Predict (cos_alpha, sin_alpha) for circular regression
        self.angle_head = nn.Linear(hidden, 2)
        # Predict concentration kappa > 0
        self.kappa_head = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )
        nn.init.zeros_(self.angle_head.weight)
        nn.init.zeros_(self.angle_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x      : (B, C, H, W) - aligned equivariant features after GroupMaxPool
        returns: mu_alpha (B,), kappa (B,)
        """
        h = self.net(x)
        cs = self.angle_head(h)           # (B, 2)  [cos_alpha, sin_alpha]
        mu_alpha = torch.atan2(cs[:, 1], cs[:, 0])   # (B,) in [-pi, pi]
        kappa = self.kappa_head(h).squeeze(1) + 1e-3   # (B,) > 0
        return mu_alpha, kappa


# ---------------------------------------------------------------------------
# CompSTN
# ---------------------------------------------------------------------------

class CompSTN(nn.Module):
    """
    Composite Spatial Transformer Network.

    Follows the two-step decomposition in Section 4.2:

        Step 1: Estimate (t, s) from invariant features.
                Resample equivariant features at (t, s).
        Step 2: Estimate alpha from the resampled equivariant features.
                Resample at alpha to produce a fully aligned glimpse.

    Parameters
    ----------
    in_channels    : C (number of group channels)
    N              : group size
    glimpse_size   : spatial size of the output glimpse (H_g, W_g)
    ts_hidden      : hidden size for translation/scale localiser
    rot_hidden     : hidden size for rotation localiser
    """

    def __init__(self,
                 in_channels: int,
                 N: int = 8,
                 glimpse_size: int = 28,
                 ts_hidden: int = 128,
                 rot_hidden: int = 128):
        super().__init__()
        self.N = N
        self.glimpse_size = glimpse_size

        # Step 1: translation + scale
        self.ts_localiser = TransScaleLocaliser(in_channels, ts_hidden)

        # Step 2: rotation (operates on GroupMaxPool of equivariant features)
        self.rot_localiser = RotationLocaliser(in_channels, N, rot_hidden)

    # ------------------------------------------------------------------
    def _group_max_pool(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, N, H, W) -> (B, C, H, W)"""
        return feat.max(dim=2).values

    # ------------------------------------------------------------------
    def forward(self, feat_equiv: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                     torch.Tensor, torch.Tensor]:
        """
        feat_equiv : (B, C, N, H, W) -- rotation-equivariant features from
                     LeaRN + ReResNet

        Returns
        -------
        glimpse_aligned : (B, C, H_g, W_g)  fully aligned crop (invariant)
        t               : (B, 2)  predicted translation
        log_s           : (B, 1)  predicted log-scale
        mu_alpha        : (B,)    predicted rotation angle
        kappa           : (B,)    Von Mises concentration
        """
        B, C, N, H, W = feat_equiv.shape
        gs = self.glimpse_size

        # -- Invariant features for Step 1 --
        feat_inv = self._group_max_pool(feat_equiv)  # (B, C, H, W)

        # -- Step 1: predict (t, s) --
        t, log_s = self.ts_localiser(feat_inv)       # (B,2), (B,1)
        theta_ts = _translation_scale_matrix(t, log_s)   # (B, 2, 3)

        # Resample equivariant features at (t, s)
        # We resample all N orientation channels jointly
        feat_flat = feat_equiv.view(B * N, C, H, W)
        theta_ts_exp = theta_ts.unsqueeze(1).expand(
            B, N, 2, 3).reshape(B * N, 2, 3)
        grid_ts = F.affine_grid(theta_ts_exp,
                                (B * N, C, gs, gs), align_corners=False)
        feat_trans = F.grid_sample(feat_flat, grid_ts,
                                   mode='bilinear', padding_mode='zeros',
                                   align_corners=False)
        feat_trans = feat_trans.view(B, N, C, gs, gs).permute(0, 2, 1, 3, 4)
        # (B, C, N, gs, gs)

        # -- Step 2: predict alpha from GroupMaxPool of aligned features --
        feat_trans_inv = self._group_max_pool(feat_trans)  # (B, C, gs, gs)
        mu_alpha, kappa = self.rot_localiser(feat_trans_inv)

        # Resample with rotation (on invariant features for simplicity,
        # as justified in Section 4.2: avoids interpolation artifacts)
        theta_rot = _rotation_matrix(mu_alpha)             # (B, 2, 3)
        grid_rot = F.affine_grid(theta_rot, (B, C, gs, gs),
                                 align_corners=False)
        glimpse_aligned = F.grid_sample(feat_trans_inv, grid_rot,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=False)
        # (B, C, gs, gs)

        return glimpse_aligned, t, log_s, mu_alpha, kappa

    # ------------------------------------------------------------------
    def reverse_sample(self, glimpse: torch.Tensor,
                       t: torch.Tensor,
                       log_s: torch.Tensor,
                       mu_alpha: torch.Tensor,
                       out_size: Tuple[int, int]) -> torch.Tensor:
        """
        Place the reconstructed glimpse back into the full image canvas.
        Inverts the (t, s, alpha) transformation.
        """
        B, C, H_g, W_g = glimpse.shape
        H_out, W_out = out_size

        # Invert rotation
        theta_rot_inv = _rotation_matrix(-mu_alpha)
        grid_rot_inv = F.affine_grid(theta_rot_inv,
                                     (B, C, H_g, W_g), align_corners=False)
        unrotated = F.grid_sample(glimpse, grid_rot_inv,
                                  mode='bilinear', padding_mode='zeros',
                                  align_corners=False)

        # Invert translation + scale
        scale = torch.exp(log_s).clamp(0.2, 5.0)
        t_inv = -t / scale
        log_s_inv = -log_s
        theta_ts_inv = _translation_scale_matrix(t_inv, log_s_inv)
        grid_ts_inv = F.affine_grid(theta_ts_inv,
                                    (B, C, H_out, W_out), align_corners=False)
        placed = F.grid_sample(unrotated, grid_ts_inv,
                               mode='bilinear', padding_mode='zeros',
                               align_corners=False)
        return placed   # (B, C, H_out, W_out)
