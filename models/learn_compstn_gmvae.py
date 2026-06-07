"""
LeaRN-CompSTN + GMVAE
======================
Full representation learning pipeline as described in Section 4.4 of the paper.

End-to-end forward pass:

    Input image x (B, C_in, H, W)
         |
    ReResNet (LiftingConv -> GroupResBlocks)
         |
    LeaRN  (learnable Riesz enrichment in group space)
         |    (B, C, N, H', W')  -- RT-Equiv. features
    GroupMaxPool  ->  (B, C, H', W')  -- rotation-invariant features
         |
    CompSTN Step 1: predict (t, s) from invariant features
                    resample equivariant features at (t, s)
         |
    CompSTN Step 2: predict alpha from aligned equivariant features
                    resample invariant features at alpha
         |
    Glimpse (B, C, H_g, W_g)  -- fully aligned crop
         |
    GMVAE Encoder: q(z_what, z_cls, z_alpha, z_pres | glimpse)
         |
    GMVAE Decoder: p(glimpse_recon | z_what)
         |
    Reverse Sampler: place glimpse back in full image -> x_hat

The model outputs both the full image reconstruction x_hat and the
glimpse reconstruction for computing the ELBO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .reresnet import ReResNet
from .learn import LeaRN
from .compstn import CompSTN
from .gmvae import GMVAE
from .group_conv import GroupMaxPool


# ---------------------------------------------------------------------------
# Utility: glimpse projection head
# ---------------------------------------------------------------------------

class GlimpseProjection(nn.Module):
    """
    Projects group-pooled features (B, C_reresnet, H_g, W_g) to
    (B, C_gmvae, H_g, W_g) for the GMVAE encoder.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class LeaRN_CompSTN_GMVAE(nn.Module):
    """
    Full LeaRN-CompSTN+GMVAE for unsupervised equivariant
    representation learning and reconstruction.

    Parameters
    ----------
    in_channels    : image channels (1 for MNIST)
    image_size     : (H, W) of input images
    N              : cyclic group order
    base_channels  : ReResNet width
    reresnet_layers: tuple of block counts per ReResNet stage
    learn_K        : Riesz Laplace mixture components
    riesz_order    : 1 or 2
    glimpse_size   : spatial size of glimpse
    latent_dim     : z_what dimension
    num_classes    : GMVAE mixture components
    kappa_prior    : Von Mises prior concentration
    beta           : KL annealing weight for ELBO
    """

    def __init__(self,
                 in_channels: int = 1,
                 image_size: Tuple[int, int] = (64, 64),
                 N: int = 8,
                 base_channels: int = 32,
                 reresnet_layers: tuple = (2, 2, 2, 2),
                 learn_K: int = 8,
                 riesz_order: int = 1,
                 glimpse_size: int = 28,
                 latent_dim: int = 32,
                 num_classes: int = 10,
                 kappa_prior: float = 0.5,
                 beta: float = 1.0):
        super().__init__()
        self.N = N
        self.image_size = image_size
        self.glimpse_size = glimpse_size
        self.beta = beta

        C = base_channels

        # -- Feature extraction backbone --
        self.reresnet = ReResNet(
            in_channels=in_channels,
            base_channels=C,
            N=N,
            layers=reresnet_layers,
            keep_group_dim=True,     # keep (B, C, N, H, W) for LeaRN
        )
        # ReResNet output channels after 4 stages = C * 8
        feat_channels = C * 8

        # -- Learnable Riesz enrichment --
        self.learn = LeaRN(
            in_channels=feat_channels,
            N=N,
            K=learn_K,
            order=riesz_order,
            out_channels=feat_channels,
        )

        # -- Group-invariant pooling --
        self.group_pool = GroupMaxPool()

        # -- Composite STN --
        self.compstn = CompSTN(
            in_channels=feat_channels,
            N=N,
            glimpse_size=glimpse_size,
        )

        # -- Glimpse projection to GMVAE input size --
        gmvae_in_ch = 64   # projected channels for GMVAE encoder
        self.glimpse_proj = GlimpseProjection(feat_channels, gmvae_in_ch)

        # -- GMVAE --
        self.gmvae = GMVAE(
            in_channels=gmvae_in_ch,
            out_channels=in_channels,
            latent_dim=latent_dim,
            num_classes=num_classes,
            glimpse_size=glimpse_size,
            kappa_prior=kappa_prior,
        )

        # -- Image-level reconstruction projection --
        # Lifts image channels back to feat_channels for reverse sampler
        self.recon_proj = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        x : (B, C_in, H, W)

        Returns
        -------
        x_hat      : (B, C_in, H, W)  full image reconstruction
        glimpse_recon : (B, C_in, H_g, W_g) glimpse reconstruction
        info       : dict with all latents, losses, and diagnostics
        """
        B, _, H, W = x.shape

        # -- ReResNet: (B, C_in, H, W) -> (B, C*8, N, H', W') --
        feat_equiv = self.reresnet(x)           # (B, C*8, N, H', W')

        # -- LeaRN: Riesz enrichment --
        feat_equiv = self.learn(feat_equiv)     # (B, C*8, N, H', W')

        # -- CompSTN: aligned glimpse + pose parameters --
        (glimpse_feat,
         t, log_s,
         mu_alpha, kappa_stn) = self.compstn(feat_equiv)
        # glimpse_feat : (B, C*8, H_g, W_g)

        # -- Project to GMVAE input --
        glimpse_proj = self.glimpse_proj(glimpse_feat)  # (B, 64, H_g, W_g)

        # -- GMVAE: encode + decode glimpse --
        glimpse_recon, vae_info = self.gmvae(glimpse_proj)
        # glimpse_recon : (B, C_in, H_g, W_g)

        # -- Reverse sample: place reconstruction back in image --
        # Use the GMVAE decoder output (image space) for reverse sampling
        x_hat = self.compstn.reverse_sample(
            glimpse_recon.expand(-1, glimpse_feat.shape[1], -1, -1)
            if glimpse_recon.shape[1] != glimpse_feat.shape[1]
            else self.recon_proj(glimpse_recon),
            t, log_s, mu_alpha,
            out_size=(H, W)
        )
        # Project back to image channels
        x_hat = F.interpolate(x_hat[:, :1], size=(H, W),
                              mode='bilinear', align_corners=False)
        x_hat = x_hat.clamp(0, 1)

        # Merge STN angle prediction with GMVAE angle for consistency
        info = {
            **vae_info,
            't':          t,
            'log_s':      log_s,
            'mu_alpha_stn':  mu_alpha,
            'kappa_stn':  kappa_stn,
        }
        return x_hat, glimpse_recon, info

    # ------------------------------------------------------------------
    def loss(self, x: torch.Tensor,
             x_hat: torch.Tensor,
             glimpse_recon: torch.Tensor,
             info: Dict,
             glimpse_target: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss:
          - Image reconstruction (MSE on full image)
          - GMVAE ELBO on glimpse
          - Riesz regularisation from LeaRN
          - Angle consistency between STN prediction and GMVAE z_alpha
        """
        # Full image reconstruction
        recon_img = F.mse_loss(x_hat, x)

        # GMVAE ELBO (glimpse-level)
        # If we have a ground-truth glimpse target use it; otherwise skip
        if glimpse_target is not None:
            elbo = self.gmvae.elbo_loss(
                glimpse_target, glimpse_recon, info, beta=self.beta)
        else:
            elbo = self.gmvae.elbo_loss(
                glimpse_recon.detach(), glimpse_recon, info, beta=self.beta)

        # Riesz regularisation
        reg_riesz = self.learn.riesz_regularization()

        # Angle consistency: STN alpha vs GMVAE alpha
        angle_consistency = (
            1 - torch.cos(info['mu_alpha_stn'] - info['mu_alpha'])
        ).mean()

        total = recon_img + elbo + reg_riesz + 0.1 * angle_consistency

        return {
            'total':             total,
            'recon_img':         recon_img,
            'elbo':              elbo,
            'kl_what':           info['kl_what'],
            'kl_cls':            info['kl_cls'],
            'kl_alpha':          info['kl_alpha'],
            'kl_pres':           info['kl_pres'],
            'reg_riesz':         reg_riesz,
            'angle_consistency': angle_consistency,
        }

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> Dict:
        """
        Inference only: returns latent variables without reconstruction.
        Useful for clustering / NMI / ARI evaluation (Table 3).
        """
        with torch.no_grad():
            _, _, info = self.forward(x)
        return info
