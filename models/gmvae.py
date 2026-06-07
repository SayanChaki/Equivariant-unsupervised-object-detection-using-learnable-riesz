"""
Gaussian Mixture VAE (GMVAE)
==============================
Implements the variational autoencoder backbone used in
LeaRN-CompSTN+GMVAE (Section 4.4, Table 3).

References:
  - Dilokthanakul et al., "Deep Unsupervised Clustering with
    Gaussian Mixture VAEs", arXiv 1611.02648, 2016.
  - Yang et al., "Deep Clustering by GMVAE with Graph Embedding",
    ICCV 2019.

Latent structure (Section 4.4):
  z_pres  : Bernoulli  -- object presence
  z_what  : Gaussian   -- object appearance (C-dim)
  z_cls   : Categorical (Gaussian mixture index)
  z_alpha : Von Mises  -- rotation angle

The scale (s) and translation (t) come deterministically from CompSTN.

The GMVAE loss is:
  ELBO = E_q[log p(x|z)] - KL[q(z_what|x) || p(z_what|z_cls)]
       - KL[q(z_cls|x) || p(z_cls)]
       - KL[q(z_alpha|x) || p(z_alpha)]

The Von Mises KL is computed in closed form using the Bessel function
approximation from Davidson et al., UAI 2018.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# Von Mises distribution utilities
# ---------------------------------------------------------------------------

def _i0(x: torch.Tensor) -> torch.Tensor:
    """Modified Bessel function I_0(x), approximated for KL computation."""
    # Series approximation valid for moderate x
    return torch.special.i0(x)


def _i1(x: torch.Tensor) -> torch.Tensor:
    return torch.special.i1(x)


def von_mises_kl(mu_q: torch.Tensor, kappa_q: torch.Tensor,
                 mu_p: torch.Tensor, kappa_p: torch.Tensor) -> torch.Tensor:
    """
    KL divergence KL[VM(mu_q, kappa_q) || VM(mu_p, kappa_p)].

    Closed-form approximation:
    KL = log(I_0(kappa_p) / I_0(kappa_q))
         + kappa_q * A(kappa_q) * (1 - cos(mu_q - mu_p)) * kappa_p
         ... (simplified first-order term)

    Here we use the Mardia (1975) approximation.
    """
    i0_q = _i0(kappa_q).clamp(min=1e-8)
    i0_p = _i0(kappa_p).clamp(min=1e-8)
    i1_q = _i1(kappa_q).clamp(min=1e-8)

    A_kq = i1_q / i0_q    # mean resultant length A(kappa_q)

    kl = (torch.log(i0_p) - torch.log(i0_q)
          + kappa_q * A_kq * (1 - torch.cos(mu_q - mu_p)) * kappa_p)
    return kl


# ---------------------------------------------------------------------------
# Encoder / Decoder CNN blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, output_padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=stride, padding=1,
                               output_padding=output_padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Appearance encoder q(z_what, z_cls, z_alpha | x_glimpse)
# ---------------------------------------------------------------------------

class AppearanceEncoder(nn.Module):
    """
    Encodes a glimpse (B, C_feat, H_g, W_g) into latent variables:
      - z_what  : (B, A) Gaussian
      - z_cls   : (B, num_classes) Categorical logits
      - z_alpha : Von Mises parameters (mu, kappa)

    C_feat is the number of channels from LeaRN output.
    """

    def __init__(self,
                 in_channels: int,
                 glimpse_size: int = 28,
                 latent_dim: int = 32,
                 num_classes: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Shared CNN trunk
        self.trunk = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        flat_dim = 256 * 16

        # Appearance z_what
        self.what_mu     = nn.Linear(flat_dim, latent_dim)
        self.what_logvar = nn.Linear(flat_dim, latent_dim)

        # Class z_cls (categorical logits)
        self.cls_head = nn.Linear(flat_dim, num_classes)

        # Rotation z_alpha (Von Mises: cos/sin + kappa)
        self.alpha_head  = nn.Linear(flat_dim, 2)   # [cos, sin]
        self.kappa_head  = nn.Sequential(
            nn.Linear(flat_dim, 1), nn.Softplus()
        )

        # Presence z_pres (Bernoulli logit)
        self.pres_head = nn.Linear(flat_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)

        # z_what
        mu_what     = self.what_mu(h)
        logvar_what = self.what_logvar(h).clamp(-6, 2)

        # z_cls
        cls_logits  = self.cls_head(h)

        # z_alpha
        cs         = self.alpha_head(h)
        mu_alpha   = torch.atan2(cs[:, 1], cs[:, 0])
        kappa      = self.kappa_head(h).squeeze(1) + 1e-3

        # z_pres
        pres_logit = self.pres_head(h).squeeze(1)

        return mu_what, logvar_what, cls_logits, mu_alpha, kappa, pres_logit


# ---------------------------------------------------------------------------
# Decoder p(x_glimpse | z_what)
# ---------------------------------------------------------------------------

class AppearanceDecoder(nn.Module):
    """
    Decodes z_what -> reconstructed glimpse (B, out_channels, H_g, W_g).
    """

    def __init__(self,
                 latent_dim: int = 32,
                 out_channels: int = 1,
                 glimpse_size: int = 28):
        super().__init__()
        self.glimpse_size = glimpse_size
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            ConvTransposeBlock(256, 128, stride=2, output_padding=1),
            ConvTransposeBlock(128, 64,  stride=2, output_padding=1),
            ConvTransposeBlock(64,  32),
            nn.Conv2d(32, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.shape[0], 256, 4, 4)
        out = self.decoder(h)
        return F.interpolate(out, size=(self.glimpse_size, self.glimpse_size),
                             mode='bilinear', align_corners=False)


# ---------------------------------------------------------------------------
# GMVAE
# ---------------------------------------------------------------------------

class GMVAE(nn.Module):
    """
    Gaussian Mixture VAE for appearance modelling.

    The mixture prior is:
        p(z_what | z_cls=c) = N(mu_c, diag(sigma_c^2))
        p(z_cls)             = Cat(1/K, ..., 1/K)
        p(z_alpha)           = VM(0, kappa_0)  (uninformative prior)

    Parameters
    ----------
    in_channels  : feature channels of the glimpse (from LeaRN output)
    out_channels : image channels for reconstruction
    latent_dim   : dimension of z_what
    num_classes  : number of mixture components K
    glimpse_size : spatial size of glimpse
    kappa_prior  : concentration of the Von Mises prior
    """

    def __init__(self,
                 in_channels: int = 32,
                 out_channels: int = 1,
                 latent_dim: int = 32,
                 num_classes: int = 10,
                 glimpse_size: int = 28,
                 kappa_prior: float = 0.5):
        super().__init__()
        self.latent_dim   = latent_dim
        self.num_classes  = num_classes
        self.kappa_prior  = kappa_prior

        self.encoder = AppearanceEncoder(in_channels, glimpse_size,
                                         latent_dim, num_classes)
        self.decoder = AppearanceDecoder(latent_dim, out_channels, glimpse_size)

        # Learnable Gaussian mixture priors mu_c, log_sigma_c
        self.prior_mu    = nn.Parameter(torch.randn(num_classes, latent_dim))
        self.prior_log_s = nn.Parameter(torch.zeros(num_classes, latent_dim))

    # ------------------------------------------------------------------
    @staticmethod
    def _reparameterise(mu: torch.Tensor,
                        logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    # ------------------------------------------------------------------
    def forward(self, glimpse: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        glimpse : (B, C_feat, H_g, W_g)

        Returns
        -------
        recon   : (B, out_channels, H_g, W_g)  reconstructed glimpse
        info    : dict with losses and latent variables
        """
        B = glimpse.shape[0]

        # --- Encode ---
        (mu_what, logvar_what,
         cls_logits, mu_alpha, kappa,
         pres_logit) = self.encoder(glimpse)

        # --- Sample z_what ---
        z_what = self._reparameterise(mu_what, logvar_what)

        # --- Sample z_cls (Gumbel-Softmax) ---
        cls_probs = F.softmax(cls_logits, dim=-1)           # (B, K)
        z_cls = F.gumbel_softmax(cls_logits, tau=1.0, hard=False)  # (B, K)

        # --- Decode ---
        recon = self.decoder(z_what)

        # --- ELBO components ---
        # 1. KL(z_what | z_cls):  sum over mixture
        prior_mu    = self.prior_mu.unsqueeze(0)       # (1, K, A)
        prior_sigma = self.prior_log_s.exp().unsqueeze(0)  # (1, K, A)

        # q(z_what): N(mu_what, exp(logvar_what))
        # p(z_what|c): N(prior_mu_c, prior_sigma_c)
        mu_q    = mu_what.unsqueeze(1).expand(B, self.num_classes, -1)
        var_q   = logvar_what.exp().unsqueeze(1).expand_as(mu_q)
        var_p   = prior_sigma ** 2

        # KL per mixture component: (B, K, A)
        kl_c = 0.5 * (
            (prior_sigma.log() * 2 - logvar_what.unsqueeze(1))
            + (var_q + (mu_q - prior_mu) ** 2) / (var_p + 1e-8)
            - 1
        )
        # Weight by q(z_cls): (B,)
        kl_what = (z_cls.unsqueeze(-1) * kl_c).sum(dim=[1, 2])

        # 2. KL(z_cls || Cat(1/K)):
        log_uniform = torch.log(torch.tensor(1.0 / self.num_classes,
                                             device=glimpse.device))
        kl_cls = (cls_probs * (cls_probs.log() - log_uniform)).sum(-1)

        # 3. KL(z_alpha || VM(0, kappa_0)):
        kappa_prior = torch.full_like(kappa, self.kappa_prior)
        mu_prior    = torch.zeros_like(mu_alpha)
        kl_alpha = von_mises_kl(mu_alpha, kappa, mu_prior, kappa_prior)

        # 4. Bernoulli z_pres KL (uniform prior):
        p_pres = torch.sigmoid(pres_logit)
        kl_pres = F.binary_cross_entropy_with_logits(
            pres_logit, torch.full_like(p_pres, 0.5), reduction='none')

        info = {
            'kl_what':  kl_what.mean(),
            'kl_cls':   kl_cls.mean(),
            'kl_alpha': kl_alpha.mean(),
            'kl_pres':  kl_pres.mean(),
            'mu_what':  mu_what,
            'z_what':   z_what,
            'z_cls':    z_cls,
            'mu_alpha': mu_alpha,
            'kappa':    kappa,
            'pres_prob': p_pres,
        }
        return recon, info

    def elbo_loss(self, x: torch.Tensor, recon: torch.Tensor,
                  info: Dict, beta: float = 1.0) -> torch.Tensor:
        """
        ELBO = -E[log p(x|z)] + beta * KL terms
        Uses MSE for reconstruction (Gaussian likelihood) on glimpse.
        """
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl = (info['kl_what'] + info['kl_cls']
              + info['kl_alpha'] + info['kl_pres'])
        return recon_loss + beta * kl
