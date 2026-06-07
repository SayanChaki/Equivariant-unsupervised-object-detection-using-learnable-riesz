"""
Learnable Riesz Transform (LeaRN)
===================================
Implements the learnable, steerable Riesz-transform-based feature extractor
described in Section 4.1 of the paper.

Key equations
-------------
Standard first-order Riesz transform (Fourier domain):
    F(R1(I))(xi) = -i * xi_1 / ||xi|| * F(I)(xi)          [Eq. 1]
    F(R2(I))(xi) = -i * xi_2 / ||xi|| * F(I)(xi)

Learnable Riesz (Eq. 2):
    F(LR1(I))(xi) = w1(xi) * F(R1(I))(xi)

Weight function as mixture of radial Laplace distributions (Eq. 3):
    w1(xi) = sum_{k=1}^{K} w1k * exp(-| ||xi|| - mu_k |)

where mu_k are fixed log-spaced frequency centres and w1k are learnable.

The second-order Riesz components (R11, R12, R22) are also included for
richer directional information (they appear in the steerable wavelet
literature as the Hessian-Riesz).

This module operates on group feature maps produced by ReResNet:
    input  : (B, C, N, H, W)   group equivariant features
    output : (B, C', N, H, W)  Riesz-enhanced features

The Riesz transform acts on the spatial (H, W) plane for each
(B, C, N) slice independently.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Frequency grid helper
# ---------------------------------------------------------------------------

def _freq_grid(H: int, W: int, device: torch.device,
               dtype: torch.dtype) -> torch.Tensor:
    """
    Build 2-D frequency grid (xi1, xi2) normalised to [-pi, pi].
    Returns tensor of shape (H, W, 2).
    """
    fy = torch.fft.fftfreq(H, d=1.0 / (2 * math.pi), device=device, dtype=dtype)
    fx = torch.fft.fftfreq(W, d=1.0 / (2 * math.pi), device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing='ij')  # (H, W)
    return torch.stack([grid_x, grid_y], dim=-1)             # (H, W, 2)


def _riesz_kernels(H: int, W: int, device: torch.device,
                   dtype: torch.dtype, order: int = 1):
    """
    Compute Riesz filter kernels in the Fourier domain.

    order=1 : returns (R1, R2)          each (H, W) complex
    order=2 : returns (R11, R12, R22)   each (H, W) complex

    The DC component (||xi|| = 0) is set to zero to avoid division by zero.
    """
    grid = _freq_grid(H, W, device, dtype)   # (H, W, 2)
    xi1 = grid[..., 0]
    xi2 = grid[..., 1]
    norm = (xi1 ** 2 + xi2 ** 2).sqrt()
    # avoid division by zero at DC
    norm_safe = norm.clone()
    norm_safe[norm_safe == 0] = 1.0

    j = torch.tensor(0 + 1j, dtype=torch.complex64, device=device)

    if order == 1:
        R1 = (-j * xi1 / norm_safe).to(torch.complex64)
        R2 = (-j * xi2 / norm_safe).to(torch.complex64)
        R1[norm == 0] = 0
        R2[norm == 0] = 0
        return R1, R2
    else:
        R11 = (-xi1 * xi1 / norm_safe ** 2).to(torch.complex64)
        R12 = (-xi1 * xi2 / norm_safe ** 2).to(torch.complex64)
        R22 = (-xi2 * xi2 / norm_safe ** 2).to(torch.complex64)
        R11[norm == 0] = 0
        R12[norm == 0] = 0
        R22[norm == 0] = 0
        return R11, R12, R22


# ---------------------------------------------------------------------------
# Learnable weight function w(xi)
# ---------------------------------------------------------------------------

class RadialLaplaceWeights(nn.Module):
    """
    Learnable radial weight function in the Fourier domain:
        w(xi) = sum_k  w_k * exp( -| ||xi|| - mu_k | )

    mu_k are fixed, log-spaced frequency centres.
    w_k are non-negative learnable scalars (enforced via softplus).
    """

    def __init__(self, K: int = 8, freq_min: float = 0.1,
                 freq_max: float = math.pi):
        super().__init__()
        self.K = K
        # Fixed log-spaced centres
        mu = torch.exp(
            torch.linspace(math.log(freq_min), math.log(freq_max), K)
        )
        self.register_buffer('mu', mu)           # (K,)
        # Learnable log-weights (softplus ensures positivity)
        self.log_w = nn.Parameter(torch.zeros(K))

    @property
    def w(self) -> torch.Tensor:
        return F.softplus(self.log_w)            # (K,)  positive

    def forward(self, norm: torch.Tensor) -> torch.Tensor:
        """
        norm : (H, W)  - radial frequency coordinate ||xi||
        returns : (H, W)  weight map
        """
        # norm (H, W) -> (H, W, 1), mu (K,) -> (1, 1, K)
        diff = (norm.unsqueeze(-1) - self.mu.view(1, 1, self.K)).abs()
        weight_map = (self.w.view(1, 1, self.K) * torch.exp(-diff)).sum(-1)
        return weight_map   # (H, W)

    def regularization(self) -> torch.Tensor:
        """L1 + L2 regularisation on weights as described in the paper."""
        w = self.w
        return w.abs().sum() + (w ** 2).sum()


# ---------------------------------------------------------------------------
# LeaRN module
# ---------------------------------------------------------------------------

class LeaRN(nn.Module):
    """
    Learnable Riesz-transform Network.

    Wraps `in_channels` group feature maps with Riesz-enhanced features.
    For each (B, C, n, H, W) input slice the Riesz transform is applied
    in the spatial FFT domain, producing 2 (order=1) or 3 (order=2)
    additional feature maps per channel.

    A learnable 1x1 group convolution then fuses the Riesz features back
    to `out_channels`.

    Parameters
    ----------
    in_channels  : C (group channels from ReResNet)
    N            : group size
    K            : number of Laplace mixture components
    order        : Riesz order (1 or 2)
    out_channels : output group channels; defaults to in_channels
    reg_lambda   : weight for frequency regularisation loss
    """

    def __init__(self,
                 in_channels: int,
                 N: int = 8,
                 K: int = 8,
                 order: int = 1,
                 out_channels: int = None,
                 reg_lambda: float = 1e-4):
        super().__init__()
        self.N = N
        self.order = order
        self.reg_lambda = reg_lambda
        out_channels = out_channels or in_channels

        # One set of radial weights per Riesz component per input channel
        n_components = 2 if order == 1 else 3
        self.n_components = n_components

        # Shared weight function across channels (paper Eq. 3):
        # independent weight functions for each Riesz component
        self.riesz_weights = nn.ModuleList(
            [RadialLaplaceWeights(K=K) for _ in range(n_components)]
        )

        # Fusion: (in_channels * (1 + n_components), N) -> (out_channels, N)
        # Implemented as a grouped 1x1 conv over C dimension
        total_in = in_channels * (1 + n_components)
        self.fusion = nn.Sequential(
            nn.Conv3d(total_in, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------
    def _apply_riesz(self, x_slice: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable Riesz to a single (B*N, C, H, W) tensor.
        Returns (B*N, C * n_components, H, W) real-valued features.
        """
        BN, C, H, W = x_slice.shape

        # FFT
        X_f = torch.fft.rfft2(x_slice.float())          # (BN, C, H, W//2+1)

        # Frequency norm grid for rfft2 output size
        fy = torch.fft.fftfreq(H, d=1.0 / (2 * math.pi),
                                device=x_slice.device, dtype=torch.float32)
        fx = torch.fft.rfftfreq(W, d=1.0 / (2 * math.pi),
                                 device=x_slice.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')   # (H, W//2+1)
        norm = (gx ** 2 + gy ** 2).sqrt()                 # (H, W//2+1)
        norm_safe = norm.clone(); norm_safe[norm_safe == 0] = 1.0

        xi1, xi2 = gx, gy

        results = []
        if self.order == 1:
            # R1, R2 kernels (purely imaginary -> real output after ifft)
            for i, (xi, rw) in enumerate(
                    zip([xi1, xi2], self.riesz_weights)):
                kernel = (-1j * xi / norm_safe)               # (H, W//2+1) complex
                kernel[norm == 0] = 0
                w_map = rw(norm).to(torch.float32)            # (H, W//2+1) real
                learnable_kernel = w_map * kernel             # broadcast over BN,C
                Y_f = X_f * learnable_kernel.unsqueeze(0).unsqueeze(0)
                Y = torch.fft.irfft2(Y_f, s=(H, W))          # (BN, C, H, W)
                results.append(Y)
        else:
            for i, (xi_pair, rw) in enumerate(
                    zip([(xi1, xi1), (xi1, xi2), (xi2, xi2)],
                        self.riesz_weights)):
                xa, xb = xi_pair
                kernel = (-xa * xb / norm_safe ** 2).to(torch.complex64)
                kernel[norm == 0] = 0
                w_map = rw(norm).to(torch.float32)
                learnable_kernel = w_map * kernel
                Y_f = X_f.to(torch.complex64) * \
                      learnable_kernel.unsqueeze(0).unsqueeze(0)
                Y = torch.fft.irfft2(Y_f, s=(H, W)).real
                results.append(Y)

        return torch.cat(results, dim=1)   # (BN, C * n_components, H, W)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, N, H, W)
        returns : (B, out_channels, N, H, W)
        """
        B, C, N, H, W = x.shape

        # Reshape to (B*N, C, H, W) to apply Riesz per spatial slice
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * N, C, H, W)

        # Apply Riesz
        riesz_feats = self._apply_riesz(x_flat)      # (B*N, C*n, H, W)

        # Concatenate original + Riesz features
        combined = torch.cat([x_flat, riesz_feats], dim=1)  # (B*N, C*(1+n), H, W)

        # Reshape back to group format
        total_C = combined.shape[1]
        combined = combined.reshape(B, N, total_C, H, W).permute(0, 2, 1, 3, 4)
        # (B, total_C, N, H, W)

        # 1x1 fusion across channels (treats N as depth dim for Conv3d)
        out = self.fusion(combined)   # (B, out_channels, N, H, W)
        return out

    def riesz_regularization(self) -> torch.Tensor:
        loss = sum(rw.regularization() for rw in self.riesz_weights)
        return self.reg_lambda * loss
