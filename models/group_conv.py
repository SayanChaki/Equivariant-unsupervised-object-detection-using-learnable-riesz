"""
Cyclic Group Convolution Primitives
====================================
Implements G-CNN group convolutions over the cyclic group G_N
(N-fold discrete rotations) following:
  - Cohen & Welling, "Group Equivariant CNNs", ICML 2016
  - Han et al., "ReDet: A Rotation-Equivariant Detector", CVPR 2021

Convention
----------
Feature maps have shape  (B, C, N, H, W)
where N is the number of rotation channels (group size).

Two convolution types:
  LiftingConv   : R^2 -> G_N   (first layer, input has no group dim)
  GroupConv     : G_N -> G_N   (subsequent layers)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _rotate_filter(weight: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    Rotate a 2-D convolutional kernel by `angle_deg` degrees using bilinear
    interpolation in the Fourier / spatial domain.

    weight : (C_out, C_in, kH, kW)
    returns the same shape
    """
    angle_rad = torch.tensor(angle_deg * math.pi / 180.0)
    cos_a = torch.cos(angle_rad).item()
    sin_a = torch.sin(angle_rad).item()

    theta = torch.tensor(
        [[cos_a, -sin_a, 0.0],
         [sin_a,  cos_a, 0.0]],
        dtype=weight.dtype, device=weight.device
    ).unsqueeze(0)  # (1, 2, 3)

    # flatten C_out * C_in into batch dim for grid_sample
    C_out, C_in, kH, kW = weight.shape
    w = weight.view(C_out * C_in, 1, kH, kW)
    grid = F.affine_grid(theta.expand(C_out * C_in, -1, -1),
                         w.size(), align_corners=True)
    rotated = F.grid_sample(w, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
    return rotated.view(C_out, C_in, kH, kW)


def _build_rotated_filters(weight: torch.Tensor, N: int) -> torch.Tensor:
    """
    Build N rotated copies of `weight`.
    weight  : (C_out, C_in, kH, kW)
    returns : (N, C_out, C_in, kH, kW)
    """
    copies = []
    for k in range(N):
        angle = k * 360.0 / N
        copies.append(_rotate_filter(weight, angle))
    return torch.stack(copies, dim=0)  # (N, C_out, C_in, kH, kW)


# ---------------------------------------------------------------------------
# Lifting convolution  R^2 -> G_N
# ---------------------------------------------------------------------------

class LiftingConv(nn.Module):
    """
    First layer: maps a standard feature map (B, C_in, H, W) to a
    group feature map (B, C_out, N, H, W).

    It applies the base filter at N orientations, producing N
    equivariant feature maps.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, N: int = 8,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.N = N
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
            * math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        )
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C_in, H, W)
        rotated = _build_rotated_filters(self.weight, self.N)
        # rotated : (N, C_out, C_in, kH, kW)
        N, C_out, C_in, kH, kW = rotated.shape
        # merge N and C_out -> treat as independent output channels
        w = rotated.view(N * C_out, C_in, kH, kW)
        out = F.conv2d(x, w, stride=self.stride, padding=self.padding)
        # out : (B, N*C_out, H', W')
        B, _, H_, W_ = out.shape
        out = out.view(B, C_out, N, H_, W_)
        if self.bias_param is not None:
            out = out + self.bias_param.view(1, C_out, 1, 1, 1)
        return out  # (B, C_out, N, H', W')


# ---------------------------------------------------------------------------
# Group convolution  G_N -> G_N
# ---------------------------------------------------------------------------

class GroupConv(nn.Module):
    """
    Subsequent layers: maps (B, C_in, N, H, W) -> (B, C_out, N, H, W).

    For each output orientation g, the input is cyclically shifted so
    that the convolution kernel is applied in the "frame" of g.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, N: int = 8,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.N = N
        self.stride = stride
        self.padding = padding
        # weight shape: (C_out, C_in * N, kH, kW)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, N, kernel_size, kernel_size)
            * math.sqrt(2.0 / (in_channels * N * kernel_size * kernel_size))
        )
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C_in, N, H, W)
        B, C_in, N, H, W = x.shape
        assert N == self.N

        outputs = []
        for k in range(N):
            # Cyclically shift the orientation axis by k
            x_shifted = torch.roll(x, shifts=-k, dims=2)         # (B, C_in, N, H, W)
            x_flat = x_shifted.reshape(B, C_in * N, H, W)

            # Rotate the spatial kernel by k * 360/N degrees
            angle = k * 360.0 / N
            w = self.weight.reshape(
                self.weight.shape[0], C_in * N,
                self.weight.shape[3], self.weight.shape[4]
            )
            w_rot = _rotate_filter(w, angle)
            out_k = F.conv2d(x_flat, w_rot,
                             stride=self.stride, padding=self.padding)
            outputs.append(out_k)  # (B, C_out, H', W')

        out = torch.stack(outputs, dim=2)  # (B, C_out, N, H', W')
        if self.bias_param is not None:
            out = out + self.bias_param.view(1, -1, 1, 1, 1)
        return out


# ---------------------------------------------------------------------------
# Group Batch Norm
# ---------------------------------------------------------------------------

class GroupBatchNorm(nn.Module):
    """BatchNorm applied jointly over the N orientation channels."""

    def __init__(self, num_features: int, N: int, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, N, H, W)  treat (C) as channels for BN3d
        return self.bn(x)


# ---------------------------------------------------------------------------
# Group Max-Pooling over orientations (rotation-invariant pooling)
# ---------------------------------------------------------------------------

class GroupMaxPool(nn.Module):
    """
    Pools over the N orientation dimension to produce rotation-invariant
    features: A_inv(t,s) = max_r  A_RT(t, s, r)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, N, H, W)
        return x.max(dim=2).values  # (B, C, H, W)
