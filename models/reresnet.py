"""
ReResNet -- Rotation-Equivariant ResNet
========================================
Implements the rotation-equivariant ResNet backbone used in ReDet
(Han et al., CVPR 2021) and adopted in LeaRN-CompSTN.

All residual blocks operate in group feature space (B, C, N, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .group_conv import LiftingConv, GroupConv, GroupBatchNorm, GroupMaxPool


class GroupResBlock(nn.Module):
    """Residual block on group feature maps (B, C, N, H, W)."""

    def __init__(self, in_channels: int, out_channels: int, N: int,
                 stride: int = 1):
        super().__init__()
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                GroupConv(in_channels, out_channels, kernel_size=1, N=N,
                          stride=stride, padding=0),
                GroupBatchNorm(out_channels, N),
            )
        self.bn1   = GroupBatchNorm(in_channels, N)
        self.conv1 = GroupConv(in_channels, out_channels, 3, N=N,
                               stride=stride, padding=1)
        self.bn2   = GroupBatchNorm(out_channels, N)
        self.conv2 = GroupConv(out_channels, out_channels, 3, N=N,
                               stride=1, padding=1)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        return out + identity


def _make_layer(in_ch, out_ch, N, num_blocks, stride=1):
    layers = [GroupResBlock(in_ch, out_ch, N, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(GroupResBlock(out_ch, out_ch, N, stride=1))
    return nn.Sequential(*layers)


class ReResNet(nn.Module):
    """
    Rotation-Equivariant ResNet.

    Parameters
    ----------
    in_channels   : input image channels (default 1 for MNIST)
    base_channels : width of first group-conv layer
    N             : cyclic group size
    layers        : block counts per stage
    keep_group_dim: if True return (B, C, N, H, W), else apply GroupMaxPool
    """

    def __init__(self, in_channels=1, base_channels=32, N=8,
                 layers=(2,2,2,2), keep_group_dim=True):
        super().__init__()
        self.N = N
        self.keep_group_dim = keep_group_dim
        C = base_channels

        self.stem = nn.Sequential(
            LiftingConv(in_channels, C, kernel_size=7, N=N,
                        stride=2, padding=3),
            GroupBatchNorm(C, N),
            nn.ReLU(inplace=True),
        )

        self.stage1 = _make_layer(C,     C,     N, layers[0], stride=1)
        self.stage2 = _make_layer(C,     C*2,   N, layers[1], stride=2)
        self.stage3 = _make_layer(C*2,   C*4,   N, layers[2], stride=2)
        self.stage4 = _make_layer(C*4,   C*8,   N, layers[3], stride=2)

        self.group_pool = GroupMaxPool()
        self.out_channels = C * 8

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (LiftingConv, GroupConv)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        out = self.stem(x)                       # (B, C, N, H/2, W/2)

        # Apply spatial maxpool on each orientation
        B, C, N, H, W = out.shape
        out_flat = out.permute(0, 2, 1, 3, 4).reshape(B*N, C, H, W)
        out_flat = F.max_pool2d(out_flat, kernel_size=3, stride=2, padding=1)
        _, _, Hp, Wp = out_flat.shape
        out = out_flat.reshape(B, N, C, Hp, Wp).permute(0, 2, 1, 3, 4)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        if not self.keep_group_dim:
            out = self.group_pool(out)
        return out
