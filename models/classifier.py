"""
LeaRN + GCNN Classification Model
====================================
Implements the classification pipeline from Section 5.4 / Table 4 of the paper:

    LeaRN + GCNN  ->  98.44% on rotated-MNIST

Architecture
------------
    Input (B, 1, H, W)
        |
    LiftingConv  (R^2 -> G_N)
        |
    LeaRN        (learnable Riesz enhancement in G_N)
        |
    Stage 1-3    (GroupResBlocks)
        |
    GroupMaxPool (G_N -> R^2, rotation-invariant)
        |
    AdaptiveAvgPool
        |
    Linear head  -> num_classes logits

This module is self-contained and runnable without the full LeaRN-CompSTN
pipeline.  It reproduces the entry "LeaRN + GCNN" in Table 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .group_conv import LiftingConv, GroupConv, GroupBatchNorm, GroupMaxPool
from .learn import LeaRN


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class LeaRN_GCNN_Classifier(nn.Module):
    """
    LeaRN-enhanced G-CNN classifier.

    Parameters
    ----------
    in_channels  : image channels (1 for MNIST)
    num_classes  : output classes
    N            : cyclic group order (8 for 45-degree steps)
    base_channels: width of feature maps
    K            : number of Laplace mixture components in LeaRN
    riesz_order  : 1 or 2
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10,
                 N: int = 8,
                 base_channels: int = 32,
                 K: int = 8,
                 riesz_order: int = 1):
        super().__init__()
        self.N = N
        C = base_channels

        # --- Lifting: image -> group feature map ---
        self.lift = nn.Sequential(
            LiftingConv(in_channels, C, kernel_size=7, N=N,
                        stride=2, padding=3),
            GroupBatchNorm(C, N),
            nn.ReLU(inplace=True),
        )

        # --- LeaRN: enrich with Riesz features ---
        self.learn = LeaRN(in_channels=C, N=N, K=K, order=riesz_order,
                           out_channels=C)

        # --- Group residual stages ---
        self.stage1 = self._make_stage(C,     C * 2, N, n_blocks=2)
        self.stage2 = self._make_stage(C * 2, C * 4, N, n_blocks=2, stride=2)
        self.stage3 = self._make_stage(C * 4, C * 8, N, n_blocks=2, stride=2)

        # --- Invariant pooling ---
        self.group_pool = GroupMaxPool()                   # G_N -> R^2
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- Classifier head ---
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, N: int,
                    n_blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        # First block handles channel change and optional spatial stride
        downsample = None
        if in_ch != out_ch or stride != 1:
            downsample = nn.Sequential(
                GroupConv(in_ch, out_ch, kernel_size=1, N=N,
                          stride=stride, padding=0),
                GroupBatchNorm(out_ch, N),
            )

        class _ResBlock(nn.Module):
            def __init__(self, ic, oc, ds, s):
                super().__init__()
                self.bn1   = GroupBatchNorm(ic, N)
                self.conv1 = GroupConv(ic, oc, 3, N=N, stride=s, padding=1)
                self.bn2   = GroupBatchNorm(oc, N)
                self.conv2 = GroupConv(oc, oc, 3, N=N, stride=1, padding=1)
                self.ds    = ds
                self.relu  = nn.ReLU(inplace=True)

            def forward(self, x):
                identity = self.ds(x) if self.ds else x
                out = self.relu(self.bn1(x))
                out = self.conv1(out)
                out = self.relu(self.bn2(out))
                out = self.conv2(out)
                return out + identity

        layers.append(_ResBlock(in_ch, out_ch, downsample, stride))
        for _ in range(1, n_blocks):
            layers.append(_ResBlock(out_ch, out_ch, None, 1))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (LiftingConv, GroupConv)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, C_in, H, W)
        returns: (B, num_classes) logits
        """
        # Lifting to group space
        out = self.lift(x)           # (B, C, N, H/2, W/2)

        # Riesz enrichment
        out = self.learn(out)        # (B, C, N, H/2, W/2)

        # Group residual stages
        out = self.stage1(out)       # (B, 2C, N, H/2,  W/2)
        out = self.stage2(out)       # (B, 4C, N, H/4,  W/4)
        out = self.stage3(out)       # (B, 8C, N, H/8,  W/8)

        # Rotation-invariant pooling
        out = self.group_pool(out)   # (B, 8C, H/8, W/8)
        out = self.spatial_pool(out) # (B, 8C, 1, 1)

        return self.head(out)        # (B, num_classes)

    def riesz_loss(self) -> torch.Tensor:
        """Frequency regularisation from LeaRN (Eq. 3 weight penalty)."""
        return self.learn.riesz_regularization()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, riesz_weight=1e-4):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        ce_loss = F.cross_entropy(logits, labels)
        reg_loss = model.riesz_loss()
        loss = ce_loss + riesz_weight * reg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_classifier(num_classes: int = 10, **kwargs) -> LeaRN_GCNN_Classifier:
    return LeaRN_GCNN_Classifier(num_classes=num_classes, **kwargs)
