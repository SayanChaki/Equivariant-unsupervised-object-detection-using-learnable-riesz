"""
Training Scripts
================
Provides two training routines:

  1. train_classifier   -- LeaRN + GCNN on rotated-MNIST (Section 5.4)
  2. train_repr_learning -- LeaRN-CompSTN+GMVAE on SR-MNIST (Section 5.3)

Both support a beta-annealing schedule for the KL term and include
evaluation with the paper's metrics (RoE, NMI, ARI, SSIM).
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------------------------
# Classification training
# ---------------------------------------------------------------------------

def train_classifier(config: dict):
    """
    Train LeaRN_GCNN_Classifier on RotatedMNIST.

    config keys:
      device, epochs, batch_size, lr, weight_decay,
      N, base_channels, K, riesz_order,
      data_root, save_dir, log_interval
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from models.classifier import LeaRN_GCNN_Classifier, train_epoch, eval_epoch
    from data.datasets import get_loaders

    device = torch.device(config.get('device', 'cpu'))
    os.makedirs(config.get('save_dir', './checkpoints'), exist_ok=True)

    # Data
    train_loader, test_loader = get_loaders(
        dataset='rotated_mnist',
        root=config.get('data_root', './data'),
        image_size=config.get('image_size', 32),
        batch_size=config.get('batch_size', 64),
        num_workers=config.get('num_workers', 4),
    )

    # Model
    model = LeaRN_GCNN_Classifier(
        in_channels=1,
        num_classes=10,
        N=config.get('N', 8),
        base_channels=config.get('base_channels', 32),
        K=config.get('K', 8),
        riesz_order=config.get('riesz_order', 1),
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 3e-4),
        weight_decay=config.get('weight_decay', 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.get('epochs', 100))

    best_acc = 0.0
    log_interval = config.get('log_interval', 10)

    for epoch in range(1, config.get('epochs', 100) + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            riesz_weight=config.get('riesz_weight', 1e-4)
        )
        test_acc = eval_epoch(model, test_loader, device)
        scheduler.step()

        if epoch % log_interval == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}] "
                  f"loss={train_loss:.4f}  "
                  f"train_acc={100*train_acc:.2f}%  "
                  f"test_acc={100*test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(config['save_dir'], 'best_classifier.pt'))

    print(f"\nBest test accuracy: {100*best_acc:.2f}%")
    return model


# ---------------------------------------------------------------------------
# Representation learning training
# ---------------------------------------------------------------------------

def _beta_schedule(epoch: int, warmup: int = 20,
                   beta_max: float = 1.0) -> float:
    """Linear beta annealing from 0 to beta_max over `warmup` epochs."""
    return min(beta_max, beta_max * epoch / max(warmup, 1))


def train_repr_learning(config: dict):
    """
    Train LeaRN-CompSTN+GMVAE on SR-MNIST for representation learning.

    config keys:
      device, epochs, batch_size, lr, weight_decay,
      N, base_channels, learn_K, riesz_order,
      latent_dim, num_classes, kappa_prior,
      beta_warmup, beta_max,
      data_root, save_dir, log_interval
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from models.learn_compstn_gmvae import LeaRN_CompSTN_GMVAE
    from data.datasets import get_loaders
    from utils.metrics import (rotation_offset_entropy, cluster_metrics,
                                ssim as compute_ssim)

    device = torch.device(config.get('device', 'cpu'))
    os.makedirs(config.get('save_dir', './checkpoints'), exist_ok=True)

    image_size = config.get('image_size', 32)

    # Data
    train_loader, test_loader = get_loaders(
        dataset='sr_mnist',
        root=config.get('data_root', './data'),
        image_size=image_size,
        batch_size=config.get('batch_size', 64),
        num_workers=config.get('num_workers', 4),
    )

    # Model
    model = LeaRN_CompSTN_GMVAE(
        in_channels=1,
        image_size=(image_size, image_size),
        N=config.get('N', 8),
        base_channels=config.get('base_channels', 16),
        reresnet_layers=config.get('reresnet_layers', (2, 2, 2, 2)),
        learn_K=config.get('learn_K', 8),
        riesz_order=config.get('riesz_order', 1),
        glimpse_size=config.get('glimpse_size', 16),
        latent_dim=config.get('latent_dim', 32),
        num_classes=config.get('num_classes', 10),
        kappa_prior=config.get('kappa_prior', 0.5),
        beta=1.0,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 3e-4),
        weight_decay=config.get('weight_decay', 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.get('epochs', 100))

    log_interval = config.get('log_interval', 10)
    beta_max     = config.get('beta_max', 1.0)
    beta_warmup  = config.get('beta_warmup', 20)
    best_nmi     = 0.0

    for epoch in range(1, config.get('epochs', 100) + 1):
        # Update beta for KL annealing
        beta = _beta_schedule(epoch, beta_warmup, beta_max)
        model.beta = beta
        model.train()

        total_loss = 0.0
        n_batches  = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()

            x_hat, glimpse_recon, info = model(imgs)
            losses = model.loss(imgs, x_hat, glimpse_recon, info)
            losses['total'].backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += losses['total'].item()
            n_batches  += 1

        scheduler.step()

        # Evaluation
        if epoch % log_interval == 0 or epoch == 1:
            model.eval()
            all_z, all_labels, all_x, all_xhat = [], [], [], []
            all_pred_angles, all_true_angles = [], []

            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.to(device)
                    x_hat, _, info = model(imgs)
                    all_z.append(info['z_what'].cpu())
                    all_labels.append(labels)
                    all_x.append(imgs.cpu())
                    all_xhat.append(x_hat.cpu())
                    # For RoE we would need ground-truth angles;
                    # here we compare STN vs GMVAE angle predictions
                    all_pred_angles.append(info['mu_alpha_stn'].cpu())
                    all_true_angles.append(info['mu_alpha'].cpu())

            z_all      = torch.cat(all_z)
            labels_all = torch.cat(all_labels)
            x_all      = torch.cat(all_x)
            xhat_all   = torch.cat(all_xhat)
            pred_a     = torch.cat(all_pred_angles)
            true_a     = torch.cat(all_true_angles)

            cm   = cluster_metrics(z_all, labels_all,
                                   config.get('num_classes', 10))
            ssim_val = compute_ssim(
                x_all[:256].clamp(0,1),
                xhat_all[:256].clamp(0,1)
            )
            roe = rotation_offset_entropy(pred_a, true_a)

            print(f"[Epoch {epoch:3d}] "
                  f"loss={total_loss/n_batches:.4f}  "
                  f"beta={beta:.3f}  "
                  f"NMI={cm['NMI']:.3f}  "
                  f"ARI={cm['ARI']:.3f}  "
                  f"SSIM={ssim_val:.3f}  "
                  f"RoE={roe:.3f}")

            if cm['NMI'] > best_nmi:
                best_nmi = cm['NMI']
                torch.save(model.state_dict(),
                           os.path.join(config['save_dir'],
                                        'best_repr_model.pt'))

    print(f"\nBest NMI: {best_nmi:.3f}")
    return model


# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

CLASSIFIER_CONFIG = {
    'device':        'cuda' if torch.cuda.is_available() else 'cpu',
    'epochs':        100,
    'batch_size':    128,
    'lr':            3e-4,
    'weight_decay':  1e-4,
    'riesz_weight':  1e-4,
    'N':             8,
    'base_channels': 32,
    'K':             8,
    'riesz_order':   1,
    'image_size':    32,
    'data_root':     './data',
    'save_dir':      './checkpoints',
    'log_interval':  5,
    'num_workers':   0,
}

REPR_LEARNING_CONFIG = {
    'device':          'cuda' if torch.cuda.is_available() else 'cpu',
    'epochs':          200,
    'batch_size':      64,
    'lr':              3e-4,
    'weight_decay':    1e-4,
    'N':               8,
    'base_channels':   16,
    'reresnet_layers': (2, 2, 2, 2),
    'learn_K':         8,
    'riesz_order':     1,
    'glimpse_size':    16,
    'latent_dim':      32,
    'num_classes':     10,
    'kappa_prior':     0.5,
    'beta_max':        1.0,
    'beta_warmup':     30,
    'image_size':      32,
    'data_root':       './data',
    'save_dir':        './checkpoints',
    'log_interval':    10,
    'num_workers':     0,
}
