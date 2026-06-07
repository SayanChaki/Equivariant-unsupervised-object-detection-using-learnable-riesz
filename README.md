# LeaRN-CompSTN Codebase

Implementation of **"Equivariant Unsupervised Object Detection with Learnable
Riesz Transform and Composite Spatial Transformers"** (CVPR Findings 2026).

## Structure

```
learncompstn/
  models/
    group_conv.py          # LiftingConv, GroupConv, GroupBatchNorm, GroupMaxPool
    reresnet.py            # Rotation-Equivariant ResNet (Han et al., CVPR 2021)
    learn.py               # Learnable Riesz Transform (LeaRN) -- Eq. 2-3
    compstn.py             # Composite Spatial Transformer (CompSTN) -- Sec. 4.2
    gmvae.py               # Gaussian Mixture VAE -- Sec. 4.4
    classifier.py          # LeaRN+GCNN for classification (Table 4)
    learn_compstn_gmvae.py # Full pipeline for representation learning (Table 3)
  data/
    datasets.py            # RotatedMNIST, SR-MNIST, MQRT-MNIST
  utils/
    metrics.py             # RoE, NMI, ARI, SSIM, LEE
  train.py                 # Training loops for both tasks
  main.py                  # Entry point
```

## Tasks

### 1. Classification -- Table 4 (LeaRN + GCNN)

```bash
python main.py --task classify --epochs 100 --N 8 --device cuda
```

Architecture: `LiftingConv -> LeaRN -> GroupResBlocks -> GroupMaxPool -> Linear`

### 2. Representation Learning -- Table 3 (LeaRN-CompSTN+GMVAE)

```bash
python main.py --task repr --epochs 200 --N 8 --device cuda
```

Architecture: `ReResNet -> LeaRN -> CompSTN (step1: t,s) -> CompSTN (step2: alpha) -> GMVAE`

## Key Design Choices

| Component | Paper Section | Implementation |
|-----------|--------------|----------------|
| LiftingConv | Sec. 3.2 | Applies kernel at N rotations, R² -> G_N |
| GroupConv | Sec. 3.2 | Cyclic shift + rotated kernel, G_N -> G_N |
| LeaRN | Sec. 4.1, Eq. 2-3 | Learnable Laplace mixture weights on Riesz spectrum |
| CompSTN | Sec. 4.2 | Two-step: (t,s) from invariant features, alpha from equivariant |
| Von Mises z_alpha | Sec. 4.4 | Circular normal prior/posterior for rotation angle |
| GMVAE | Sec. 4.4 | Gaussian mixture prior on z_what, Gumbel-softmax z_cls |
| RoE metric | Sec. 5.1 | Entropy of angle-difference histogram, normalised to [0,1] |

## Install

```bash
pip install -r requirements.txt
```
