"""
LeaRN-CompSTN Codebase
========================
Main entry point.

Usage
-----
# Classification (Table 4: LeaRN + GCNN on rotated-MNIST)
python main.py --task classify

# Representation learning (Table 3: LeaRN-CompSTN+GMVAE on SR-MNIST)
python main.py --task repr

# Custom config overrides
python main.py --task classify --epochs 50 --batch_size 256 --device cpu
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from train import (train_classifier, train_repr_learning,
                   CLASSIFIER_CONFIG, REPR_LEARNING_CONFIG)


def parse_args():
    p = argparse.ArgumentParser(description='LeaRN-CompSTN')
    p.add_argument('--task', choices=['classify', 'repr'],
                   default='classify')
    p.add_argument('--epochs',      type=int,   default=None)
    p.add_argument('--batch_size',  type=int,   default=None)
    p.add_argument('--lr',          type=float, default=None)
    p.add_argument('--device',      type=str,   default=None)
    p.add_argument('--N',           type=int,   default=None,
                   help='Cyclic group order')
    p.add_argument('--base_channels', type=int, default=None)
    p.add_argument('--data_root',   type=str,   default=None)
    p.add_argument('--save_dir',    type=str,   default=None)
    p.add_argument('--num_workers', type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()

    if args.task == 'classify':
        config = dict(CLASSIFIER_CONFIG)
    else:
        config = dict(REPR_LEARNING_CONFIG)

    # Apply overrides
    for key in ['epochs', 'batch_size', 'lr', 'device',
                'N', 'base_channels', 'data_root', 'save_dir', 'num_workers']:
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    print(f"\n{'='*60}")
    print(f"Task      : {args.task}")
    print(f"Device    : {config['device']}")
    print(f"Epochs    : {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Group N   : {config['N']}")
    print(f"{'='*60}\n")

    os.makedirs(config['save_dir'], exist_ok=True)

    if args.task == 'classify':
        train_classifier(config)
    else:
        train_repr_learning(config)


if __name__ == '__main__':
    main()
