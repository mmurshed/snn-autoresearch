"""Data preparation for SNN experiments.

Downloads and preprocesses neuromorphic datasets. Run once before training.

Usage:
    uv run prepare.py --dataset shd --data-dir data/
    uv run prepare.py --dataset shd --placeholder   # synthetic data for testing

Supported datasets:
    shd         Spiking Heidelberg Digits (700 input channels, 20 classes)
    cifar10dvs  CIFAR10-DVS (2 polarity channels, 10 classes)
    nmnist      Neuromorphic MNIST (2 polarity channels, 10 classes)
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ── Dataset configs ─────────────────────────────────────────────────────

DATASETS = {
    "shd": {
        "n_inputs": 700,
        "n_classes": 20,
        "n_steps": 100,
        "arch": "recurrent",
        "train_url": "https://zenodo.org/records/7044500/files/shd_train.h5",
        "test_url": "https://zenodo.org/records/7044500/files/shd_test.h5",
    },
    "cifar10dvs": {
        "in_channels": 2,
        "n_classes": 10,
        "n_steps": 10,
        "spatial": (48, 48),
        "arch": "resnet18",
    },
    "nmnist": {
        "in_channels": 2,
        "n_classes": 10,
        "n_steps": 10,
        "spatial": (34, 34),
        "arch": "resnet18",
    },
}


def download_shd(data_dir: Path):
    """Download SHD dataset from Zenodo."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = DATASETS["shd"]

    for split, url in [("train", cfg["train_url"]), ("test", cfg["test_url"])]:
        dest = data_dir / f"shd_{split}.h5"
        if dest.exists():
            print(f"  {dest} already exists, skipping")
            continue
        print(f"  Downloading {split} split...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")


def load_shd(data_dir: Path, batch_size: int = 128, val_split: float = 0.15):
    """Load SHD dataset from h5 files. Returns (train_loader, val_loader, test_loader)."""
    import h5py

    cfg = DATASETS["shd"]
    n_steps, n_inputs = cfg["n_steps"], cfg["n_inputs"]

    def _load_h5(path):
        with h5py.File(path, "r") as f:
            times = f["spikes"]["times"]
            units = f["spikes"]["units"]
            labels = np.array(f["labels"])

        n_samples = len(labels)
        data = np.zeros((n_samples, n_steps, n_inputs), dtype=np.float32)
        for i in range(n_samples):
            t = np.array(times[i])
            u = np.array(units[i])
            # Bin spikes into time bins
            t_bins = np.clip((t * n_steps).astype(int), 0, n_steps - 1)
            u_bins = np.clip(u.astype(int), 0, n_inputs - 1)
            data[i, t_bins, u_bins] = 1.0
        return torch.tensor(data), torch.tensor(labels, dtype=torch.long)

    train_x, train_y = _load_h5(data_dir / "shd_train.h5")
    test_x, test_y = _load_h5(data_dir / "shd_test.h5")

    # Train/val split
    n = len(train_x)
    n_val = int(n * val_split)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(
        TensorDataset(train_x[train_idx], train_y[train_idx]),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(train_x[val_idx], train_y[val_idx]),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=batch_size,
    )
    return train_loader, val_loader, test_loader


def make_placeholder_loaders(dataset: str = "shd", batch_size: int = 64):
    """Create synthetic data loaders for testing (no download needed).

    Returns (train_loader, test_loader).
    """
    cfg = DATASETS[dataset]

    if dataset == "shd":
        n_steps, n_inputs, n_classes = cfg["n_steps"], cfg["n_inputs"], cfg["n_classes"]
        train_x = torch.rand(256, n_steps, n_inputs)
        train_y = torch.randint(0, n_classes, (256,))
        test_x = torch.rand(64, n_steps, n_inputs)
        test_y = torch.randint(0, n_classes, (64,))
    else:
        in_ch = cfg["in_channels"]
        h, w = cfg["spatial"]
        n_steps, n_classes = cfg["n_steps"], cfg["n_classes"]
        train_x = torch.rand(256, n_steps, in_ch, h, w)
        train_y = torch.randint(0, n_classes, (256,))
        test_x = torch.rand(64, n_steps, in_ch, h, w)
        test_y = torch.randint(0, n_classes, (64,))

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=batch_size,
    )
    return train_loader, test_loader


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare SNN datasets")
    parser.add_argument("--dataset", default="shd", choices=list(DATASETS))
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--placeholder", action="store_true", help="Skip download, verify placeholder works")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / args.dataset

    if args.placeholder:
        print(f"Creating placeholder loaders for {args.dataset}...")
        train_loader, test_loader = make_placeholder_loaders(args.dataset)
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches:  {len(test_loader)}")
        batch = next(iter(train_loader))
        print(f"  Input shape:   {batch[0].shape}")
        print(f"  Label shape:   {batch[1].shape}")
        print("Placeholder data ready.")
        return

    if args.dataset == "shd":
        print(f"Downloading SHD dataset to {data_dir}/...")
        download_shd(data_dir)
        print("Loading and verifying...")
        train_loader, val_loader, test_loader = load_shd(data_dir)
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches:   {len(val_loader)}")
        print(f"  Test batches:  {len(test_loader)}")
        print("SHD data ready.")
    else:
        print(f"Dataset {args.dataset!r} requires manual setup or tonic library.")
        print("Use --placeholder for synthetic data.")


if __name__ == "__main__":
    main()
