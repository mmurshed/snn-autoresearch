"""Train an SNN with a single surrogate gradient function.

Runs one training session and outputs structured results. This script can be
used standalone for manual experiments or called by loop.py for automated
discovery.

Usage:
    # Train with a baseline surrogate
    uv run train.py --surrogate sigmoid --dataset shd --epochs 30

    # Train with a custom expression
    uv run train.py --expr "(1 - np.tanh(x / alpha)**2) / (4 * alpha)" \
                     --params '{"alpha": 2.0}' --dataset shd --epochs 30

    # Quick test with placeholder data
    uv run train.py --surrogate sigmoid --placeholder --epochs 3
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from prepare import DATASETS, load_shd, make_placeholder_loaders
from snn_autoresearch.candidate import BASELINES, SurrogateCandidate
from snn_autoresearch.evaluate import train_and_evaluate
from snn_autoresearch.spike import make_spike_fn
from snn_autoresearch.models import RecurrentSNN, SpikingResNet18, SpikingVGG11

# ── Hyperparameters ─────────────────────────────────────────────────────

LEARNING_RATE = 5e-3
BATCH_SIZE = 128
BETA = 0.9
DROPOUT = 0.2


def build_model(dataset: str, spike_fn):
    """Build the appropriate SNN model for a dataset."""
    cfg = DATASETS[dataset]
    if dataset == "shd":
        return RecurrentSNN(
            n_inputs=cfg["n_inputs"],
            n_hidden=256,
            n_classes=cfg["n_classes"],
            n_steps=cfg["n_steps"],
            spike_fn=spike_fn,
            beta=BETA,
            dropout=DROPOUT,
        )
    else:
        in_ch = cfg.get("in_channels", 2)
        arch = cfg.get("arch", "resnet18")
        if arch == "vgg11":
            return SpikingVGG11(
                in_channels=in_ch,
                n_classes=cfg["n_classes"],
                n_steps=cfg["n_steps"],
                spike_fn=spike_fn,
            )
        else:
            return SpikingResNet18(
                in_channels=in_ch,
                n_classes=cfg["n_classes"],
                n_steps=cfg["n_steps"],
                spike_fn=spike_fn,
            )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train SNN with a surrogate gradient")
    parser.add_argument("--surrogate", default=None, help="Baseline name: sigmoid, fast_sigmoid, gaussian, arctan")
    parser.add_argument("--expr", default=None, help="Custom python_expr for surrogate")
    parser.add_argument("--params", default="{}", help="JSON params for custom expr")
    parser.add_argument("--name", default="custom", help="Name for custom surrogate")
    parser.add_argument("--dataset", default="shd", choices=list(DATASETS))
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--placeholder", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Resolve surrogate
    if args.surrogate:
        matches = [b for b in BASELINES if b.name == args.surrogate]
        if not matches:
            print(f"Unknown baseline: {args.surrogate!r}", file=sys.stderr)
            print(f"Available: {', '.join(b.name for b in BASELINES)}", file=sys.stderr)
            sys.exit(1)
        candidate = matches[0]
    elif args.expr:
        candidate = SurrogateCandidate(
            name=args.name,
            symbolic_expr="",
            python_expr=args.expr,
            params=json.loads(args.params),
            source="custom",
        )
    else:
        print("Provide --surrogate or --expr", file=sys.stderr)
        sys.exit(1)

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    # Load data
    if args.placeholder:
        train_loader, test_loader = make_placeholder_loaders(args.dataset, BATCH_SIZE)
    elif args.dataset == "shd":
        from pathlib import Path
        train_loader, _, test_loader = load_shd(Path(args.data_dir) / "shd", BATCH_SIZE)
    else:
        print(f"Real data for {args.dataset!r} requires manual setup. Use --placeholder.", file=sys.stderr)
        sys.exit(1)

    # Build model and train
    spike_fn = make_spike_fn(candidate)
    model = build_model(args.dataset, spike_fn)
    result = train_and_evaluate(model, train_loader, test_loader, args.epochs, args.lr, device)

    # Output (autoresearch-style structured output)
    print("---")
    print(f"surrogate:         {candidate.name}")
    print(f"test_accuracy:     {result.best_accuracy:.6f}")
    print(f"training_seconds:  {result.training_time:.1f}")
    print(f"convergence_epoch: {result.convergence_epoch}")
    print(f"total_spikes:      {result.total_spikes}")
    print(f"n_epochs:          {args.epochs}")
    print(f"dataset:           {args.dataset}")
    print(f"device:            {device}")
    print(f"seed:              {args.seed}")


if __name__ == "__main__":
    main()
