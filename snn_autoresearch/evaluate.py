"""Training and evaluation for SNN surrogate gradient candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class TrainingResult:
    """Metrics from a single training run."""

    epoch_losses: list[float] = field(default_factory=list)
    epoch_accuracies: list[float] = field(default_factory=list)
    best_accuracy: float = 0.0
    convergence_epoch: int = 0
    total_spikes: int = 0
    training_time: float = 0.0
    grad_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "best_accuracy": self.best_accuracy,
            "convergence_epoch": self.convergence_epoch,
            "total_spikes": self.total_spikes,
            "training_time": round(self.training_time, 1),
            "final_loss": self.epoch_losses[-1] if self.epoch_losses else None,
            "n_epochs": len(self.epoch_losses),
        }


def train_and_evaluate(
    model: nn.Module,
    train_loader,
    test_loader,
    n_epochs: int,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TrainingResult:
    """Train a model and return metrics."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.CrossEntropyLoss()

    result = TrainingResult()
    start = time.time()

    for epoch in range(n_epochs):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            spk_rec, _ = model(data)
            out = spk_rec.sum(dim=0)  # rate-coded readout
            loss = loss_fn(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            correct += (out.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
        scheduler.step()

        train_loss = total_loss / max(total, 1)
        result.epoch_losses.append(train_loss)

        # ── Evaluate ──
        acc, spikes = _evaluate(model, test_loader, device)
        result.epoch_accuracies.append(acc)
        if acc > result.best_accuracy:
            result.best_accuracy = acc
            result.convergence_epoch = epoch

    # Final spike count from last evaluation
    _, result.total_spikes = _evaluate(model, test_loader, device)
    result.training_time = time.time() - start
    result.grad_summary = _grad_summary(model)
    return result


@torch.no_grad()
def _evaluate(model: nn.Module, loader, device: str) -> tuple[float, int]:
    """Evaluate accuracy and spike count on a dataset."""
    model.eval()
    correct, total, total_spikes = 0, 0, 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_rec, info = model(data)
        out = spk_rec.sum(dim=0)
        correct += (out.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
        if isinstance(info, dict):
            for v in info.values():
                if isinstance(v, (int, float)):
                    total_spikes += int(v)
    return correct / max(total, 1), total_spikes


def _grad_summary(model: nn.Module) -> dict:
    """Snapshot of current gradient magnitudes per parameter."""
    summary = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            summary[name] = {
                "mean": round(float(p.grad.abs().mean()), 6),
                "std": round(float(p.grad.std()), 6),
            }
    return summary
