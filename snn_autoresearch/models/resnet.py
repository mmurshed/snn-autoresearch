"""Spiking ResNet-18 for spatial event-based classification (e.g., CIFAR10-DVS).

Architecture: Conv → [ResBlock, ResBlock] → AvgPool → FC → LIF
Input shape:  (batch, n_steps, channels, H, W)
Output shape: (n_steps, batch, n_classes) spike record + info dict
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..spike import LIFNeuron


class _SpikingResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, spike_fn, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.lif1 = LIFNeuron(spike_fn)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.lif2 = LIFNeuron(spike_fn)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x, mem1=None, mem2=None):
        identity = x
        out = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(out, mem1)
        out = self.bn2(self.conv2(spk1))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        spk2, mem2 = self.lif2(out, mem2)
        return spk2, mem1, mem2


class SpikingResNet18(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        n_classes: int = 10,
        n_steps: int = 10,
        spike_fn: callable = None,
    ):
        super().__init__()
        self.n_steps = n_steps

        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.lif_in = LIFNeuron(spike_fn)

        self.block1 = _SpikingResBlock(64, 64, spike_fn)
        self.block2 = _SpikingResBlock(64, 128, spike_fn, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, n_classes)
        self.lif_out = LIFNeuron(spike_fn)

    def forward(self, x: torch.Tensor):
        batch = x.size(0)
        device = x.device
        spk_rec = []
        total_spikes = 0

        # Persistent membrane states
        mem_in = None
        b1_m1 = b1_m2 = b2_m1 = b2_m2 = None
        mem_out = None

        for t in range(self.n_steps):
            inp = x[:, t] if x.dim() == 5 else x
            out = self.pool(self.bn1(self.conv1(inp)))
            spk_in, mem_in = self.lif_in(out, mem_in)

            spk1, b1_m1, b1_m2 = self.block1(spk_in, b1_m1, b1_m2)
            spk2, b2_m1, b2_m2 = self.block2(spk1, b2_m1, b2_m2)

            pooled = self.avgpool(spk2).flatten(1)
            logits = self.fc(pooled)
            spk_out, mem_out = self.lif_out(logits, mem_out)

            spk_rec.append(spk_out)
            total_spikes += int(spk_in.sum().item()) + int(spk_out.sum().item())

        return torch.stack(spk_rec), {"total_spikes": total_spikes}
