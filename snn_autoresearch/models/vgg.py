"""Spiking VGG-11 for spatial event-based classification (e.g., CIFAR10-DVS).

Architecture: 6 Conv+LIF blocks with pooling → Global AvgPool → FC → LIF
Input shape:  (batch, n_steps, channels, H, W)
Output shape: (n_steps, batch, n_classes) spike record + info dict
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..spike import LIFNeuron

# VGG-11 channel config; pool after layers at indices 0, 1, 3, 5
_CHANNELS = [64, 128, 256, 256, 512, 512]
_POOL_AFTER = {0, 1, 3, 5}


class SpikingVGG11(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        n_classes: int = 10,
        n_steps: int = 10,
        spike_fn: callable = None,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_layers = len(_CHANNELS)

        # Build conv layers
        convs = []
        lifs = []
        ch_in = in_channels
        for ch_out in _CHANNELS:
            convs.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
                    nn.BatchNorm2d(ch_out),
                )
            )
            lifs.append(LIFNeuron(spike_fn))
            ch_in = ch_out
        self.convs = nn.ModuleList(convs)
        self.lifs = nn.ModuleList(lifs)
        self.pool = nn.MaxPool2d(2, 2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(_CHANNELS[-1], n_classes)
        self.lif_out = LIFNeuron(spike_fn)

    def forward(self, x: torch.Tensor):
        batch = x.size(0)
        spk_rec = []
        total_spikes = 0

        # Persistent membrane states per layer
        mems = [None] * self.n_layers
        mem_out = None

        for t in range(self.n_steps):
            inp = x[:, t] if x.dim() == 5 else x
            out = inp

            for i, (conv, lif) in enumerate(zip(self.convs, self.lifs)):
                out = conv(out)
                spk, mems[i] = lif(out, mems[i])
                if i in _POOL_AFTER:
                    spk = self.pool(spk)
                out = spk

            pooled = self.global_pool(out).flatten(1)
            logits = self.fc(pooled)
            spk_out, mem_out = self.lif_out(logits, mem_out)

            spk_rec.append(spk_out)
            total_spikes += int(spk_out.sum().item())

        return torch.stack(spk_rec), {"total_spikes": total_spikes}
