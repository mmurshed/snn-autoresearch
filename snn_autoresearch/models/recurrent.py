"""Recurrent SNN for temporal classification (e.g., SHD).

Architecture: FC → LIF (with recurrence) → FC → LIF
Input shape:  (batch, n_steps, n_inputs)
Output shape: (n_steps, batch, n_classes) spike record + info dict
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..spike import LIFNeuron


class RecurrentSNN(nn.Module):
    def __init__(
        self,
        n_inputs: int = 700,
        n_hidden: int = 256,
        n_classes: int = 20,
        n_steps: int = 100,
        spike_fn: callable = None,
        beta: float = 0.9,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.fc2 = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(dropout)

        self.lif1 = LIFNeuron(spike_fn, beta=beta)
        self.lif2 = LIFNeuron(spike_fn, beta=beta)

    def forward(self, x: torch.Tensor):
        batch = x.size(0)
        device = x.device

        mem1 = torch.zeros(batch, self.n_hidden, device=device)
        mem2 = torch.zeros(batch, self.n_classes, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)

        spk_rec = []
        total_spikes = 0

        for t in range(self.n_steps):
            inp = x[:, t] if t < x.size(1) else torch.zeros(batch, x.size(2), device=device)

            cur1 = self.fc1(inp) + self.rec(spk1)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_drop = self.dropout(spk1)

            cur2 = self.fc2(spk1_drop)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_rec.append(spk2)
            total_spikes += int(spk1.sum().item()) + int(spk2.sum().item())

        return torch.stack(spk_rec), {"total_spikes": total_spikes}
