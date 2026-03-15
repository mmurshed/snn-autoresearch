"""Custom autograd for surrogate gradient injection and LIF neuron layer.

SurrogateSpike replaces the Heaviside step's zero derivative with an arbitrary
surrogate gradient during backpropagation. LIFNeuron wraps this into a reusable
leaky integrate-and-fire neuron module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .candidate import SurrogateCandidate


class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, surrogate gradient backward."""

    @staticmethod
    def forward(ctx, membrane, threshold, surrogate_fn):
        ctx.save_for_backward(membrane, threshold)
        ctx.surrogate_fn = surrogate_fn
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold = ctx.saved_tensors
        centered = membrane - threshold
        return grad_output * ctx.surrogate_fn(centered), None, None


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with configurable surrogate gradient."""

    def __init__(
        self,
        spike_fn: callable,
        beta: float = 0.9,
        threshold: float = 1.0,
        learn_beta: bool = False,
    ):
        super().__init__()
        self.spike_fn = spike_fn
        if learn_beta:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("threshold", torch.tensor(threshold))

    def forward(self, input_current: torch.Tensor, mem: torch.Tensor | None = None):
        if mem is None:
            mem = torch.zeros_like(input_current)
        mem = self.beta * mem + input_current
        spk = self.spike_fn(mem, self.threshold)
        mem = mem * (1 - spk)  # soft reset
        return spk, mem


def make_spike_fn(candidate: SurrogateCandidate) -> callable:
    """Build a spike function with surrogate gradient from a candidate.

    Returns a callable(membrane, threshold) -> spikes.
    """
    surrogate_fn = _build_torch_fn(candidate)

    def spike_fn(membrane, threshold=torch.tensor(1.0)):
        return SurrogateSpike.apply(membrane, threshold, surrogate_fn)

    return spike_fn


def _build_torch_fn(candidate: SurrogateCandidate) -> callable:
    """Convert a candidate's python_expr to a torch-compatible function."""
    ns = {
        "__builtins__": {},
        "torch": torch,
        "np": torch,  # allow np.exp etc. to resolve to torch
        "exp": torch.exp,
        "tanh": torch.tanh,
        "abs": torch.abs,
        "sigmoid": torch.sigmoid,
        "clip": torch.clamp,
        **{k: torch.tensor(v) for k, v in candidate.params.items()},
    }
    expr = candidate.python_expr.replace("np.clip", "torch.clamp").replace("np.", "torch.")
    try:
        fn = eval(f"lambda x: {expr}", ns)
        # Smoke test
        fn(torch.tensor([0.0]))
        return fn
    except Exception:
        # Fallback: standard sigmoid surrogate
        alpha = torch.tensor(10.0)
        return lambda x: alpha * torch.sigmoid(alpha * x) * (1 - torch.sigmoid(alpha * x))
