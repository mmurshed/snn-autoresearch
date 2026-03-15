from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import md5
from time import time_ns


@dataclass
class SurrogateCandidate:
    """A candidate surrogate gradient function for SNN training."""

    name: str
    symbolic_expr: str
    python_expr: str
    params: dict[str, float] = field(default_factory=dict)
    source: str = "llm"  # "llm" | "baseline" | "random"
    generation: int = 0
    reasoning: str = ""
    uid: str = field(init=False)

    def __post_init__(self):
        raw = f"{self.symbolic_expr}{self.params}{time_ns()}"
        self.uid = md5(raw.encode()).hexdigest()[:10]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "symbolic_expr": self.symbolic_expr,
            "python_expr": self.python_expr,
            "params": self.params,
            "source": self.source,
            "generation": self.generation,
            "reasoning": self.reasoning,
            "uid": self.uid,
        }


# ── Standard baselines with experimentally-validated parameters ─────────

BASELINES = [
    SurrogateCandidate(
        name="sigmoid",
        symbolic_expr="alpha * sigma(alpha*x) * (1 - sigma(alpha*x))",
        python_expr="alpha * np.exp(np.clip(-alpha * x, -50, 50)) / (1 + np.exp(np.clip(-alpha * x, -50, 50)))**2",
        params={"alpha": 2.0},
        source="baseline",
    ),
    SurrogateCandidate(
        name="fast_sigmoid",
        symbolic_expr="1 / (1 + alpha * |x|)^2",
        python_expr="1.0 / (1.0 + alpha * np.abs(x)) ** 2",
        params={"alpha": 2.0},
        source="baseline",
    ),
    SurrogateCandidate(
        name="gaussian",
        symbolic_expr="exp(-x^2 / (2 * sigma^2))",
        python_expr="np.exp(-x**2 / (2.0 * sigma**2))",
        params={"sigma": 0.5},
        source="baseline",
    ),
    SurrogateCandidate(
        name="arctan",
        symbolic_expr="alpha / (pi * (1 + (alpha*x)^2))",
        python_expr="alpha / (3.14159265 * (1.0 + (alpha * x)**2))",
        params={"alpha": 2.0},
        source="baseline",
    ),
]
