"""Prompt templates for surrogate gradient discovery."""

SYSTEM_PROMPT = """\
You are an expert in spiking neural network optimization, specializing in \
surrogate gradient design for training SNNs with backpropagation through time.

Design surrogate gradient functions g(x) that replace the Heaviside step \
function's zero derivative during backpropagation. Your functions must satisfy:

1. g(x) >= 0 for all x  (non-negative)
2. g(x) -> 0 as |x| -> inf  (locality / tail decay)
3. Single peak near x = 0  (unimodal, strongest gradient at threshold)
4. Numerically stable for all finite inputs  (no NaN/Inf)

Available parameters: alpha (0.5-50), beta (0.01-5), sigma (0.1-10).

Output each candidate as a JSON object:
{
  "name": "descriptive_name",
  "symbolic_expr": "mathematical expression",
  "python_expr": "numpy expression using np.exp, np.tanh, np.abs, etc.",
  "params": {"alpha": 2.0},
  "reasoning": "brief explanation of design rationale"
}
"""


def build_generation_prompt(n: int = 8) -> str:
    """Initial generation prompt — produce N diverse candidates."""
    return f"""\
Generate {n} diverse surrogate gradient functions for spiking neural networks.

Requirements:
- At least 2 must be structurally different (not just parameter variations)
- At least 1 should prioritize deep network training (smooth, wide support)
- At least 1 should prioritize spike efficiency (sharp, narrow support)
- All must use only: np.exp, np.tanh, np.abs, np.clip, basic arithmetic
- Parameters must be inlined in python_expr (e.g., use the variable name 'alpha')

Output exactly {n} JSON objects, one per candidate."""


def build_refinement_prompt(top_results: list[dict], n_new: int = 4) -> str:
    """Refinement prompt — improve on top performers."""
    lines = []
    for r in top_results:
        parts = [f"  - {r['name']}: accuracy={r['accuracy']:.4f}"]
        if "grad_norm" in r:
            parts.append(f"grad_norm={r['grad_norm']:.4f}")
        if "total_spikes" in r:
            parts.append(f"spikes={r['total_spikes']}")
        if "python_expr" in r:
            parts.append(f"expr={r['python_expr']}")
        lines.append(", ".join(parts))
    results_block = "\n".join(lines)

    return f"""\
Here are the top-performing surrogate gradient functions from evaluation:

{results_block}

Analyze what makes the top performers work well, then generate {n_new} improved candidates:
1. Combine successful properties from top performers
2. Explore at least one novel structural variation
3. Address weaknesses (high gradient variance, excessive spikes, poor tail decay)

Output exactly {n_new} JSON objects."""
