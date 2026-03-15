# SNN-AutoResearch

LLM-driven discovery of surrogate gradient functions for spiking neural networks (SNNs).

An automated research loop that uses large language models (Claude, GPT-4o) to generate, verify, and refine surrogate gradient functions — the key component enabling backpropagation in SNNs.

## How It Works

SNNs use binary spikes, which have zero gradient almost everywhere. Surrogate gradients replace this zero derivative during backpropagation, enabling gradient-based training. This project automates the search for better surrogates:

```
┌─────────┐     ┌──────────┐     ┌──────────┐
│ 1.PROMPT│────>│2.GENERATE│────>│3.EVALUATE│
│ Define  │     │ LLM makes│     │ Verify + │
│ problem │     │candidates│     │ train SNN│
└─────────┘     └──────────┘     └────┬─────┘
     ^                                │
     │          ┌──────────┐     ┌────v─────┐
     └──────────│5. REFINE │<────│4.FEEDBACK│
     (iterate)  │ LLM uses │     │ Rank by  │
                │ feedback │     │ accuracy │
                └──────────┘     └────┬─────┘
                                      │
                                ┌─────v────┐
                                │6.CONVERGE│
                                │ Accept or│
                                │ iterate  │
                                └──────────┘
```

## Quick Start

```bash
# Install dependencies
uv sync

# Prepare data (placeholder for testing)
uv run prepare.py --dataset shd --placeholder

# Run discovery loop
export ANTHROPIC_API_KEY="your-key"
uv run loop.py --dataset shd --llm claude --placeholder --max-rounds 3

# Or run a single training with a known surrogate
uv run train.py --surrogate sigmoid --placeholder --epochs 5
```

## With Real Data

```bash
# Download SHD dataset
uv run prepare.py --dataset shd --data-dir data/

# Full discovery (Claude)
uv run loop.py --dataset shd --llm claude --max-rounds 5

# Full discovery (OpenAI)
export OPENAI_API_KEY="your-key"
uv run loop.py --dataset shd --llm openai --max-rounds 5

# Full training with a discovered surrogate
uv run train.py --expr "(1 - np.tanh(x / alpha)**2) / (4 * alpha)" \
                --params '{"alpha": 2.0}' --name llm1_tanh \
                --dataset shd --epochs 30
```

## Project Structure

```
snn-autoresearch/
├── prepare.py              # Data download and loading (do not modify)
├── train.py                # Single training run (editable hyperparameters)
├── loop.py                 # Discovery loop (main entry point)
├── program.md              # Agent instructions for automated experiments
├── snn_autoresearch/       # Core library
│   ├── candidate.py        # SurrogateCandidate dataclass + baselines
│   ├── verify.py           # Numerical verification (4 hard checks)
│   ├── spike.py            # Custom autograd + LIF neuron
│   ├── llm.py              # LLM backends (Claude, OpenAI)
│   ├── prompts.py          # Prompt templates
│   ├── evaluate.py         # Training and evaluation
│   └── models/
│       ├── recurrent.py    # RecurrentSNN (for SHD)
│       ├── resnet.py       # SpikingResNet18 (for CIFAR10-DVS)
│       └── vgg.py          # SpikingVGG11 (for CIFAR10-DVS)
```

## Supported Datasets

| Dataset    | Task           | Architecture | Input                   |
|------------|----------------|--------------|-------------------------|
| shd        | Spoken digits  | RecurrentSNN | 700 channels, 100 steps |
| cifar10dvs | Image (events) | ResNet18     | 2×48×48, 10 steps       |
| nmnist     | Digits (events)| ResNet18     | 2×34×34, 10 steps       |

## Verification Checks

Every LLM-generated surrogate must pass four numerical checks before training:

1. **Non-negativity**: g(x) >= 0 for all x
2. **Bounded integral**: ∫|g(x)|dx < 1000
3. **Locality**: g(x) → 0 for large |x|
4. **Numerical stability**: No NaN/Inf for any finite input

## Environment Variables

- `ANTHROPIC_API_KEY` — required for `--llm claude`
- `OPENAI_API_KEY` — required for `--llm openai`

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- [uv](https://docs.astral.sh/uv/) for dependency management
