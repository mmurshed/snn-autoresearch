# SNN-AutoResearch Agent Program

You are an AI agent running an automated surrogate gradient discovery loop for
spiking neural networks. Your goal is to find surrogate gradient functions that
maximize SNN test accuracy.

## Setup

1. Read `prepare.py`, `train.py`, `loop.py`, and the `snn_autoresearch/` package.
2. Verify dependencies: `uv sync`
3. Verify data: `uv run prepare.py --dataset shd --placeholder`
4. Create a results tracking file: `results.tsv` with header:
   ```
   commit	surrogate	accuracy	spikes	status	description
   ```

## Discovery Loop

The loop runs automatically via `loop.py`. To start:

```bash
uv run loop.py --dataset shd --llm claude --max-rounds 5 > discovery.log 2>&1
```

### Loop Steps

1. **PROMPT** — Build the design problem prompt (initial generation or refinement)
2. **GENERATE** — LLM generates N candidate surrogate gradient functions as JSON
3. **EVALUATE** — Verify each candidate (non-negativity, bounded integral, locality, stability), then train an SNN for `--eval-epochs` epochs and measure accuracy
4. **FEEDBACK** — Rank candidates by accuracy, select top-K for feedback
5. **REFINE** — On subsequent rounds, the prompt includes top performer data so the LLM can improve upon them
6. **CONVERGE** — If accuracy improvement drops below `--convergence-threshold`, stop

### Manual Experimentation

You can also run individual surrogates manually:

```bash
# Run a baseline
uv run train.py --surrogate sigmoid --dataset shd --epochs 30 > run.log 2>&1

# Run a custom surrogate found by the loop
uv run train.py --expr "(1 - np.tanh(x / alpha)**2) / (4 * alpha)" \
                 --params '{"alpha": 2.0}' --name llm1_tanh \
                 --dataset shd --epochs 30 > run.log 2>&1

# Extract results
grep "^test_accuracy:\|^total_spikes:" run.log
```

After each run, log results to `results.tsv` and commit:

```
git add train.py results.tsv
git commit -m "experiment: <description>"
```

## Rules

- **DO NOT** modify `prepare.py` or the `snn_autoresearch/` package
- **DO** edit `train.py` to tune hyperparameters (LR, batch size, dropout, beta)
- **DO** log every experiment to `results.tsv`
- Commit after each meaningful experiment
- If a run exceeds 10 minutes, kill it and discard
- Prefer simpler surrogates over complex ones at similar accuracy
- **NEVER STOP** — continue iterating until manually stopped
