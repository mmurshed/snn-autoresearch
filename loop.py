"""SNN surrogate gradient discovery loop.

Automated loop that uses an LLM to discover and refine surrogate gradient
functions for training spiking neural networks.

Loop steps:
  1. PROMPT     — Define the design problem (initial or refined)
  2. GENERATE   — LLM generates candidate surrogate functions
  3. EVALUATE   — Verify constraints, train SNN, score by accuracy
  4. FEEDBACK   — Rank results, package scores for next prompt
  5. REFINE     — LLM revises candidates using feedback (next iteration)
  6. CONVERGE   — Check if improvement has plateaued; accept or iterate

Usage:
    # Full discovery loop with Claude
    uv run loop.py --dataset shd --llm claude --max-rounds 5

    # Quick test with placeholder data
    uv run loop.py --dataset shd --llm claude --placeholder --eval-epochs 2

    # Use OpenAI backend
    uv run loop.py --dataset shd --llm openai --max-rounds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from prepare import DATASETS, load_shd, make_placeholder_loaders
from snn_autoresearch.candidate import SurrogateCandidate
from snn_autoresearch.evaluate import train_and_evaluate
from snn_autoresearch.llm import get_llm, parse_candidates
from snn_autoresearch.prompts import SYSTEM_PROMPT, build_generation_prompt, build_refinement_prompt
from snn_autoresearch.spike import make_spike_fn
from snn_autoresearch.verify import verify
from train import build_model, resolve_device

# ── Loop config ─────────────────────────────────────────────────────────

POPULATION_SIZE = 8          # candidates per generation
MAX_ROUNDS = 5               # maximum discovery rounds
TOP_K = 3                    # top candidates for feedback
NEW_PER_ROUND = 4            # new candidates per refinement
EVAL_EPOCHS = 3              # quick evaluation epochs
CONVERGENCE_THRESHOLD = 0.005  # minimum improvement to continue


def run_loop(args):
    llm = get_llm(args.llm)
    device = resolve_device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.placeholder:
        train_loader, test_loader = make_placeholder_loaders(args.dataset, args.batch_size)
    elif args.dataset == "shd":
        data_dir = Path(args.data_dir) / "shd"
        train_loader, _, test_loader = load_shd(data_dir, args.batch_size)
    else:
        print(f"Real data for {args.dataset!r} requires manual setup. Use --placeholder.", file=sys.stderr)
        sys.exit(1)

    # ── Discovery loop ──────────────────────────────────────────────────

    population: list[tuple[SurrogateCandidate, dict]] = []
    history: list[dict] = []
    best_score = 0.0

    for round_num in range(args.max_rounds):
        round_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"  Round {round_num + 1}/{args.max_rounds}")
        print(f"{'=' * 60}")

        # ── Step 1: PROMPT ──────────────────────────────────────────
        if round_num == 0:
            prompt = build_generation_prompt(args.population_size)
        else:
            # Step 5: REFINE — build prompt from feedback
            top = _get_top(population, args.top_k)
            feedback = [_format_result(c, m) for c, m in top]
            prompt = build_refinement_prompt(feedback, n_new=args.new_per_round)

        # ── Step 2: GENERATE ────────────────────────────────────────
        print("\n[Generate] Querying LLM...")
        response = llm.generate(SYSTEM_PROMPT, prompt)
        candidates = parse_candidates(response)
        for c in candidates:
            c.generation = round_num
        print(f"[Generate] Received {len(candidates)} candidates")

        if not candidates:
            print("[Generate] No candidates parsed — retrying with simpler prompt")
            response = llm.generate(SYSTEM_PROMPT, build_generation_prompt(4))
            candidates = parse_candidates(response)
            for c in candidates:
                c.generation = round_num
            if not candidates:
                print("[Generate] Still no candidates — skipping round")
                continue

        # ── Step 3: EVALUATE ────────────────────────────────────────
        n_verified, n_failed = 0, 0
        for candidate in candidates:
            # Verify constraints
            checks = verify(candidate)
            if not checks["valid"]:
                failed = [k for k, v in checks.items() if k != "valid" and not v]
                print(f"  x {candidate.name} — failed: {', '.join(failed)}")
                n_failed += 1
                continue

            n_verified += 1
            print(f"  + {candidate.name} — verified, training {args.eval_epochs} epochs...")

            # Train and score
            torch.manual_seed(0)
            spike_fn = make_spike_fn(candidate)
            model = build_model(args.dataset, spike_fn)
            result = train_and_evaluate(
                model, train_loader, test_loader,
                n_epochs=args.eval_epochs,
                lr=args.lr,
                device=device,
            )

            metrics = result.to_dict()
            metrics["python_expr"] = candidate.python_expr
            metrics["params"] = candidate.params
            population.append((candidate, metrics))
            print(f"    accuracy: {result.best_accuracy:.4f}  spikes: {result.total_spikes}")

        print(f"\n[Evaluate] {n_verified} verified, {n_failed} rejected")

        # ── Step 4: FEEDBACK ────────────────────────────────────────
        population.sort(key=lambda x: x[1]["best_accuracy"], reverse=True)
        # Keep population manageable
        population = population[: args.population_size * 2]

        if population:
            best_candidate, best_metrics = population[0]
            current_best = best_metrics["best_accuracy"]
            print(f"\n[Feedback] Best: {best_candidate.name} ({current_best:.4f})")

            history.append({
                "round": round_num,
                "name": best_candidate.name,
                "accuracy": current_best,
                "n_verified": n_verified,
                "n_rejected": n_failed,
                "time_seconds": round(time.time() - round_start, 1),
            })

            # ── Step 6: CONVERGE ────────────────────────────────────
            improvement = current_best - best_score
            if round_num > 0 and improvement < args.convergence_threshold:
                print(f"\n[Converge] Improvement {improvement:.4f} < threshold {args.convergence_threshold}")
                print(f"[Converge] Search converged after {round_num + 1} rounds")
                break

            best_score = current_best
        else:
            print("\n[Feedback] No valid candidates this round")
            history.append({
                "round": round_num,
                "name": "none",
                "accuracy": best_score,
                "n_verified": 0,
                "n_rejected": n_failed,
                "time_seconds": round(time.time() - round_start, 1),
            })

    # ── Save results ────────────────────────────────────────────────────
    _save_results(population, history, output_dir)

    # ── Final report ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Discovery Complete")
    print(f"{'=' * 60}")
    print(f"\nResults saved to {output_dir}/")

    if population:
        print("\nTop candidates:")
        for i, (c, m) in enumerate(population[:5]):
            print(f"  {i + 1}. {c.name}: {m['best_accuracy']:.4f}  (gen {c.generation})")
            print(f"     expr: {c.python_expr}")

    # Structured output
    if population:
        best_c, best_m = population[0]
        print("\n---")
        print(f"best_surrogate:    {best_c.name}")
        print(f"best_accuracy:     {best_m['best_accuracy']:.6f}")
        print(f"total_rounds:      {len(history)}")
        print(f"total_candidates:  {sum(h.get('n_verified', 0) for h in history)}")
        print(f"dataset:           {args.dataset}")
        print(f"llm_backend:       {args.llm}")
        print(f"python_expr:       {best_c.python_expr}")


# ── Helpers ─────────────────────────────────────────────────────────────

def _get_top(population, k):
    return sorted(population, key=lambda x: x[1]["best_accuracy"], reverse=True)[:k]


def _format_result(candidate: SurrogateCandidate, metrics: dict) -> dict:
    return {
        "name": candidate.name,
        "accuracy": metrics["best_accuracy"],
        "total_spikes": metrics.get("total_spikes", 0),
        "python_expr": candidate.python_expr,
    }


def _save_results(population, history, output_dir: Path):
    results = {
        "history": history,
        "candidates": [
            {**c.to_dict(), "metrics": m}
            for c, m in population
        ],
    }
    with open(output_dir / "discovery_log.json", "w") as f:
        json.dump(results, f, indent=2)

    if population:
        best_c, best_m = population[0]
        with open(output_dir / "best_surrogate.json", "w") as f:
            json.dump({**best_c.to_dict(), "metrics": best_m}, f, indent=2)


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-driven surrogate gradient discovery loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="shd", choices=list(DATASETS))
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--llm", default="claude", choices=["claude", "openai"])
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--population-size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--new-per-round", type=int, default=NEW_PER_ROUND)
    parser.add_argument("--eval-epochs", type=int, default=EVAL_EPOCHS)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--convergence-threshold", type=float, default=CONVERGENCE_THRESHOLD)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--placeholder", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    run_loop(args)


if __name__ == "__main__":
    main()
