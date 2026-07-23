"""Phase 3, Hessian-only: fixed-point conditioning by arm, no reranker training.

Runs the same scheduling + Ollama judging + Bradley-Terry fit as full Phase 3, and
computes the Hessian eigenspectrum/condition number per query -- but stops there. Skips
build_training_examples/train_reranker (the CrossEncoderReranker distillation step) and
the downstream NDCG/ECE eval + H3 correlation, since those require a trained model.

Produces a standalone "how well-determined is the MLE fit" result per arm: condition
number, spectral gap, effective rank -- with per-query granularity retained (not just
arm-level aggregates) so eigenvalue-spectrum and per-query scatter figures are possible
later without re-running anything.

Usage:
    python scripts/run_phase3_hessian_only.py --config configs/fiqa.yaml
    python scripts/run_phase3_hessian_only.py --config configs/smoke.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase2 import build_eval_calibration_pairs, build_pair_universe, judge_scheduled_pairs
from src.beir_loader import build_candidate_pools, generate_synthetic_dataset, load_beir
from src.curricula import SCHEDULERS
from src.elo_fit import fit_ratings
from src.hessian_analysis import aggregate_conditioning, eigenspectrum
from src.judge import PairwiseJudge
from src.utils import apply_smoke_overrides, load_config, set_seed


def run_arm_seed_hessian(
    arm: str, seed: int, pair_universe: list, pools_by_query: dict, judge: PairwiseJudge, config: dict,
) -> dict:
    phase2_cfg = config["phase2"]
    budget = phase2_cfg["comparison_budget"]
    scheduled = SCHEDULERS[arm](pair_universe, budget=budget, seed=seed)
    outcomes_by_query, _ = judge_scheduled_pairs(scheduled, pools_by_query, judge)

    per_query = []
    spectra = []
    for qid, outcomes in outcomes_by_query.items():
        if not outcomes.win_counts:
            continue
        fit = fit_ratings(
            outcomes, method=config["rating_fit"]["method"], optimizer=config["rating_fit"]["optimizer"],
            max_iters=config["rating_fit"]["max_iters"], tol=config["rating_fit"]["tol"],
            l2_reg=config["rating_fit"]["l2_reg"],
        )
        spec = eigenspectrum(fit.hessian_analytic)
        spectra.append(spec)
        per_query.append({
            "query_id": qid,
            "n_docs": outcomes.n_docs,
            "condition_number": spec.condition_number,
            "spectral_gap": spec.spectral_gap,
            "effective_rank": spec.effective_rank,
            "eigenvalues": spec.eigenvalues.tolist(),
        })

    conditioning = aggregate_conditioning(spectra)
    return {"arm": arm, "seed": seed, "per_query": per_query, "conditioning": conditioning}


def run(config: dict, smoke: bool) -> dict:
    config = apply_smoke_overrides(config, smoke)
    set_seed(config["seed"])

    ds_cfg = config["dataset"]
    if smoke:
        corpus, queries, qrels = generate_synthetic_dataset(
            n_queries=ds_cfg["max_queries"], pool_size=ds_cfg["candidate_pool_size"], seed=config["seed"]
        )
    else:
        corpus, queries, qrels = load_beir(ds_cfg["name"], ds_cfg["beir_data_dir"], ds_cfg["split"])

    all_pools = build_candidate_pools(
        corpus, queries, qrels, pool_size=ds_cfg["candidate_pool_size"],
        seed=config["seed"], max_queries=ds_cfg.get("max_queries"),
    )
    phase2_cfg = config["phase2"]
    n_eval = max(1, round(len(all_pools) * phase2_cfg["eval_query_fraction"]))
    train_pools = all_pools[n_eval:] or all_pools
    pools_by_query = {p.query_id: p for p in train_pools}

    judge = PairwiseJudge(config["judge"])
    pair_universe = build_pair_universe(train_pools)

    seeds = config.get("seeds", [config["seed"]])
    all_runs = []
    total = len(phase2_cfg["arms"]) * len(seeds)
    for i, arm in enumerate(phase2_cfg["arms"]):
        for j, seed in enumerate(seeds):
            n = i * len(seeds) + j + 1
            print(f"[{n}/{total}] Hessian pass: arm={arm} seed={seed} ...", flush=True)
            all_runs.append(run_arm_seed_hessian(arm, seed, pair_universe, pools_by_query, judge, config))
            print(f"[{n}/{total}] Done arm={arm} seed={seed}", flush=True)

    by_arm = defaultdict(list)
    for r in all_runs:
        by_arm[r["arm"]].append(r)

    results_table = {}
    for arm, runs in by_arm.items():
        conds = [r["conditioning"]["condition_number_mean"] for r in runs if r["conditioning"].get("condition_number_mean") is not None]
        ranks = [r["conditioning"]["effective_rank_mean"] for r in runs if r["conditioning"].get("effective_rank_mean") is not None]
        results_table[arm] = {
            "condition_number_mean": float(np.mean(conds)) if conds else None,
            "condition_number_std": float(np.std(conds)) if conds else None,
            "condition_number_median": float(np.median(conds)) if conds else None,
            "effective_rank_mean": float(np.mean(ranks)) if ranks else None,
            "n_seeds": len(runs),
        }

    return {"config": config, "runs": all_runs, "results_table": results_table}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run(config, smoke=args.smoke)

    dataset_name = Path(config["phase2"]["output_dir"]).name
    out_dir = Path("results/phase3_hessian_only") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke_" if args.smoke else ""
    (out_dir / f"{suffix}results.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"Phase 3 (Hessian-only) {'(smoke) ' if args.smoke else ''}complete.")
    print("Condition number / effective rank by arm (no training, no downstream eval):")
    for arm, row in results["results_table"].items():
        print(f"  {arm}: {row}")
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
