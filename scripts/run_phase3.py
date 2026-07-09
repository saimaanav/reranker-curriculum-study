"""Phase 3: dynamics analysis of the rating fixed point (H3).

At equal comparison budget, compares the Hessian eigenspectrum/conditioning of the
Bradley-Terry fixed point across scheduling arms, and tests whether better conditioning
predicts downstream reranker calibration (ECE) and near-tie pairwise accuracy. Also
tracks a lightweight score-distribution geometry cross-check (score separation between
relevant/irrelevant held-out docs).

Produces the single key figure: curriculum vs. random on calibration + near-tie accuracy
(left panel), beside fixed-point conditioning by arm (right panel, log-scale) -- the
plan.md "definition of done" figure.

Usage:
    python scripts/run_phase3.py --config configs/fiqa.yaml
    python scripts/run_phase3.py --config configs/smoke.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase2 import build_eval_calibration_pairs, build_pair_universe, judge_scheduled_pairs
from src.beir_loader import build_candidate_pools, generate_synthetic_dataset, load_beir
from src.curricula import SCHEDULERS
from src.distill import build_training_examples, train_reranker
from src.elo_fit import fit_ratings
from src.hessian_analysis import (
    aggregate_conditioning,
    correlate_conditioning_with_downstream,
    eigenspectrum,
    score_separation,
)
from src.judge import PairwiseJudge
from src.metrics import evaluate_ranking_on_pools, pairwise_calibration
from src.utils import WandbLogger, apply_smoke_overrides, load_config, set_seed


def run_arm_seed_dynamics(
    arm: str, seed: int, pair_universe: list, pools_by_query: dict, eval_pools: list,
    calibration_pairs: list[dict], judge: PairwiseJudge, config: dict, smoke: bool,
) -> dict:
    phase2_cfg = config["phase2"]
    budget = phase2_cfg["comparison_budget"]
    scheduled = SCHEDULERS[arm](pair_universe, budget=budget, seed=seed)
    outcomes_by_query, doc_index_by_query = judge_scheduled_pairs(scheduled, pools_by_query, judge)

    spectra = []
    ratings_by_query = {}
    for qid, outcomes in outcomes_by_query.items():
        if not outcomes.win_counts:
            continue
        fit = fit_ratings(
            outcomes, method=config["rating_fit"]["method"], optimizer=config["rating_fit"]["optimizer"],
            max_iters=config["rating_fit"]["max_iters"], tol=config["rating_fit"]["tol"],
            l2_reg=config["rating_fit"]["l2_reg"],
        )
        ratings_by_query[qid] = fit.ratings
        spectra.append(eigenspectrum(fit.hessian_analytic))

    if not ratings_by_query:
        return {"arm": arm, "seed": seed, "conditioning": {}, "downstream": {}}

    conditioning = aggregate_conditioning(spectra)

    examples = build_training_examples(
        {qid: pools_by_query[qid] for qid in ratings_by_query}, ratings_by_query,
        {qid: doc_index_by_query[qid] for qid in ratings_by_query},
    )
    reranker = train_reranker(examples, smoke=smoke, seed=seed, base_model=phase2_cfg.get("base_model"))

    ranking_metrics = evaluate_ranking_on_pools(reranker, eval_pools)
    calibration_metrics = pairwise_calibration(reranker, calibration_pairs)

    sep_values = []
    for pool in eval_pools:
        if not pool.doc_ids:
            continue
        scores = reranker.predict([(pool.query_text, t) for t in pool.doc_texts])
        labels = np.array([1.0 if d in pool.relevant_doc_ids else 0.0 for d in pool.doc_ids])
        sep = score_separation(scores, labels)
        if sep is not None:
            sep_values.append(sep)
    mean_score_separation = float(np.mean(sep_values)) if sep_values else None

    downstream = {
        "ndcg@10": ranking_metrics["ndcg@10"],
        "recall@10": ranking_metrics["recall@10"],
        "mrr": ranking_metrics["mrr"],
        "ece": calibration_metrics["ece"],
        "near_tie_pairwise_accuracy": calibration_metrics["near_tie_pairwise_accuracy"],
        "score_separation": mean_score_separation,
    }
    return {"arm": arm, "seed": seed, "conditioning": conditioning, "downstream": downstream}


def make_key_figure(results_table: dict, out_path: Path) -> None:
    arms = list(results_table.keys())
    ece_means = [results_table[a]["ece_mean"] or 0.0 for a in arms]
    ece_stds = [results_table[a]["ece_std"] or 0.0 for a in arms]
    near_tie_means = [results_table[a]["near_tie_pairwise_accuracy_mean"] or 0.0 for a in arms]
    near_tie_stds = [results_table[a]["near_tie_pairwise_accuracy_std"] or 0.0 for a in arms]
    cond_means = [results_table[a]["condition_number_mean"] or 1.0 for a in arms]
    cond_stds = [results_table[a]["condition_number_std"] or 0.0 for a in arms]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(arms))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width / 2, ece_means, width, yerr=ece_stds, label="ECE (lower better)", color="tab:red")
    ax.bar(x + width / 2, near_tie_means, width, yerr=near_tie_stds, label="near-tie pairwise acc.", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=20, ha="right")
    ax.set_title("Calibration & near-tie accuracy at equal budget")
    ax.legend()

    ax2 = axes[1]
    ax2.bar(x, cond_means, yerr=cond_stds, color="tab:green")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(arms, rotation=20, ha="right")
    ax2.set_title("Rating fixed-point Hessian condition number\n(mechanism, log scale)")

    fig.suptitle("Difficulty-scheduled curriculum vs. random: calibration/near-tie performance\nand fixed-point conditioning at equal comparison budget")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    eval_pools = all_pools[:n_eval]
    train_pools = all_pools[n_eval:] or all_pools
    pools_by_query = {p.query_id: p for p in train_pools}

    judge = PairwiseJudge(config["judge"])
    pair_universe = build_pair_universe(train_pools)
    calibration_pairs = build_eval_calibration_pairs(
        eval_pools, judge, n_pairs_per_query=max(2, phase2_cfg["comparison_budget"] // 10),
        seed=config["seed"], rating_cfg=config["rating_fit"],
    )

    seeds = config.get("seeds", [config["seed"]])
    all_runs = []
    for arm in phase2_cfg["arms"]:
        for seed in seeds:
            all_runs.append(
                run_arm_seed_dynamics(arm, seed, pair_universe, pools_by_query, eval_pools,
                                       calibration_pairs, judge, config, smoke)
            )

    by_arm = defaultdict(list)
    for r in all_runs:
        by_arm[r["arm"]].append(r)

    results_table = {}
    for arm, runs in by_arm.items():
        conds = [r["conditioning"]["condition_number_mean"] for r in runs if r["conditioning"].get("condition_number_mean") is not None]
        eces = [r["downstream"]["ece"] for r in runs if r["downstream"].get("ece") is not None]
        near_ties = [r["downstream"]["near_tie_pairwise_accuracy"] for r in runs if r["downstream"].get("near_tie_pairwise_accuracy") is not None]
        ndcgs = [r["downstream"]["ndcg@10"] for r in runs if r["downstream"].get("ndcg@10") is not None]
        seps = [r["downstream"]["score_separation"] for r in runs if r["downstream"].get("score_separation") is not None]
        results_table[arm] = {
            "condition_number_mean": float(np.mean(conds)) if conds else None,
            "condition_number_std": float(np.std(conds)) if conds else None,
            "ece_mean": float(np.mean(eces)) if eces else None,
            "ece_std": float(np.std(eces)) if eces else None,
            "near_tie_pairwise_accuracy_mean": float(np.mean(near_ties)) if near_ties else None,
            "near_tie_pairwise_accuracy_std": float(np.std(near_ties)) if near_ties else None,
            "ndcg@10_mean": float(np.mean(ndcgs)) if ndcgs else None,
            "ndcg@10_std": float(np.std(ndcgs)) if ndcgs else None,
            "score_separation_mean": float(np.mean(seps)) if seps else None,
            "n_seeds": len(runs),
        }

    h3_records = [
        {
            "condition_number": r["conditioning"]["condition_number_mean"],
            "ece": r["downstream"]["ece"],
            "near_tie_accuracy": r["downstream"]["near_tie_pairwise_accuracy"],
        }
        for r in all_runs
        if r["conditioning"].get("condition_number_mean") is not None
        and r["downstream"].get("ece") is not None
        and r["downstream"].get("near_tie_pairwise_accuracy") is not None
    ]
    h3_correlation = correlate_conditioning_with_downstream(h3_records)

    return {
        "config": config,
        "runs": all_runs,
        "results_table": results_table,
        "h3_correlation": h3_correlation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = WandbLogger(config, run_name=config.get("run_name", "phase3") + "_phase3")

    results = run(config, smoke=args.smoke)

    dataset_name = Path(config["phase2"]["output_dir"]).name
    out_dir = Path("results/phase3") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke_" if args.smoke else ""
    (out_dir / f"{suffix}results.json").write_text(json.dumps(results, indent=2, default=str))
    make_key_figure(results["results_table"], out_dir / f"{suffix}key_figure.png")

    for arm, row in results["results_table"].items():
        logger.log({f"{arm}/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
    logger.finish()

    print(f"Phase 3 {'(smoke) ' if args.smoke else ''}complete.")
    print("Results table (condition number, ECE, near-tie accuracy by arm):")
    for arm, row in results["results_table"].items():
        print(f"  {arm}: {row}")
    print(f"H3 correlation (conditioning vs. downstream): {results['h3_correlation']}")
    print(f"Key figure written to {out_dir / f'{suffix}key_figure.png'}")


if __name__ == "__main__":
    main()
