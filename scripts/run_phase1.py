"""Phase 1: comparison difficulty in Elo coordinates.

For each query: build a BEIR (or synthetic, for --smoke) candidate pool, run an initial
random round of pairwise judge calls, fit per-query Bradley-Terry ratings (+ Hessian),
score every judged pair's margin-difficulty, and check that margin predicts judge
disagreement (flip-rate) via repeated/ensemble judge calls.

Reranker-error correlation (the other half of the Phase 1 analysis in plan.md) is deferred
to Phase 2, once distill.py produces a trained pointwise cross-encoder to measure error
against -- this script only covers what's available before any training: margin vs.
judge-disagreement.

Usage:
    python scripts/run_phase1.py --config configs/fiqa.yaml
    python scripts/run_phase1.py --config configs/smoke.yaml --smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.beir_loader import build_candidate_pools, generate_synthetic_dataset, load_beir
from src.comparison_difficulty import judge_flip_rate, margin_difficulty, score_pairs
from src.elo_fit import PairwiseOutcomes, fit_ratings
from src.judge import PairwiseJudge
from src.utils import WandbLogger, apply_smoke_overrides, load_config, set_seed


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

    pools = build_candidate_pools(
        corpus, queries, qrels,
        pool_size=ds_cfg["candidate_pool_size"],
        seed=config["seed"],
        max_queries=ds_cfg.get("max_queries"),
    )

    judge = PairwiseJudge(config["judge"])
    n_repeats = config["judge"].get("repeat_calls_for_flip_rate", 1)
    n_random_pairs = config["comparisons"]["initial_random_round_per_query"]
    rng = np.random.default_rng(config["seed"])

    per_query_results = []
    all_margins: list[float] = []
    all_flip_rates: list[float] = []

    for pool in pools:
        n = len(pool.doc_ids)
        if n < 2:
            continue
        doc_index = {d: i for i, d in enumerate(pool.doc_ids)}
        all_possible_pairs = list(itertools.combinations(range(n), 2))
        rng.shuffle(all_possible_pairs)
        sampled = all_possible_pairs[: min(n_random_pairs, len(all_possible_pairs))]

        outcomes = PairwiseOutcomes(n_docs=n)
        verdicts_per_pair = {}
        for i, j in sampled:
            doc_a_id, doc_b_id = pool.doc_ids[i], pool.doc_ids[j]
            repeats = []
            for r in range(n_repeats):
                verdicts = judge.judge_pair(
                    pool.query_id, pool.query_text,
                    doc_a_id, pool.doc_texts[i],
                    doc_b_id, pool.doc_texts[j],
                    order="ab", repeat_idx=r,
                )
                repeats.extend(verdicts)
            for v in repeats:
                if v.winner == "A":
                    outcomes.add(i, j)
                else:
                    outcomes.add(j, i)
            verdicts_per_pair[(pool.query_id, doc_a_id, doc_b_id)] = repeats

        fit = fit_ratings(
            outcomes,
            method=config["rating_fit"]["method"],
            optimizer=config["rating_fit"]["optimizer"],
            max_iters=config["rating_fit"]["max_iters"],
            tol=config["rating_fit"]["tol"],
            l2_reg=config["rating_fit"]["l2_reg"],
        )

        pair_keys = [(pool.query_id, pool.doc_ids[i], pool.doc_ids[j]) for i, j in sampled]
        scored = score_pairs(
            pair_keys,
            ratings_per_query={pool.query_id: fit.ratings},
            doc_index_per_query={pool.query_id: doc_index},
            verdicts_per_pair=verdicts_per_pair,
        )

        eigvals = np.linalg.eigvalsh(fit.hessian_analytic)
        condition_number = float(eigvals.max() / max(eigvals.min(), 1e-12))

        per_query_results.append({
            "query_id": pool.query_id,
            "n_docs": n,
            "n_pairs_judged": len(sampled),
            "converged": fit.converged,
            "n_iters": fit.n_iters,
            "condition_number": condition_number,
            "pairs": [
                {"doc_a": p.doc_a_id, "doc_b": p.doc_b_id, "margin": p.margin, "flip_rate": p.flip_rate}
                for p in scored
            ],
        })
        for p in scored:
            all_margins.append(p.margin)
            all_flip_rates.append(p.flip_rate if p.flip_rate is not None else 0.0)

    correlation = {}
    if len(all_margins) >= 3 and n_repeats > 1:
        pear_r, pear_p = pearsonr(all_margins, all_flip_rates)
        spear_r, spear_p = spearmanr(all_margins, all_flip_rates)
        correlation = {
            "pearson_r": float(pear_r), "pearson_p": float(pear_p),
            "spearman_r": float(spear_r), "spearman_p": float(spear_p),
            "n_pairs": len(all_margins),
            "note": "expect negative correlation: larger margin -> lower judge disagreement",
        }
    else:
        correlation = {"note": "insufficient repeats or pairs to compute margin-vs-flip-rate correlation"}

    return {
        "config": config,
        "per_query_results": per_query_results,
        "margin_vs_flip_rate_correlation": correlation,
        "n_queries": len(per_query_results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = WandbLogger(config, run_name=config.get("run_name", "phase1"))

    results = run(config, smoke=args.smoke)

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("smoke_results.json" if args.smoke else "results.json")
    out_path.write_text(json.dumps(results, indent=2, default=str))

    corr = results["margin_vs_flip_rate_correlation"]
    logger.log({"n_queries": results["n_queries"], **{k: v for k, v in corr.items() if isinstance(v, float)}})
    logger.finish()

    print(f"Phase 1 {'(smoke) ' if args.smoke else ''}complete: {results['n_queries']} queries processed.")
    print(f"Results written to {out_path}")
    print(f"Margin-vs-flip-rate correlation: {corr}")


if __name__ == "__main__":
    main()
