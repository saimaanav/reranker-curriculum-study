"""Ad-hoc check: does compositional's NDCG@10 advantage survive at NDCG@20 (the full
20-doc candidate pool), or is it specific to the top-10 cutoff?

Rebuilds the exact same schedule/judging/BT-fit pipeline as run_phase2.py for each
arm/seed (cache-only, no new Ollama calls since this is the same config already run),
then evaluates NDCG at k=10 and k=20 side by side.

Usage:
    python scripts/check_ndcg_at_full_pool.py --config configs/fiqa.yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase2 import BTRatingReranker, build_pair_universe, judge_scheduled_pairs
from src.beir_loader import build_candidate_pools, load_beir
from src.curricula import SCHEDULERS
from src.elo_fit import fit_ratings
from src.judge import PairwiseJudge
from src.metrics import ndcg_at_k
from src.utils import load_config, set_seed


def evaluate_at_k(reranker, pools: list, k: int) -> float:
    ndcgs = []
    for pool in pools:
        if not pool.doc_ids:
            continue
        pairs = [(pool.query_text, text) for text in pool.doc_texts]
        scores = reranker.predict(pairs)
        order = np.argsort(-scores)
        relevances = [1.0 if pool.doc_ids[i] in pool.relevant_doc_ids else 0.0 for i in order]
        ndcgs.append(ndcg_at_k(relevances, k=k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    ds_cfg = config["dataset"]
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
    pool_size = ds_cfg["candidate_pool_size"]

    budget = phase2_cfg["comparison_budget"]
    seeds = config.get("seeds", [config["seed"]])

    by_arm_k10 = defaultdict(list)
    by_arm_k20 = defaultdict(list)

    total = len(phase2_cfg["arms"]) * len(seeds)
    n = 0
    for arm in phase2_cfg["arms"]:
        for seed in seeds:
            n += 1
            print(f"[{n}/{total}] arm={arm} seed={seed} (cache-only, refitting) ...", flush=True)
            scheduled = SCHEDULERS[arm](pair_universe, budget=budget, seed=seed)
            outcomes_by_query, doc_index_by_query = judge_scheduled_pairs(scheduled, pools_by_query, judge)
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
            if not ratings_by_query:
                continue
            reranker = BTRatingReranker(ratings_by_query, doc_index_by_query, pools_by_query)
            by_arm_k10[arm].append(evaluate_at_k(reranker, train_pools, k=10))
            by_arm_k20[arm].append(evaluate_at_k(reranker, train_pools, k=pool_size))

    print()
    print(f"{'arm':<20} {'NDCG@10':<20} {'NDCG@' + str(pool_size) + ' (full pool)':<20}")
    for arm in phase2_cfg["arms"]:
        k10 = by_arm_k10[arm]
        k20 = by_arm_k20[arm]
        print(f"{arm:<20} {np.mean(k10):.4f}±{np.std(k10):.4f}     {np.mean(k20):.4f}±{np.std(k20):.4f}")


if __name__ == "__main__":
    main()
