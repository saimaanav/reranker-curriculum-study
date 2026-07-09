"""Phase 2: curriculum vs. baseline scheduling.

At a fixed comparison budget, compares four pairwise-scheduling arms (random baseline,
difficulty-scheduled curriculum, anti-curriculum, compositional sub-skill curriculum)
across 3+ seeds: fit Bradley-Terry ratings from the arm's scheduled comparisons and use
those ratings directly as the reranker (no cross-encoder training).

Reports NDCG@10/Recall/MRR, comparisons-to-target (H1), ECE + near-tie pairwise accuracy
(H2), and per-arm learning curves (metric vs. comparisons spent, at configurable budget
fraction checkpoints).

Usage:
    python scripts/run_phase2.py --config configs/fiqa.yaml
    python scripts/run_phase2.py --config configs/smoke.yaml --smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.beir_loader import build_candidate_pools, generate_synthetic_dataset, load_beir
from src.curricula import SCHEDULERS, PairCandidate
from src.elo_fit import PairwiseOutcomes, fit_ratings
from src.judge import OracleJudge, PairwiseJudge
from src.metrics import comparisons_to_target, evaluate_ranking_on_pools, pairwise_calibration
from src.utils import WandbLogger, apply_smoke_overrides, load_config, set_seed


class BTRatingReranker:
    """Uses fitted Bradley-Terry ratings directly as relevance scores.

    Keyed by (query_text, doc_text) for O(1) lookup during eval.
    Unseen pairs score 0.0 (treated as unranked).
    """

    def __init__(self, ratings_by_query: dict[str, np.ndarray],
                 doc_index_by_query: dict[str, dict[str, int]],
                 pools_by_query: dict):
        self._scores: dict[tuple[str, str], float] = {}
        for qid, ratings in ratings_by_query.items():
            pool = pools_by_query.get(qid)
            if pool is None:
                continue
            for doc_id, idx in doc_index_by_query[qid].items():
                doc_idx = pool.doc_ids.index(doc_id)
                self._scores[(pool.query_text, pool.doc_texts[doc_idx])] = float(ratings[idx])

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.array([self._scores.get(p, 0.0) for p in pairs], dtype=np.float64)


def _tf_cosine_sim(text_a: str, text_b: str) -> float:
    """Cheap bag-of-words cosine similarity, used as a "semantic" proxy signal for the
    compositional curriculum -- avoids depending on a neural embedding model (extra
    network download) to keep --smoke fully offline under the $0 budget constraint."""
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    vocab = set(tokens_a) | set(tokens_b)
    if not vocab:
        return 0.0
    vec_a = np.array([tokens_a.count(t) for t in vocab], dtype=np.float64)
    vec_b = np.array([tokens_b.count(t) for t in vocab], dtype=np.float64)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(np.dot(vec_a, vec_b) / denom) if denom > 0 else 0.0


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not (ta | tb):
        return 0.0
    return len(ta & tb) / len(ta | tb)


def build_pair_universe(pools: list) -> list[PairCandidate]:
    """All C(n,2) candidate pairs per query pool, annotated with cheap pre-fit difficulty
    and sub-skill-categorization features."""
    pairs = []
    for pool in pools:
        n = len(pool.doc_ids)
        for i, j in itertools.combinations(range(n), 2):
            doc_a_id, doc_b_id = pool.doc_ids[i], pool.doc_ids[j]
            bm25_a = pool.bm25_scores.get(doc_a_id, 0.0)
            bm25_b = pool.bm25_scores.get(doc_b_id, 0.0)
            lex_overlap = (
                _jaccard(pool.query_text, pool.doc_texts[i]) + _jaccard(pool.query_text, pool.doc_texts[j])
            ) / 2.0
            semantic_sim = _tf_cosine_sim(pool.doc_texts[i], pool.doc_texts[j])
            pairs.append(
                PairCandidate(
                    query_id=pool.query_id, doc_a_id=doc_a_id, doc_b_id=doc_b_id,
                    bootstrap_margin=abs(bm25_a - bm25_b),
                    lexical_overlap=lex_overlap, semantic_sim=semantic_sim,
                )
            )
    return pairs


def judge_scheduled_pairs(
    scheduled: list[PairCandidate], pools_by_query: dict, judge,
) -> tuple[dict, dict]:
    """Judges all scheduled pairs in parallel and accumulates per-query win counts."""
    outcomes_by_query: dict[str, PairwiseOutcomes] = {}
    doc_index_by_query: dict[str, dict[str, int]] = {}

    # build index structures
    pair_args = []
    for pc in scheduled:
        pool = pools_by_query[pc.query_id]
        if pc.query_id not in doc_index_by_query:
            doc_index_by_query[pc.query_id] = {d: i for i, d in enumerate(pool.doc_ids)}
            outcomes_by_query[pc.query_id] = PairwiseOutcomes(n_docs=len(pool.doc_ids))
        text_a = pool.doc_texts[pool.doc_ids.index(pc.doc_a_id)]
        text_b = pool.doc_texts[pool.doc_ids.index(pc.doc_b_id)]
        pair_args.append((pc.query_id, pool.query_text, pc.doc_a_id, text_a, pc.doc_b_id, text_b))

    # judge all pairs in parallel
    if hasattr(judge, 'judge_pairs_parallel'):
        all_verdicts = judge.judge_pairs_parallel(pair_args)
    else:
        all_verdicts = [judge.judge_pair(*args) for args in pair_args]

    for pc, verdicts in zip(scheduled, all_verdicts):
        doc_index = doc_index_by_query[pc.query_id]
        i, j = doc_index[pc.doc_a_id], doc_index[pc.doc_b_id]
        a_votes = sum(1 for v in verdicts if v.winner == "A")
        if a_votes * 2 >= len(verdicts):
            outcomes_by_query[pc.query_id].add(i, j)
        else:
            outcomes_by_query[pc.query_id].add(j, i)

    return outcomes_by_query, doc_index_by_query


def fit_ratings_for_touched_queries(
    outcomes_by_query: dict[str, PairwiseOutcomes], rating_cfg: dict,
) -> dict[str, np.ndarray]:
    ratings = {}
    for qid, outcomes in outcomes_by_query.items():
        if not outcomes.win_counts:
            continue
        fit = fit_ratings(
            outcomes, method=rating_cfg["method"], optimizer=rating_cfg["optimizer"],
            max_iters=rating_cfg["max_iters"], tol=rating_cfg["tol"], l2_reg=rating_cfg["l2_reg"],
        )
        ratings[qid] = fit.ratings
    return ratings


def build_eval_calibration_pairs(
    eval_pools: list, judge: PairwiseJudge, n_pairs_per_query: int, seed: int, rating_cfg: dict,
) -> list[dict]:
    """A small fixed sample of judged pairs on held-out queries, used only for the
    calibration/near-tie evaluation (H2) -- independent of any arm's training budget."""
    rng = np.random.default_rng(seed)
    calibration_pairs = []
    for pool in eval_pools:
        n = len(pool.doc_ids)
        if n < 2:
            continue
        all_pairs = list(itertools.combinations(range(n), 2))
        rng.shuffle(all_pairs)
        sampled = all_pairs[: min(n_pairs_per_query, len(all_pairs))]
        outcomes = PairwiseOutcomes(n_docs=n)
        pair_meta = []
        pair_args = [
            (pool.query_id, pool.query_text, pool.doc_ids[i], pool.doc_texts[i],
             pool.doc_ids[j], pool.doc_texts[j])
            for i, j in sampled
        ]
        if hasattr(judge, 'judge_pairs_parallel'):
            all_verdicts = judge.judge_pairs_parallel(pair_args)
        else:
            all_verdicts = [judge.judge_pair(*args) for args in pair_args]

        for (i, j), verdicts in zip(sampled, all_verdicts):
            doc_a_id, doc_b_id = pool.doc_ids[i], pool.doc_ids[j]
            a_votes = sum(1 for v in verdicts if v.winner == "A")
            outcome = 1 if a_votes * 2 >= len(verdicts) else 0
            if outcome == 1:
                outcomes.add(i, j)
            else:
                outcomes.add(j, i)
            pair_meta.append((i, j, doc_a_id, doc_b_id, outcome))
        if not outcomes.win_counts:
            continue
        fit = fit_ratings(
            outcomes, method=rating_cfg["method"], optimizer=rating_cfg["optimizer"],
            max_iters=rating_cfg["max_iters"], tol=rating_cfg["tol"], l2_reg=rating_cfg["l2_reg"],
        )
        for i, j, doc_a_id, doc_b_id, outcome in pair_meta:
            calibration_pairs.append({
                "query_text": pool.query_text,
                "doc_a_text": pool.doc_texts[pool.doc_ids.index(doc_a_id)],
                "doc_b_text": pool.doc_texts[pool.doc_ids.index(doc_b_id)],
                "outcome": outcome,
                "margin": abs(fit.ratings[i] - fit.ratings[j]),
            })
    return calibration_pairs


def run_arm_seed(
    arm: str, seed: int, pair_universe: list[PairCandidate], pools_by_query: dict,
    train_pools: list, eval_pools: list, calibration_pairs: list[dict], config: dict, smoke: bool,
) -> dict:
    phase2_cfg = config["phase2"]
    budget = phase2_cfg["comparison_budget"]
    scheduler = SCHEDULERS[arm]
    scheduled = scheduler(pair_universe, budget=budget, seed=seed)

    checkpoints = []
    for fraction in phase2_cfg["checkpoints_fractions"]:
        n = max(1, round(fraction * len(scheduled)))
        prefix = scheduled[:n]
        print(f"  checkpoint {fraction:.0%}: judging {len(prefix)} pairs ...", flush=True)
        outcomes_by_query, doc_index_by_query = judge_scheduled_pairs(prefix, pools_by_query, judge=_JUDGE)
        ratings_by_query = fit_ratings_for_touched_queries(outcomes_by_query, config["rating_fit"])
        if not ratings_by_query:
            continue
        reranker = BTRatingReranker(ratings_by_query, doc_index_by_query, pools_by_query)
        print(f"  checkpoint {fraction:.0%}: evaluating ...", flush=True)
        ranking_metrics = evaluate_ranking_on_pools(reranker, train_pools)
        calibration_metrics = pairwise_calibration(reranker, calibration_pairs)
        checkpoints.append({
            "n_comparisons": len(prefix),
            "metric_value": ranking_metrics["ndcg@10"],
            **ranking_metrics,
            **{k: v for k, v in calibration_metrics.items() if k != "reliability_curve"},
        })

    if not checkpoints:
        return {"arm": arm, "seed": seed, "checkpoints": [], "final": {}, "comparisons_to_target_ndcg": None}

    final = checkpoints[-1]
    ctt = comparisons_to_target(checkpoints, target=phase2_cfg["target_ndcg"], higher_is_better=True)
    return {"arm": arm, "seed": seed, "checkpoints": checkpoints, "final": final, "comparisons_to_target_ndcg": ctt}


_JUDGE: PairwiseJudge | None = None


def run(config: dict, smoke: bool) -> dict:
    global _JUDGE
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
    train_pools = all_pools[n_eval:]
    if not train_pools:
        train_pools = all_pools
    pools_by_query = {p.query_id: p for p in train_pools}

    _JUDGE = PairwiseJudge(config["judge"])

    pair_universe = build_pair_universe(train_pools)
    print(f"Building eval calibration pairs ({len(eval_pools)} eval queries) ...", flush=True)
    calibration_pairs = build_eval_calibration_pairs(
        eval_pools, _JUDGE, n_pairs_per_query=5,
        seed=config["seed"], rating_cfg=config["rating_fit"],
    )
    print(f"Calibration pairs built: {len(calibration_pairs)}", flush=True)

    seeds = config.get("seeds", [config["seed"]])
    all_runs = []
    total = len(phase2_cfg["arms"]) * len(seeds)
    for i, arm in enumerate(phase2_cfg["arms"]):
        for j, seed in enumerate(seeds):
            n = i * len(seeds) + j + 1
            print(f"[{n}/{total}] Running arm={arm} seed={seed} ...", flush=True)
            all_runs.append(
                run_arm_seed(arm, seed, pair_universe, pools_by_query, train_pools, eval_pools, calibration_pairs, config, smoke)
            )
            print(f"[{n}/{total}] Done arm={arm} seed={seed}", flush=True)

    aggregated = defaultdict(list)
    for r in all_runs:
        aggregated[r["arm"]].append(r)

    results_table = {}
    for arm, runs in aggregated.items():
        finals = [r["final"] for r in runs if r["final"]]
        ctts = [r["comparisons_to_target_ndcg"] for r in runs if r["comparisons_to_target_ndcg"] is not None]
        metric_keys = ["ndcg@10", "recall@10", "mrr", "ece", "pairwise_accuracy", "near_tie_pairwise_accuracy"]
        row = {}
        for k in metric_keys:
            vals = [f[k] for f in finals if f.get(k) is not None]
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{k}_std"] = float(np.std(vals)) if vals else None
        row["comparisons_to_target_ndcg_mean"] = float(np.mean(ctts)) if ctts else None
        row["n_seeds"] = len(runs)
        results_table[arm] = row

    return {
        "config": config,
        "n_train_queries": len(train_pools),
        "n_eval_queries": len(eval_pools),
        "runs": all_runs,
        "results_table": results_table,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = WandbLogger(config, run_name=config.get("run_name", "phase2") + "_phase2")

    results = run(config, smoke=args.smoke)

    out_dir = Path(config["phase2"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke_" if args.smoke else ""
    (out_dir / f"{suffix}results.json").write_text(json.dumps(results, indent=2, default=str))

    lines = ["| arm | NDCG@10 | Recall@10 | MRR | ECE | near-tie acc | comparisons-to-target |",
              "|---|---|---|---|---|---|---|"]
    for arm, row in results["results_table"].items():
        lines.append(
            f"| {arm} | {row['ndcg@10_mean']:.3f}±{row['ndcg@10_std']:.3f} | "
            f"{row['recall@10_mean']:.3f}±{row['recall@10_std']:.3f} | "
            f"{row['mrr_mean']:.3f}±{row['mrr_std']:.3f} | "
            f"{row['ece_mean']:.3f}±{row['ece_std']:.3f} | "
            f"{row['near_tie_pairwise_accuracy_mean']} | "
            f"{row['comparisons_to_target_ndcg_mean']} |"
            if row['ndcg@10_mean'] is not None else f"| {arm} | (no data) |"
        )
    table_md = "\n".join(lines)
    (out_dir / f"{suffix}results_table.md").write_text(table_md)

    for arm, row in results["results_table"].items():
        logger.log({f"{arm}/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
    logger.finish()

    print(f"Phase 2 {'(smoke) ' if args.smoke else ''}complete.")
    print(table_md)
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
