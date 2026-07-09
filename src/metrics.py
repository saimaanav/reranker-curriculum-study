"""Ranking and calibration metrics shared across phases.

Phase 1 only needs a slice of this (used by the margin-vs-error analysis); Phase 2 uses
the full set (NDCG@10, Recall, MRR, ECE, reliability curves, near-tie slicing).
"""
from __future__ import annotations

import numpy as np


def dcg_at_k(relevances: list[float], k: int) -> float:
    relevances = relevances[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_relevances: list[float], k: int = 10) -> float:
    """ranked_relevances: relevance labels in the order the system ranked them (best first)."""
    dcg = dcg_at_k(ranked_relevances, k)
    ideal = dcg_at_k(sorted(ranked_relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_relevances: list[float], n_relevant_total: int, k: int = 10) -> float:
    if n_relevant_total == 0:
        return 0.0
    hits = sum(1 for rel in ranked_relevances[:k] if rel > 0)
    return hits / n_relevant_total


def mrr(ranked_relevances: list[float]) -> float:
    for i, rel in enumerate(ranked_relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE: bins predictions by confidence, compares mean confidence to accuracy per bin."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_curve(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict]:
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    curve = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        curve.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "count": int(mask.sum()),
            "mean_confidence": float(confidences[mask].mean()) if mask.sum() else None,
            "accuracy": float(correct[mask].mean()) if mask.sum() else None,
        })
    return curve


def evaluate_ranking_on_pools(reranker, pools: list) -> dict:
    """Macro-averaged NDCG@10, Recall@10, MRR of `reranker` over a list of BEIR CandidatePool
    objects, using each pool's qrels-derived `relevant_doc_ids` as binary relevance labels."""
    ndcgs, recalls, mrrs = [], [], []
    for pool in pools:
        if not pool.doc_ids:
            continue
        pairs = [(pool.query_text, text) for text in pool.doc_texts]
        scores = reranker.predict(pairs)
        order = np.argsort(-scores)
        relevances = [1.0 if pool.doc_ids[i] in pool.relevant_doc_ids else 0.0 for i in order]
        n_relevant_total = len(pool.relevant_doc_ids & set(pool.doc_ids))
        ndcgs.append(ndcg_at_k(relevances, k=10))
        recalls.append(recall_at_k(relevances, n_relevant_total, k=10))
        mrrs.append(mrr(relevances))
    return {
        "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "n_queries": len(ndcgs),
    }


def pairwise_calibration(
    reranker,
    judged_pairs: list[dict],  # each: {query_text, doc_a_text, doc_b_text, outcome (1=a wins), margin}
    near_tie_quantile: float = 0.25,
) -> dict:
    """ECE + reliability curve + pairwise ranking accuracy (overall and on the near-tie slice)
    of `reranker`'s implied pairwise predictions (sigmoid(score_a - score_b)) against judged
    ground-truth outcomes. This is the calibration/near-tie evaluation for H2."""
    if not judged_pairs:
        return {"ece": None, "reliability_curve": [], "pairwise_accuracy": None, "near_tie_pairwise_accuracy": None}

    a_texts = [(p["query_text"], p["doc_a_text"]) for p in judged_pairs]
    b_texts = [(p["query_text"], p["doc_b_text"]) for p in judged_pairs]
    scores_a = reranker.predict(a_texts)
    scores_b = reranker.predict(b_texts)
    diff = scores_a - scores_b
    p_a_wins = 1.0 / (1.0 + np.exp(-diff))
    outcomes = np.array([p["outcome"] for p in judged_pairs], dtype=np.float64)
    margins = np.array([p["margin"] for p in judged_pairs], dtype=np.float64)

    predicted_a = (p_a_wins >= 0.5).astype(np.float64)
    correct = (predicted_a == outcomes).astype(np.float64)
    confidence = np.where(predicted_a == 1.0, p_a_wins, 1.0 - p_a_wins)

    ece = expected_calibration_error(confidence, correct)
    curve = reliability_curve(confidence, correct)
    overall_acc = float(correct.mean())

    near_tie_mask = near_tie_slice(margins, near_tie_quantile)
    near_tie_acc = float(correct[near_tie_mask].mean()) if near_tie_mask.sum() > 0 else None

    return {
        "ece": ece,
        "reliability_curve": curve,
        "pairwise_accuracy": overall_acc,
        "near_tie_pairwise_accuracy": near_tie_acc,
        "n_pairs": len(judged_pairs),
        "n_near_tie_pairs": int(near_tie_mask.sum()),
    }


def comparisons_to_target(
    checkpoints: list[dict],  # each: {"n_comparisons": int, "metric_value": float}
    target: float,
    higher_is_better: bool = True,
) -> int | None:
    """First checkpoint's `n_comparisons` at which `metric_value` reaches `target`, or None
    if the target was never reached within the budget (H1)."""
    for cp in sorted(checkpoints, key=lambda c: c["n_comparisons"]):
        reached = cp["metric_value"] >= target if higher_is_better else cp["metric_value"] <= target
        if reached:
            return cp["n_comparisons"]
    return None


def near_tie_slice(margins: np.ndarray, threshold_quantile: float = 0.25) -> np.ndarray:
    """Boolean mask selecting the hardest (smallest-margin) pairs, i.e. the bottom quantile
    of the margin distribution -- used to slice NDCG/ECE onto the near-tie subset (H2)."""
    margins = np.asarray(margins, dtype=np.float64)
    threshold = np.quantile(margins, threshold_quantile)
    return margins <= threshold
