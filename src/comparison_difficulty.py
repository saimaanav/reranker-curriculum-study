"""Scores the difficulty of pairwise comparisons in Elo/rating coordinates.

Primary signal is the rating margin |r_i - r_j| (small margin = hard/near-tie), which is
the coordinate the curriculum schedules on. Judge flip-rate and cross-query bias magnitude
are secondary validation signals used to check margin is a real difficulty proxy (Phase 1
analysis), not used to schedule comparisons themselves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.judge import JudgeVerdict


@dataclass
class PairDifficulty:
    query_id: str
    doc_a_id: str
    doc_b_id: str
    margin: float              # |r_a - r_b| under current/bootstrap ratings
    flip_rate: float | None    # fraction of repeated/ensemble judge calls that disagree
    cross_query_bias_mag: float | None  # |bias_q| for this pair's query, if fitted


def margin_difficulty(ratings: np.ndarray, doc_index: dict[str, int], doc_a_id: str, doc_b_id: str) -> float:
    """|r_a - r_b|: small margin = hard/near-tie pair, in the query's own rating scale."""
    return float(abs(ratings[doc_index[doc_a_id]] - ratings[doc_index[doc_b_id]]))


def bootstrap_margin_estimate(bm25_scores: dict[str, float], doc_a_id: str, doc_b_id: str) -> float:
    """Before any ratings exist (first bootstrap round), approximate margin with the gap in a
    cheap retrieval signal (e.g. BM25 score) so the curriculum has something to schedule on."""
    return float(abs(bm25_scores[doc_a_id] - bm25_scores[doc_b_id]))


def judge_flip_rate(verdicts: list[JudgeVerdict]) -> float:
    """Fraction of judge calls (across repeats and/or ensemble members) that disagree with
    the majority verdict for this pair. 0.0 = perfect agreement, up to 0.5 = maximal disagreement
    for a binary A/B outcome."""
    if len(verdicts) <= 1:
        return 0.0
    winners = [v.winner for v in verdicts]
    a_votes = winners.count("A")
    b_votes = winners.count("B")
    minority = min(a_votes, b_votes)
    return minority / len(winners)


def score_pairs(
    pairs: list[tuple[str, str, str]],  # (query_id, doc_a_id, doc_b_id)
    ratings_per_query: dict[str, np.ndarray],
    doc_index_per_query: dict[str, dict[str, int]],
    verdicts_per_pair: dict[tuple[str, str, str], list[JudgeVerdict]] | None = None,
    cross_query_bias: dict[str, float] | None = None,
) -> list[PairDifficulty]:
    """Scores an arbitrary set of (query, docA, docB) pairs for difficulty and, where available,
    the secondary validation signals."""
    results = []
    verdicts_per_pair = verdicts_per_pair or {}
    cross_query_bias = cross_query_bias or {}
    for qid, a, b in pairs:
        ratings = ratings_per_query[qid]
        doc_index = doc_index_per_query[qid]
        margin = margin_difficulty(ratings, doc_index, a, b)
        verdicts = verdicts_per_pair.get((qid, a, b))
        flip_rate = judge_flip_rate(verdicts) if verdicts else None
        bias_mag = abs(cross_query_bias[qid]) if qid in cross_query_bias else None
        results.append(PairDifficulty(qid, a, b, margin, flip_rate, bias_mag))
    return results


def rank_by_difficulty(scored_pairs: list[PairDifficulty], hardest_first: bool = True) -> list[PairDifficulty]:
    """Sorts pairs by margin difficulty. hardest_first=True -> near-ties (small margin) last
    is the default curriculum ordering in curricula.py; this helper just sorts either direction."""
    return sorted(scored_pairs, key=lambda p: p.margin, reverse=not hardest_first)
