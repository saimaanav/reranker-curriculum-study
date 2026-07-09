"""Comparison scheduling arms for Phase 2: given a universe of candidate pairs and a fixed
comparison budget, decide which pairs get judged and in what order.

All schedulers are deterministic given `seed` and truncate to exactly `budget` pairs (or
fewer if the universe is smaller). Difficulty is approximated pre-fit via a bootstrap
signal (e.g. BM25 score gap) since true Elo margins aren't known before any ratings exist.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class PairCandidate:
    query_id: str
    doc_a_id: str
    doc_b_id: str
    bootstrap_margin: float     # cheap pre-fit difficulty proxy (e.g. |BM25_a - BM25_b|)
    lexical_overlap: float      # e.g. mean query-doc token overlap for a and b
    semantic_sim: float         # e.g. embedding cosine similarity between doc a and doc b


def schedule_random(pairs: list[PairCandidate], budget: int, seed: int) -> list[PairCandidate]:
    """Baseline: uniform random order, truncated to budget."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    return shuffled[:budget]


def schedule_difficulty_curriculum(pairs: list[PairCandidate], budget: int, seed: int) -> list[PairCandidate]:
    """Large-margin (easy) battles first, near-ties (hard) last -- builds a stable rating
    skeleton before spending budget on the most informative, hardest pairs."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)  # break ties within equal margin deterministically per seed
    ordered = sorted(shuffled, key=lambda p: p.bootstrap_margin, reverse=True)
    return ordered[:budget]


def schedule_anti_curriculum(pairs: list[PairCandidate], budget: int, seed: int) -> list[PairCandidate]:
    """Ablation: near-ties (hardest) first, large-margin (easiest) last -- the reverse
    ordering direction, to test whether ordering direction matters or just exposure."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda p: p.bootstrap_margin)
    return ordered[:budget]


def _quantile_bins(values: list[float], n_bins: int) -> list[float]:
    if not values:
        return [0.0] * (n_bins - 1)
    sorted_vals = sorted(values)
    edges = []
    for i in range(1, n_bins):
        idx = int(len(sorted_vals) * i / n_bins)
        idx = min(idx, len(sorted_vals) - 1)
        edges.append(sorted_vals[idx])
    return edges


def _compositional_stage(p: PairCandidate, lex_edges: list[float], margin_edges: list[float]) -> int:
    """Heuristic 4-stage sub-skill categorization, approximating:
      stage 0 (lexical):              high lexical overlap, large margin -- easy keyword matches
      stage 1 (semantic):             low lexical overlap, large-to-mid margin -- paraphrase/semantic
      stage 2 (instruction-following): mid lexical overlap, mid margin -- nuanced relevance judgments
      stage 3 (hard-distractor):      high lexical overlap, small margin -- near-tie hard negatives
    This is a proxy categorization from cheap signals (BM25 lexical overlap, embedding
    similarity, bootstrap margin); it is not a ground-truth skill taxonomy.
    """
    lex_hi = p.lexical_overlap >= lex_edges[-1] if lex_edges else False
    lex_lo = p.lexical_overlap <= lex_edges[0] if lex_edges else False
    margin_hi = p.bootstrap_margin >= margin_edges[-1] if margin_edges else False
    margin_lo = p.bootstrap_margin <= margin_edges[0] if margin_edges else False

    if margin_lo and lex_hi:
        return 3  # hard-distractor: near-tie + lexically similar (hardest, saved for last)
    if lex_lo and not margin_lo:
        return 1  # semantic: low lexical overlap but not a near-tie
    if lex_hi and margin_hi:
        return 0  # lexical: clear lexical + clear margin (easiest, goes first)
    return 2  # instruction-following (residual middle bucket)


def schedule_compositional_curriculum(
    pairs: list[PairCandidate], budget: int, seed: int, n_quantile_bins: int = 3
) -> list[PairCandidate]:
    """Sophisticated variant: lexical -> semantic -> instruction-following -> hard-distractor,
    the compositional sub-skill ordering described in plan.md."""
    lex_edges = _quantile_bins([p.lexical_overlap for p in pairs], n_quantile_bins)
    margin_edges = _quantile_bins([p.bootstrap_margin for p in pairs], n_quantile_bins)

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda p: _compositional_stage(p, lex_edges, margin_edges))
    return ordered[:budget]


SCHEDULERS = {
    "random": schedule_random,
    "difficulty_curriculum": schedule_difficulty_curriculum,
    "anti_curriculum": schedule_anti_curriculum,
    "compositional_curriculum": schedule_compositional_curriculum,
}
