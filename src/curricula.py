"""Comparison scheduling arms for Phase 2.

Three arms:
  - random_cycles: zELO-style coverage-guaranteed scheduling (every doc compared equally)
  - compositional: weighted difficulty score, easy pairs first
  - anti_compositional: exact reverse of compositional, hard pairs first

Difficulty score (higher = easier to judge):
    difficulty = 0.7 * margin + 0.3 * (1 - semantic_sim)
where:
    margin       = |BM25_a - BM25_b|, min-max normalized to [0,1] (large gap = easy)
    semantic_sim = embedding cosine similarity between doc_a and doc_b, normalized to [0,1]
                   (high similarity = docs are confusingly close = hard, so we invert)
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class PairCandidate:
    query_id: str
    doc_a_id: str
    doc_b_id: str
    bootstrap_margin: float   # |BM25_a - BM25_b|, normalized to [0,1]
    semantic_sim: float       # embedding cosine similarity between doc_a and doc_b, normalized to [0,1]


def _difficulty_score(p: PairCandidate, alpha: float = 0.7, beta: float = 0.3) -> float:
    """Higher score = easier pair to judge."""
    return alpha * p.bootstrap_margin + beta * (1.0 - p.semantic_sim)


def schedule_random_cycles(pairs: list[PairCandidate], budget: int, seed: int) -> list[PairCandidate]:
    """zELO-style: build random cycles within each query pool so every doc gets compared
    roughly equally. Guarantees no doc is left with zero win/loss record."""
    rng = random.Random(seed)

    # group pairs and collect docs per query
    by_query: dict[str, list[PairCandidate]] = defaultdict(list)
    pair_lookup: dict[tuple, PairCandidate] = {}
    for p in pairs:
        by_query[p.query_id].append(p)
        pair_lookup[(p.query_id, p.doc_a_id, p.doc_b_id)] = p
        pair_lookup[(p.query_id, p.doc_b_id, p.doc_a_id)] = p

    # proportional budget per query
    total = len(pairs)
    selected: list[PairCandidate] = []
    selected_keys: set[tuple] = set()

    query_ids = list(by_query.keys())
    rng.shuffle(query_ids)

    for qid in query_ids:
        q_pairs = by_query[qid]
        # dict.fromkeys (not a set) so doc order is insertion-order, not Python's
        # per-process randomized string-hash order -- keeps the schedule reproducible
        # across separate process invocations given the same seed, not just within one run.
        docs = list(dict.fromkeys(d for p in q_pairs for d in (p.doc_a_id, p.doc_b_id)))
        n = len(docs)
        if n < 2:
            continue
        q_budget = min(max(n, round(budget * len(q_pairs) / total)), len(q_pairs))

        q_selected: list[PairCandidate] = []
        q_keys: set[tuple] = set()

        while len(q_selected) < q_budget:
            perm = docs[:]
            rng.shuffle(perm)
            for i in range(len(perm)):
                a, b = perm[i], perm[(i + 1) % len(perm)]
                key = (qid, min(a, b), max(a, b))
                if key not in q_keys and key not in selected_keys:
                    p = pair_lookup.get((qid, a, b)) or pair_lookup.get((qid, b, a))
                    if p:
                        q_selected.append(p)
                        q_keys.add(key)
                if len(q_selected) >= q_budget:
                    break
            else:
                continue
            break

        selected.extend(q_selected)
        selected_keys.update(q_keys)

    rng.shuffle(selected)
    return selected[:budget]


PER_QUERY_FLOOR = 10   # min pairs guaranteed per query before free budget clusters by difficulty
N_DIFFICULTY_BINS = 10   # quantile buckets; macro order (bin sequence) is deterministic per
                          # arm direction, but which pairs land at the front of a bin -- and
                          # therefore which pairs survive a mid-bin budget/floor cutoff -- is
                          # shuffled per seed. A strict full sort by a continuous difficulty
                          # score has no real ties to break, so shuffling before sorting was a
                          # no-op; bucketing gives the seed something real to randomize.


def _bucket_by_difficulty(pairs: list[PairCandidate], seed: int, reverse: bool) -> list[PairCandidate]:
    """Bins pairs into N_DIFFICULTY_BINS quantiles by difficulty score, orders the bins by
    arm direction (easiest-first for compositional, hardest-first for anti_compositional),
    and shuffles within each bin using the seed's RNG."""
    rng = random.Random(seed)
    ascending = sorted(pairs, key=_difficulty_score)  # low score (hard) -> high score (easy)
    n = len(ascending)
    bin_size = max(1, n // N_DIFFICULTY_BINS)
    bins = [ascending[i:i + bin_size] for i in range(0, n, bin_size)]
    # bins[0] = hardest bin ... bins[-1] = easiest bin
    bin_order = list(reversed(bins)) if reverse else bins

    ordered: list[PairCandidate] = []
    for b in bin_order:
        b_shuffled = b[:]
        rng.shuffle(b_shuffled)
        ordered.extend(b_shuffled)
    return ordered


def _difficulty_scheduled(pairs: list[PairCandidate], budget: int, seed: int, reverse: bool) -> list[PairCandidate]:
    """Guarantees each query at least PER_QUERY_FLOOR pairs (its own best-by-difficulty
    pairs, in the arm's direction), then fills the remaining budget by taking leftover
    pairs in bucketed-difficulty order -- letting hard/easy pairs cluster in whichever
    queries have the most of them."""
    ordered = _bucket_by_difficulty(pairs, seed, reverse)

    by_query: dict[str, list[PairCandidate]] = defaultdict(list)
    for p in ordered:
        by_query[p.query_id].append(p)

    selected: list[PairCandidate] = []
    selected_keys: set[tuple] = set()
    for qid, q_pairs in by_query.items():
        floor_pairs = q_pairs[:PER_QUERY_FLOOR]
        selected.extend(floor_pairs)
        selected_keys.update((qid, p.doc_a_id, p.doc_b_id) for p in floor_pairs)

    remaining_budget = budget - len(selected)
    if remaining_budget > 0:
        leftover = [p for p in ordered if (p.query_id, p.doc_a_id, p.doc_b_id) not in selected_keys]
        selected.extend(leftover[:remaining_budget])

    return selected[:budget]


def schedule_compositional(pairs: list[PairCandidate], budget: int, seed: int) -> list[PairCandidate]:
    """Easy pairs first (highest difficulty score), hard pairs last.
    Builds a stable B-T rating skeleton before spending budget on confusing near-ties."""
    return _difficulty_scheduled(pairs, budget, seed, reverse=True)


def schedule_anti_compositional(pairs: list[PairCandidate], budget: int, seed: int) -> list[PairCandidate]:
    """Exact reverse of compositional: hard pairs first, easy pairs last.
    True ablation — same scoring function, opposite order."""
    return _difficulty_scheduled(pairs, budget, seed, reverse=False)


SCHEDULERS = {
    "random_cycles": schedule_random_cycles,
    "compositional": schedule_compositional,
    "anti_compositional": schedule_anti_compositional,
}
