import numpy as np

from src.comparison_difficulty import (
    judge_flip_rate,
    margin_difficulty,
    rank_by_difficulty,
    score_pairs,
)
from src.judge import JudgeVerdict


def test_margin_difficulty_basic():
    ratings = np.array([2.0, 0.0, -2.0])
    idx = {"d0": 0, "d1": 1, "d2": 2}
    assert margin_difficulty(ratings, idx, "d0", "d2") == 4.0
    assert margin_difficulty(ratings, idx, "d0", "d1") == 2.0


def test_judge_flip_rate_unanimous_and_split():
    unanimous = [JudgeVerdict("A", 0.9, "m1", False), JudgeVerdict("A", 0.8, "m2", False)]
    assert judge_flip_rate(unanimous) == 0.0

    split = [JudgeVerdict("A", 0.9, "m1", False), JudgeVerdict("B", 0.6, "m2", False)]
    assert judge_flip_rate(split) == 0.5


def test_score_pairs_and_rank_by_difficulty():
    ratings_per_query = {"q1": np.array([3.0, 0.1, 0.0, -3.0])}
    doc_index = {"q1": {"d0": 0, "d1": 1, "d2": 2, "d3": 3}}
    pairs = [("q1", "d0", "d3"), ("q1", "d1", "d2")]

    scored = score_pairs(pairs, ratings_per_query, doc_index)
    margins = {(-1): None}
    by_pair = {(p.doc_a_id, p.doc_b_id): p.margin for p in scored}
    assert by_pair[("d0", "d3")] == 6.0
    assert abs(by_pair[("d1", "d2")] - 0.1) < 1e-9

    ranked = rank_by_difficulty(scored, hardest_first=True)
    assert ranked[0].doc_a_id == "d1" and ranked[0].doc_b_id == "d2"  # near-tie is hardest
    assert ranked[-1].doc_a_id == "d0" and ranked[-1].doc_b_id == "d3"
