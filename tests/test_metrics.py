import numpy as np

from src.metrics import (
    comparisons_to_target,
    expected_calibration_error,
    evaluate_ranking_on_pools,
    mrr,
    ndcg_at_k,
    near_tie_slice,
    pairwise_calibration,
    recall_at_k,
    reliability_curve,
)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 1, 0, 0], k=4) == 1.0


def test_ndcg_worst_ranking_less_than_one():
    assert ndcg_at_k([0, 0, 1, 1], k=4) < 1.0


def test_recall_at_k():
    assert recall_at_k([1, 0, 1, 0], n_relevant_total=3, k=4) == 2 / 3


def test_mrr_first_hit_position():
    assert mrr([0, 0, 1, 0]) == 1 / 3
    assert mrr([0, 0, 0]) == 0.0


def test_ece_perfect_calibration_is_zero():
    confidences = np.array([0.1, 0.5, 0.9])
    correct = np.array([0.1, 0.5, 0.9])  # matches confidence exactly in expectation per-bin
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece < 1e-6


def test_reliability_curve_shape():
    confidences = np.random.default_rng(0).uniform(0, 1, 50)
    correct = (np.random.default_rng(1).uniform(0, 1, 50) < confidences).astype(float)
    curve = reliability_curve(confidences, correct, n_bins=5)
    assert len(curve) == 5


def test_near_tie_slice_selects_smallest_margins():
    margins = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    mask = near_tie_slice(margins, threshold_quantile=0.4)
    assert mask[0]  # smallest margin always in the near-tie slice
    assert not mask[-1]


class _FakePool:
    def __init__(self, query_text, doc_ids, doc_texts, relevant_doc_ids):
        self.query_text = query_text
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self.relevant_doc_ids = relevant_doc_ids


class _PerfectReranker:
    """Scores docs by a precomputed dict, perfectly recovering relevance order."""

    def __init__(self, score_by_text):
        self.score_by_text = score_by_text

    def predict(self, pairs):
        return np.array([self.score_by_text[doc] for _, doc in pairs])


def test_evaluate_ranking_on_pools_perfect_reranker_gets_ndcg_one():
    pool = _FakePool(
        query_text="q",
        doc_ids=["d0", "d1", "d2"],
        doc_texts=["relevant text", "somewhat relevant", "irrelevant text"],
        relevant_doc_ids={"d0"},
    )
    reranker = _PerfectReranker({"relevant text": 1.0, "somewhat relevant": 0.5, "irrelevant text": 0.0})
    result = evaluate_ranking_on_pools(reranker, [pool])
    assert result["ndcg@10"] == 1.0
    assert result["mrr"] == 1.0


def test_pairwise_calibration_and_near_tie_slicing():
    reranker = _PerfectReranker({"docA": 2.0, "docB": 0.0})
    judged_pairs = [
        {"query_text": "q", "doc_a_text": "docA", "doc_b_text": "docB", "outcome": 1, "margin": 5.0},
        {"query_text": "q", "doc_a_text": "docA", "doc_b_text": "docB", "outcome": 1, "margin": 0.1},
    ]
    result = pairwise_calibration(reranker, judged_pairs, near_tie_quantile=0.5)
    assert result["pairwise_accuracy"] == 1.0
    assert result["near_tie_pairwise_accuracy"] == 1.0
    assert result["n_pairs"] == 2


def test_comparisons_to_target():
    checkpoints = [
        {"n_comparisons": 10, "metric_value": 0.5},
        {"n_comparisons": 20, "metric_value": 0.7},
        {"n_comparisons": 30, "metric_value": 0.9},
    ]
    assert comparisons_to_target(checkpoints, target=0.7) == 20
    assert comparisons_to_target(checkpoints, target=0.99) is None
