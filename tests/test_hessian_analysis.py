import numpy as np

from src.hessian_analysis import (
    EigenSpectrum,
    aggregate_conditioning,
    correlate_conditioning_with_downstream,
    eigenspectrum,
    score_separation,
)


def test_eigenspectrum_well_conditioned_identity():
    H = np.eye(4)
    spec = eigenspectrum(H)
    assert np.isclose(spec.condition_number, 1.0)
    assert np.isclose(spec.effective_rank, 4.0)


def test_eigenspectrum_ill_conditioned():
    H = np.diag([1e-6, 1.0, 1.0, 1.0])
    spec = eigenspectrum(H)
    assert spec.condition_number > 1e5


def test_aggregate_conditioning_mean_std():
    spectra = [
        EigenSpectrum(eigenvalues=np.array([1, 2]), condition_number=2.0, spectral_gap=1.0, effective_rank=1.8),
        EigenSpectrum(eigenvalues=np.array([1, 4]), condition_number=4.0, spectral_gap=1.0, effective_rank=1.5),
    ]
    agg = aggregate_conditioning(spectra)
    assert agg["condition_number_mean"] == 3.0
    assert agg["n_queries"] == 2


def test_aggregate_conditioning_empty():
    agg = aggregate_conditioning([])
    assert agg["condition_number_mean"] is None


def test_correlate_conditioning_positive_with_ece_negative_with_near_tie():
    # construct records where higher condition number -> higher ECE, lower near-tie accuracy
    records = [
        {"condition_number": 10.0, "ece": 0.05, "near_tie_accuracy": 0.9},
        {"condition_number": 20.0, "ece": 0.15, "near_tie_accuracy": 0.7},
        {"condition_number": 30.0, "ece": 0.30, "near_tie_accuracy": 0.5},
        {"condition_number": 40.0, "ece": 0.40, "near_tie_accuracy": 0.4},
    ]
    result = correlate_conditioning_with_downstream(records)
    assert result["condition_vs_ece"]["pearson_r"] > 0.9
    assert result["condition_vs_near_tie_accuracy"]["pearson_r"] < -0.9


def test_correlate_conditioning_insufficient_points():
    result = correlate_conditioning_with_downstream([{"condition_number": 1.0, "ece": 0.1, "near_tie_accuracy": 0.9}])
    assert "note" in result


def test_score_separation():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    sep = score_separation(scores, labels)
    assert np.isclose(sep, 0.7)


def test_score_separation_no_relevant_docs_returns_none():
    scores = np.array([0.1, 0.2])
    labels = np.array([0, 0])
    assert score_separation(scores, labels) is None
