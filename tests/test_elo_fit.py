"""Correctness cross-check of our PyTorch Bradley-Terry MLE against `choix`.

`choix` is only ever used here, as an external sanity check on synthetic data with a
known ground-truth rating vector -- never inside the load-bearing fit in src/elo_fit.py,
because Phase 3 needs the Hessian as a first-class output and choix does not expose it.
"""
import choix
import numpy as np
import pytest
import torch

from src.elo_fit import PairwiseOutcomes, fit_cross_query_bias, fit_ratings


def _synthetic_outcomes(true_ratings: np.ndarray, n_battles_per_pair: int, seed: int) -> PairwiseOutcomes:
    rng = np.random.default_rng(seed)
    n = len(true_ratings)
    outcomes = PairwiseOutcomes(n_docs=n)
    pairwise_data = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_i_beats_j = 1.0 / (1.0 + np.exp(-(true_ratings[i] - true_ratings[j])))
            wins_i = rng.binomial(n_battles_per_pair, p_i_beats_j)
            if wins_i > 0:
                outcomes.add(i, j, wins_i)
                pairwise_data.extend([(i, j)] * wins_i)
    return outcomes, pairwise_data


def test_bradley_terry_matches_choix_on_synthetic_data():
    true_ratings = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    outcomes, pairwise_data = _synthetic_outcomes(true_ratings, n_battles_per_pair=200, seed=0)

    ours = fit_ratings(outcomes, method="bradley_terry", optimizer="newton", l2_reg=1e-4)

    choix_params = choix.ilsr_pairwise(len(true_ratings), pairwise_data, alpha=1e-4)

    # Both scales are only identified up to an additive constant (before our L2 anchor
    # pulls it toward 0) -- compare after centering, and via rank correlation + predicted
    # win-probability agreement rather than raw equality.
    ours_centered = ours.ratings - ours.ratings.mean()
    choix_centered = np.array(choix_params) - np.mean(choix_params)

    assert np.corrcoef(ours_centered, choix_centered)[0, 1] > 0.99

    # predicted pairwise win probabilities should closely agree pointwise
    for i in range(len(true_ratings)):
        for j in range(len(true_ratings)):
            if i == j:
                continue
            p_ours = 1.0 / (1.0 + np.exp(-(ours_centered[i] - ours_centered[j])))
            p_choix = 1.0 / (1.0 + np.exp(-(choix_centered[i] - choix_centered[j])))
            assert abs(p_ours - p_choix) < 0.05


def test_hessian_analytic_matches_autograd():
    true_ratings = np.array([1.5, 0.5, -0.5, -1.5])
    outcomes, _ = _synthetic_outcomes(true_ratings, n_battles_per_pair=50, seed=1)
    res = fit_ratings(outcomes, method="bradley_terry", optimizer="newton", l2_reg=1e-4)
    assert np.allclose(res.hessian_analytic, res.hessian_autograd, atol=1e-6)
    # Hessian of a negative log-likelihood at an MLE optimum should be positive semi-definite.
    eigvals = np.linalg.eigvalsh(res.hessian_analytic)
    assert eigvals.min() > -1e-6


def test_fit_converges_and_recovers_ranking():
    true_ratings = np.array([3.0, 1.0, -1.0, -3.0])
    outcomes, _ = _synthetic_outcomes(true_ratings, n_battles_per_pair=100, seed=2)
    res = fit_ratings(outcomes, method="bradley_terry", optimizer="newton")
    assert res.converged
    order = np.argsort(-res.ratings)
    assert list(order) == [0, 1, 2, 3]


def test_cross_query_bias_recovers_known_offset():
    # query "a" ratings are on a scale shifted by +5 relative to query "b"'s true scale
    ratings_per_query = {
        "a": np.array([5.0, 4.0]),   # true underlying values [0.0, -1.0] + offset 5
        "b": np.array([0.0, -1.0]),
    }
    # cross-query calibration pairs: doc 0 of query a vs doc 0 of query b, etc, generated
    # from the TRUE unbiased scale so the fit should recover offset_a ~ +5, offset_b ~ 0.
    true_a = np.array([0.0, -1.0])
    true_b = np.array([0.0, -1.0])
    rng = np.random.default_rng(0)
    calibration_pairs = []
    for _ in range(200):
        i, j = rng.integers(0, 2), rng.integers(0, 2)
        p = 1.0 / (1.0 + np.exp(-(true_a[i] - true_b[j])))
        outcome = 1 if rng.random() < p else 0
        calibration_pairs.append(("a", i, "b", j, outcome))

    bias = fit_cross_query_bias(ratings_per_query, calibration_pairs)
    # bias_a - bias_b should correct the +5 offset back toward 0, i.e. bias_a ~ -5 relative to bias_b
    assert abs((bias["a"] - bias["b"]) - (-5.0)) < 1.0
