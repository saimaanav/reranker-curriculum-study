"""Dynamics analysis of the rating fixed point (Phase 3, H3): eigenspectrum and
conditioning of the Bradley-Terry Hessian at convergence, under different comparison
scheduling arms, and whether that conditioning predicts downstream calibration/near-tie
performance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass
class EigenSpectrum:
    eigenvalues: np.ndarray       # ascending order
    condition_number: float       # lambda_max / lambda_min (lambda_min clamped away from 0)
    spectral_gap: float           # lambda_min(nonzero) - 0, i.e. smallest eigenvalue itself
    effective_rank: float         # participation ratio: (sum(lambda))^2 / sum(lambda^2)


def eigenspectrum(hessian: np.ndarray, eig_floor: float = 1e-12) -> EigenSpectrum:
    """Computes the eigenspectrum of a (symmetric) Hessian at the converged rating fixed
    point. `eig_floor` guards against division by ~0 for the smallest eigenvalue, which
    can be tiny given the Bradley-Terry translation-invariance + small L2 anchor."""
    eigvals = np.linalg.eigvalsh(hessian)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    lam_min = max(eigvals.min(), eig_floor)
    lam_max = eigvals.max()
    condition_number = float(lam_max / lam_min)
    sum_lam = eigvals.sum()
    sum_lam_sq = (eigvals ** 2).sum()
    effective_rank = float((sum_lam ** 2) / sum_lam_sq) if sum_lam_sq > 0 else 0.0
    return EigenSpectrum(
        eigenvalues=eigvals, condition_number=condition_number,
        spectral_gap=float(eigvals.min()), effective_rank=effective_rank,
    )


def aggregate_conditioning(spectra: list[EigenSpectrum]) -> dict:
    """Summarizes condition number / effective rank across queries for one arm+seed."""
    if not spectra:
        return {"condition_number_mean": None, "condition_number_std": None, "effective_rank_mean": None}
    conds = np.array([s.condition_number for s in spectra])
    ranks = np.array([s.effective_rank for s in spectra])
    return {
        "condition_number_mean": float(conds.mean()),
        "condition_number_std": float(conds.std()),
        "condition_number_median": float(np.median(conds)),
        "effective_rank_mean": float(ranks.mean()),
        "n_queries": len(spectra),
    }


def correlate_conditioning_with_downstream(
    records: list[dict],  # each: {"condition_number": float, "ece": float, "near_tie_accuracy": float}
) -> dict:
    """H3: does worse fixed-point conditioning (higher condition number) predict worse
    downstream calibration (higher ECE) and worse near-tie accuracy (lower accuracy)?
    Expects a positive correlation with ECE and a negative correlation with near-tie
    accuracy if H3 holds."""
    conds = np.array([r["condition_number"] for r in records], dtype=np.float64)
    eces = np.array([r["ece"] for r in records], dtype=np.float64)
    near_tie = np.array([r["near_tie_accuracy"] for r in records], dtype=np.float64)

    result = {"n_points": len(records)}
    if len(records) < 3:
        result["note"] = "insufficient arm/seed points to compute a meaningful correlation"
        return result

    def _safe_correlate(x: np.ndarray, y: np.ndarray) -> dict:
        if np.std(x) == 0 or np.std(y) == 0:
            return {"note": "constant input (e.g. every arm/seed got the same downstream value) -- correlation undefined"}
        pear_r, pear_p = pearsonr(x, y)
        spear_r, spear_p = spearmanr(x, y)
        return {"pearson_r": float(pear_r), "pearson_p": float(pear_p),
                "spearman_r": float(spear_r), "spearman_p": float(spear_p)}

    result["condition_vs_ece"] = {
        **_safe_correlate(conds, eces),
        "expected_direction": "positive (worse conditioning -> higher ECE)",
    }
    result["condition_vs_near_tie_accuracy"] = {
        **_safe_correlate(conds, near_tie),
        "expected_direction": "negative (worse conditioning -> lower near-tie accuracy)",
    }
    return result


def score_separation(scores: np.ndarray, relevance_labels: np.ndarray) -> float | None:
    """Lightweight score-distribution geometry cross-check: mean reranker score on
    relevant docs minus mean score on irrelevant docs (in the trained reranker's own
    score units). Larger separation suggests a more confidently discriminative model."""
    relevant_scores = scores[relevance_labels > 0]
    irrelevant_scores = scores[relevance_labels == 0]
    if len(relevant_scores) == 0 or len(irrelevant_scores) == 0:
        return None
    return float(relevant_scores.mean() - irrelevant_scores.mean())
