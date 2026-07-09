"""Bradley-Terry / Thurstone rating fit with the Hessian at the converged fixed point
exposed as a first-class output.

Design requirement (see plan.md): the Hessian of the negative log-likelihood at the
converged rating vector must be directly accessible, computed via our own PyTorch
implementation (closed-form analytic Bradley-Terry Hessian, cross-checked against
torch.autograd.functional.hessian). `choix` is used only in tests/test_elo_fit.py as an
external correctness check on synthetic data -- never as the load-bearing fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class PairwiseOutcomes:
    """Aggregated win counts for one query's candidate pool.

    n_docs: number of candidate documents for this query.
    win_counts: dict[(i, j)] = number of times doc i beat doc j, for i != j.
    """

    n_docs: int
    win_counts: dict[tuple[int, int], int] = field(default_factory=dict)

    def add(self, winner: int, loser: int, n: int = 1) -> None:
        key = (winner, loser)
        self.win_counts[key] = self.win_counts.get(key, 0) + n

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (i_idx, j_idx, w_ij) for every unordered pair with at least one battle,
        where w_ij is wins of i over j and the caller also needs w_ji (looked up separately)."""
        pairs = sorted({tuple(sorted(k)) for k in self.win_counts})
        i_idx = np.array([p[0] for p in pairs], dtype=np.int64)
        j_idx = np.array([p[1] for p in pairs], dtype=np.int64)
        return i_idx, j_idx, pairs


@dataclass
class RatingFitResult:
    ratings: np.ndarray            # (n_docs,) converged rating vector r*
    hessian_analytic: np.ndarray   # (n_docs, n_docs) closed-form Bradley-Terry Hessian at r*
    hessian_autograd: np.ndarray   # (n_docs, n_docs) autograd cross-check
    n_iters: int
    converged: bool
    neg_log_likelihood: float


def _neg_log_likelihood(
    ratings: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    w_ij: torch.Tensor,
    w_ji: torch.Tensor,
    l2_reg: float,
    method: str,
) -> torch.Tensor:
    diff = ratings[i_idx] - ratings[j_idx]
    if method == "bradley_terry":
        # log P(i beats j) = log sigmoid(diff); log P(j beats i) = log sigmoid(-diff)
        log_p_i = torch.nn.functional.logsigmoid(diff)
        log_p_j = torch.nn.functional.logsigmoid(-diff)
    elif method == "thurstone":
        # probit link: P(i beats j) = Phi(diff / sqrt(2))
        normal = torch.distributions.Normal(0.0, 1.0)
        scaled = diff / (2.0 ** 0.5)
        eps = 1e-9
        log_p_i = torch.log(normal.cdf(scaled).clamp_min(eps))
        log_p_j = torch.log(normal.cdf(-scaled).clamp_min(eps))
    else:
        raise ValueError(f"unknown rating fit method: {method}")
    nll = -(w_ij * log_p_i + w_ji * log_p_j).sum()
    nll = nll + l2_reg * (ratings ** 2).sum()
    return nll


def fit_ratings(
    outcomes: PairwiseOutcomes,
    method: str = "bradley_terry",
    optimizer: str = "newton",
    max_iters: int = 200,
    tol: float = 1e-8,
    l2_reg: float = 1e-4,
) -> RatingFitResult:
    """Fits a single query's document ratings by MLE on pairwise outcomes.

    The rating scale is only identified up to an additive constant in the pure
    Bradley-Terry model; `l2_reg` anchors the scale (ridge toward 0) so both the
    optimization and the Hessian are numerically well-posed and invertible.
    """
    n = outcomes.n_docs
    i_idx_np, j_idx_np, pairs = outcomes.as_arrays()
    w_ij_np = np.array([outcomes.win_counts.get((i, j), 0) for i, j in pairs], dtype=np.float64)
    w_ji_np = np.array([outcomes.win_counts.get((j, i), 0) for i, j in pairs], dtype=np.float64)

    i_idx = torch.tensor(i_idx_np, dtype=torch.long)
    j_idx = torch.tensor(j_idx_np, dtype=torch.long)
    w_ij = torch.tensor(w_ij_np, dtype=torch.float64)
    w_ji = torch.tensor(w_ji_np, dtype=torch.float64)

    ratings = torch.zeros(n, dtype=torch.float64, requires_grad=True)

    def nll_fn(r: torch.Tensor) -> torch.Tensor:
        return _neg_log_likelihood(r, i_idx, j_idx, w_ij, w_ji, l2_reg, method)

    converged = False
    n_iters = 0
    if optimizer == "newton":
        r = ratings.detach().clone()
        for it in range(max_iters):
            n_iters = it + 1
            r.requires_grad_(True)
            loss = nll_fn(r)
            grad = torch.autograd.grad(loss, r, create_graph=True)[0]
            hess = torch.autograd.functional.hessian(nll_fn, r.detach())
            # damped Newton step: solve H @ delta = grad, r_new = r - delta
            hess_damped = hess + 1e-6 * torch.eye(n, dtype=torch.float64)
            delta = torch.linalg.solve(hess_damped, grad.detach())
            step_size = 1.0
            r_new = (r.detach() - step_size * delta)
            step_norm = torch.norm(r_new - r.detach()).item()
            r = r_new.detach()
            if step_norm < tol:
                converged = True
                break
        ratings = r
    elif optimizer == "gradient_descent":
        r = ratings.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([r], lr=0.05)
        prev_loss = float("inf")
        for it in range(max_iters):
            n_iters = it + 1
            opt.zero_grad()
            loss = nll_fn(r)
            loss.backward()
            opt.step()
            cur_loss = loss.item()
            if abs(prev_loss - cur_loss) < tol:
                converged = True
                break
            prev_loss = cur_loss
        ratings = r.detach()
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")

    r_star = ratings.detach().clone()
    final_nll = nll_fn(r_star).item()

    hessian_autograd = torch.autograd.functional.hessian(nll_fn, r_star).numpy()
    hessian_analytic = _analytic_bradley_terry_hessian(
        r_star.numpy(), i_idx_np, j_idx_np, w_ij_np, w_ji_np, n, l2_reg, method
    )

    return RatingFitResult(
        ratings=r_star.numpy(),
        hessian_analytic=hessian_analytic,
        hessian_autograd=hessian_autograd,
        n_iters=n_iters,
        converged=converged,
        neg_log_likelihood=final_nll,
    )


def _analytic_bradley_terry_hessian(
    ratings: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    w_ij: np.ndarray,
    w_ji: np.ndarray,
    n: int,
    l2_reg: float,
    method: str,
) -> np.ndarray:
    """Closed-form Hessian of the Bradley-Terry negative log-likelihood.

    For the logistic (Bradley-Terry) link, d^2(NLL)/dr_i dr_j = -(w_ij + w_ji) * p * (1-p)
    for i != j, where p = sigmoid(r_i - r_j); diagonal entries are minus the row sum of the
    off-diagonal entries (since NLL is translation-invariant before the L2 anchor), plus 2*l2_reg.
    Thurstone (probit) falls back to the autograd Hessian since the closed form is messier;
    callers should treat hessian_autograd as authoritative for that method.
    """
    H = np.zeros((n, n), dtype=np.float64)
    if method != "bradley_terry":
        return H  # analytic form not implemented for thurstone; use hessian_autograd
    diff = ratings[i_idx] - ratings[j_idx]
    p = 1.0 / (1.0 + np.exp(-diff))
    total_w = w_ij + w_ji
    off = -total_w * p * (1 - p)
    for a, (i, j, val) in enumerate(zip(i_idx, j_idx, off)):
        H[i, j] += val
        H[j, i] += val
        H[i, i] -= val
        H[j, j] -= val
    H += 2.0 * l2_reg * np.eye(n)
    return H


def fit_cross_query_bias(
    ratings_per_query: dict[str, np.ndarray],
    calibration_pairs: list[tuple[str, int, str, int, int]],
) -> dict[str, float]:
    """Fits an additive per-query bias offset so ratings are comparable across queries/domains.

    `calibration_pairs` is a list of (query_a, doc_idx_a, query_b, doc_idx_b, outcome) tuples
    from a small set of cross-query judged comparisons (e.g. shared anchor documents judged
    against candidates from two different queries); outcome=1 means doc_a beat doc_b.
    Mirrors zELO's cross-query calibration: without it, a rating of X in one query's local
    scale need not mean the same thing as X in another query's scale.
    """
    query_ids = sorted(ratings_per_query.keys())
    idx_of = {q: k for k, q in enumerate(query_ids)}
    n_q = len(query_ids)
    if not calibration_pairs or n_q <= 1:
        return {q: 0.0 for q in query_ids}

    bias = torch.zeros(n_q, dtype=torch.float64, requires_grad=True)
    qa_idx = torch.tensor([idx_of[c[0]] for c in calibration_pairs], dtype=torch.long)
    qb_idx = torch.tensor([idx_of[c[2]] for c in calibration_pairs], dtype=torch.long)
    ra = torch.tensor([ratings_per_query[c[0]][c[1]] for c in calibration_pairs], dtype=torch.float64)
    rb = torch.tensor([ratings_per_query[c[2]][c[3]] for c in calibration_pairs], dtype=torch.float64)
    y = torch.tensor([float(c[4]) for c in calibration_pairs], dtype=torch.float64)

    opt = torch.optim.Adam([bias], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        diff = (ra + bias[qa_idx]) - (rb + bias[qb_idx])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(diff, y)
        loss = loss + 1e-4 * (bias ** 2).sum()
        loss.backward()
        opt.step()

    return {q: bias[idx_of[q]].item() for q in query_ids}
