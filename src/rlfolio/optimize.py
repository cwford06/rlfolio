"""Long-only portfolio construction: covariance shrinkage and optimizers."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog
from scipy.sparse import eye as sparse_eye, hstack, csr_matrix
from scipy.spatial.distance import squareform


def ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward scaled identity."""
    x = returns - returns.mean(axis=0)
    t, n = x.shape
    sample = x.T @ x / t
    mu = np.trace(sample) / n
    target = mu * np.eye(n)
    d2 = np.sum((sample - target) ** 2) / n
    if d2 <= 0:
        return sample
    b2_bar = sum(np.sum((np.outer(row, row) - sample) ** 2) for row in x) / (n * t**2)
    b2 = min(b2_bar, d2)
    shrinkage = b2 / d2
    return shrinkage * target + (1 - shrinkage) * sample


def project_capped_simplex(v: np.ndarray, cap: float, total: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {w : sum(w) = total, 0 <= w <= cap}."""
    if cap * len(v) < total - 1e-12:
        raise ValueError(f"cap {cap} is infeasible for {len(v)} assets")
    low, high = float(v.min() - cap), float(v.max())
    for _ in range(60):
        mid = 0.5 * (low + high)
        if np.clip(v - mid, 0.0, cap).sum() > total:
            low = mid
        else:
            high = mid
    return np.clip(v - 0.5 * (low + high), 0.0, cap)


def solve_long_only_qp(
    cov: np.ndarray,
    mu: np.ndarray | None = None,
    risk_aversion: float = 1.0,
    cap: float = 1.0,
    iterations: int = 500,
) -> np.ndarray:
    """Maximize ``mu'w - risk_aversion/2 * w'Cov w`` on the capped simplex (FISTA)."""
    n = len(cov)
    mu = np.zeros(n) if mu is None else mu
    hessian = risk_aversion * cov
    step = 1.0 / max(float(np.linalg.eigvalsh(hessian)[-1]), 1e-12)
    w = project_capped_simplex(np.full(n, 1.0 / n), cap)
    y, momentum = w.copy(), 1.0
    for _ in range(iterations):
        next_w = project_capped_simplex(y - step * (hessian @ y - mu), cap)
        if np.abs(next_w - w).max() < 1e-9:
            return next_w
        next_momentum = 0.5 * (1 + np.sqrt(1 + 4 * momentum**2))
        y = next_w + (momentum - 1) / next_momentum * (next_w - w)
        w, momentum = next_w, next_momentum
    return w


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def min_variance(cov: np.ndarray, cap: float = 1.0) -> np.ndarray:
    return solve_long_only_qp(cov, cap=cap)


def mean_variance(
    cov: np.ndarray, mu: np.ndarray, risk_aversion: float = 4.0, cap: float = 1.0
) -> np.ndarray:
    return solve_long_only_qp(cov, mu, risk_aversion, cap)


def risk_parity(cov: np.ndarray, cap: float = 1.0, iterations: int = 1000) -> np.ndarray:
    """Equal risk contribution: coordinate descent on 0.5 x'Cov x - sum(b * log x)."""
    n = len(cov)
    diag = np.diag(cov)
    x = 1.0 / np.sqrt(diag)
    budget = 1.0 / n
    for _ in range(iterations):
        previous = x.copy()
        for i in range(n):
            c = cov[i] @ x - diag[i] * x[i]
            x[i] = (-c + np.sqrt(c**2 + 4 * diag[i] * budget)) / (2 * diag[i])
        if np.abs(x / previous - 1).max() < 1e-10:
            break
    x /= x.sum()
    return project_capped_simplex(x, cap) if x.max() > cap else x


def hierarchical_risk_parity(cov: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """Lopez de Prado (2016) hierarchical risk parity."""
    n = len(cov)
    vol = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(vol, vol), -1.0, 1.0)
    distance = np.sqrt(np.clip(0.5 * (1 - corr), 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="single"))

    def cluster_variance(items: np.ndarray) -> float:
        inverse_var = 1.0 / np.diag(cov)[items]
        w = inverse_var / inverse_var.sum()
        return float(w @ cov[np.ix_(items, items)] @ w)

    weights = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [c for c in clusters if len(c) > 1]
        next_clusters = []
        for cluster in clusters:
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            var_left, var_right = cluster_variance(left), cluster_variance(right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_clusters += [left, right]
        clusters = next_clusters
    return project_capped_simplex(weights, cap) if weights.max() > cap else weights


def min_cvar(returns: np.ndarray, beta: float = 0.95, cap: float = 1.0) -> np.ndarray:
    """Minimize historical CVaR with the Rockafellar-Uryasev linear program."""
    s, n = returns.shape
    objective = np.concatenate([np.zeros(n), [1.0], np.full(s, 1.0 / ((1 - beta) * s))])
    # Loss constraint: -r_s'w - alpha - u_s <= 0
    a_ub = hstack(
        [csr_matrix(-returns), csr_matrix(-np.ones((s, 1))), -sparse_eye(s, format="csr")],
        format="csr",
    )
    a_eq = csr_matrix(np.concatenate([np.ones(n), np.zeros(s + 1)])[None, :])
    bounds = [(0.0, cap)] * n + [(None, None)] + [(0.0, None)] * s
    solution = linprog(
        objective, A_ub=a_ub, b_ub=np.zeros(s), A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs"
    )
    if not solution.success:
        return project_capped_simplex(equal_weight(n), cap)
    return project_capped_simplex(solution.x[:n], cap)
