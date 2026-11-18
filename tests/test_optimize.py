import numpy as np
import pytest
from scipy.optimize import minimize

from rlfolio import optimize as opt


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(0)
    factor = rng.normal(0, 0.01, (500, 1))
    return factor @ rng.uniform(0.2, 1.2, (1, 6)) + rng.normal(0, 0.006, (500, 6))


@pytest.fixture(scope="module")
def cov(returns):
    return opt.ledoit_wolf(returns) * 252


def test_ledoit_wolf_is_symmetric_positive_definite(cov):
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_ledoit_wolf_rescues_singular_sample():
    x = np.random.default_rng(1).normal(size=(10, 30))
    assert np.linalg.eigvalsh(opt.ledoit_wolf(x)).min() > 0


def test_capped_simplex_projection_is_feasible():
    w = opt.project_capped_simplex(np.array([5.0, 1.0, -2.0, 0.3]), cap=0.4)
    assert w.sum() == pytest.approx(1.0) and w.max() <= 0.4 + 1e-12 and w.min() >= 0


def test_capped_simplex_rejects_infeasible_cap():
    with pytest.raises(ValueError):
        opt.project_capped_simplex(np.zeros(3), cap=0.2)


def test_min_variance_matches_scipy(cov):
    n = len(cov)
    reference = minimize(
        lambda w: w @ cov @ w, np.full(n, 1 / n), jac=lambda w: 2 * cov @ w,
        bounds=[(0, 0.35)] * n, constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP", options={"ftol": 1e-12, "maxiter": 500},
    )
    w = opt.min_variance(cov, cap=0.35)
    assert w @ cov @ w == pytest.approx(reference.fun, rel=1e-4)


def test_risk_parity_equalizes_risk_contributions(cov):
    w = opt.risk_parity(cov)
    contributions = w * (cov @ w)
    assert contributions.std() / contributions.mean() < 1e-5


def test_hrp_is_a_valid_portfolio(cov):
    w = opt.hierarchical_risk_parity(cov, cap=0.4)
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all() and w.max() <= 0.4 + 1e-12


def test_min_cvar_has_no_worse_tail_than_equal_weight(returns):
    w = opt.min_cvar(returns, cap=0.5)

    def cvar(weights):
        losses = np.sort(-(returns @ weights))
        return losses[int(0.95 * len(losses)) :].mean()

    assert w.sum() == pytest.approx(1.0)
    assert cvar(w) <= cvar(opt.equal_weight(returns.shape[1])) + 1e-9
