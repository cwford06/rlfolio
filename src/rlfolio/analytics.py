"""Performance statistics, including tests that account for selection bias."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PERIODS = 252
EULER_GAMMA = 0.5772156649


def sharpe(returns: np.ndarray, periods: int = PERIODS) -> float:
    returns = np.asarray(returns, dtype=float)
    std = returns.std(ddof=1)
    return float(returns.mean() / std * np.sqrt(periods)) if std > 0 else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    return float((1 - equity / np.maximum.accumulate(equity)).max())


def performance(equity: pd.Series) -> dict[str, float]:
    values = equity.to_numpy(float)
    returns = values[1:] / values[:-1] - 1
    years = len(returns) / PERIODS
    cagr = (values[-1] / values[0]) ** (1 / years) - 1
    downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)) * np.sqrt(PERIODS)
    drawdown = max_drawdown(values)
    return {
        "cagr": cagr,
        "volatility": float(returns.std(ddof=1) * np.sqrt(PERIODS)),
        "sharpe": sharpe(returns),
        "sortino": float(returns.mean() * PERIODS / downside) if downside > 0 else 0.0,
        "max_drawdown": drawdown,
        "calmar": cagr / drawdown if drawdown > 0 else 0.0,
        "worst_day": float(returns.min()),
    }


def probabilistic_sharpe(returns: np.ndarray, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark_sr), per Bailey and Lopez de Prado (2012)."""
    returns = np.asarray(returns, dtype=float)
    sr = returns.mean() / returns.std(ddof=1)
    skew = stats.skew(returns)
    kurtosis = stats.kurtosis(returns, fisher=False)
    denominator = np.sqrt(max(1 - skew * sr + (kurtosis - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(len(returns) - 1) / denominator))


def deflated_sharpe(returns: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """Probabilistic Sharpe against the best Sharpe expected from luck across trials.

    ``trial_sharpes`` holds the per-period Sharpe of every strategy tried.
    """
    trials = np.asarray(trial_sharpes, dtype=float)
    n = len(trials)
    if n < 2:
        return probabilistic_sharpe(returns)
    z1, z2 = stats.norm.ppf(1 - 1 / n), stats.norm.ppf(1 - 1 / (n * np.e))
    expected_max = np.sqrt(trials.var(ddof=1)) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    return probabilistic_sharpe(returns, float(expected_max))


def bootstrap_sharpe_difference(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    mean_block: int = 21,
    seed: int = 0,
) -> dict[str, float]:
    """Stationary bootstrap of Sharpe(a) - Sharpe(b) on paired daily returns."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    index = np.zeros((n_boot, n), dtype=int)
    index[:, 0] = rng.integers(n, size=n_boot)
    for t in range(1, n):
        restart = rng.random(n_boot) < 1 / mean_block
        index[:, t] = np.where(restart, rng.integers(n, size=n_boot), (index[:, t - 1] + 1) % n)

    def boot_sharpe(x: np.ndarray) -> np.ndarray:
        sample = x[index]
        return sample.mean(axis=1) / sample.std(axis=1, ddof=1) * np.sqrt(PERIODS)

    diffs = boot_sharpe(a) - boot_sharpe(b)
    return {
        "diff": sharpe(a) - sharpe(b),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_not_better": float((diffs <= 0).mean()),
    }
