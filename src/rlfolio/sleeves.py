"""Causal regime features and the book of candidate portfolios the RL agents pick from.

Rather than asking an agent to output raw weights, which is hopelessly noisy on a
few thousand daily observations, each action selects a *sleeve* (an optimizer's
portfolio) and an exposure level. The optimizers handle cross-sectional allocation
and the agent learns when each one deserves capital.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import optimize as opt
from .data import MarketData

WARMUP = 253  # bars of history required before the first decision

FEATURE_NAMES = (
    "trend_21", "trend_63", "trend_252", "vol_21", "vol_ratio",
    "drawdown", "breadth", "avg_corr", "dispersion",
)
TABULAR_FEATURES = (1, 3, 7)  # trend_63, vol_21, avg_corr: the Q-table's regime axes

ACTION_SLEEVES = ("ew", "minvar", "erc", "mvo", "trend")
BENCHMARK_SLEEVES = ("hrp", "cvar")
EXPOSURES = (1.0, 0.5)


def regime_features(close: np.ndarray, t: int) -> np.ndarray:
    """Market-state vector at the close of bar ``t``, reading only bars up to ``t``."""
    if t < WARMUP:
        raise ValueError(f"need at least {WARMUP} bars of history, got {t}")
    px = close[t - WARMUP : t + 1]
    rets = px[1:] / px[:-1] - 1
    market = rets.mean(axis=1)
    index = np.cumprod(1 + market)
    daily_vol = max(float(market[-21:].std(ddof=1)), 1e-6)
    long_vol = max(float(market.std(ddof=1)), 1e-6)

    def trend(h: int) -> float:
        value = (index[-1] / index[-h - 1] - 1) / (daily_vol * np.sqrt(h))
        return float(np.clip(value, -3.0, 3.0))

    corr = np.nan_to_num(np.corrcoef(rets[-63:].T))
    n = corr.shape[0]
    avg_corr = (corr.sum() - n) / max(n * (n - 1), 1)
    dispersion = max(float((px[-1] / px[-22] - 1).std()), 1e-4)
    return np.array(
        [
            trend(21),
            trend(63),
            trend(252),
            np.log(daily_vol * np.sqrt(252) / 0.15),
            np.log(daily_vol / long_vol),
            (index[-1] / index[-252:].max() - 1) * 5.0,
            (px[-1] > px[-200:].mean(axis=0)).mean() * 2.0 - 1.0,
            avg_corr * 2.0 - 1.0,
            np.log(dispersion / 0.03),
        ]
    )


@dataclass(frozen=True)
class BookConfig:
    rebalance_every: int = 5
    lookback: int = 252
    max_weight: float = 0.35
    risk_aversion: float = 4.0
    information_coefficient: float = 0.10


def sleeve_weights(close: np.ndarray, t: int, config: BookConfig) -> dict[str, np.ndarray]:
    """Compute every sleeve's weights at bar ``t`` from trailing data only."""
    n = close.shape[1]
    cap = max(config.max_weight, 1.0 / n)
    window = close[t - config.lookback : t + 1]
    rets = window[1:] / window[:-1] - 1
    cov = opt.ledoit_wolf(rets) * 252
    vol = np.sqrt(np.diag(cov))

    # Grinold alpha: IC * volatility * cross-sectional momentum score, added to the
    # equilibrium returns implied by an equal-weight prior.
    momentum = close[t - 21] / close[t - 252] - 1
    spread = momentum.std()
    score = np.clip((momentum - momentum.mean()) / spread, -2, 2) if spread > 0 else np.zeros(n)
    equilibrium = config.risk_aversion * cov @ opt.equal_weight(n)
    mu = equilibrium + config.information_coefficient * vol * score

    # Absolute trend: inverse-vol weights, with assets in a downtrend moved to cash.
    inverse_vol = (1.0 / vol) / (1.0 / vol).sum()
    in_trend = (close[t] > close[t - 199 : t + 1].mean(axis=0)) & (momentum > 0)

    return {
        "ew": opt.equal_weight(n),
        "minvar": opt.min_variance(cov, cap),
        "erc": opt.risk_parity(cov, cap),
        "mvo": opt.mean_variance(cov, mu, config.risk_aversion, cap),
        "trend": np.minimum(inverse_vol * in_trend, cap),
        "hrp": opt.hierarchical_risk_parity(cov, cap),
        "cvar": opt.min_cvar(rets, cap=cap),
    }


@dataclass(eq=False)
class SleeveBook:
    """Pre-computed decision data. Index ``k`` counts decision points."""

    decision_bars: np.ndarray  # (K,)
    features: np.ndarray  # (K, F)
    sleeves: dict[str, np.ndarray]  # name -> (K, N)
    actions: tuple[str, ...]
    W: np.ndarray  # (K, A, N) target weights per action
    R: np.ndarray  # (K, N) open-to-open return earned by decision k; NaN if unrealized
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        self.k_of = {int(bar): k for k, bar in enumerate(self.decision_bars)}

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def baseline_action(self) -> int:
        return self.actions.index("ew@100")

    def realized_decisions(self, cut_bar: int) -> int:
        """Number of leading decisions whose full holding period ended by ``cut_bar``.

        Decision ``k`` trades at the open of ``d[k] + 1`` and is held until the open
        of ``d[k + 1] + 1``, so it is usable only when ``d[k + 1] + 1 <= cut_bar``.
        """
        known = int(np.searchsorted(self.decision_bars, cut_bar - 1, side="right"))
        return max(known - 1, 0)


def build_book(
    md: MarketData, config: BookConfig = BookConfig(), include_last_bar: bool = False
) -> SleeveBook:
    """Evaluate features and sleeves at every decision bar."""
    first = max(WARMUP, config.lookback)
    bars = list(range(first, md.n_bars, config.rebalance_every))
    if include_last_bar and bars and bars[-1] != md.n_bars - 1:
        bars.append(md.n_bars - 1)
    if not bars:
        raise ValueError(f"need more than {first} bars of data, got {md.n_bars}")
    decision_bars = np.array(bars)

    features = np.array([regime_features(md.close, int(t)) for t in decision_bars])
    computed = [sleeve_weights(md.close, int(t), config) for t in decision_bars]
    sleeves = {name: np.array([c[name] for c in computed]) for name in computed[0]}

    actions = ["cash"]
    columns = [np.zeros((len(bars), md.n_assets))]
    for name in ACTION_SLEEVES:
        for exposure in EXPOSURES:
            actions.append(f"{name}@{int(exposure * 100)}")
            columns.append(sleeves[name] * exposure)
    W = np.stack(columns, axis=1)

    R = np.full((len(bars), md.n_assets), np.nan)
    for k in range(len(bars) - 1):
        entry, exit_ = bars[k] + 1, bars[k + 1] + 1
        if exit_ < md.n_bars:
            R[k] = md.open[exit_] / md.open[entry] - 1
    return SleeveBook(decision_bars, features, sleeves, tuple(actions), W, R, md.symbols)
