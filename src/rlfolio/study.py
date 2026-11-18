"""Walk-forward training and out-of-sample comparison against benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import analytics
from .agents import AGENTS, Agent, AllocationMDP, MDPConfig
from .data import MarketData
from .engine import BacktestEngine, BacktestResult
from .execution import ExecutionConfig, RiskLimits
from .sleeves import BookConfig, SleeveBook, build_book


@dataclass(frozen=True)
class Fold:
    train_cut: int  # training may only use prices from bars <= train_cut
    test_start: int
    test_end: int


def make_folds(n_bars: int, initial_train: int, test_bars: int, embargo: int) -> list[Fold]:
    """Expanding-window folds; an embargo separates each training cut from its test."""
    folds = []
    for start in range(initial_train, n_bars, test_bars):
        folds.append(Fold(start - embargo, start, min(start + test_bars, n_bars)))
    if not folds:
        raise ValueError("not enough data for a single walk-forward fold")
    return folds


class SleeveStrategy:
    """Holds one sleeve from the book, rebalancing at every decision bar."""

    def __init__(self, book: SleeveBook, sleeve: str):
        self.book, self.weights = book, book.sleeves[sleeve]

    def on_bar_close(self, bar: int) -> np.ndarray | None:
        k = self.book.k_of.get(bar)
        return None if k is None else self.weights[k]


class FixedMixStrategy:
    """Rebalances to constant weights; with ``once=True``, buys and holds."""

    def __init__(self, book: SleeveBook, weights: np.ndarray, once: bool = False):
        self.book, self.weights, self.once, self.done = book, weights, once, False

    def on_bar_close(self, bar: int) -> np.ndarray | None:
        if bar not in self.book.k_of or (self.once and self.done):
            return None
        self.done = True
        return self.weights


class WalkForwardPolicy:
    """Acts with whichever agent was trained for the fold containing the current bar."""

    def __init__(self, book: SleeveBook, folds: list[Fold], agents: list[Agent]):
        self.book, self.folds, self.agents = book, folds, agents
        self.prev_action = 0  # starts in cash
        self.history: list[tuple[int, str]] = []

    def on_bar_close(self, bar: int) -> np.ndarray | None:
        k = self.book.k_of.get(bar)
        if k is None:
            return None
        for fold, agent in zip(self.folds, self.agents):
            if fold.test_start <= bar < fold.test_end:
                action = agent.act(self.book.features[k], self.prev_action)
                self.prev_action = action
                self.history.append((bar, self.book.actions[action]))
                return self.book.W[k, action]
        return None


def train_walk_forward(
    book: SleeveBook, folds: list[Fold], agent_key: str, mdp_config: MDPConfig, seed: int
) -> list[Agent]:
    """Fit a fresh agent per fold using only decisions realized by the training cut."""
    agents: list[Agent] = []
    for i, fold in enumerate(folds):
        mdp = AllocationMDP(book, book.realized_decisions(fold.train_cut), mdp_config)
        agents.append(AGENTS[agent_key]().fit(mdp, seed + i))
    return agents


@dataclass(frozen=True)
class StudyConfig:
    book: BookConfig = BookConfig()
    mdp: MDPConfig = MDPConfig()
    execution: ExecutionConfig = ExecutionConfig()
    limits: RiskLimits = RiskLimits()
    initial_capital: float = 1_000_000.0
    initial_train_bars: int = 1260
    test_bars: int = 252
    embargo: int = 5
    agents: tuple[str, ...] = ("qlearning", "dqn")
    seed: int = 7


@dataclass(eq=False)
class StudyResult:
    results: dict[str, BacktestResult]
    summary: pd.DataFrame
    comparisons: pd.DataFrame
    action_counts: dict[str, pd.Series] = field(default_factory=dict)
    folds: list[Fold] = field(default_factory=list)


def _benchmarks(md: MarketData, book: SleeveBook) -> dict[str, object]:
    strategies: dict[str, object] = {
        name: SleeveStrategy(book, name)
        for name in ("ew", "minvar", "erc", "hrp", "cvar", "mvo", "trend")
    }
    symbols = list(md.symbols)
    if "SPY" in symbols:
        spy = np.zeros(md.n_assets)
        spy[symbols.index("SPY")] = 1.0
        strategies["spy_buy_hold"] = FixedMixStrategy(book, spy, once=True)
        if "IEF" in symbols:
            mix = np.zeros(md.n_assets)
            mix[symbols.index("SPY")], mix[symbols.index("IEF")] = 0.6, 0.4
            strategies["60_40"] = FixedMixStrategy(book, mix)
    return strategies


def summarize(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Build the metrics table. Deflated Sharpe counts every strategy as a trial."""
    returns = {name: r.returns.to_numpy() for name, r in results.items()}
    trial_sharpes = np.array([x.mean() / x.std(ddof=1) for x in returns.values()])
    rows = {}
    for name, result in results.items():
        rows[name] = {
            **analytics.performance(result.equity),
            "turnover": result.annual_turnover,
            "cost_drag": result.cost_drag,
            "psr": analytics.probabilistic_sharpe(returns[name]),
            "dsr": analytics.deflated_sharpe(returns[name], trial_sharpes),
        }
    return pd.DataFrame(rows).T.sort_values("sharpe", ascending=False)


def run_study(md: MarketData, config: StudyConfig = StudyConfig()) -> StudyResult:
    book = build_book(md, config.book)
    folds = make_folds(md.n_bars, config.initial_train_bars, config.test_bars, config.embargo)
    start = folds[0].test_start

    def backtest(name: str, strategy: object) -> BacktestResult:
        # Fixed-mix benchmarks are concentrated by definition, so the cap must not alter them.
        uncapped = isinstance(strategy, FixedMixStrategy)
        limits = replace(config.limits, max_weight=1.0) if uncapped else config.limits
        engine = BacktestEngine(
            md, strategy, config.initial_capital, config.execution, limits,
            start_bar=start, name=name,
        )
        return engine.run()

    results: dict[str, BacktestResult] = {}
    action_counts: dict[str, pd.Series] = {}
    for key in config.agents:
        agents = train_walk_forward(book, folds, key, config.mdp, config.seed)
        policy = WalkForwardPolicy(book, folds, agents)
        results[key] = backtest(key, policy)
        action_counts[key] = pd.Series([a for _, a in policy.history]).value_counts()
    for name, strategy in _benchmarks(md, book).items():
        results[name] = backtest(name, strategy)

    comparisons = []
    references = [n for n in ("ew", "erc", "60_40", "spy_buy_hold") if n in results]
    for key in config.agents:
        for reference in references:
            stats = analytics.bootstrap_sharpe_difference(
                results[key].returns.to_numpy(), results[reference].returns.to_numpy(), seed=config.seed
            )
            comparisons.append({"agent": key, "versus": reference, **stats})
    return StudyResult(results, summarize(results), pd.DataFrame(comparisons), action_counts, folds)
