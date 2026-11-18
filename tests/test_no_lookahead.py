"""Leakage tests: nothing computed at time t may depend on data after t."""

import numpy as np
import pytest

from rlfolio.agents import AllocationMDP, DoubleQLearningAgent, DQNConfig, DuelingDQNAgent
from rlfolio.data import MarketData, synthetic_market
from rlfolio.engine import BacktestEngine
from rlfolio.execution import ExecutionConfig, RiskLimits
from rlfolio.sleeves import WARMUP, build_book, regime_features, sleeve_weights, BookConfig


@pytest.fixture(scope="module")
def md():
    return synthetic_market(700, 6, seed=3)


def scrambled_after(md: MarketData, cut: int) -> MarketData:
    """Same history through bar ``cut``, wildly different prices afterwards."""
    rng = np.random.default_rng(99)
    factor = np.ones((md.n_bars, md.n_assets))
    factor[cut + 1 :] = np.exp(np.cumsum(rng.normal(0, 0.05, (md.n_bars - cut - 1, md.n_assets)), axis=0))
    return MarketData(
        md.dates, md.symbols, md.open * factor, md.high * factor, md.low * factor,
        md.close * factor, md.volume,
    )


def test_features_and_sleeves_ignore_future_bars(md):
    t = WARMUP + 100
    cut = md.close[: t + 1]
    np.testing.assert_array_equal(regime_features(md.close, t), regime_features(cut, t))
    full, truncated = sleeve_weights(md.close, t, BookConfig()), sleeve_weights(cut, t, BookConfig())
    for name in full:
        np.testing.assert_array_equal(full[name], truncated[name])


def test_book_decisions_unchanged_by_future_prices(md):
    cut = 500
    a, b = build_book(md), build_book(scrambled_after(md, cut))
    known = a.decision_bars <= cut
    np.testing.assert_array_equal(a.features[known], b.features[known])
    np.testing.assert_array_equal(a.W[known], b.W[known])


@pytest.mark.parametrize(
    "make_agent",
    [lambda: DoubleQLearningAgent(), lambda: DuelingDQNAgent(DQNConfig(total_steps=600, ensemble=1))],
    ids=["qlearning", "dqn"],
)
def test_training_is_invariant_to_prices_after_the_cut(md, make_agent):
    cut = 550
    book_a, book_b = build_book(md), build_book(scrambled_after(md, cut))
    k_end = book_a.realized_decisions(cut)
    agent_a = make_agent().fit(AllocationMDP(book_a, k_end), seed=1)
    agent_b = make_agent().fit(AllocationMDP(book_b, k_end), seed=1)
    probe = book_a.features[:k_end]
    actions_a = [agent_a.act(x, 0) for x in probe]
    actions_b = [agent_b.act(x, 0) for x in probe]
    assert actions_a == actions_b


def test_realized_decisions_never_use_returns_past_the_cut(md):
    book = build_book(md)
    for cut in (400, 451, 600):
        k_end = book.realized_decisions(cut)
        assert book.decision_bars[k_end] + 1 <= cut  # exit of the last usable decision


def test_orders_fill_at_the_next_open_only(md):
    class BuyOnce:
        def __init__(self):
            self.sent = False

        def on_bar_close(self, bar):
            if self.sent:
                return None
            self.sent = True
            return np.full(md.n_assets, 1 / md.n_assets)

    engine = BacktestEngine(
        md, BuyOnce(), execution=ExecutionConfig(), limits=RiskLimits(), start_bar=100, end_bar=110
    )
    result = engine.run()
    assert len(result.fills) > 0
    assert (result.fills["date"] == md.dates[101]).all()
