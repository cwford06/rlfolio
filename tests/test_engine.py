import numpy as np
import pandas as pd
import pytest

from rlfolio.data import MarketData, synthetic_market
from rlfolio.engine import BacktestEngine
from rlfolio.events import Order
from rlfolio.execution import ExecutionConfig, Portfolio, RiskLimits, RiskManager, SimulatedBroker

FREE = ExecutionConfig(
    commission_per_share=0, min_commission=0, max_commission_pct=0,
    half_spread_bps=0, impact_coef=0, max_participation=1.0,
)
OPEN_LIMITS = RiskLimits(max_weight=1.0, cash_buffer=0.0, no_trade_band=0.0)


class Once:
    def __init__(self, weights):
        self.weights, self.sent = weights, False

    def on_bar_close(self, bar):
        if self.sent:
            return None
        self.sent = True
        return self.weights


class EveryBar:
    def __init__(self, n, seed=0):
        self.n, self.rng = n, np.random.default_rng(seed)

    def on_bar_close(self, bar):
        return self.rng.dirichlet(np.ones(self.n))


@pytest.fixture(scope="module")
def md():
    return synthetic_market(400, 4, seed=1)


def test_frictionless_buy_and_hold_matches_analytic_value(md):
    weights = np.array([1.0, 0, 0, 0])
    result = BacktestEngine(
        md, Once(weights), 1_000_000, FREE, OPEN_LIMITS, start_bar=50
    ).run()
    # Sized at the close, then limited by cash if the next open gaps higher.
    shares = min(np.floor(1_000_000 / md.close[50, 0]), np.floor(1_000_000 / md.open[51, 0]))
    cash = 1_000_000 - shares * md.open[51, 0]
    expected = cash + shares * md.close[-1, 0]
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_accounting_identity_holds_with_costs(md):
    engine = BacktestEngine(md, EveryBar(md.n_assets), 500_000, start_bar=30)
    result = engine.run()
    pf = engine.portfolio
    assert pf.nav == pytest.approx(pf.cash + pf.positions @ md.close[-1])
    flows = (result.fills["qty"] * result.fills["price"]).sum() + result.fills["commission"].sum()
    assert pf.cash == pytest.approx(500_000 - flows)
    assert pf.cash >= 0 and (pf.positions >= 0).all()


def test_costs_reduce_final_nav(md):
    free = BacktestEngine(md, EveryBar(4), 1e6, FREE, start_bar=30).run()
    costly = BacktestEngine(md, EveryBar(4), 1e6, ExecutionConfig(half_spread_bps=10), start_bar=30).run()
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert costly.commissions + costly.slippage > 0


def test_participation_cap_limits_fill_size():
    dates = pd.bdate_range("2020-01-01", periods=30)
    prices = np.full((30, 1), 100.0)
    md = MarketData(dates, ("X",), prices, prices, prices, prices, np.full((30, 1), 1000.0))
    broker = SimulatedBroker(md, ExecutionConfig(max_participation=0.1))
    broker.submit(Order(24, 0, 500.0))
    fills = broker.execute(25, 1e9, np.zeros(1))
    assert fills[0].qty == 100


def test_buys_are_scaled_to_available_cash():
    dates = pd.bdate_range("2020-01-01", periods=30)
    prices = np.full((30, 1), 100.0)
    md = MarketData(dates, ("X",), prices, prices, prices, prices, np.full((30, 1), 1e7))
    broker = SimulatedBroker(md)
    broker.submit(Order(24, 0, 1000.0))
    fills = broker.execute(25, 5_000.0, np.zeros(1))
    assert 0 < fills[0].qty < 50


def test_sells_cannot_exceed_holdings():
    dates = pd.bdate_range("2020-01-01", periods=30)
    prices = np.full((30, 1), 100.0)
    md = MarketData(dates, ("X",), prices, prices, prices, prices, np.full((30, 1), 1e7))
    broker = SimulatedBroker(md)
    broker.submit(Order(24, 0, -50.0))
    assert broker.execute(25, 0.0, np.array([20.0]))[0].qty == -20


def test_risk_manager_caps_weights_and_trips_on_drawdown():
    manager = RiskManager(RiskLimits(max_weight=0.5, drawdown_limit=0.2, derisk_gross=0.25))
    assert manager.apply(np.array([0.9, 0.1]), 0.0).max() == pytest.approx(0.5)
    assert manager.apply(np.array([0.5, 0.5]), 0.25).sum() == pytest.approx(0.25)
    assert manager.apply(np.array([0.5, 0.5]), 0.15).sum() == pytest.approx(0.25)  # still tripped
    assert manager.apply(np.array([0.5, 0.5]), 0.05).sum() == pytest.approx(0.995)


def test_portfolio_marks_nav():
    pf = Portfolio(2, 1000.0)
    pf.mark(np.array([10.0, 20.0]))
    assert pf.nav == 1000.0 and pf.drawdown == 0.0
