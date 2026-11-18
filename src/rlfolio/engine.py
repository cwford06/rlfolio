"""The event-driven backtest loop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .data import MarketData
from .events import BarClose, BarOpen, Event, Fill, Order, TargetWeights
from .execution import (
    ExecutionConfig,
    Portfolio,
    RiskLimits,
    RiskManager,
    SimulatedBroker,
)


class Strategy(Protocol):
    def on_bar_close(self, bar: int) -> np.ndarray | None:
        """Return target weights after the close of ``bar``, or None to hold."""


@dataclass(eq=False)
class BacktestResult:
    name: str
    equity: pd.Series
    weights: pd.DataFrame
    fills: pd.DataFrame
    commissions: float
    slippage: float
    traded_notional: float

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()

    @property
    def annual_turnover(self) -> float:
        """One-way turnover per year, as a multiple of average NAV."""
        years = max(len(self.equity) / 252, 1e-9)
        return self.traded_notional / 2 / float(self.equity.mean()) / years

    @property
    def cost_drag(self) -> float:
        """Annual trading costs as a fraction of average NAV."""
        years = max(len(self.equity) / 252, 1e-9)
        return (self.commissions + self.slippage) / float(self.equity.mean()) / years


class BacktestEngine:
    """Runs a strategy over ``[start_bar, end_bar)`` by dispatching events."""

    def __init__(
        self,
        md: MarketData,
        strategy: Strategy,
        initial_capital: float = 1_000_000.0,
        execution: ExecutionConfig = ExecutionConfig(),
        limits: RiskLimits = RiskLimits(),
        start_bar: int = 0,
        end_bar: int | None = None,
        name: str = "strategy",
    ):
        self.md = md
        self.strategy = strategy
        self.broker = SimulatedBroker(md, execution)
        self.portfolio = Portfolio(md.n_assets, initial_capital)
        self.risk = RiskManager(limits)
        self.start_bar = start_bar
        self.end_bar = md.n_bars if end_bar is None else end_bar
        self.name = name
        self.queue: deque[Event] = deque()
        self.fills: list[Fill] = []

    def _drain(self) -> None:
        while self.queue:
            self._dispatch(self.queue.popleft())

    def _dispatch(self, event: Event) -> None:
        match event:
            case BarOpen(bar=bar):
                self.queue.extend(
                    self.broker.execute(bar, self.portfolio.cash, self.portfolio.positions)
                )
            case Fill():
                self.portfolio.apply_fill(event)
                self.fills.append(event)
            case BarClose(bar=bar):
                self.portfolio.mark(self.md.close[bar])
                target = self.strategy.on_bar_close(bar)
                if target is not None:
                    self.queue.append(TargetWeights(bar, target))
            case TargetWeights():
                self._rebalance(event)
            case Order():
                self.broker.submit(event)

    def _rebalance(self, event: TargetWeights) -> None:
        """Convert target weights into share orders at the latest close."""
        target = self.risk.apply(event.weights, self.portfolio.drawdown)
        current = self.portfolio.weights
        prices = self.md.close[event.bar]
        band = self.risk.limits.no_trade_band
        for asset, weight in enumerate(target):
            if weight > 0 and abs(weight - current[asset]) < band:
                continue
            shares = np.floor(weight * self.portfolio.nav / prices[asset])
            delta = shares - self.portfolio.positions[asset]
            if delta != 0:
                self.queue.append(Order(event.bar, asset, float(delta)))

    def run(self) -> BacktestResult:
        navs, weights = [], []
        for bar in range(self.start_bar, self.end_bar):
            self.queue.append(BarOpen(bar))
            self._drain()
            self.queue.append(BarClose(bar))
            self._drain()
            navs.append(self.portfolio.nav)
            weights.append(self.portfolio.weights.copy())

        index = self.md.dates[self.start_bar : self.end_bar]
        fills = pd.DataFrame(
            [
                {
                    "date": self.md.dates[f.bar],
                    "symbol": self.md.symbols[f.asset],
                    "qty": f.qty,
                    "price": f.price,
                    "commission": f.commission,
                    "slippage": f.slippage,
                }
                for f in self.fills
            ],
            columns=["date", "symbol", "qty", "price", "commission", "slippage"],
        )
        return BacktestResult(
            name=self.name,
            equity=pd.Series(navs, index=index, name=self.name),
            weights=pd.DataFrame(weights, index=index, columns=list(self.md.symbols)),
            fills=fills,
            commissions=self.portfolio.commissions,
            slippage=self.portfolio.slippage,
            traded_notional=self.portfolio.traded_notional,
        )
