"""Simulated broker, portfolio accounting and pre-trade risk limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import MarketData
from .events import Fill, Order


@dataclass(frozen=True)
class ExecutionConfig:
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    max_commission_pct: float = 0.005
    half_spread_bps: float = 1.0
    impact_coef: float = 0.1  # square-root impact: coef * daily_vol * sqrt(qty / ADV)
    max_participation: float = 0.05  # maximum fraction of trailing ADV per day
    window: int = 20  # bars used for trailing ADV and volatility


class SimulatedBroker:
    """Fills market-on-open orders using only information available before the open."""

    def __init__(self, md: MarketData, config: ExecutionConfig = ExecutionConfig()):
        self.md = md
        self.config = config
        self.pending: list[Order] = []

    def submit(self, order: Order) -> None:
        self.pending.append(order)

    def _liquidity(self, bar: int, asset: int) -> tuple[float, float]:
        """Trailing ADV (shares) and daily volatility, excluding the current bar."""
        start = max(0, bar - self.config.window)
        volumes = self.md.volume[start:bar, asset]
        adv = float(volumes.mean()) if volumes.size else float("inf")
        closes = self.md.close[start:bar, asset]
        vol = float(np.std(closes[1:] / closes[:-1] - 1, ddof=1)) if closes.size > 2 else 0.01
        return adv, vol

    def _commission(self, qty: float, price: float) -> float:
        c = self.config
        shares = abs(qty)
        fee = max(shares * c.commission_per_share, c.min_commission)
        return min(fee, c.max_commission_pct * shares * price)

    def execute(self, bar: int, cash: float, positions: np.ndarray) -> list[Fill]:
        """Fill all pending orders at this bar's open. Unfilled quantity is cancelled."""
        orders, self.pending = self.pending, []
        fills: list[Fill] = []
        # Sells go first so their proceeds can fund the buys.
        for order in sorted(orders, key=lambda o: o.qty):
            adv, vol = self._liquidity(bar, order.asset)
            cap = np.floor(self.config.max_participation * adv)
            qty = np.sign(order.qty) * min(abs(order.qty), cap)
            if qty < 0:
                qty = -min(-qty, positions[order.asset])  # long-only: never oversell
            if qty == 0:
                continue

            open_px = float(self.md.open[bar, order.asset])
            cost_frac = self.config.half_spread_bps / 1e4
            if np.isfinite(adv) and adv > 0:
                cost_frac += self.config.impact_coef * vol * np.sqrt(abs(qty) / adv)
            price = open_px * (1 + np.sign(qty) * cost_frac)

            if qty > 0:  # scale buys down to the cash actually available
                unit_cost = price * (1 + self.config.max_commission_pct)
                qty = min(qty, np.floor(max(cash, 0.0) / unit_cost))
                if qty <= 0:
                    continue

            commission = self._commission(qty, price)
            cash -= qty * price + commission
            fills.append(
                Fill(bar, order.asset, float(qty), price, commission, abs(qty) * abs(price - open_px))
            )
        return fills


class Portfolio:
    """Tracks cash, share positions, NAV and cumulative trading costs."""

    def __init__(self, n_assets: int, initial_cash: float):
        self.cash = float(initial_cash)
        self.positions = np.zeros(n_assets)
        self.last_prices = np.full(n_assets, np.nan)
        self.nav = float(initial_cash)
        self.peak_nav = float(initial_cash)
        self.commissions = 0.0
        self.slippage = 0.0
        self.traded_notional = 0.0

    def apply_fill(self, fill: Fill) -> None:
        self.cash -= fill.qty * fill.price + fill.commission
        self.positions[fill.asset] += fill.qty
        self.commissions += fill.commission
        self.slippage += fill.slippage
        self.traded_notional += abs(fill.qty) * fill.price

    def mark(self, prices: np.ndarray) -> float:
        self.last_prices = prices
        self.nav = self.cash + float(self.positions @ prices)
        self.peak_nav = max(self.peak_nav, self.nav)
        return self.nav

    @property
    def weights(self) -> np.ndarray:
        if self.nav <= 0 or np.isnan(self.last_prices).any():
            return np.zeros_like(self.positions)
        return self.positions * self.last_prices / self.nav

    @property
    def drawdown(self) -> float:
        return 1.0 - self.nav / self.peak_nav


@dataclass(frozen=True)
class RiskLimits:
    max_weight: float = 0.40
    max_gross: float = 1.0
    cash_buffer: float = 0.005  # held back so overnight gaps do not cause failed buys
    no_trade_band: float = 0.005  # skip trades smaller than this fraction of NAV
    drawdown_limit: float | None = None  # e.g. 0.20 enables the circuit breaker
    derisk_gross: float = 0.25  # gross exposure while the circuit breaker is active


class RiskManager:
    """Applies position, leverage and drawdown limits to strategy targets."""

    def __init__(self, limits: RiskLimits = RiskLimits()):
        self.limits = limits
        self.tripped = False

    def apply(self, target: np.ndarray, drawdown: float) -> np.ndarray:
        limits = self.limits
        weights = np.clip(target, 0.0, limits.max_weight)
        if limits.drawdown_limit is not None:
            if drawdown >= limits.drawdown_limit:
                self.tripped = True
            elif drawdown <= limits.drawdown_limit / 2:
                self.tripped = False
        max_gross = limits.derisk_gross if self.tripped else limits.max_gross
        max_gross = min(max_gross, 1.0 - limits.cash_buffer)
        gross = weights.sum()
        return weights * max_gross / gross if gross > max_gross else weights
