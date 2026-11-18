"""Event types that flow through the backtest engine.

One trading day is two phases, always in this order:

``BarOpen``  -> pending orders fill at the open  -> ``Fill``
``BarClose`` -> portfolio marks, strategy decides -> ``TargetWeights`` -> ``Order``

Orders created at the close of bar ``t`` can therefore only fill at the open of
bar ``t + 1``. That ordering is what rules out same-bar lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BarOpen:
    bar: int


@dataclass(frozen=True, slots=True)
class BarClose:
    bar: int


@dataclass(frozen=True, slots=True, eq=False)
class TargetWeights:
    bar: int
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class Order:
    bar: int
    asset: int
    qty: float  # signed shares: positive buys, negative sells


@dataclass(frozen=True, slots=True)
class Fill:
    bar: int
    asset: int
    qty: float  # signed shares actually executed
    price: float  # all-in price including spread and impact
    commission: float
    slippage: float  # dollars paid versus the raw open price


Event = BarOpen | BarClose | TargetWeights | Order | Fill
