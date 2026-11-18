"""Turn the latest model decision into orders, with optional Alpaca paper execution.

Only Alpaca's paper-trading endpoint is supported. There is deliberately no setting
that routes orders to a live account.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .agents import AGENTS, AllocationMDP, MDPConfig
from .data import MarketData
from .sleeves import BookConfig, build_book

PAPER_URL = "https://paper-api.alpaca.markets"

Transport = Callable[[str, str, Mapping[str, str], bytes | None], tuple[int, bytes]]


@dataclass(frozen=True)
class Signal:
    as_of: pd.Timestamp
    action: str
    previous_action: str
    weights: pd.Series


def latest_signal(
    md: MarketData,
    agent_key: str = "qlearning",
    current_weights: np.ndarray | None = None,
    book_config: BookConfig = BookConfig(),
    mdp_config: MDPConfig = MDPConfig(),
    seed: int = 7,
) -> Signal:
    """Train on all realized history and decide at the most recent close."""
    book = build_book(md, book_config, include_last_bar=True)
    mdp = AllocationMDP(book, book.realized_decisions(md.n_bars - 1), mdp_config)
    agent = AGENTS[agent_key]().fit(mdp, seed)
    held = np.zeros(md.n_assets) if current_weights is None else current_weights
    prev = int(np.abs(book.W[-1] - held).sum(axis=1).argmin())  # nearest action to holdings
    action = agent.act(book.features[-1], prev)
    return Signal(
        md.dates[-1], book.actions[action], book.actions[prev],
        pd.Series(book.W[-1, action], index=list(md.symbols), name="target_weight"),
    )


def order_ticket(
    target: pd.Series,
    prices: pd.Series,
    capital: float,
    positions: Mapping[str, float] | None = None,
    min_notional: float = 100.0,
) -> pd.DataFrame:
    """Whole-share orders that move ``positions`` to ``target`` at ``prices``."""
    positions = positions or {}
    rows = []
    for symbol, weight in target.items():
        price = float(prices[symbol])
        held = float(positions.get(symbol, 0.0))
        desired = np.floor(weight * capital / price)
        delta = desired - held
        if abs(delta) * price < min_notional:
            continue
        rows.append(
            {
                "symbol": symbol,
                "side": "buy" if delta > 0 else "sell",
                "qty": int(abs(delta)),
                "reference_price": round(price, 4),
                "notional": round(abs(delta) * price, 2),
                "target_weight": round(float(weight), 4),
            }
        )
    columns = ["symbol", "side", "qty", "reference_price", "notional", "target_weight"]
    ticket = pd.DataFrame(rows, columns=columns)
    return ticket.sort_values("side", ascending=False, ignore_index=True)  # sells first


def _urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class AlpacaPaperBroker:
    """Minimal client for Alpaca's paper-trading REST API."""

    def __init__(self, key: str, secret: str, transport: Transport = _urllib_transport):
        self.headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Content-Type": "application/json",
        }
        self.transport = transport

    @classmethod
    def from_env(cls) -> "AlpacaPaperBroker":
        try:
            return cls(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"])
        except KeyError as missing:
            raise RuntimeError(f"environment variable {missing} is not set") from None

    def _request(self, method: str, path: str, payload: dict | None = None):
        body = json.dumps(payload).encode() if payload is not None else None
        status, raw = self.transport(method, PAPER_URL + path, self.headers, body)
        if status >= 300:
            raise RuntimeError(f"Alpaca {method} {path} failed ({status}): {raw[:300]!r}")
        return json.loads(raw)

    def equity(self) -> float:
        return float(self._request("GET", "/v2/account")["equity"])

    def positions(self) -> dict[str, float]:
        return {p["symbol"]: float(p["qty"]) for p in self._request("GET", "/v2/positions")}

    def submit_ticket(self, ticket: pd.DataFrame, batch_id: str) -> list[dict]:
        """Submit market orders. ``batch_id`` makes a repeated submission idempotent."""
        return [
            self._request(
                "POST",
                "/v2/orders",
                {
                    "symbol": row.symbol,
                    "qty": str(row.qty),
                    "side": row.side,
                    "type": "market",
                    "time_in_force": "day",
                    "client_order_id": f"rlfolio-{batch_id}-{row.symbol}-{row.side}",
                },
            )
            for row in ticket.itertuples()
        ]
