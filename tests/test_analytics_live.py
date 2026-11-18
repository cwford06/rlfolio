import json

import numpy as np
import pandas as pd
import pytest

from rlfolio import analytics
from rlfolio.data import MarketData, load_csv, synthetic_market
from rlfolio.live import PAPER_URL, AlpacaPaperBroker, latest_signal, order_ticket
from rlfolio.study import StudyConfig, make_folds, run_study
from rlfolio.agents import DQNConfig


def test_max_drawdown_known_value():
    assert analytics.max_drawdown(np.array([100, 120, 60, 90, 130])) == pytest.approx(0.5)


def test_sharpe_known_value():
    returns = np.array([0.01, -0.01] * 50) + 0.001
    expected = 0.001 / returns.std(ddof=1) * np.sqrt(252)
    assert analytics.sharpe(returns) == pytest.approx(expected)


def test_deflated_sharpe_is_stricter_with_more_trials():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0006, 0.01, 2000)
    psr = analytics.probabilistic_sharpe(returns)
    dsr = analytics.deflated_sharpe(returns, rng.normal(0.02, 0.03, 50))
    assert dsr < psr


def test_bootstrap_detects_clear_winner_and_ties():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, 2000)
    strong = analytics.bootstrap_sharpe_difference(noise + 0.002, noise, n_boot=300)
    assert strong["p_not_better"] < 0.01 and strong["ci_low"] > 0
    tie = analytics.bootstrap_sharpe_difference(noise, noise, n_boot=100)
    assert tie["diff"] == 0.0


def test_folds_are_embargoed_and_contiguous():
    folds = make_folds(2000, 1000, 250, embargo=5)
    assert all(f.train_cut == f.test_start - 5 for f in folds)
    assert all(a.test_end == b.test_start for a, b in zip(folds, folds[1:]))
    assert folds[-1].test_end == 2000


def test_csv_loader_inner_joins_dates(tmp_path):
    rows = {"date": ["2020-01-02", "2020-01-03"], "open": [1, 2], "high": [1, 2],
            "low": [1, 2], "close": [1, 2], "volume": [5, 5]}
    pd.DataFrame(rows).to_csv(tmp_path / "AAA.csv", index=False)
    pd.DataFrame(rows).iloc[1:].to_csv(tmp_path / "BBB.csv", index=False)
    md = load_csv(tmp_path)
    assert md.n_bars == 1 and md.symbols == ("AAA", "BBB")


def test_order_ticket_nets_positions_and_sells_first():
    target = pd.Series({"AAA": 0.5, "BBB": 0.0})
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0})
    ticket = order_ticket(target, prices, 10_000, {"AAA": 10, "BBB": 40})
    assert ticket.iloc[0].to_dict()["side"] == "sell"
    assert dict(zip(ticket.symbol, ticket.qty)) == {"BBB": 40, "AAA": 40}


def test_alpaca_client_only_calls_paper_endpoint_and_is_idempotent():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, json.loads(body) if body else None))
        return 200, b'{"id": "1"}'

    broker = AlpacaPaperBroker("key", "secret", transport)
    ticket = order_ticket(pd.Series({"AAA": 1.0}), pd.Series({"AAA": 10.0}), 1000)
    broker.submit_ticket(ticket, "20240102")
    method, url, payload = calls[0]
    assert method == "POST" and url == f"{PAPER_URL}/v2/orders"
    assert payload["client_order_id"] == "rlfolio-20240102-AAA-buy"
    assert payload["qty"] == "100"


def test_alpaca_errors_are_raised():
    broker = AlpacaPaperBroker("k", "s", lambda *_: (403, b"forbidden"))
    with pytest.raises(RuntimeError, match="403"):
        broker.equity()


def test_latest_signal_produces_valid_weights():
    md = synthetic_market(700, 5, seed=2)
    signal = latest_signal(md, "qlearning")
    assert signal.as_of == md.dates[-1]
    assert (signal.weights >= 0).all() and signal.weights.sum() <= 1 + 1e-9


def test_full_study_runs_end_to_end():
    md = synthetic_market(1100, 5, seed=4)
    config = StudyConfig(initial_train_bars=700, test_bars=200, agents=("qlearning",))
    study = run_study(md, config)
    assert {"qlearning", "ew", "erc", "minvar"} <= set(study.summary.index)
    lengths = {len(r.equity) for r in study.results.values()}
    assert lengths == {md.n_bars - 700}
    assert len(study.comparisons) == 2


def test_fixed_mix_benchmarks_are_not_clipped_by_the_position_cap():
    base = synthetic_market(1000, 5, seed=5)
    md = MarketData(
        base.dates, ("SPY", "IEF", "AAA", "BBB", "CCC"),
        base.open, base.high, base.low, base.close, base.volume,
    )
    study = run_study(md, StudyConfig(initial_train_bars=700, test_bars=300, agents=()))
    assert study.results["spy_buy_hold"].weights["SPY"].iloc[10] > 0.95
    assert study.results["60_40"].weights["SPY"].iloc[10] == pytest.approx(0.6, abs=0.03)
