"""Command-line interface: ``rlfolio backtest`` and ``rlfolio signal``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_UNIVERSE, MarketData, load_csv, load_yahoo, synthetic_market
from .live import AlpacaPaperBroker, latest_signal, order_ticket
from .study import StudyConfig, StudyResult, run_study


def _load(args: argparse.Namespace) -> MarketData:
    if args.source == "synthetic":
        return synthetic_market(args.synthetic_bars, seed=args.seed)
    if args.source == "csv":
        return load_csv(args.csv_dir, args.symbols)
    return load_yahoo(args.symbols or DEFAULT_UNIVERSE, args.start, args.end, args.cache_dir)


def _markdown(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join(" --- " for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.to_numpy()]
    return "\n".join([header, rule, *rows])


def _summary_table(study: StudyResult) -> pd.DataFrame:
    s = study.summary
    return pd.DataFrame(
        {
            "strategy": s.index,
            "CAGR": s["cagr"].map("{:.2%}".format),
            "Vol": s["volatility"].map("{:.2%}".format),
            "Sharpe": s["sharpe"].map("{:.2f}".format),
            "Sortino": s["sortino"].map("{:.2f}".format),
            "Max DD": s["max_drawdown"].map("{:.2%}".format),
            "Calmar": s["calmar"].map("{:.2f}".format),
            "Turnover/yr": s["turnover"].map("{:.2f}x".format),
            "Costs/yr": s["cost_drag"].map("{:.2%}".format),
            "DSR": s["dsr"].map("{:.2f}".format),
        }
    )


def _write_report(md: MarketData, study: StudyResult, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    equity = pd.DataFrame({name: r.equity for name, r in study.results.items()})
    equity.to_csv(out / "equity_curves.csv")
    study.summary.to_csv(out / "summary.csv")
    study.comparisons.to_csv(out / "comparisons.csv", index=False)

    first = equity.index[0].date()
    last = equity.index[-1].date()
    comparisons = study.comparisons.copy()
    for column in ("diff", "ci_low", "ci_high", "p_not_better"):
        comparisons[column] = comparisons[column].map("{:.2f}".format)
    lines = [
        "# Walk-forward results",
        "",
        f"Out-of-sample period {first} to {last}, {len(study.folds)} folds, "
        f"universe: {', '.join(md.symbols)}.",
        "",
        _markdown(_summary_table(study)),
        "",
        "## Agent Sharpe versus benchmarks (stationary bootstrap, 95% CI)",
        "",
        _markdown(comparisons),
        "",
        "## Actions chosen out of sample",
        "",
    ]
    for key, counts in study.action_counts.items():
        share = (counts / counts.sum()).map("{:.0%}".format)
        lines.append(f"- **{key}**: " + ", ".join(f"{a} {v}" for a, v in share.items()))
    (out / "report.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    agents = list(study.action_counts)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 1])
    for name in equity.columns:
        highlight = name in agents
        curve = equity[name] / equity[name].iloc[0]
        axes[0].plot(curve, lw=1.8 if highlight else 0.8, alpha=1.0 if highlight else 0.55, label=name)
        if highlight or name == "ew":
            axes[1].plot(curve / curve.cummax() - 1, lw=1.0, label=name)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Growth of $1 (log scale)")
    axes[0].set_title("Out-of-sample equity, net of trading costs")
    axes[0].legend(ncol=4, fontsize=8, frameon=False)
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(ncol=3, fontsize=8, frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=130)
    plt.close(fig)


def cmd_backtest(args: argparse.Namespace) -> int:
    md = _load(args)
    print(f"Loaded {md.n_bars} bars x {md.n_assets} assets: {md.dates[0].date()} to {md.dates[-1].date()}")
    study = run_study(md, StudyConfig(agents=tuple(args.agents), seed=args.seed))
    _write_report(md, study, Path(args.out))
    print(_markdown(_summary_table(study)))
    print(f"\nReport written to {args.out}/report.md")
    return 0


def cmd_signal(args: argparse.Namespace) -> int:
    md = _load(args)
    broker = AlpacaPaperBroker.from_env() if args.alpaca_paper else None
    if broker:
        positions, capital = broker.positions(), broker.equity()
    else:
        positions = json.loads(Path(args.positions).read_text()) if args.positions else {}
        capital = args.capital

    prices = pd.Series(md.close[-1], index=list(md.symbols))
    held = np.array([positions.get(s, 0.0) * prices[s] / capital for s in md.symbols])
    signal = latest_signal(md, args.agent, held, seed=args.seed)
    ticket = order_ticket(signal.weights * (1 - args.cash_buffer), prices, capital, positions)

    print(f"As of {signal.as_of.date()}: {signal.previous_action} -> {signal.action}")
    print(ticket.to_string(index=False) if len(ticket) else "No trades required.")
    ticket.to_csv(args.out, index=False)
    print(f"Order ticket written to {args.out}")

    if args.submit:
        if broker is None:
            raise SystemExit("--submit requires --alpaca-paper")
        placed = broker.submit_ticket(ticket, signal.as_of.strftime("%Y%m%d"))
        print(f"Submitted {len(placed)} orders to Alpaca paper trading.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlfolio", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", choices=["yahoo", "csv", "synthetic"], default="yahoo")
    common.add_argument("--symbols", nargs="+", help="defaults to a 12-ETF multi-asset universe")
    common.add_argument("--start", default="2007-05-01")
    common.add_argument("--end", default=None)
    common.add_argument("--csv-dir", default="data/csv")
    common.add_argument("--cache-dir", default="data/cache")
    common.add_argument("--synthetic-bars", type=int, default=3000)
    common.add_argument("--seed", type=int, default=7)

    backtest = sub.add_parser("backtest", parents=[common], help="run the walk-forward study")
    backtest.add_argument("--agents", nargs="+", choices=["qlearning", "dqn"], default=["qlearning", "dqn"])
    backtest.add_argument("--out", default="results")
    backtest.set_defaults(func=cmd_backtest)

    signal = sub.add_parser("signal", parents=[common], help="generate today's target orders")
    signal.add_argument("--agent", choices=["qlearning", "dqn"], default="qlearning")
    signal.add_argument("--capital", type=float, default=100_000.0)
    signal.add_argument("--positions", help='JSON file of current shares, e.g. {"SPY": 10}')
    signal.add_argument("--cash-buffer", type=float, default=0.01)
    signal.add_argument("--out", default="orders.csv")
    signal.add_argument("--alpaca-paper", action="store_true", help="read account state from Alpaca paper")
    signal.add_argument("--submit", action="store_true", help="send the orders to Alpaca paper")
    signal.set_defaults(func=cmd_signal)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
