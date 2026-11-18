"""Re-run the walk-forward study across seeds and trading-cost assumptions.

A single training seed can be lucky, so conclusions should rest on the distribution.

    python scripts/robustness.py --end 2026-11-14 --out results
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

from rlfolio.data import load_yahoo
from rlfolio.execution import ExecutionConfig
from rlfolio.study import StudyConfig, run_study

SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
COST_MULTIPLIERS = (1.0, 2.0, 4.0)


def run(job: tuple[str | None, int, float]) -> list[dict]:
    end, seed, multiplier = job
    base = ExecutionConfig()
    execution = replace(
        base,
        half_spread_bps=base.half_spread_bps * multiplier,
        impact_coef=base.impact_coef * multiplier,
        commission_per_share=base.commission_per_share * multiplier,
    )
    study = run_study(load_yahoo(end=end), StudyConfig(seed=seed, execution=execution))
    rows = []
    for name, row in study.summary.iterrows():
        rows.append(
            {
                "seed": seed, "cost_multiplier": multiplier, "strategy": name,
                "cagr": row["cagr"], "sharpe": row["sharpe"],
                "max_drawdown": row["max_drawdown"], "turnover": row["turnover"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="results")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = [(args.end, seed, 1.0) for seed in SEEDS]
    jobs += [(args.end, 7, m) for m in COST_MULTIPLIERS if m != 1.0]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [row for result in pool.map(run, jobs) for row in result]

    frame = pd.DataFrame(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    frame.to_csv(Path(args.out) / "robustness.csv", index=False)
    base = frame[frame.cost_multiplier == 1.0]
    print(base.groupby("strategy")["sharpe"].agg(["mean", "std", "min", "max"]).round(3))
    print(frame[frame.seed == 7].pivot_table(index="strategy", columns="cost_multiplier", values="sharpe").round(3))


if __name__ == "__main__":
    main()
