# Walk-forward results

Out-of-sample period 2012-04-30 to 2026-11-13, 15 folds, universe: SPY, QQQ, IWM, EFA, EEM, VNQ, TLT, IEF, LQD, HYG, GLD, DBC.

| strategy | CAGR | Vol | Sharpe | Sortino | Max DD | Calmar | Turnover/yr | Costs/yr | DSR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60_40 | 9.18% | 9.77% | 0.95 | 1.35 | 21.11% | 0.43 | 0.16x | 0.01% | 1.00 |
| mvo | 11.39% | 12.45% | 0.93 | 1.30 | 20.02% | 0.57 | 5.27x | 0.18% | 1.00 |
| qlearning | 9.20% | 10.15% | 0.92 | 1.32 | 20.64% | 0.45 | 8.85x | 0.32% | 1.00 |
| spy_buy_hold | 14.28% | 16.55% | 0.89 | 1.26 | 33.60% | 0.42 | 0.01x | 0.00% | 0.99 |
| trend | 3.69% | 4.61% | 0.81 | 1.09 | 10.00% | 0.37 | 1.76x | 0.07% | 0.98 |
| dqn | 7.25% | 9.34% | 0.80 | 1.15 | 14.54% | 0.50 | 9.56x | 0.36% | 0.99 |
| ew | 7.22% | 9.99% | 0.75 | 1.05 | 21.67% | 0.33 | 0.12x | 0.01% | 0.98 |
| erc | 5.05% | 7.21% | 0.72 | 1.01 | 20.28% | 0.25 | 0.23x | 0.01% | 0.97 |
| minvar | 3.12% | 5.35% | 0.60 | 0.84 | 17.08% | 0.18 | 0.51x | 0.02% | 0.92 |
| hrp | 3.74% | 6.46% | 0.60 | 0.84 | 19.63% | 0.19 | 1.29x | 0.05% | 0.92 |
| cvar | 3.13% | 5.53% | 0.58 | 0.83 | 17.02% | 0.18 | 1.11x | 0.05% | 0.91 |

## Agent Sharpe versus benchmarks (stationary bootstrap, 95% CI)

| agent | versus | diff | ci_low | ci_high | p_not_better |
| --- | --- | --- | --- | --- | --- |
| qlearning | ew | 0.17 | -0.12 | 0.45 | 0.12 |
| qlearning | erc | 0.20 | -0.09 | 0.47 | 0.09 |
| qlearning | 60_40 | -0.03 | -0.35 | 0.27 | 0.57 |
| qlearning | spy_buy_hold | 0.03 | -0.36 | 0.37 | 0.43 |
| dqn | ew | 0.05 | -0.28 | 0.35 | 0.38 |
| dqn | erc | 0.08 | -0.27 | 0.42 | 0.32 |
| dqn | 60_40 | -0.15 | -0.52 | 0.19 | 0.80 |
| dqn | spy_buy_hold | -0.09 | -0.49 | 0.26 | 0.69 |

## Actions chosen out of sample

- **qlearning**: mvo@100 43%, ew@100 16%, erc@100 13%, cash 10%, trend@100 7%, minvar@100 6%, trend@50 2%, minvar@50 1%, mvo@50 1%, ew@50 1%
- **dqn**: mvo@100 25%, ew@100 21%, cash 12%, trend@100 10%, minvar@50 8%, trend@50 7%, erc@100 7%, erc@50 5%, minvar@100 2%, mvo@50 1%, ew@50 1%
