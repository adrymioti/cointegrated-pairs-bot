# Results summary - SYNTHETIC

> **SYNTHETIC DATA - pipeline validation only. These numbers say nothing about crypto markets.**

All headline numbers below are from the **final test period** (out of sample), frozen model fitted on development data only, net of the stated costs.

- Split: {'dev_start': '2022-01-01 00:00:00+00:00', 'val_start': '2023-06-10 22:00:00+00:00', 'test_start': '2024-05-26 14:00:00+00:00', 'test_end': '2026-01-01 00:00:00+00:00'}
- Pairs tested: 78, selected after Holm (5% FWER) + filters: 3
- Selected pairs: [{'pair': 'ADAUSDT-SOLUSDT', 'alpha': -0.1880794741368983, 'beta': 0.6773340452411051}, {'pair': 'BTCUSDT-ETHUSDT', 'alpha': 0.01724041091324806, 'beta': 1.0935016760418597}, {'pair': 'ETHUSDT-SOLUSDT', 'alpha': 3.17328565230351, 'beta': 0.27271945911417145}]

## Gross vs net (frozen model, final test)
| series | n_hours | start | end | total_return | ann_vol | sharpe | max_drawdown | mean_hourly_return | note | n_trades | win_rate | mean_win | mean_loss | mean_net_pnl | mean_hold_hours | exposure | turnover | fees | slippage | borrow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gross | 14026 | 2024-05-26 14:00:00+00:00 | 2025-12-31 23:00:00+00:00 | 1.325 | 0.1155 | 4.618 | 0.04509 | 6.091e-05 | Sharpe uses sqrt(K) scaling; serial correlation can make it misleading | 243 | 0.8395 | 729.7 | -847.5 | 476.6 | 50.14 | 0.6366 | 96.95 | 0 | 0 | 0 |
| net | 14026 | 2024-05-26 14:00:00+00:00 | 2025-12-31 23:00:00+00:00 | 1.158 | 0.1202 | 4.057 | 0.04858 | 5.567e-05 | Sharpe uses sqrt(K) scaling; serial correlation can make it misleading | 243 | 0.8395 | 729.7 | -847.5 | 476.6 | 50.14 | 0.6366 | 96.95 | 7309 | 7309 | 2049 |

## Walk-forward (same period)
| series | n_hours | start | end | total_return | ann_vol | sharpe | max_drawdown | mean_hourly_return | note | n_trades | win_rate | mean_win | mean_loss | mean_net_pnl | mean_hold_hours | exposure | turnover | fees | slippage | borrow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gross | 14026 | 2024-05-26 14:00:00+00:00 | 2025-12-31 23:00:00+00:00 | 0.8387 | 0.08614 | 4.46 | 0.03047 | 4.385e-05 | Sharpe uses sqrt(K) scaling; serial correlation can make it misleading | 210 | 0.8143 | 560.9 | -632.6 | 339.3 | 43.06 | 0.4887 | 84.29 | 0 | 0 | 0 |
| net | 14026 | 2024-05-26 14:00:00+00:00 | 2025-12-31 23:00:00+00:00 | 0.7125 | 0.08949 | 3.799 | 0.03389 | 3.881e-05 | Sharpe uses sqrt(K) scaling; serial correlation can make it misleading | 210 | 0.8143 | 560.9 | -632.6 | 339.3 | 43.06 | 0.4887 | 84.29 | 5657 | 5657 | 1311 |

## Cost stress (total fee+slippage bps per executed notional)
| total_cost_bps | n_trades | net_return | sharpe | max_dd | net_pnl |
|---|---|---|---|---|---|
| 0 | 243 | 1.377 | 4.561 | 0.04647 | 1.377e+05 |
| 5 | 243 | 1.265 | 4.309 | 0.04752 | 1.265e+05 |
| 10 | 243 | 1.158 | 4.057 | 0.04858 | 1.158e+05 |
| 20 | 243 | 0.9597 | 3.55 | 0.05068 | 9.597e+04 |
| 30 | 243 | 0.7793 | 3.042 | 0.05278 | 7.793e+04 |
| 40 | 243 | 0.6155 | 2.534 | 0.05488 | 6.155e+04 |
| 60 | 243 | 0.3314 | 1.524 | 0.05907 | 3.314e+04 |

Break-even cost: inf bps (0 = unprofitable even at zero cost; inf = profitable at largest tested cost)

## Regimes (descriptive)
| regime | hours | share_of_hours | net_return_sum | mean_hourly_return | max_drawdown_within_regime | trades_entered | net_pnl_of_entered_trades | mean_hold_hours | exposure |
|---|---|---|---|---|---|---|---|---|---|
| high_vol|falling | 2765 | 0.1971 | 0.1967 | 7.115e-05 | 0.02365 | 48 | 3.028e+04 | 60.85 | 0.4933 |
| high_vol|rising | 2155 | 0.1536 | 0.09216 | 4.277e-05 | 0.03629 | 42 | 1.135e+04 | 51.48 | 0.6172 |
| low_vol|falling | 2117 | 0.1509 | 0.1626 | 7.682e-05 | 0.03839 | 54 | 3.2e+04 | 42.19 | 0.5829 |
| low_vol|rising | 2184 | 0.1557 | 0.1619 | 7.412e-05 | 0.04946 | 51 | 2.404e+04 | 45.29 | 0.7386 |

Block-bootstrap (5th-95th pct): {'sharpe_5': 3.168318677876533, 'sharpe_95': 4.974190265597659, 'total_return_5': 0.8140393711965239, 'total_return_95': 1.5623799868104733, 'block': 168, 'reps': 1000}
References: {'cash_zero_return': 0.0, 'buy_hold_total_return': -0.2765939386032996, 'buy_hold_max_dd': 0.54627202864762}

Prespecified: pair-selection method, Holm correction, 60/40 split, walk-forward schedule, grid, cost scenarios.
Exploratory: regime analysis, rolling diagnostics, sub-period and leg-failure stress tests.