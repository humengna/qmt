import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from leadlag.backtest import BacktestConfig, build_follower_scores, simulate_portfolio
from leadlag.data import make_synthetic_market
from leadlag.factor import (
    MiningConfig,
    compute_returns,
    compute_up_indicator,
    mine_lead_lag_pairs,
    select_for_deployment,
    validate_out_of_sample,
)
from leadlag.metrics import performance_summary


class TestBacktest(unittest.TestCase):
    def test_backtest_runs_end_to_end_on_synthetic_signal(self):
        open_px, close_px, injected = make_synthetic_market(
            n_stocks=30, n_days=1200, n_lead_lag_pairs=4, flip_prob=0.5, boost=0.05, seed=3
        )
        returns = compute_returns(close_px)
        split = int(len(returns) * 0.6)
        train, test = returns.iloc[:split], returns.iloc[split:]

        cfg = MiningConfig(lag=1, min_obs=40, alpha=0.1, min_lift=0.02)
        up_train, valid_train = compute_up_indicator(train, mode="absolute")
        candidates = mine_lead_lag_pairs(up_train, valid_train, cfg)
        up_test, valid_test = compute_up_indicator(test, mode="absolute")
        validated = validate_out_of_sample(up_test, valid_test, candidates, cfg)
        pairs = select_for_deployment(validated, max_unique_symbols=500, min_oos_n=5)
        self.assertGreater(len(pairs), 0, "synthetic signal should yield at least one deployable pair")

        up_full, _ = compute_up_indicator(returns, mode="absolute")
        score = build_follower_scores(up_full, pairs, lag=1, weight_col="oos_z")

        bt_cfg = BacktestConfig(lag=1, top_k=5, holding_period=1, initial_capital=1_000_000.0)
        equity, trades = simulate_portfolio(score, open_px, close_px, bt_cfg)

        self.assertEqual(len(equity), len(score))
        self.assertTrue((equity > 0).all(), "equity should never go non-positive")
        self.assertGreater(len(trades), 0, "expected at least some trades to fire")

        summary = performance_summary(equity)
        self.assertIn("sharpe", summary)
        self.assertIn("max_drawdown", summary)
        self.assertLessEqual(summary["max_drawdown"], 0)

    def test_costs_reduce_equity_vs_zero_cost(self):
        open_px, close_px, _ = make_synthetic_market(
            n_stocks=20, n_days=300, n_lead_lag_pairs=3, flip_prob=0.6, boost=0.05, seed=4
        )
        returns = compute_returns(close_px)
        up, _ = compute_up_indicator(returns, mode="absolute")

        # Fabricate a pair table directly instead of mining, to isolate cost effects.
        pairs = pd.DataFrame({
            "leader": [close_px.columns[0]],
            "follower": [close_px.columns[1]],
            "oos_z": [2.0],
        })
        score = build_follower_scores(up, pairs, lag=1, weight_col="oos_z")

        cheap = BacktestConfig(lag=1, top_k=1, holding_period=1, commission_bps=0,
                                stamp_tax_bps=0, slippage_bps=0)
        pricey = BacktestConfig(lag=1, top_k=1, holding_period=1, commission_bps=50,
                                 stamp_tax_bps=50, slippage_bps=50)

        equity_cheap, _ = simulate_portfolio(score, open_px, close_px, cheap)
        equity_pricey, _ = simulate_portfolio(score, open_px, close_px, pricey)

        self.assertGreaterEqual(equity_cheap.iloc[-1], equity_pricey.iloc[-1])


if __name__ == "__main__":
    unittest.main()
