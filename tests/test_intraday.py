import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from intraday.backtest import IntradayBacktestConfig, build_intraday_scores, simulate_intraday_portfolio
from intraday.data import broadcast_prev_close_to_bars, limit_pct_for_code, make_synthetic_intraday_market
from intraday.event import build_follower_frames, build_leader_frames
from leadlag.factor import (
    MiningConfig,
    compute_pairwise_stats_same_row,
    filter_oos_significant,
    select_for_deployment,
    select_top_n_candidates,
    validate_out_of_sample_same_row,
)
from leadlag.metrics import performance_summary


class TestIntradayDetector(unittest.TestCase):
    def test_first_touch_matches_limit_price_formula_exactly(self):
        minute_close, minute_suspend, daily_close, pairs = make_synthetic_intraday_market(
            n_days=300, bars_per_day=48, n_stocks=30, n_pairs=5, lag_bars=6, seed=1
        )
        leader_triggered, leader_valid = build_leader_frames(minute_close, minute_suspend, daily_close)
        prev_close_by_bar = broadcast_prev_close_to_bars(minute_close.index, daily_close)

        checked = 0
        for leader, _ in pairs:
            pct = limit_pct_for_code(leader)
            trig_times = leader_triggered[leader][leader_triggered[leader]].index
            self.assertGreater(len(trig_times), 0)
            for t in trig_times:
                expected = round(prev_close_by_bar.at[t, leader] * (1 + pct), 2)
                self.assertAlmostEqual(minute_close.at[t, leader], expected, places=2)
                checked += 1
        self.assertGreater(checked, 0)

    def test_at_most_one_trigger_per_stock_per_day(self):
        minute_close, minute_suspend, daily_close, _ = make_synthetic_intraday_market(
            n_days=300, n_stocks=30, n_pairs=5, seed=1
        )
        leader_triggered, _ = build_leader_frames(minute_close, minute_suspend, daily_close)
        per_day = leader_triggered.groupby(leader_triggered.index.normalize()).sum()
        self.assertLessEqual(per_day.max().max(), 1)

    def test_non_leader_stocks_never_trigger_by_noise(self):
        minute_close, minute_suspend, daily_close, pairs = make_synthetic_intraday_market(
            n_days=300, n_stocks=30, n_pairs=5, seed=1
        )
        leader_triggered, _ = build_leader_frames(minute_close, minute_suspend, daily_close)
        non_leaders = [c for c in minute_close.columns if c not in dict(pairs)]
        self.assertEqual(int(leader_triggered[non_leaders].to_numpy().sum()), 0)

    def test_forward_outcome_excludes_windows_crossing_day_boundary(self):
        minute_close, minute_suspend, daily_close, _ = make_synthetic_intraday_market(
            n_days=10, bars_per_day=48, n_stocks=5, n_pairs=0, seed=2
        )
        lag_bars = 6
        _, follower_valid = build_follower_frames(minute_close, minute_suspend, lag_bars=lag_bars)
        # the last `lag_bars` bars of every trading day can't have a same-day outcome
        last_bars_of_day = follower_valid.groupby(follower_valid.index.normalize()).tail(lag_bars)
        self.assertFalse(last_bars_of_day.to_numpy().any())


class TestIntradayMining(unittest.TestCase):
    def _mine(self, minute_close, minute_suspend, daily_close, lag_bars, cfg, train_frac=0.7, top_n=40):
        leader_triggered, leader_valid = build_leader_frames(minute_close, minute_suspend, daily_close)
        follower_outcome, follower_valid = build_follower_frames(
            minute_close, minute_suspend, lag_bars=lag_bars, mode="absolute"
        )
        split = int(len(minute_close) * train_frac)
        train, test = slice(0, split), slice(split, None)

        stats = compute_pairwise_stats_same_row(
            leader_triggered.iloc[train], leader_valid.iloc[train],
            follower_outcome.iloc[train], follower_valid.iloc[train], cfg,
        )
        candidates = select_top_n_candidates(stats, cfg, top_n=top_n)
        return validate_out_of_sample_same_row(
            leader_triggered.iloc[test], leader_valid.iloc[test],
            follower_outcome.iloc[test], follower_valid.iloc[test], candidates, cfg,
        )

    def test_recovers_injected_intraday_pairs(self):
        minute_close, minute_suspend, daily_close, pairs = make_synthetic_intraday_market(
            n_days=500, bars_per_day=48, n_stocks=40, n_pairs=6, lag_bars=6,
            limitup_prob=0.1, flip_prob=0.6, boost=0.025, seed=3,
        )
        cfg = MiningConfig(lag=6, min_obs=15, min_lift=0.03)
        validated = self._mine(minute_close, minute_suspend, daily_close, lag_bars=6, cfg=cfg)

        oos_significant = filter_oos_significant(validated, alpha=0.1, min_oos_n=5)
        deployable = select_for_deployment(oos_significant, max_unique_symbols=500, min_oos_n=5)

        found = set(zip(deployable["leader"], deployable["follower"]))
        recovered = found & set(pairs)
        self.assertGreaterEqual(
            len(recovered), len(pairs) * 2 // 3,
            f"expected to recover at least 2/3 of {set(pairs)}, got {found}",
        )
        for row in deployable.itertuples(index=False):
            self.assertGreater(row.oos_lift, 0)

    def test_pure_noise_yields_no_survivors_after_oos_fdr(self):
        minute_close, minute_suspend, daily_close, pairs = make_synthetic_intraday_market(
            n_days=500, bars_per_day=48, n_stocks=40, n_pairs=0, lag_bars=6, seed=4,
        )
        self.assertEqual(pairs, [])
        cfg = MiningConfig(lag=6, min_obs=15, min_lift=0.03)
        validated = self._mine(minute_close, minute_suspend, daily_close, lag_bars=6, cfg=cfg, top_n=40)

        naive_survivors = (validated["oos_lift"] > 0).sum()
        oos_significant = filter_oos_significant(validated, alpha=0.1, min_oos_n=5)
        self.assertLessEqual(len(oos_significant), max(naive_survivors // 2, 2))


class TestIntradayBacktest(unittest.TestCase):
    def test_backtest_runs_and_reflects_the_injected_signal(self):
        minute_close, minute_suspend, daily_close, pairs = make_synthetic_intraday_market(
            n_days=500, bars_per_day=48, n_stocks=40, n_pairs=6, lag_bars=6,
            limitup_prob=0.1, flip_prob=0.6, boost=0.025, seed=3,
        )
        leader_triggered, leader_valid = build_leader_frames(minute_close, minute_suspend, daily_close)
        follower_outcome, follower_valid = build_follower_frames(
            minute_close, minute_suspend, lag_bars=6, mode="absolute"
        )

        cfg = MiningConfig(lag=6, min_obs=10, min_lift=0.02)
        split = int(len(minute_close) * 0.6)
        train, test = slice(0, split), slice(split, None)
        stats = compute_pairwise_stats_same_row(
            leader_triggered.iloc[train], leader_valid.iloc[train],
            follower_outcome.iloc[train], follower_valid.iloc[train], cfg,
        )
        candidates = select_top_n_candidates(stats, cfg, top_n=30)
        validated = validate_out_of_sample_same_row(
            leader_triggered.iloc[test], leader_valid.iloc[test],
            follower_outcome.iloc[test], follower_valid.iloc[test], candidates, cfg,
        )
        oos_significant = filter_oos_significant(validated, alpha=0.15, min_oos_n=3)
        pairs_df = select_for_deployment(oos_significant, max_unique_symbols=500, min_oos_n=3)
        self.assertGreater(len(pairs_df), 0, "synthetic intraday signal should yield a deployable pair")

        score = build_intraday_scores(leader_triggered, pairs_df, weight_col="oos_z")
        bt_cfg = IntradayBacktestConfig(lag_bars=6, top_k=5, initial_capital=1_000_000.0)
        equity, trades = simulate_intraday_portfolio(score, minute_close, bt_cfg)

        self.assertEqual(len(equity), len(score))
        self.assertTrue((equity > 0).all())
        self.assertGreater(len(trades), 0)
        # every exit must land on or before its trading day's last bar (never spills
        # into the next day - the whole point of this backtest's exit-scheduling)
        dates = equity.index.normalize()
        last_bar_idx_of_day = pd.Series(range(len(equity)), index=equity.index).groupby(dates).transform("max")
        buys = trades[trades["side"] == "buy"].copy()
        buys["bar_idx"] = buys["bar"].map({t: i for i, t in enumerate(equity.index)})
        for _, row in buys.iterrows():
            self.assertLessEqual(row["bar_idx"] + bt_cfg.lag_bars, last_bar_idx_of_day.iloc[row["bar_idx"]])

        summary = performance_summary(equity, periods_per_year=252 * 48)
        self.assertIn("sharpe", summary)
        self.assertGreater(summary["sharpe"], 0)


if __name__ == "__main__":
    unittest.main()
