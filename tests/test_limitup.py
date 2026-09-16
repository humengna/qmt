import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leadlag.backtest import BacktestConfig, build_follower_scores, simulate_portfolio
from leadlag.factor import (
    MiningConfig,
    compute_pairwise_stats_asymmetric,
    compute_up_indicator,
    filter_oos_significant,
    select_for_deployment,
    select_top_n_candidates,
    validate_out_of_sample_asymmetric,
)
from leadlag.metrics import performance_summary
from limitup.data import limit_pct_for_code, make_synthetic_limitup_market
from limitup.event import build_leader_frames


class TestLimitPct(unittest.TestCase):
    def test_board_based_limit_percentage(self):
        self.assertEqual(limit_pct_for_code("300001.SZ"), 0.20)  # ChiNext
        self.assertEqual(limit_pct_for_code("301001.SZ"), 0.20)  # ChiNext (new codes)
        self.assertEqual(limit_pct_for_code("688001.SH"), 0.20)  # STAR market
        self.assertEqual(limit_pct_for_code("600001.SH"), 0.10)  # main board
        self.assertEqual(limit_pct_for_code("000001.SZ"), 0.10)  # main board (SZ)
        self.assertEqual(limit_pct_for_code("830001.BJ"), 0.30)  # BSE (approx)


class TestLimitUpDetector(unittest.TestCase):
    def test_detector_matches_injected_trigger_days_exactly(self):
        raw_close, raw_preclose, suspend, _, pairs = make_synthetic_limitup_market(
            n_stocks=30, n_days=600, n_pairs=4, limitup_prob=0.05, seed=2
        )
        hit, valid = build_leader_frames(raw_close, raw_preclose, suspend)

        for leader, _ in pairs:
            pct = limit_pct_for_code(leader)
            expected_price = (raw_preclose[leader] * (1 + pct)).round(2)
            actual_hit_days = hit[leader][hit[leader]].index
            self.assertGreater(len(actual_hit_days), 0)
            for day in actual_hit_days:
                self.assertAlmostEqual(raw_close.at[day, leader], expected_price.at[day], places=2)

        # a stock never designated as a leader should essentially never hit its limit
        # by pure noise (idio std 0.015 vs. a 10-20% move is an enormous outlier)
        non_leaders = [c for c in raw_close.columns if c not in dict(pairs)]
        total_noise_hits = hit[non_leaders].sum().sum()
        self.assertEqual(total_noise_hits, 0)


class TestLimitUpMining(unittest.TestCase):
    def _mine(self, raw_close, raw_preclose, suspend, adj_close, cfg, train_frac=0.7, top_n=40):
        leader_up, leader_valid = build_leader_frames(raw_close, raw_preclose, suspend)
        returns = adj_close.sort_index().pct_change()
        follower_up, follower_valid = compute_up_indicator(returns, mode="absolute")

        split = int(len(returns) * train_frac)
        train, test = slice(0, split), slice(split, None)

        stats = compute_pairwise_stats_asymmetric(
            leader_up.iloc[train], leader_valid.iloc[train],
            follower_up.iloc[train], follower_valid.iloc[train], cfg,
        )
        candidates = select_top_n_candidates(stats, cfg, top_n=top_n)
        return validate_out_of_sample_asymmetric(
            leader_up.iloc[test], leader_valid.iloc[test],
            follower_up.iloc[test], follower_valid.iloc[test], candidates, cfg,
        )

    def test_recovers_injected_trigger_pairs(self):
        raw_close, raw_preclose, suspend, adj_close, pairs = make_synthetic_limitup_market(
            n_stocks=50, n_days=1800, n_pairs=6, limitup_prob=0.06, flip_prob=0.6, boost=0.05, seed=3
        )
        cfg = MiningConfig(lag=1, min_obs=20, min_lift=0.03)
        validated = self._mine(raw_close, raw_preclose, suspend, adj_close, cfg)

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
        raw_close, raw_preclose, suspend, adj_close, pairs = make_synthetic_limitup_market(
            n_stocks=50, n_days=1800, n_pairs=0, seed=4
        )
        self.assertEqual(pairs, [])
        cfg = MiningConfig(lag=1, min_obs=20, min_lift=0.03)
        validated = self._mine(raw_close, raw_preclose, suspend, adj_close, cfg, top_n=40)

        naive_survivors = (validated["oos_lift"] > 0).sum()
        oos_significant = filter_oos_significant(validated, alpha=0.1, min_oos_n=5)
        # Same coin-flip check as leadlag's regression test: a naive sign check on a
        # large rank-selected candidate pool of pure noise passes roughly half of it,
        # while the real (FDR-controlled) out-of-sample gate should reject nearly all
        # (allow a couple through by chance rather than requiring exactly 0).
        self.assertLessEqual(len(oos_significant), max(naive_survivors // 2, 2))


class TestLimitUpBacktest(unittest.TestCase):
    def test_backtest_runs_and_uses_the_limit_up_signal(self):
        raw_close, raw_preclose, suspend, adj_close, pairs = make_synthetic_limitup_market(
            n_stocks=40, n_days=1200, n_pairs=5, limitup_prob=0.06, flip_prob=0.6, boost=0.05, seed=5
        )
        leader_up, leader_valid = build_leader_frames(raw_close, raw_preclose, suspend)
        returns = adj_close.sort_index().pct_change()
        follower_up, follower_valid = compute_up_indicator(returns, mode="absolute")

        cfg = MiningConfig(lag=1, min_obs=15, min_lift=0.02)
        split = int(len(returns) * 0.6)
        train, test = slice(0, split), slice(split, None)
        stats = compute_pairwise_stats_asymmetric(
            leader_up.iloc[train], leader_valid.iloc[train],
            follower_up.iloc[train], follower_valid.iloc[train], cfg,
        )
        candidates = select_top_n_candidates(stats, cfg, top_n=30)
        validated = validate_out_of_sample_asymmetric(
            leader_up.iloc[test], leader_valid.iloc[test],
            follower_up.iloc[test], follower_valid.iloc[test], candidates, cfg,
        )
        oos_significant = filter_oos_significant(validated, alpha=0.15, min_oos_n=3)
        pairs_df = select_for_deployment(oos_significant, max_unique_symbols=500, min_oos_n=3)
        self.assertGreater(len(pairs_df), 0, "synthetic trigger signal should yield a deployable pair")

        open_px = adj_close.shift(1).bfill()
        score = build_follower_scores(leader_up, pairs_df, lag=1, weight_col="oos_z")
        bt_cfg = BacktestConfig(lag=1, top_k=5, holding_period=1, initial_capital=1_000_000.0)
        equity, trades = simulate_portfolio(score, open_px, adj_close, bt_cfg)

        self.assertEqual(len(equity), len(score))
        self.assertTrue((equity > 0).all())
        self.assertGreater(len(trades), 0)
        summary = performance_summary(equity)
        self.assertIn("sharpe", summary)


if __name__ == "__main__":
    unittest.main()
