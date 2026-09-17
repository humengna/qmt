import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from leadlag.data import make_synthetic_market
from leadlag.factor import (
    MiningConfig,
    compute_returns,
    compute_up_indicator,
    exclude_hub_followers,
    filter_oos_significant,
    mine_lead_lag_pairs,
    select_for_deployment,
    select_top_n_candidates,
    compute_pairwise_stats,
    validate_out_of_sample,
)


def _mine_and_validate(open_px, close_px, cfg, mode="absolute", train_frac=0.7):
    returns = compute_returns(close_px)
    split = int(len(returns) * train_frac)
    train, test = returns.iloc[:split], returns.iloc[split:]
    up_train, valid_train = compute_up_indicator(train, mode=mode)
    up_test, valid_test = compute_up_indicator(test, mode=mode)
    candidates = mine_lead_lag_pairs(up_train, valid_train, cfg)
    return validate_out_of_sample(up_test, valid_test, candidates, cfg)


class TestLeadLagMining(unittest.TestCase):
    def test_recovers_injected_pairs(self):
        open_px, close_px, injected = make_synthetic_market(
            n_stocks=40, n_days=1500, n_lead_lag_pairs=5, flip_prob=0.45, boost=0.04, seed=1
        )
        cfg = MiningConfig(lag=1, min_obs=50, alpha=0.05, min_lift=0.03)
        validated = _mine_and_validate(open_px, close_px, cfg)
        oos_significant = filter_oos_significant(validated, alpha=0.05, min_oos_n=10)
        deployable = select_for_deployment(oos_significant, max_unique_symbols=500, min_oos_n=10)

        found = set(zip(deployable["leader"], deployable["follower"]))
        injected_set = set(injected)
        recovered = injected_set & found
        self.assertGreaterEqual(
            len(recovered), len(injected_set) // 2,
            f"expected to recover at least half of {injected_set}, got {found}",
        )
        # every recovered pair should keep the correct sign out-of-sample
        for row in deployable.itertuples(index=False):
            self.assertGreater(row.oos_lift, 0)

    def test_pure_noise_yields_almost_no_survivors(self):
        open_px, close_px, injected = make_synthetic_market(
            n_stocks=40, n_days=1500, n_lead_lag_pairs=0, seed=2
        )
        self.assertEqual(injected, [])
        cfg = MiningConfig(lag=1, min_obs=50, alpha=0.05, min_lift=0.03)
        validated = _mine_and_validate(open_px, close_px, cfg)
        oos_significant = filter_oos_significant(validated, alpha=0.05, min_oos_n=10)
        deployable = select_for_deployment(oos_significant, max_unique_symbols=500, min_oos_n=10)
        # some false positives can still survive by chance; there should be very few
        # relative to the ~40*39 = 1560 ordered pairs tested.
        self.assertLess(len(deployable), 10)

    def test_same_sector_only_excludes_cross_sector_pairs(self):
        open_px, close_px, injected = make_synthetic_market(
            n_stocks=40, n_days=800, n_lead_lag_pairs=5, flip_prob=0.45, boost=0.04, seed=5
        )
        returns = compute_returns(close_px)
        up, valid = compute_up_indicator(returns, mode="absolute")
        cfg = MiningConfig(lag=1, min_obs=50, min_lift=0.0)

        codes = list(close_px.columns)
        # Split the universe into two disjoint sectors; deliberately put each injected
        # pair's leader and follower on OPPOSITE sides, so a correct implementation
        # must find nothing (the "true" cross-sector relationships get filtered out
        # by construction), while an unfiltered run would find them easily.
        sector_map = {}
        for leader, follower in injected:
            sector_map[leader] = "SECTOR_A"
            sector_map[follower] = "SECTOR_B"
        for code in codes:
            sector_map.setdefault(code, "SECTOR_A" if codes.index(code) % 2 == 0 else "SECTOR_B")

        stats_all = compute_pairwise_stats(up, valid, cfg)
        stats_grouped = compute_pairwise_stats(up, valid, cfg, sector_map=sector_map)

        self.assertGreater(len(stats_all), len(stats_grouped))
        for row in stats_grouped.itertuples(index=False):
            self.assertEqual(sector_map.get(row.leader), sector_map.get(row.follower))
        # none of the (deliberately cross-sector) injected pairs should survive
        found = set(zip(stats_grouped["leader"], stats_grouped["follower"]))
        self.assertFalse(found & set(injected))

    def test_oos_sign_check_alone_is_not_a_real_filter(self):
        # Regression test for a real bug found running this pipeline against actual
        # A-share data: --candidate-mode=top-n (or a loose --alpha) can hand the
        # out-of-sample step hundreds of thousands of largely-spurious in-sample
        # candidates. Checking only sign(oos_lift) passes ~50% of those by chance no
        # matter how many were tested - filter_oos_significant (real FDR control on
        # oos_p_value) must reject almost all of them instead.
        open_px, close_px, injected = make_synthetic_market(
            n_stocks=80, n_days=800, n_lead_lag_pairs=0, seed=7
        )
        self.assertEqual(injected, [])
        cfg = MiningConfig(lag=1, min_obs=30, min_lift=0.0)

        returns = compute_returns(close_px)
        split = int(len(returns) * 0.7)
        train, test = returns.iloc[:split], returns.iloc[split:]
        up_train, valid_train = compute_up_indicator(train, mode="absolute")
        up_test, valid_test = compute_up_indicator(test, mode="absolute")

        stats = compute_pairwise_stats(up_train, valid_train, cfg)
        # Deliberately pick a large slice of pure noise by rank, the way
        # --candidate-mode=top-n does on a real, much bigger universe.
        candidates = select_top_n_candidates(stats, cfg, top_n=300)
        self.assertEqual(len(candidates), 300)

        validated = validate_out_of_sample(up_test, valid_test, candidates, cfg)
        naive_survivors = (validated["oos_lift"] > 0).sum()
        # sanity check that this really does reproduce the ~coin-flip behavior
        self.assertGreater(naive_survivors, 60)

        oos_significant = filter_oos_significant(validated, alpha=0.05, min_oos_n=10)
        self.assertLess(
            len(oos_significant), naive_survivors // 3,
            "FDR-controlled out-of-sample filtering should reject the vast majority "
            "of a candidate set built purely from noise, unlike a naive sign check",
        )

    def test_excess_mode_removes_common_market_day(self):
        # A day where every stock jumps together should not, by itself, register as
        # a lead-lag relationship once the common move is netted out.
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2021-01-01", periods=300)
        codes = [f"X{i}" for i in range(10)]
        common = rng.normal(0, 0.02, len(dates))
        idio = rng.normal(0, 0.001, (len(dates), len(codes)))
        rets = pd.DataFrame(common[:, None] + idio, index=dates, columns=codes)

        up_abs, valid_abs = compute_up_indicator(rets, mode="absolute")
        up_exc, valid_exc = compute_up_indicator(rets, mode="excess")

        cfg = MiningConfig(lag=1, min_obs=50, alpha=0.05, min_lift=0.03)
        pairs_abs = mine_lead_lag_pairs(up_abs, valid_abs, cfg)
        pairs_exc = mine_lead_lag_pairs(up_exc, valid_exc, cfg)
        self.assertLessEqual(len(pairs_exc), len(pairs_abs))


class TestExcludeHubFollowers(unittest.TestCase):
    def _pairs(self, rows):
        return pd.DataFrame(rows, columns=["leader", "follower"])

    def test_drops_follower_tied_to_too_many_distinct_leaders(self):
        # HUB is paired with 4 distinct leaders (> max=3); NORMAL only ever has 1.
        rows = [
            {"leader": "L1", "follower": "HUB"}, {"leader": "L2", "follower": "HUB"},
            {"leader": "L3", "follower": "HUB"}, {"leader": "L4", "follower": "HUB"},
            {"leader": "L5", "follower": "NORMAL"},
        ]
        out = exclude_hub_followers(self._pairs(rows), max_leaders_per_follower=3)
        self.assertNotIn("HUB", set(out["follower"]))
        self.assertIn("NORMAL", set(out["follower"]))
        self.assertEqual(len(out), 1)

    def test_follower_at_exactly_the_limit_is_kept(self):
        rows = [
            {"leader": "L1", "follower": "F"}, {"leader": "L2", "follower": "F"},
            {"leader": "L3", "follower": "F"},
        ]
        out = exclude_hub_followers(self._pairs(rows), max_leaders_per_follower=3)
        self.assertEqual(len(out), 3)

    def test_zero_disables_the_filter(self):
        rows = [{"leader": f"L{i}", "follower": "HUB"} for i in range(20)]
        out = exclude_hub_followers(self._pairs(rows), max_leaders_per_follower=0)
        self.assertEqual(len(out), 20)

    def test_empty_input_is_a_no_op(self):
        empty = self._pairs([])
        out = exclude_hub_followers(empty, max_leaders_per_follower=3)
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
