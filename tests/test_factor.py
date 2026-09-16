import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leadlag.data import make_synthetic_market
from leadlag.factor import (
    MiningConfig,
    compute_returns,
    compute_up_indicator,
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


if __name__ == "__main__":
    unittest.main()
