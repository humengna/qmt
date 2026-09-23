import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from intraday.backtest import (
    IntradayBacktestConfig,
    _overnight_exit_rows,
    build_intraday_scores,
    simulate_intraday_portfolio,
    simulate_overnight_portfolio,
)
from intraday.data import (
    broadcast_prev_close_to_bars,
    limit_pct_for_code,
    make_synthetic_intraday_market,
    make_synthetic_surge_market,
    make_synthetic_threshold_market,
)
from intraday.event import (
    build_follower_frames,
    build_follower_frames_overnight,
    build_leader_frames,
    build_leader_frames_surge,
    build_leader_frames_threshold,
)
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


class TestOvernightHold(unittest.TestCase):
    """The T+1-executable variant: trigger intraday, exit the NEXT trading day."""

    def _market(self, **kw):
        return make_synthetic_surge_market(
            n_days=400, n_stocks=40, n_pairs=6, lag_bars=6, threshold=0.02, window_bars=5,
            trigger_prob=0.12, flip_prob=0.6, boost=0.02, seed=3, **kw
        )

    def test_outcome_measures_next_day_not_same_day(self):
        minute_close, minute_suspend, _ = self._market()
        outcome, valid = build_follower_frames_overnight(
            minute_close, minute_suspend, exit_at="next_open", mode="absolute"
        )
        dates = minute_close.index.normalize()
        day_first = minute_close.groupby(dates).first()

        # pick a bar on the first day and verify against the SECOND day's opening bar
        code = minute_close.columns[0]
        t = minute_close.index[5]
        expected = day_first[code].iloc[1] / minute_close.at[t, code] - 1
        self.assertEqual(bool(outcome.at[t, code]), expected > 0)
        self.assertTrue(bool(valid.at[t, code]))

    def test_final_trading_day_has_no_valid_outcome(self):
        minute_close, minute_suspend, _ = self._market()
        _, valid = build_follower_frames_overnight(minute_close, minute_suspend, mode="absolute")
        last_day = minute_close.index.normalize().max()
        self.assertFalse(valid[minute_close.index.normalize() == last_day].to_numpy().any())

    def test_next_close_exits_later_than_next_open(self):
        minute_close, minute_suspend, _ = self._market()
        dates = minute_close.index.normalize().to_numpy()
        opens = _overnight_exit_rows(minute_close.index, "next_open")
        closes = _overnight_exit_rows(minute_close.index, "next_close")
        tradeable = opens >= 0
        self.assertTrue((closes[tradeable] > opens[tradeable]).all())
        # and the exit always lands on a LATER day than the entry bar's
        self.assertTrue((dates[opens[tradeable]] > dates[tradeable]).all())

    def test_every_trade_spans_a_day_boundary(self):
        # the whole point: a position held across a day boundary satisfies T+1,
        # which the same-day engine by construction cannot.
        minute_close, minute_suspend, pairs = self._market()
        leader_triggered, _ = build_leader_frames_surge(
            minute_close, minute_suspend, threshold=0.02, window_bars=5
        )
        pairs_df = pd.DataFrame(
            [{"leader": l, "follower": f, "oos_z": 3.0} for l, f in pairs]
        )
        score = build_intraday_scores(leader_triggered, pairs_df, weight_col="oos_z")
        cfg = IntradayBacktestConfig(lag_bars=6, top_k=5, initial_capital=1_000_000.0)
        equity, trades = simulate_overnight_portfolio(score, minute_close, cfg, exit_at="next_open")

        self.assertGreater(len(trades), 0)
        self.assertEqual(len(equity), len(score))
        self.assertTrue((equity > 0).all())

        buys = trades[trades["side"] == "buy"].reset_index(drop=True)
        sells = trades[trades["side"] == "sell"].reset_index(drop=True)
        self.assertEqual(len(buys), len(sells))
        for code, grp in trades.groupby("code"):
            grp = grp.sort_values("bar")
            b = grp[grp["side"] == "buy"]["bar"].to_numpy()
            s = grp[grp["side"] == "sell"]["bar"].to_numpy()
            n = min(len(b), len(s))
            for entry, exit_ in zip(b[:n], s[:n]):
                self.assertGreater(
                    pd.Timestamp(exit_).normalize(), pd.Timestamp(entry).normalize(),
                    f"{code} entered {entry} and exited {exit_} on the same day - violates T+1",
                )


class TestSurgeDetector(unittest.TestCase):
    """Covers the '急拉' leader trigger: a move of a given SIZE within a given TIME."""

    def _panel(self, closes_by_code, bars_per_day=None):
        """Build a one-or-two-day minute panel by hand from explicit price paths."""
        n = len(next(iter(closes_by_code.values())))
        bars_per_day = bars_per_day or n
        index = pd.DatetimeIndex([
            pd.Timestamp("2024-01-01") + pd.Timedelta(days=i // bars_per_day)
            + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=5 * (i % bars_per_day))
            for i in range(n)
        ])
        close = pd.DataFrame(closes_by_code, index=index)
        return close, pd.DataFrame(0, index=index, columns=close.columns)

    def test_slow_grind_does_not_trigger_but_a_spike_does(self):
        # THE distinction from compute_first_threshold_cross_indicator: both paths end
        # up +4% on the day, but only one of them is a surge.
        n = 24
        grind = [20.0 * (1 + 0.04 * i / (n - 1)) for i in range(n)]   # +4% spread over 24 bars
        spike = [20.0] * 10 + [20.8] * 14                             # +4% in a single bar
        close, suspend = self._panel({"GRIND.SH": grind, "SPIKE.SH": spike})

        triggered, _ = build_leader_frames_surge(close, suspend, threshold=0.02, window_bars=5)
        self.assertEqual(int(triggered["GRIND.SH"].sum()), 0)
        self.assertEqual(int(triggered["SPIKE.SH"].sum()), 1)

    def test_lookback_never_crosses_into_the_previous_day(self):
        # Day 1 ends at 20.0, day 2 opens at 21.0 (+5% overnight gap) and then goes flat.
        # An overnight gap is not a surge, so nothing may trigger.
        close, suspend = self._panel({"GAP.SH": [20.0] * 10 + [21.0] * 10}, bars_per_day=10)
        triggered, _ = build_leader_frames_surge(close, suspend, threshold=0.02, window_bars=5)
        self.assertEqual(int(triggered["GAP.SH"].sum()), 0)

    def test_matches_the_injected_surge_bars_exactly(self):
        minute_close, minute_suspend, pairs = make_synthetic_surge_market(
            n_days=200, n_stocks=30, n_pairs=5, threshold=0.02, window_bars=5, seed=1
        )
        triggered, _ = build_leader_frames_surge(minute_close, minute_suspend, threshold=0.02, window_bars=5)

        checked = 0
        for leader, _ in pairs:
            trig_times = triggered[leader][triggered[leader]].index
            self.assertGreater(len(trig_times), 0)
            for t in trig_times:
                pos = minute_close.index.get_loc(t)
                past = minute_close[leader].iloc[pos - 5]
                self.assertGreaterEqual(minute_close[leader].iloc[pos] / past - 1, 0.02)
                checked += 1
        self.assertGreater(checked, 0)

    def test_at_most_one_trigger_per_stock_per_day(self):
        minute_close, minute_suspend, _ = make_synthetic_surge_market(
            n_days=200, n_stocks=30, n_pairs=5, seed=1
        )
        triggered, _ = build_leader_frames_surge(minute_close, minute_suspend, threshold=0.02, window_bars=5)
        per_day = triggered.groupby(triggered.index.normalize()).sum()
        self.assertLessEqual(per_day.max().max(), 1)

    def test_uninvolved_stocks_never_trigger_by_noise(self):
        minute_close, minute_suspend, pairs = make_synthetic_surge_market(
            n_days=200, n_stocks=30, n_pairs=5, seed=1
        )
        triggered, _ = build_leader_frames_surge(minute_close, minute_suspend, threshold=0.02, window_bars=5)
        involved = set(dict(pairs).keys()) | set(dict(pairs).values())
        uninvolved = [c for c in minute_close.columns if c not in involved]
        self.assertEqual(int(triggered[uninvolved].to_numpy().sum()), 0)

    def test_recovers_injected_pairs_with_a_sector_map(self):
        minute_close, minute_suspend, pairs = make_synthetic_surge_market(
            n_days=400, n_stocks=40, n_pairs=6, lag_bars=6, threshold=0.02, window_bars=5,
            trigger_prob=0.12, flip_prob=0.6, boost=0.02, seed=3,
        )
        # put each injected pair in its own sector, everything else elsewhere, so the
        # sector mask must keep the true pairs and can only drop noise
        sector_map = {}
        for i, (leader, follower) in enumerate(pairs):
            sector_map[leader] = sector_map[follower] = f"SW1_{i}"
        for code in minute_close.columns:
            sector_map.setdefault(code, "SW1_OTHER")

        leader_triggered, leader_valid = build_leader_frames_surge(
            minute_close, minute_suspend, threshold=0.02, window_bars=5
        )
        follower_outcome, follower_valid = build_follower_frames(
            minute_close, minute_suspend, lag_bars=6, mode="absolute"
        )
        cfg = MiningConfig(lag=6, min_obs=15, min_lift=0.03)
        split = int(len(minute_close) * 0.7)
        train, test = slice(0, split), slice(split, None)

        stats = compute_pairwise_stats_same_row(
            leader_triggered.iloc[train], leader_valid.iloc[train],
            follower_outcome.iloc[train], follower_valid.iloc[train], cfg, sector_map,
        )
        for row in stats.itertuples(index=False):
            self.assertEqual(sector_map[row.leader], sector_map[row.follower])

        candidates = select_top_n_candidates(stats, cfg, top_n=40)
        validated = validate_out_of_sample_same_row(
            leader_triggered.iloc[test], leader_valid.iloc[test],
            follower_outcome.iloc[test], follower_valid.iloc[test], candidates, cfg,
        )
        deployable = select_for_deployment(
            filter_oos_significant(validated, alpha=0.1, min_oos_n=5), max_unique_symbols=500, min_oos_n=5
        )
        recovered = set(zip(deployable["leader"], deployable["follower"])) & set(pairs)
        self.assertGreaterEqual(len(recovered), len(pairs) // 2)


class TestThresholdDetector(unittest.TestCase):
    """Covers the T0-ETF leader trigger (run_etf_mining.py): cumulative return since
    the day's own first bar crossing a plain threshold, instead of a limit-up formula.
    """

    def test_trigger_bars_actually_cleared_the_threshold_and_the_prior_bar_did_not(self):
        minute_close, minute_suspend, pairs = make_synthetic_threshold_market(
            n_days=300, bars_per_day=48, n_stocks=30, n_pairs=5, lag_bars=6, threshold=0.01, seed=1,
        )
        leader_triggered, leader_valid = build_leader_frames_threshold(minute_close, minute_suspend, threshold=0.01)
        dates = minute_close.index.normalize()
        day_open = minute_close.where(leader_valid).groupby(dates).transform("first")

        checked = 0
        for leader, _ in pairs:
            trig_times = leader_triggered[leader][leader_triggered[leader]].index
            self.assertGreater(len(trig_times), 0)
            for t in trig_times:
                cum_ret = minute_close.at[t, leader] / day_open.at[t, leader] - 1
                self.assertGreaterEqual(cum_ret, 0.01)
                pos = minute_close.index.get_loc(t)
                if pos > 0 and minute_close.index.normalize()[pos - 1] == minute_close.index.normalize()[pos]:
                    prev_t = minute_close.index[pos - 1]
                    prev_ret = minute_close.at[prev_t, leader] / day_open.at[prev_t, leader] - 1
                    self.assertLess(prev_ret, 0.01)
                checked += 1
        self.assertGreater(checked, 0)

    def test_at_most_one_trigger_per_stock_per_day(self):
        minute_close, minute_suspend, _ = make_synthetic_threshold_market(
            n_days=300, n_stocks=30, n_pairs=5, threshold=0.01, seed=1,
        )
        leader_triggered, _ = build_leader_frames_threshold(minute_close, minute_suspend, threshold=0.01)
        per_day = leader_triggered.groupby(leader_triggered.index.normalize()).sum()
        self.assertLessEqual(per_day.max().max(), 1)

    def test_uninvolved_stocks_never_trigger_by_noise(self):
        # Excludes followers too, not just leaders: a follower's injected boost (2.5%)
        # comfortably clears the 1% leader-threshold on its own, so it legitimately
        # triggers as a "leader" on its boosted bar - that's the injected signal
        # working as intended, not noise. Only stocks with NO role in any pair should
        # never trigger at this (deliberately low, see the generator's docstring)
        # background noise scale.
        minute_close, minute_suspend, pairs = make_synthetic_threshold_market(
            n_days=300, n_stocks=30, n_pairs=5, threshold=0.01, seed=1,
        )
        leader_triggered, _ = build_leader_frames_threshold(minute_close, minute_suspend, threshold=0.01)
        involved = set(dict(pairs).keys()) | set(dict(pairs).values())
        uninvolved = [c for c in minute_close.columns if c not in involved]
        self.assertEqual(int(leader_triggered[uninvolved].to_numpy().sum()), 0)

    def test_first_bar_of_day_can_never_trigger(self):
        # bar 0 IS the day's own anchor - it can't have "crossed" anything relative to itself.
        minute_close, minute_suspend, _ = make_synthetic_threshold_market(
            n_days=50, bars_per_day=48, n_stocks=10, n_pairs=3, threshold=0.01, seed=7,
        )
        leader_triggered, _ = build_leader_frames_threshold(minute_close, minute_suspend, threshold=0.01)
        first_bars = leader_triggered.groupby(leader_triggered.index.normalize()).head(1)
        self.assertFalse(first_bars.to_numpy().any())


class TestEtfMining(unittest.TestCase):
    def _mine(self, minute_close, minute_suspend, lag_bars, leader_threshold, cfg, train_frac=0.7, top_n=40):
        leader_triggered, leader_valid = build_leader_frames_threshold(
            minute_close, minute_suspend, threshold=leader_threshold
        )
        follower_outcome, follower_valid = build_follower_frames(
            minute_close, minute_suspend, lag_bars=lag_bars, mode="absolute"
        )
        split = int(len(minute_close) * train_frac)
        train, test = slice(0, split), slice(split, None)

        stats = compute_pairwise_stats_same_row(
            leader_triggered.iloc[train], leader_valid.iloc[train],
            follower_outcome.iloc[train], follower_valid.iloc[train], cfg, sector_map=None,
        )
        candidates = select_top_n_candidates(stats, cfg, top_n=top_n)
        return validate_out_of_sample_same_row(
            leader_triggered.iloc[test], leader_valid.iloc[test],
            follower_outcome.iloc[test], follower_valid.iloc[test], candidates, cfg,
        )

    def test_recovers_injected_pairs_with_no_sector_restriction(self):
        minute_close, minute_suspend, pairs = make_synthetic_threshold_market(
            n_days=500, bars_per_day=48, n_stocks=40, n_pairs=6, lag_bars=6, threshold=0.01,
            trigger_prob=0.1, flip_prob=0.6, boost=0.025, seed=3,
        )
        cfg = MiningConfig(lag=6, min_obs=15, min_lift=0.03)
        validated = self._mine(minute_close, minute_suspend, lag_bars=6, leader_threshold=0.01, cfg=cfg)

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
        minute_close, minute_suspend, pairs = make_synthetic_threshold_market(
            n_days=500, bars_per_day=48, n_stocks=40, n_pairs=0, lag_bars=6, threshold=0.01, seed=4,
        )
        self.assertEqual(pairs, [])
        cfg = MiningConfig(lag=6, min_obs=15, min_lift=0.03)
        validated = self._mine(minute_close, minute_suspend, lag_bars=6, leader_threshold=0.01, cfg=cfg, top_n=40)

        naive_survivors = (validated["oos_lift"] > 0).sum()
        oos_significant = filter_oos_significant(validated, alpha=0.1, min_oos_n=5)
        self.assertLessEqual(len(oos_significant), max(naive_survivors // 2, 2))


class TestEtfBacktest(unittest.TestCase):
    def test_backtest_runs_and_reflects_the_injected_signal(self):
        minute_close, minute_suspend, pairs = make_synthetic_threshold_market(
            n_days=500, bars_per_day=48, n_stocks=40, n_pairs=6, lag_bars=6, threshold=0.01,
            trigger_prob=0.1, flip_prob=0.6, boost=0.025, seed=3,
        )
        leader_triggered, leader_valid = build_leader_frames_threshold(minute_close, minute_suspend, threshold=0.01)
        follower_outcome, follower_valid = build_follower_frames(
            minute_close, minute_suspend, lag_bars=6, mode="absolute"
        )

        cfg = MiningConfig(lag=6, min_obs=10, min_lift=0.02)
        split = int(len(minute_close) * 0.6)
        train, test = slice(0, split), slice(split, None)
        stats = compute_pairwise_stats_same_row(
            leader_triggered.iloc[train], leader_valid.iloc[train],
            follower_outcome.iloc[train], follower_valid.iloc[train], cfg, sector_map=None,
        )
        candidates = select_top_n_candidates(stats, cfg, top_n=30)
        validated = validate_out_of_sample_same_row(
            leader_triggered.iloc[test], leader_valid.iloc[test],
            follower_outcome.iloc[test], follower_valid.iloc[test], candidates, cfg,
        )
        oos_significant = filter_oos_significant(validated, alpha=0.15, min_oos_n=3)
        pairs_df = select_for_deployment(oos_significant, max_unique_symbols=500, min_oos_n=3)
        self.assertGreater(len(pairs_df), 0, "synthetic ETF threshold signal should yield a deployable pair")

        score = build_intraday_scores(leader_triggered, pairs_df, weight_col="oos_z")
        bt_cfg = IntradayBacktestConfig(lag_bars=6, top_k=5, initial_capital=1_000_000.0)
        equity, trades = simulate_intraday_portfolio(score, minute_close, bt_cfg)

        self.assertEqual(len(equity), len(score))
        self.assertTrue((equity > 0).all())
        self.assertGreater(len(trades), 0)

        summary = performance_summary(equity, periods_per_year=252 * 48)
        self.assertIn("sharpe", summary)
        self.assertGreater(summary["sharpe"], 0)


class TestIntradayTurnoverRanking(unittest.TestCase):
    """The ETF pipeline must stay purely minute-level: its universe selection ranks by
    turnover summed from the SAME minute bars it mines on, never from daily bars (which
    xtdata caches separately - ranking off 1d silently returned nothing for a fixed ETF
    list whose 5m history was fully downloaded).
    """

    def _stub_xtdata(self, amount_by_code, bars_per_day=2, n_days=3):
        import sys
        import types
        from unittest import mock

        periods_seen = []

        def get_market_data_ex(fields, stock_list, period="5m", start_time="", end_time="", **kwargs):
            periods_seen.append(period)
            stamps = [
                f"2024010{d + 1}10{b:02d}00"
                for d in range(n_days) for b in range(bars_per_day)
            ]
            return {
                code: pd.DataFrame({"amount": values}, index=stamps)
                for code, values in amount_by_code.items() if code in set(stock_list)
            }

        def download_history_data(code, period, start_time="", end_time=""):
            periods_seen.append(period)

        module = types.ModuleType("xtquant")
        module.xtdata = types.SimpleNamespace(
            get_market_data_ex=get_market_data_ex, download_history_data=download_history_data,
        )
        return mock.patch.dict(sys.modules, {"xtquant": module}), periods_seen

    def test_ranks_on_minute_bars_only_never_daily(self):
        from intraday.data import rank_by_intraday_turnover_xtdata

        # 3 days x 2 bars; per-day turnover is the SUM of that day's bars
        patcher, periods_seen = self._stub_xtdata({
            "BIG.SH": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],   # 200/day
            "MID.SZ": [10.0, 40.0, 10.0, 40.0, 10.0, 40.0],         # 50/day
            "SMALL.SH": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],             # 2/day
        })
        with patcher:
            ranked = rank_by_intraday_turnover_xtdata(
                ["BIG.SH", "MID.SZ", "SMALL.SH"], period="5m", top_n=2, download=True
            )
        self.assertEqual(ranked, ["BIG.SH", "MID.SZ"])
        self.assertTrue(periods_seen)
        self.assertNotIn("1d", periods_seen, f"daily bars must never be touched, saw {periods_seen}")

    def test_empty_minute_cache_raises_an_actionable_error(self):
        from intraday.data import rank_by_intraday_turnover_xtdata

        patcher, _ = self._stub_xtdata({})
        with patcher, self.assertRaises(ValueError) as ctx:
            rank_by_intraday_turnover_xtdata(["A.SH"], period="5m", top_n=1, download=False)
        self.assertIn("--top-liquid", str(ctx.exception))


class TestEtfParamSweep(unittest.TestCase):
    def test_run_one_combo_recovers_pairs_at_the_matching_lag(self):
        from types import SimpleNamespace

        from research.run_etf_param_sweep import run_one_combo

        minute_close, minute_suspend, pairs = make_synthetic_threshold_market(
            n_days=500, bars_per_day=48, n_stocks=40, n_pairs=6, lag_bars=6, threshold=0.01,
            trigger_prob=0.1, flip_prob=0.6, boost=0.025, seed=3,
        )
        args = SimpleNamespace(
            mode="absolute", threshold=0.0, min_obs=15, candidate_mode="top-n", top_n=40,
            alpha=0.01, min_lift=0.03, oos_alpha=0.1, min_oos_lift=0.0, train_frac=0.7,
            max_symbols=500, min_oos_n=5, max_leaders_per_follower=3,
        )
        summary, deployable, split = run_one_combo(
            minute_close, minute_suspend, leader_threshold=0.01, lag_bars=6, args=args
        )
        self.assertEqual(summary["leader_threshold"], 0.01)
        self.assertEqual(summary["lag_bars"], 6)
        self.assertGreater(summary["n_deployable"], 0)
        found = set(zip(deployable["leader"], deployable["follower"]))
        self.assertGreater(len(found & set(pairs)), 0, f"expected to recover some of {set(pairs)}, got {found}")

    def test_max_leaders_per_follower_zero_disables_hub_filter_in_the_pipeline(self):
        from types import SimpleNamespace

        from research.run_etf_param_sweep import run_one_combo

        minute_close, minute_suspend, pairs = make_synthetic_threshold_market(
            n_days=500, bars_per_day=48, n_stocks=40, n_pairs=6, lag_bars=6, threshold=0.01,
            trigger_prob=0.1, flip_prob=0.6, boost=0.025, seed=3,
        )
        kwargs = dict(
            mode="absolute", threshold=0.0, min_obs=15, candidate_mode="top-n", top_n=40,
            alpha=0.01, min_lift=0.03, oos_alpha=0.1, min_oos_lift=0.0, train_frac=0.7,
            max_symbols=500, min_oos_n=5,
        )
        summary_filtered, _, _ = run_one_combo(
            minute_close, minute_suspend, leader_threshold=0.01, lag_bars=6,
            args=SimpleNamespace(max_leaders_per_follower=1, **kwargs),
        )
        summary_unfiltered, _, _ = run_one_combo(
            minute_close, minute_suspend, leader_threshold=0.01, lag_bars=6,
            args=SimpleNamespace(max_leaders_per_follower=0, **kwargs),
        )
        # an aggressive cap (1) can only ever drop pairs relative to no filter at all
        self.assertLessEqual(summary_filtered["n_deployable"], summary_unfiltered["n_deployable"])


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


class TestDataCoverageSummary(unittest.TestCase):
    """Pure summariser behind research/check_data_coverage.py - the pre-flight check for
    whether a --no-download run can actually see anything."""

    def _panel(self, cols):
        idx = pd.date_range("2026-01-05 09:35", periods=4, freq="5min")
        return pd.DataFrame(cols, index=idx)

    def test_reports_range_and_counts_for_cached_symbols(self):
        from research.check_data_coverage import summarize_coverage

        out = summarize_coverage(self._panel({
            "A.SH": [1.0, 2.0, 3.0, 4.0],
            "B.SH": [1.0, None, 3.0, None],
        }))
        self.assertEqual(out["with_data"], 2)
        self.assertEqual(out["median_bars"], 3)
        self.assertEqual(out["first_bar"], "2026-01-05 09:35:00")
        self.assertEqual(out["last_bar"], "2026-01-05 09:50:00")

    def test_symbols_with_no_cached_bars_are_not_counted(self):
        from research.check_data_coverage import summarize_coverage

        out = summarize_coverage(self._panel({
            "HAS.SH": [1.0, 2.0, None, None],
            "EMPTY.SH": [None, None, None, None],
        }))
        self.assertEqual(out["with_data"], 1)
        self.assertEqual(out["last_bar"], "2026-01-05 09:40:00")

    def test_completely_empty_cache_reports_nothing(self):
        from research.check_data_coverage import summarize_coverage

        for panel in (None, pd.DataFrame(), self._panel({"X.SH": [None]*4})):
            out = summarize_coverage(panel)
            self.assertEqual(out["with_data"], 0)
            self.assertIsNone(out["first_bar"])


class TestSplitResolution(unittest.TestCase):
    """--split-date pins the train/test boundary to a real date instead of a bar
    fraction, so two runs over different fetched ranges stay comparable."""

    def _args(self, **kw):
        from types import SimpleNamespace
        return SimpleNamespace(train_frac=0.7, split_date=None, **kw)

    def _index(self):
        # 4 trading days x 3 bars
        return pd.DatetimeIndex([
            pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=5 * b)
            for d in ("2026-01-02", "2026-01-05", "2026-07-01", "2026-07-02") for b in range(3)
        ])

    def test_splits_at_the_first_bar_on_or_after_the_date(self):
        from research.run_intraday_mining import resolve_split

        idx = self._index()
        args = self._args()
        args.split_date = "20260701"
        split = resolve_split(idx, args)
        self.assertEqual(split, 6)
        self.assertEqual(str(idx[split - 1].date()), "2026-01-05")  # last train bar
        self.assertEqual(str(idx[split].date()), "2026-07-01")      # first test bar

    def test_falls_back_to_train_frac_without_a_date(self):
        from research.run_intraday_mining import resolve_split

        self.assertEqual(resolve_split(self._index(), self._args()), 8)

    def test_a_date_outside_the_data_is_an_error_not_an_empty_side(self):
        from research.run_intraday_mining import resolve_split

        for bad in ("20200101", "20990101"):
            args = self._args()
            args.split_date = bad
            with self.assertRaises(SystemExit):
                resolve_split(self._index(), args)
