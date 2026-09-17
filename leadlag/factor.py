"""Lead-lag correlation factor mining.

Core idea being tested: does stock A closing up today raise the probability that
stock B closes up on some later day (typically the next trading day), beyond B's
own baseline propensity to rise?

The mining is done pairwise across an entire universe using matrix multiplication
(instead of an O(N^2) Python loop over pairs) so it scales to a few thousand
symbols, then guarded by two independent defenses against false discoveries:

1. Benjamini-Hochberg FDR control on the in-sample p-values (`mine_lead_lag_pairs`).
   With N stocks there are ~N^2 ordered pairs tested at once; FDR control is
   necessary because a flat p < 0.05 threshold would pass a huge number of pairs
   by pure chance.
2. Out-of-sample validation on a held-out time window (`validate_out_of_sample`).
   FDR control alone still permits relationships that are an artifact of one
   particular sample window (regime-specific co-movement, a shared one-off event,
   survivorship in the universe, etc.). A pair is only trustworthy if the same
   effect, with the same sign, shows up again on data the mining step never saw.

Both stages must pass before a pair is considered for `select_for_deployment`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import benjamini_hochberg, normal_sf


def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close simple returns, one row per trading day, one column per stock."""
    return close.sort_index().pct_change()


def compute_up_indicator(
    returns: pd.DataFrame, mode: str = "excess", threshold: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn a return panel into a boolean "was this stock up" panel.

    mode='absolute': up iff the raw return exceeds `threshold`.
    mode='excess'  : up iff the return, net of that day's cross-sectional median
                     return, exceeds `threshold`. This is the recommended default:
                     on a day the whole market rallies, most stocks close up
                     together for reasons that have nothing to do with any specific
                     leader-follower relationship, and 'absolute' mode would count
                     every such pair as a co-occurrence. Netting out the common
                     market move isolates relative (idiosyncratic) lead-lag effects.

    Returns (up, valid); `valid` marks days with a real (non-NaN) return, e.g. not
    a newly-listed or suspended stock, so the denominators used later don't silently
    treat "no data" as "did not go up".
    """
    valid = returns.notna()
    if mode == "absolute":
        signal = returns
    elif mode == "excess":
        market = returns.median(axis=1)
        signal = returns.sub(market, axis=0)
    else:
        raise ValueError(f"unknown mode: {mode!r}, expected 'absolute' or 'excess'")
    up = (signal > threshold) & valid
    return up, valid


@dataclass
class MiningConfig:
    lag: int = 1               # follower is checked `lag` trading days after the leader
    min_obs: int = 60          # minimum sample size in EACH branch (leader-up / leader-not-up)
    alpha: float = 0.01        # BH-FDR level applied to the in-sample candidate pairs
    min_lift: float = 0.05     # minimum P(follower up | leader up) - P(follower up | leader not up)
    exclude_same: bool = True  # drop the meaningless leader == follower pair (diagonal)


_PAIR_COLUMNS = [
    "leader", "follower", "n_leader_up", "n_leader_flat",
    "p_cond", "p_base", "lift", "z", "p_value",
]


def _empty_pairs_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_PAIR_COLUMNS)


def _two_proportion_matrix_stats(
    L: np.ndarray, VL: np.ndarray, F: np.ndarray, VF: np.ndarray,
    leader_codes: np.ndarray, follower_codes: np.ndarray,
    cfg: MiningConfig, sector_map: dict[str, str] | None,
) -> pd.DataFrame:
    """Shared core: two-proportion z-test for every (leader, follower) pair, given
    ALREADY-ALIGNED 0/1 matrices (leader triggered/valid, follower outcome/valid - same
    row `t` in both means "compare-able", however that alignment was produced upstream).

    Used by both `compute_pairwise_stats_asymmetric` (which shifts by `cfg.lag` rows
    before calling this) and `compute_pairwise_stats_same_row` (whose caller has
    already baked the lag into a forward-looking `follower_outcome`, e.g. intraday's
    "did price rise over the next N bars", so no further shift is needed or correct -
    a lag-based `.iloc[:-lag]` slice on an already-aligned frame is not just redundant,
    `arr[:-0]` for a same-row lag of 0 would evaluate to an EMPTY slice in numpy/pandas,
    silently discarding all data).
    """
    n1 = L.T @ VF   # count: leader triggered & follower data valid
    x1 = L.T @ F    # count: leader triggered & follower outcome
    n0 = VL.T @ VF - n1   # count: leader valid-not-triggered & follower data valid
    x0 = VL.T @ F - x1    # count: leader valid-not-triggered & follower outcome

    if cfg.exclude_same:
        common = set(leader_codes) & set(follower_codes)
        if common:
            li_idx = {c: i for i, c in enumerate(leader_codes)}
            fi_idx = {c: i for i, c in enumerate(follower_codes)}
            for c in common:
                n1[li_idx[c], fi_idx[c]] = 0
                n0[li_idx[c], fi_idx[c]] = 0

    same_sector = None
    if sector_map is not None:
        leader_sectors = np.array([sector_map.get(c) for c in leader_codes], dtype=object)
        follower_sectors = np.array([sector_map.get(c) for c in follower_codes], dtype=object)
        has_l = leader_sectors != None  # noqa: E711
        has_f = follower_sectors != None  # noqa: E711
        same_sector = has_l[:, None] & has_f[None, :] & (leader_sectors[:, None] == follower_sectors[None, :])

    with np.errstate(divide="ignore", invalid="ignore"):
        p1 = np.divide(x1, n1, out=np.full_like(x1, np.nan), where=n1 > 0)
        p0 = np.divide(x0, n0, out=np.full_like(x0, np.nan), where=n0 > 0)
        pooled = np.divide(x1 + x0, n1 + n0, out=np.full_like(x1, np.nan), where=(n1 + n0) > 0)
        se = np.sqrt(pooled * (1 - pooled) * (1.0 / np.where(n1 > 0, n1, np.nan)
                                               + 1.0 / np.where(n0 > 0, n0, np.nan)))
        z = np.divide(p1 - p0, se, out=np.full_like(x1, np.nan), where=se > 0)

    lift = p1 - p0
    pvalue = normal_sf(z)  # one-sided: H1 = "leader trigger raises the follower's odds"

    mask = (n1 >= cfg.min_obs) & (n0 >= cfg.min_obs) & np.isfinite(z)
    if same_sector is not None:
        mask = mask & same_sector
    li, fi = np.nonzero(mask)
    if li.size == 0:
        return _empty_pairs_frame()

    return pd.DataFrame({
        "leader": leader_codes[li],
        "follower": follower_codes[fi],
        "n_leader_up": n1[li, fi].astype(int),
        "n_leader_flat": n0[li, fi].astype(int),
        "p_cond": p1[li, fi],
        "p_base": p0[li, fi],
        "lift": lift[li, fi],
        "z": z[li, fi],
        "p_value": pvalue[li, fi],
    })


def compute_pairwise_stats_asymmetric(
    leader_up: pd.DataFrame, leader_valid: pd.DataFrame,
    follower_up: pd.DataFrame, follower_valid: pd.DataFrame,
    cfg: MiningConfig,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Two-proportion z-test for every ordered (leader, follower) pair, with the leader's
    "triggered" event and the follower's "outcome" event allowed to be DIFFERENT signals,
    the follower's outcome checked `cfg.lag` ROWS after the leader's (e.g. lag=1 trading
    day for daily bars - see `compute_pairwise_stats_same_row` for the case where the
    follower's outcome is already a forward-looking quantity computed at the SAME row).

    `compute_pairwise_stats(up, valid, cfg)` is the common case of this where the same
    "up" definition is used on both sides. The asymmetric form exists for hypotheses
    like limitup's "leader closed AT LIMIT-UP" (a much rarer, more specific event than
    plain "up") predicting "follower is up" - see limitup/event.py. `leader_up.columns`
    and `follower_up.columns` may be different universes; only codes present on both
    sides can ever pair up.

    For each pair, compares:
      group 1: days the leader was triggered              -> P(follower outcome `cfg.lag` days later)
      group 0: days the leader was valid but not triggered -> P(follower outcome `cfg.lag` days later)
    using the standard pooled two-proportion z-test, computed for ALL pairs at once via
    a handful of (N x T) @ (T x N) matrix multiplications rather than a per-pair loop.

    Unlike `mine_lead_lag_pairs`, this only applies the `min_obs` sample-size floor -
    it does NOT filter on `min_lift` or run FDR control. Use it directly (e.g. from a
    notebook, or the `--diagnostics` output of research/run_mining.py) to inspect the
    raw lift/z distribution across the whole universe before deciding whether
    `min_lift`/`alpha` are set sensibly for the data at hand.

    `sector_map`, if given, restricts consideration to pairs whose leader and follower
    map to the SAME sector/industry (e.g. `{"600000.SH": "银行", "000001.SZ": "银行", ...}`,
    typically built by `leadlag.data.build_sector_map_xtdata`). A stock missing from the
    map is treated as belonging to no sector and can't form a pair with anything. This
    both targets a more economically plausible hypothesis (co-movement within a sector,
    not any two arbitrary stocks) and shrinks the multiple-testing family a lot, which
    is often the difference between everything failing FDR/OOS control and nothing
    doing so - see README's mining-diagnostics discussion.
    """
    lag = cfg.lag
    leader_codes = leader_up.columns.to_numpy()
    follower_codes = follower_up.columns.to_numpy()
    if len(leader_codes) < 1 or len(follower_codes) < 1 or len(leader_up) <= lag:
        return _empty_pairs_frame()

    L = leader_up.iloc[:-lag].to_numpy(dtype=np.float64)         # leader triggered, day t
    VL = leader_valid.iloc[:-lag].to_numpy(dtype=np.float64)     # leader valid,      day t
    F = follower_up.iloc[lag:].to_numpy(dtype=np.float64)        # follower outcome,  day t + lag
    VF = follower_valid.iloc[lag:].to_numpy(dtype=np.float64)    # follower valid,    day t + lag

    return _two_proportion_matrix_stats(L, VL, F, VF, leader_codes, follower_codes, cfg, sector_map)


def compute_pairwise_stats_same_row(
    leader_triggered: pd.DataFrame, leader_valid: pd.DataFrame,
    follower_outcome: pd.DataFrame, follower_valid: pd.DataFrame,
    cfg: MiningConfig,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Two-proportion z-test for every ordered (leader, follower) pair, when the
    follower's "outcome" is already a forward-looking quantity computed at the SAME
    row as the leader's trigger - e.g. intraday/event.py's "did the follower's price
    rise from bar t to bar t + N, without crossing into the next trading day", which
    is indexed by the trigger bar `t` itself, not by `t + N`.

    Do NOT use `compute_pairwise_stats_asymmetric` for this: it additionally shifts by
    `cfg.lag` rows, which would either double-apply the lag (if `cfg.lag` is set to the
    same N already baked into `follower_outcome`) or, if you tried to pass `cfg.lag=0`
    to opt out of the shift, silently return nothing at all (`arr[:-0]` is `arr[:0]`,
    i.e. empty, in numpy/pandas - not "no slicing").

    Otherwise identical to `compute_pairwise_stats_asymmetric` - see its docstring for
    `sector_map` and the min_obs-only, no-FDR filtering this applies.
    """
    leader_codes = leader_triggered.columns.to_numpy()
    follower_codes = follower_outcome.columns.to_numpy()
    if len(leader_codes) < 1 or len(follower_codes) < 1 or len(leader_triggered) == 0:
        return _empty_pairs_frame()

    L = leader_triggered.to_numpy(dtype=np.float64)
    VL = leader_valid.to_numpy(dtype=np.float64)
    F = follower_outcome.to_numpy(dtype=np.float64)
    VF = follower_valid.to_numpy(dtype=np.float64)

    return _two_proportion_matrix_stats(L, VL, F, VF, leader_codes, follower_codes, cfg, sector_map)


def compute_pairwise_stats(
    up: pd.DataFrame, valid: pd.DataFrame, cfg: MiningConfig,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Two-proportion z-test for every ordered (leader, follower) pair with enough data.

    For each pair, compares:
      group 1: days the leader was up            -> P(follower up `cfg.lag` days later)
      group 0: days the leader was not up (valid) -> P(follower up `cfg.lag` days later)
    using the standard pooled two-proportion z-test, computed for ALL pairs at once via
    a handful of (N x T) @ (T x N) matrix multiplications rather than a per-pair loop.
    Thin wrapper over `compute_pairwise_stats_asymmetric` for the common case where the
    leader and follower share the same "up" definition.
    """
    return compute_pairwise_stats_asymmetric(up, valid, up, valid, cfg, sector_map)


def filter_significant_pairs(stats: pd.DataFrame, cfg: MiningConfig) -> pd.DataFrame:
    """Narrow `compute_pairwise_stats` output down to economically-meaningful, FDR-significant pairs."""
    if stats.empty:
        return stats

    out = stats[stats["lift"] >= cfg.min_lift]
    if out.empty:
        return _empty_pairs_frame()

    # FDR control is applied within this already effect-size-filtered family (lift >=
    # min_lift), not the full N^2 family: we only ever cared about economically
    # meaningful positive lift, so multiplicity correction is scoped to that subset.
    sig = benjamini_hochberg(out["p_value"].to_numpy(), alpha=cfg.alpha)
    out = out[sig].sort_values("z", ascending=False).reset_index(drop=True)
    return out


def select_top_n_candidates(stats: pd.DataFrame, cfg: MiningConfig, top_n: int) -> pd.DataFrame:
    """Alternative to `filter_significant_pairs`: skip FDR, take the `top_n` pairs by z-score.

    FDR control is the right tool when the in-sample screen IS the final answer. Here it
    isn't - every candidate this returns must still independently clear
    `validate_out_of_sample` before ever being deployed. When the universe is large
    enough that testing a few thousand to a few million pairs at once makes alpha=0.01
    reject everything (routine at whole-market scale: see docs/QMT_API_NOTES.md and the
    README's mining-diagnostics discussion), rank-based screening plus a hard
    out-of-sample re-check is a more practical way to generate candidates worth testing.
    The trade-off: the CANDIDATE list here has a much higher false-discovery rate than
    `filter_significant_pairs` would allow - that's expected, and is exactly what
    `validate_out_of_sample` exists to filter back down.
    """
    if stats.empty:
        return stats
    out = stats[stats["lift"] >= cfg.min_lift]
    if out.empty:
        return _empty_pairs_frame()
    return out.sort_values("z", ascending=False).head(top_n).reset_index(drop=True)


def mine_lead_lag_pairs(
    up: pd.DataFrame, valid: pd.DataFrame, cfg: MiningConfig,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute pairwise stats and filter down to economically-meaningful, FDR-significant pairs.

    Equivalent to `filter_significant_pairs(compute_pairwise_stats(up, valid, cfg, sector_map), cfg)`;
    kept as a single call for convenience when you don't need the unfiltered stats too
    (e.g. research/run_mining.py calls the two halves separately so it can print
    diagnostics on `stats` before filtering).
    """
    return filter_significant_pairs(compute_pairwise_stats(up, valid, cfg, sector_map), cfg)


_OOS_COLS = ["oos_p_cond", "oos_p_base", "oos_lift", "oos_z", "oos_p_value", "oos_n"]


def _two_proportion_pair_stat(l: np.ndarray, vl: np.ndarray, f: np.ndarray, vf: np.ndarray) -> tuple:
    """Shared per-pair core for the OOS validators, given already-aligned 0/1 arrays
    (same alignment convention as `_two_proportion_matrix_stats`, just for one pair
    at a time via a plain loop rather than a matrix multiply - see the OOS validators'
    docstrings for why this recomputation is only ever run over a small candidate list).
    """
    d = vl - l
    n1, x1 = float((l * vf).sum()), float((l * f).sum())
    n0, x0 = float((d * vf).sum()), float((d * f).sum())
    if n1 <= 0 or n0 <= 0:
        return (np.nan,) * 6

    p1, p0 = x1 / n1, x0 / n0
    pooled = (x1 + x0) / (n1 + n0)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0)) if 0 < pooled < 1 else np.nan
    z = (p1 - p0) / se if se and se > 0 else np.nan
    pvalue = float(normal_sf(z)) if np.isfinite(z) else np.nan
    return (p1, p0, p1 - p0, z, pvalue, n1)


def validate_out_of_sample_asymmetric(
    leader_up: pd.DataFrame, leader_valid: pd.DataFrame,
    follower_up: pd.DataFrame, follower_valid: pd.DataFrame,
    candidates: pd.DataFrame, cfg: MiningConfig,
) -> pd.DataFrame:
    """Recompute the same leader/follower statistics on a held-out window, shifting the
    follower's outcome by `cfg.lag` rows after the leader's - see
    `validate_out_of_sample_same_row` for when the follower's outcome is already a
    forward-looking quantity computed at the same row as the leader.

    Asymmetric counterpart to `validate_out_of_sample`, for when the leader's trigger
    and the follower's outcome are different signals (see
    `compute_pairwise_stats_asymmetric`, limitup/event.py).

    This is the main defense against data-snooping: with N^2 pairs tested in-sample,
    FDR control bounds the false-discovery rate but does not guarantee any single
    surviving pair reflects a real, persistent relationship rather than a one-window
    artifact. Only candidates are recomputed (cheap, a plain Python loop), since by
    this point the candidate list is small.
    """
    if candidates.empty:
        return candidates.assign(**{c: pd.Series(dtype=float) for c in _OOS_COLS})

    lag = cfg.lag
    up_t, valid_t = leader_up.iloc[:-lag], leader_valid.iloc[:-lag]
    up_f, valid_f = follower_up.iloc[lag:], follower_valid.iloc[lag:]

    records = []
    for row in candidates.itertuples(index=False):
        leader, follower = row.leader, row.follower
        if leader not in up_t.columns or follower not in up_f.columns:
            records.append((np.nan,) * 6)
            continue
        records.append(_two_proportion_pair_stat(
            up_t[leader].to_numpy(dtype=np.float64), valid_t[leader].to_numpy(dtype=np.float64),
            up_f[follower].to_numpy(dtype=np.float64), valid_f[follower].to_numpy(dtype=np.float64),
        ))

    oos = pd.DataFrame(records, columns=_OOS_COLS)
    return pd.concat([candidates.reset_index(drop=True), oos], axis=1)


def validate_out_of_sample_same_row(
    leader_triggered: pd.DataFrame, leader_valid: pd.DataFrame,
    follower_outcome: pd.DataFrame, follower_valid: pd.DataFrame,
    candidates: pd.DataFrame, cfg: MiningConfig,
) -> pd.DataFrame:
    """Recompute the same leader/follower statistics on a held-out window, for the
    same-row-aligned case (see `compute_pairwise_stats_same_row`'s docstring for why
    this is NOT the same as shifting by `cfg.lag` rows).
    """
    if candidates.empty:
        return candidates.assign(**{c: pd.Series(dtype=float) for c in _OOS_COLS})

    records = []
    for row in candidates.itertuples(index=False):
        leader, follower = row.leader, row.follower
        if leader not in leader_triggered.columns or follower not in follower_outcome.columns:
            records.append((np.nan,) * 6)
            continue
        records.append(_two_proportion_pair_stat(
            leader_triggered[leader].to_numpy(dtype=np.float64), leader_valid[leader].to_numpy(dtype=np.float64),
            follower_outcome[follower].to_numpy(dtype=np.float64), follower_valid[follower].to_numpy(dtype=np.float64),
        ))

    oos = pd.DataFrame(records, columns=_OOS_COLS)
    return pd.concat([candidates.reset_index(drop=True), oos], axis=1)


def validate_out_of_sample(
    up: pd.DataFrame, valid: pd.DataFrame, candidates: pd.DataFrame, cfg: MiningConfig
) -> pd.DataFrame:
    """Recompute the same leader/follower statistics on a held-out window.

    Thin wrapper over `validate_out_of_sample_asymmetric` for the common case where the
    leader and follower share the same "up" definition. See that function's docstring
    for why this step matters.
    """
    return validate_out_of_sample_asymmetric(up, valid, up, valid, candidates, cfg)


def filter_oos_significant(
    validated: pd.DataFrame, alpha: float = 0.05, min_oos_lift: float = 0.0, min_oos_n: int = 20,
) -> pd.DataFrame:
    """The real out-of-sample gate: BH-FDR control on `oos_p_value`, not just its sign.

    `validate_out_of_sample` reports `oos_lift` for every candidate, but "oos_lift > 0"
    alone is a very weak filter: under pure noise, about half of ANY candidate set keeps
    the same sign out of sample purely by chance, no matter how large the candidate set
    is. This matters a lot together with `select_top_n_candidates`/a loose `--alpha`,
    which can hand this function hundreds of thousands of largely-spurious candidates
    (this combination is exactly what turned up testing this pipeline against a real,
    whole-market run: ~50.7% of ~250K in-sample "candidates" kept a positive oos_lift,
    which is indistinguishable from a coin flip and should NOT be read as confirmation).
    This re-applies proper FDR control on the held-out window's own p-values, so
    "out-of-sample significant" means an actual statistical test again.
    """
    df = validated.dropna(subset=["oos_p_value"])
    df = df[(df["oos_lift"] >= min_oos_lift) & (df["oos_n"] >= min_oos_n)]
    if df.empty:
        return df
    sig = benjamini_hochberg(df["oos_p_value"].to_numpy(), alpha=alpha)
    return df[sig].sort_values("oos_z", ascending=False).reset_index(drop=True)


def select_for_deployment(
    validated: pd.DataFrame,
    max_unique_symbols: int = 500,
    min_oos_lift: float = 0.0,
    min_oos_n: int = 20,
) -> pd.DataFrame:
    """Trim an already out-of-sample-significant pair table to a live-tradable symbol budget.

    Call this AFTER `filter_oos_significant` - it does no significance filtering of its
    own beyond the `min_oos_lift`/`min_oos_n` sanity floors, it only enforces the symbol
    budget. QMT's live/simulated quote subscription
    (`ContextInfo.get_market_data_ex(subscribe=True)`) caps out at 500 symbols, so the
    pair table shipped to the live strategy must respect that budget. Greedily adds pairs
    ranked by out-of-sample z-score until the budget of unique (leader ∪ follower)
    symbols would be exceeded.
    """
    required = {"oos_lift", "oos_z", "oos_n"}
    if not required.issubset(validated.columns):
        raise ValueError(f"validated frame is missing columns: {required - set(validated.columns)}")

    df = validated[(validated["oos_lift"] > min_oos_lift) & (validated["oos_n"] >= min_oos_n)]
    df = df.sort_values("oos_z", ascending=False)

    kept_symbols: set[str] = set()
    keep_rows = []
    for row in df.itertuples(index=False):
        candidate_symbols = kept_symbols | {row.leader, row.follower}
        if len(candidate_symbols) > max_unique_symbols:
            continue
        kept_symbols = candidate_symbols
        keep_rows.append(row)

    return pd.DataFrame(keep_rows, columns=df.columns).reset_index(drop=True)


def exclude_hub_followers(pairs: pd.DataFrame, max_leaders_per_follower: int) -> pd.DataFrame:
    """Drop any follower that shows up paired with MORE than `max_leaders_per_follower`
    distinct leaders in `pairs`.

    A follower that is "significant" against a dozen+ unrelated leaders at once is a red
    flag for an unremoved common factor, not a dozen+ genuine pairwise relationships: it
    means that follower is simply high-beta to whatever shared factor moves the whole
    universe (any leader crossing its trigger threshold tends to coincide with days the
    shared factor is active), not specifically driven by any one leader. This showed up
    concretely running the T0-ETF pipeline (research/run_etf_mining.py) against real
    data on a small, highly homogeneous universe (85 cross-border/commodity/bond ETFs):
    a handful of followers were paired with 10+ different leaders each, and excess-mode's
    cross-sectional median (over just that same homogeneous 85-name universe) evidently
    didn't fully net out the shared factor those high-beta names ride on.

    `max_leaders_per_follower=0` disables this (returns `pairs` unchanged) - use it when
    a follower genuinely responding to several distinct leaders is plausible for the
    universe at hand (e.g. a large, diverse full-market universe where a handful of
    independent sector leaders driving one liquid bellwether follower isn't suspicious
    the way it is in a small, homogeneous universe).

    This is a coarse, univariate safeguard - it doesn't attempt to model or remove the
    common factor itself (see `compute_up_indicator`'s `mode='excess'` for that), just
    refuses to deploy a pair where "how many other leaders also point at this follower"
    already contradicts the story of a pairwise relationship.
    """
    if max_leaders_per_follower <= 0 or pairs.empty:
        return pairs
    counts = pairs.groupby("follower")["leader"].nunique()
    hub_followers = set(counts[counts > max_leaders_per_follower].index)
    if not hub_followers:
        return pairs
    return pairs[~pairs["follower"].isin(hub_followers)].reset_index(drop=True)
