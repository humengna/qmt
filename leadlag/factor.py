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
    codes = up.columns.to_numpy()
    n = len(codes)
    if n < 2 or len(up) <= lag:
        return _empty_pairs_frame()

    L = up.iloc[:-lag].to_numpy(dtype=np.float64)       # leader up,   day t
    VL = valid.iloc[:-lag].to_numpy(dtype=np.float64)   # leader valid, day t
    F = up.iloc[lag:].to_numpy(dtype=np.float64)        # follower up, day t + lag
    VF = valid.iloc[lag:].to_numpy(dtype=np.float64)    # follower valid, day t + lag
    D = VL - L                                          # leader valid but NOT up, day t

    n1 = L.T @ VF   # count: leader up & follower data valid
    x1 = L.T @ F    # count: leader up & follower up
    n0 = D.T @ VF   # count: leader flat/down & follower data valid
    x0 = D.T @ F    # count: leader flat/down & follower up

    if cfg.exclude_same:
        np.fill_diagonal(n1, 0)
        np.fill_diagonal(n0, 0)

    same_sector = None
    if sector_map is not None:
        sectors = np.array([sector_map.get(c) for c in codes], dtype=object)
        has_sector = sectors != None  # noqa: E711 - vectorized None-check, not identity misuse
        same_sector = has_sector[:, None] & has_sector[None, :] & (sectors[:, None] == sectors[None, :])

    with np.errstate(divide="ignore", invalid="ignore"):
        p1 = np.divide(x1, n1, out=np.full_like(x1, np.nan), where=n1 > 0)
        p0 = np.divide(x0, n0, out=np.full_like(x0, np.nan), where=n0 > 0)
        pooled = np.divide(x1 + x0, n1 + n0, out=np.full_like(x1, np.nan), where=(n1 + n0) > 0)
        se = np.sqrt(pooled * (1 - pooled) * (1.0 / np.where(n1 > 0, n1, np.nan)
                                               + 1.0 / np.where(n0 > 0, n0, np.nan)))
        z = np.divide(p1 - p0, se, out=np.full_like(x1, np.nan), where=se > 0)

    lift = p1 - p0
    pvalue = normal_sf(z)  # one-sided: H1 = "leader up raises the follower's odds"

    mask = (n1 >= cfg.min_obs) & (n0 >= cfg.min_obs) & np.isfinite(z)
    if same_sector is not None:
        mask = mask & same_sector
    li, fi = np.nonzero(mask)
    if li.size == 0:
        return _empty_pairs_frame()

    return pd.DataFrame({
        "leader": codes[li],
        "follower": codes[fi],
        "n_leader_up": n1[li, fi].astype(int),
        "n_leader_flat": n0[li, fi].astype(int),
        "p_cond": p1[li, fi],
        "p_base": p0[li, fi],
        "lift": lift[li, fi],
        "z": z[li, fi],
        "p_value": pvalue[li, fi],
    })


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


def validate_out_of_sample(
    up: pd.DataFrame, valid: pd.DataFrame, candidates: pd.DataFrame, cfg: MiningConfig
) -> pd.DataFrame:
    """Recompute the same leader/follower statistics on a held-out window.

    This is the main defense against data-snooping: with N^2 pairs tested in-sample,
    FDR control bounds the false-discovery rate but does not guarantee any single
    surviving pair reflects a real, persistent relationship rather than a one-window
    artifact. Only candidates are recomputed (cheap, a plain Python loop), since by
    this point the candidate list is small.
    """
    oos_cols = ["oos_p_cond", "oos_p_base", "oos_lift", "oos_z", "oos_p_value", "oos_n"]
    if candidates.empty:
        return candidates.assign(**{c: pd.Series(dtype=float) for c in oos_cols})

    lag = cfg.lag
    up_t, valid_t = up.iloc[:-lag], valid.iloc[:-lag]
    up_f, valid_f = up.iloc[lag:], valid.iloc[lag:]

    records = []
    for row in candidates.itertuples(index=False):
        leader, follower = row.leader, row.follower
        if leader not in up_t.columns or follower not in up_f.columns:
            records.append((np.nan,) * 6)
            continue
        l = up_t[leader].to_numpy(dtype=np.float64)
        vl = valid_t[leader].to_numpy(dtype=np.float64)
        f = up_f[follower].to_numpy(dtype=np.float64)
        vf = valid_f[follower].to_numpy(dtype=np.float64)
        d = vl - l

        n1, x1 = float((l * vf).sum()), float((l * f).sum())
        n0, x0 = float((d * vf).sum()), float((d * f).sum())
        if n1 <= 0 or n0 <= 0:
            records.append((np.nan,) * 6)
            continue

        p1, p0 = x1 / n1, x0 / n0
        pooled = (x1 + x0) / (n1 + n0)
        se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0)) if 0 < pooled < 1 else np.nan
        z = (p1 - p0) / se if se and se > 0 else np.nan
        pvalue = float(normal_sf(z)) if np.isfinite(z) else np.nan
        records.append((p1, p0, p1 - p0, z, pvalue, n1))

    oos = pd.DataFrame(records, columns=oos_cols)
    return pd.concat([candidates.reset_index(drop=True), oos], axis=1)


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
