"""Performance metrics for an equity-curve series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def performance_summary(equity: pd.Series, periods_per_year: int = 252) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = len(rets) / periods_per_year
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    ann_vol = rets.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (rets.mean() * periods_per_year) / ann_vol if ann_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "win_rate": float((rets > 0).mean()),
        "n_days": int(len(rets)),
    }
