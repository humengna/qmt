"""Dependency-free statistics helpers (numpy only, no scipy).

Kept free of scipy so this module can be copy-pasted or imported unmodified
inside QMT's embedded Python interpreter, which does not ship scipy.
"""
from __future__ import annotations

import numpy as np


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorized erf approximation (Abramowitz & Stegun 7.1.26, |err| <= 1.5e-7)."""
    sign = np.sign(x)
    x = np.abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def normal_sf(z: np.ndarray) -> np.ndarray:
    """Upper-tail probability P(Z > z) for a standard normal Z."""
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * (1.0 - _erf(z / np.sqrt(2.0)))


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of which p-values are significant under BH false-discovery-rate control.

    With N stocks there are up to N^2 candidate pairs, so testing each one at a flat
    p < 0.05 would produce huge numbers of spurious "significant" pairs by chance alone.
    BH-FDR bounds the expected fraction of false discoveries among the pairs kept.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(n, dtype=bool)
    cutoff = ranked[np.nonzero(passed)[0].max()]
    return p <= cutoff
