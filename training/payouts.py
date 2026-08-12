"""
Payout tables for the bracket scorer (swappable), grounded in real DraftKings
Best Ball tournament structures.

A payout table maps FINAL-round placement (rank 1..pod) to prize equity. The
optimizer maximizes expected prize equity, so the *shape* of this curve is what
makes extreme right-tail outcomes valuable: a top-heavy curve pays far more for
1st than for 6th, which is exactly why "boom" teams are worth more than "solid"
teams in a tournament. Only the SHAPE matters for ranking rosters (argmax is
scale-invariant), so every curve here is normalized to sum 1.0 (equity units).

Real DK Best Ball tournament structure (verified 2026-06):
  - Round 1 (Weeks 1-14, cumulative): 12-team pods, TOP 2 ADVANCE.
  - Round 2 (Week 15, single week): top 1 of the pod advances.
  - Round 3 (Week 16, single week): top 1 advances.
  - Round 4 (Week 17, single week): championship final, top-heavy payout.

IMPORTANT — this is a representative PROXY, not the exact current table. The 2026
flagship is the "NFL Best Ball $20M Millionaire" with $3M to 1st, but DraftKings
does not publish the full per-place payout for it. So we build the default curve
from the most recent FULLY published flagship table (the $15M Millionaire top
places, same contest type / convexity) and expose `from_dollar_table` to drop in
any contest's exact numbers when available.

Convexity matters and varies by contest:
  - Flagship Millionaire ($15M/$20M, ~1,000+ finalists): FLATTER top — 1st/2nd
    ~2x ($2M -> $1M) because prize money spreads over a deep field. This is what
    most entrants actually play, so it is the DEFAULT.
  - High-stakes smaller Millionaire ($2.6M, $555 entry, 29 finalists): STEEPER —
    1st/2nd ~6.6x ($1M -> $150.75k). Kept as an alternative curve.

Only the SHAPE matters for ranking rosters (argmax is scale-invariant); the deep
sub-pod tail is truncated to the modeled 12-team final pod.

Sources:
  - DraftKings 2026 $20M Best Ball Millionaire ($3M to 1st) + NFL Best Ball page
  - 4for4 Best Ball overview (structure + flagship $15M top payouts)
  - DK Network $2.6M Best Ball Millionaire full prize breakdown
"""

from __future__ import annotations

import numpy as np

POD = 12


def _normalize(w: np.ndarray) -> np.ndarray:
    s = w.sum()
    return w / s if s > 0 else w


def from_dollar_table(dollars, pod: int = POD) -> np.ndarray:
    """Build a length-`pod` equity curve from a real $-by-finishing-place table.

    `dollars[k]` is the prize for finishing place k+1. The model's final round is
    a `pod`-team placement, so we take the top `pod` places and normalize to
    equity units (sum 1). Truncating the long tail at `pod` is deliberate: the
    deep tail is nearly flat and barely moves roster rank order, while the steep
    top (the 1st->2nd cliff) is the convexity that does."""
    arr = np.asarray(dollars, dtype=float)[:pod]
    if len(arr) < pod:
        arr = np.concatenate([arr, np.zeros(pod - len(arr))])
    return _normalize(arr)


def flat(pod: int = POD) -> np.ndarray:
    """Every finalist paid equally. Useful as a diagnostic baseline: under a
    flat curve, prize-EV is driven by mean/advance, not by the right tail."""
    return np.full(pod, 1.0 / pod)


def winner_take_all(pod: int = POD) -> np.ndarray:
    """Most convex curve — only 1st place pays. Maximal right-tail reward."""
    p = np.zeros(pod)
    p[0] = 1.0
    return p


def top_heavy(pod: int = POD, decay: float = 0.5) -> np.ndarray:
    """Geometric-decay top-heavy curve (synthetic diagnostic). decay<1 -> steeper."""
    w = decay ** np.arange(pod)
    return _normalize(w)


# Real published championship table — DK $2.6M Best Ball Millionaire ($555 entry,
# 5.2K entries, 29 finalists). Dollars by finishing place (1st..29th). This is the
# fully-specified reference; the flagship $15M ($20 entry, 1,021 finalists) has the
# same shape but spreads the tail over far more places (1st $2M, 2nd $1M, 3rd
# $715,600, 4th $500k, 5th $300k, ... $1,000 for 701st-1,021st).
DK_MILLIONAIRE_2P6M_DOLLARS = [
    1_000_000,                       # 1st
    150_750,                         # 2nd
    100_000,                         # 3rd
    70_000,                          # 4th
    60_000,                          # 5th
    50_000,                          # 6th
    40_000,                          # 7th
    30_000,                          # 8th
    25_000,                          # 9th
    20_000,                          # 10th
    15_000, 15_000,                  # 11th-12th
    12_000, 12_000, 12_000,          # 13th-15th
    *([10_000] * 14),                # 16th-29th
]

# Flagship Millionaire ($15M, ~1,021 finalists) published top places; the long tail
# decays to $1,000 for 701st-1,021st. The 2026 $20M flagship has the same shape with
# $3M to 1st (full per-place table unpublished). Scaling does not change the normalized
# shape, so these real $15M dollars define the flagship convexity used as the default.
DK_MILLIONAIRE_FLAGSHIP_TOP_DOLLARS = [
    2_000_000,   # 1st
    1_000_000,   # 2nd
    715_600,     # 3rd
    500_000,     # 4th
    300_000,     # 5th
]
DK_MILLIONAIRE_15M_TOP_DOLLARS = DK_MILLIONAIRE_FLAGSHIP_TOP_DOLLARS  # back-compat alias

# Default: the flagship Millionaire shape (what most entrants play), normalized over the
# modeled 12-team final pod. Flatter top than the $2.6M (1st/2nd ~2x vs ~6.6x); deep places
# beyond the published top-5 are truncated to the pod (their per-pod equity is ~0 anyway).
DK_BEST_BALL_MILLIONAIRE = from_dollar_table(DK_MILLIONAIRE_FLAGSHIP_TOP_DOLLARS, POD)
# Steeper high-stakes alternative ($2.6M / $555 / 29 finalists), fully published.
DK_BEST_BALL_MILLIONAIRE_STEEP = from_dollar_table(DK_MILLIONAIRE_2P6M_DOLLARS, POD)

# Back-compat alias for callers importing the old name.
TOP_HEAVY_PLACEHOLDER = DK_BEST_BALL_MILLIONAIRE

DEFAULT_PAYOUT = DK_BEST_BALL_MILLIONAIRE

assert abs(DEFAULT_PAYOUT.sum() - 1.0) < 1e-9
assert len(DEFAULT_PAYOUT) == POD


# ── Modeling note (read before trusting ABSOLUTE prize EV) ────────────────────
# The bracket scorer models the Week-17 championship as placement within a single
# 12-team final pod. The real championship ranks ~29-1,021 elite finalists, and
# prize money is also paid at the earlier rounds (R2/R3 cash, R4 floor). So the
# ABSOLUTE prize-EV here is a relative score, not real dollars. What this curve
# gets RIGHT is the convexity the optimizer needs: the flagship default pays 1st
# ~2x 2nd and ~3x 3rd, rewarding the extreme ceiling (boom teams) the way the real
# top-heavy championship does, without the over-steepness of the small-field $2.6M.
# To price absolute dollars, model the full finalist field + per-round cash with a
# conditional (survivor) opponent distribution and the exact contest payout table.
