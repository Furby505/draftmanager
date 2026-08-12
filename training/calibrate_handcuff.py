"""Empirically calibrate the handcuff transfer fractions used by outcome_model.

For each (season, team, position), the season-long #1 by total DK points is the
"starter" and the #2 is the "backup". We compare the backup's per-game output in
weeks the starter PLAYED vs weeks the starter was OUT (no row that week), and the
starter's output when in. The transfer fraction is the share of the starter-vs-
backup gap the backup captures when the starter is out:

    frac = (B_out - B_in) / (S_in - B_in)

This is exactly the quantity outcome_model.apply_workload_transfer applies
(eff_anchor = own + frac*(starter_anchor - own)). Pools week-level observations
across many team-seasons, per position.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import recalc_weekly_df  # noqa: E402

WEEKLY = Path(__file__).resolve().parent.parent / "data" / "raw" / "player_stats_weekly.csv"
MIN_SEASON = 2015            # recency window
POSITIONS = ("RB", "QB", "WR", "TE")
STARTER_MIN_GAMES = 8        # the #1 must look like a real starter
BACKUP_MIN_GAMES = 3         # the #2 must have a usable sample
PTS = "dk_pts"


def main():
    df = pd.read_csv(WEEKLY, low_memory=False)
    if "season_type" in df.columns:
        df = df[df["season_type"].fillna("REG") == "REG"]
    df = df[(df["season"] >= MIN_SEASON) & df["position"].isin(POSITIONS)].copy()
    df = recalc_weekly_df(df, platform="draftkings")
    df[PTS] = df["fantasy_points_ppr"] if "dk_pts" not in df else df[PTS]
    # recalc_weekly_df rewrites fantasy_points_ppr to DK points; use that column.
    df[PTS] = df["fantasy_points_ppr"]

    rows = {pos: {"B_in": [], "B_out": [], "S_in": []} for pos in POSITIONS}

    for (season, team, pos), g in df.groupby(["season", "recent_team", "position"]):
        if pos not in POSITIONS:
            continue
        totals = g.groupby("player_id")[PTS].sum().sort_values(ascending=False)
        if len(totals) < 2:
            continue
        starter_id, backup_id = totals.index[0], totals.index[1]
        s = g[g["player_id"] == starter_id]
        b = g[g["player_id"] == backup_id]
        if len(s) < STARTER_MIN_GAMES or len(b) < BACKUP_MIN_GAMES:
            continue
        starter_weeks = set(s["week"].tolist())
        for _, r in b.iterrows():
            (rows[pos]["B_in"] if r["week"] in starter_weeks else rows[pos]["B_out"]).append(r[PTS])
        for _, r in s.iterrows():
            rows[pos]["S_in"].append(r[PTS])

    print(f"Handcuff transfer calibration (REG {MIN_SEASON}-2025, DK scoring)\n")
    print(f"{'pos':<4}{'B_in':>8}{'B_out':>8}{'S_in':>8}{'frac':>8}{'n_out':>8}")
    result = {}
    for pos in POSITIONS:
        d = rows[pos]
        if not d["B_out"] or not d["S_in"]:
            print(f"{pos:<4}{'-':>8}{'-':>8}{'-':>8}{'-':>8}{0:>8}")
            continue
        b_in = float(np.mean(d["B_in"])) if d["B_in"] else 0.0
        b_out = float(np.mean(d["B_out"]))
        s_in = float(np.mean(d["S_in"]))
        denom = s_in - b_in
        frac = (b_out - b_in) / denom if denom > 1e-6 else 0.0
        frac = float(np.clip(frac, 0.0, 1.0))
        result[pos] = round(frac, 2)
        print(f"{pos:<4}{b_in:>8.2f}{b_out:>8.2f}{s_in:>8.2f}{frac:>8.2f}{len(d['B_out']):>8}")

    print("\nHANDCUFF_TRANSFER =", result)


if __name__ == "__main__":
    main()
