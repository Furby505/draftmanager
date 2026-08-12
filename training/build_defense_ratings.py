"""Build per-team, per-position defense-vs-position ratings (multipliers ~1.0) and
project them to 2026, for the playoff-matchup layer.

rating(defense D, position P, season) =
    (avg DK points D allowed to position P per game) / (league avg allowed to P)

>1.0 = soft matchup (offense scores more vs D); <1.0 = tough. Centered at 1.0 per
position-season, so a mean-1 multiplier the sim can apply per week.

2026 projection: defenses are noisy year-to-year, so we DON'T trust last year's
rating outright. We MEASURE the lag-1 predictability (corr of a defense's rating in
year N vs N+1) per position and shrink toward 1.0 by that factor:
    rating_2026 = 1 + shrink_P * (rating_2025 - 1),  shrink_P = max(0, corr_lag1_P).
A position whose defense ratings barely persist year-to-year (low corr) collapses to
~1.0 (we can't predict it); a persistent one keeps more of its signal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import recalc_weekly_df  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEEKLY = ROOT / "data" / "raw" / "player_stats_weekly.csv"
OUT = ROOT / "data" / "processed" / "defense_vs_position_2026.json"
MIN_SEASON = 2015
POSITIONS = ("QB", "RB", "WR", "TE")
# 2025's weekly export has no opponent_team (all '0'), so the most-recent VALID
# actuals are 2024. Carry a 2-year weighted blend (2024 heavier) for stability.
CARRIER = {2024: 0.6, 2023: 0.4}
RATING_CLIP = (0.6, 1.6)       # don't let any matchup multiplier get extreme


def season_ratings(df: pd.DataFrame) -> dict:
    """{(season, team, pos): rating} centered at 1.0 within each (season, pos)."""
    # points a defense allowed to a position in a game = sum of that position's DK
    # points by the offense facing it that week.
    allowed = (df.groupby(["season", "week", "opponent_team", "position"])
                 ["fantasy_points_ppr"].sum().reset_index())
    per_def = (allowed.groupby(["season", "opponent_team", "position"])
                      ["fantasy_points_ppr"].mean().reset_index())
    out = {}
    for (season, pos), g in per_def.groupby(["season", "position"]):
        base = g["fantasy_points_ppr"].mean()
        if base <= 0:
            continue
        for _, r in g.iterrows():
            out[(int(season), r["opponent_team"], pos)] = float(r["fantasy_points_ppr"] / base)
    return out


def lag1_corr(ratings: dict, pos: str) -> float:
    """Correlation of a defense's rating in season N vs N+1 for this position."""
    by_team: dict[str, dict[int, float]] = {}
    for (season, team, p), v in ratings.items():
        if p == pos:
            by_team.setdefault(team, {})[season] = v
    prev, nxt = [], []
    for team, sv in by_team.items():
        for s in sv:
            if s + 1 in sv:
                prev.append(sv[s]); nxt.append(sv[s + 1])
    if len(prev) < 30:
        return 0.0
    return float(np.corrcoef(prev, nxt)[0, 1])


def main():
    df = pd.read_csv(WEEKLY, low_memory=False)
    if "season_type" in df.columns:
        df = df[df["season_type"].fillna("REG") == "REG"]
    df = df[(df["season"] >= MIN_SEASON) & df["position"].isin(POSITIONS)].copy()
    # Drop rows with no real opponent (2025 export is all '0').
    df = df[df["opponent_team"].notna() & ~df["opponent_team"].astype(str).isin(["0", "nan", ""])]
    df = recalc_weekly_df(df, platform="draftkings")

    ratings = season_ratings(df)
    shrink = {pos: max(0.0, lag1_corr(ratings, pos)) for pos in POSITIONS}

    teams = sorted({t for (s, t, _) in ratings if s in CARRIER})
    proj: dict[str, dict[str, float]] = {}
    for team in teams:
        row = {}
        for pos in POSITIONS:
            # Weighted blend of the carrier seasons (renormalize over what's present).
            num = sum(w * ratings[(s, team, pos)] for s, w in CARRIER.items()
                      if (s, team, pos) in ratings)
            den = sum(w for s, w in CARRIER.items() if (s, team, pos) in ratings)
            if den <= 0:
                row[pos] = 1.0
            else:
                carrier = num / den
                v = 1.0 + shrink[pos] * (carrier - 1.0)
                row[pos] = round(float(np.clip(v, *RATING_CLIP)), 4)
        proj[team] = row

    OUT.write_text(json.dumps({"shrink": shrink, "carrier": CARRIER,
                               "ratings": proj}, indent=2), encoding="utf-8")

    print(f"Defense-vs-position ratings -> {OUT.name}")
    print("\nyear-to-year predictability (shrink toward 1.0):")
    for pos in POSITIONS:
        print(f"  {pos}: lag-1 corr {shrink[pos]:.2f}")
    for pos in POSITIONS:
        ranked = sorted(proj.items(), key=lambda kv: kv[1][pos])
        soft = ranked[-3:][::-1]
        tough = ranked[:3]
        print(f"\n{pos}: softest 2026 (boost) {[(t, v[pos]) for t, v in soft]}")
        print(f"{pos}: toughest 2026 (fade) {[(t, v[pos]) for t, v in tough]}")


if __name__ == "__main__":
    main()
