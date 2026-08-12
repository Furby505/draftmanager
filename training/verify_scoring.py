"""
Verify DraftKings scoring is correct across the weekly data pipeline.

Usage:
  python training/verify_scoring.py                    # summary + top milestone weeks
  python training/verify_scoring.py --player "Josh Allen"
  python training/verify_scoring.py --player "Christian McCaffrey" --season 2025
  python training/verify_scoring.py --week 3 --season 2025

Shows per-week scoring breakdown: base pts + each milestone bonus + fumble penalty.
Use this after any retrain to confirm DK rules are applied correctly.
"""

import sys
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from scoring import recalc_weekly_df, PLATFORMS, DEFAULT_PLATFORM

RAW = Path(__file__).parent.parent / "data" / "raw"


def load_weekly() -> pd.DataFrame:
    df = pd.read_csv(RAW / "player_stats_weekly.csv", low_memory=False)
    df = df[df["season_type"] == "REG"].copy()
    return df


def build_breakdown(df: pd.DataFrame, platform: str = DEFAULT_PLATFORM) -> pd.DataFrame:
    """Add per-scoring-component columns so each bonus is visible."""
    s = PLATFORMS[platform]

    def col(name, default=0.0):
        return df[name].fillna(default) if name in df.columns else pd.Series(default, index=df.index)

    pass_yds = col("passing_yards")
    rush_yds = col("rushing_yards")
    rec_yds  = col("receiving_yards")

    fumbles = (
        col("rushing_fumbles_lost")
        + col("receiving_fumbles_lost")
        + col("sack_fumbles_lost")
        + col("fumbles_lost")
    ).clip(0, 4)

    out = df[["player_display_name", "position", "recent_team", "season", "week"]].copy()
    out["pass_base"]      = pass_yds * s["pass_yd_per"] + col("passing_tds") * s["pass_td"] + col("interceptions") * s["int"]
    out["rush_base"]      = rush_yds * s["rush_yd_per"] + col("rushing_tds") * s["rush_td"]
    out["rec_base"]       = rec_yds  * s["rec_yd_per"]  + col("receiving_tds") * s["rec_td"] + col("receptions") * s["reception"]
    out["pass_bonus_300"] = np.where(pass_yds >= 300, s["pass_bonus_300"], 0.0)
    out["rush_bonus_100"] = np.where(rush_yds >= 100, s["rush_bonus_100"], 0.0)
    out["rec_bonus_100"]  = np.where(rec_yds  >= 100, s["rec_bonus_100"],  0.0)
    out["fumble_penalty"] = fumbles * s["fumble_lost"]
    out["total_dk"]       = (
        out["pass_base"] + out["rush_base"] + out["rec_base"]
        + out["pass_bonus_300"] + out["rush_bonus_100"] + out["rec_bonus_100"]
        + out["fumble_penalty"]
    )

    # Raw stat columns for reference
    out["pass_yds"] = pass_yds.astype(int)
    out["rush_yds"] = rush_yds.astype(int)
    out["rec_yds"]  = rec_yds.astype(int)
    out["ints"]     = col("interceptions").astype(int)
    out["fum_lost"] = fumbles.astype(int)

    return out


def fmt_row(row) -> str:
    parts = [f"{row.total_dk:6.1f} pts"]
    if row.pass_bonus_300: parts.append(f"+3 (300+ pass)")
    if row.rush_bonus_100: parts.append(f"+3 (100+ rush)")
    if row.rec_bonus_100:  parts.append(f"+3 (100+ rec)")
    if row.fumble_penalty: parts.append(f"{row.fumble_penalty:.0f} (fum)")
    return "  |  ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player",  default=None, help="Player display name (partial match OK)")
    parser.add_argument("--season",  type=int, default=None)
    parser.add_argument("--week",    type=int, default=None)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--top",     type=int, default=20, help="Rows to show in summary mode")
    args = parser.parse_args()

    print(f"Loading weekly data and applying {args.platform} scoring...")
    df = load_weekly()
    bd = build_breakdown(df, args.platform)

    if args.season:
        bd = bd[bd["season"] == args.season]
    if args.week:
        bd = bd[bd["week"] == args.week]

    if args.player:
        mask = bd["player_display_name"].str.contains(args.player, case=False, na=False)
        bd = bd[mask]
        if bd.empty:
            print(f"No rows found for '{args.player}'")
            return
        bd = bd.sort_values(["season", "week"])
        print(f"\n{'Wk':>3} {'Yr':>4}  {'Team':>4}  {'Score':>7}  Breakdown")
        print("-" * 70)
        for _, row in bd.iterrows():
            print(f"{int(row.week):>3} {int(row.season):>4}  {row.recent_team:>4}  {fmt_row(row)}"
                  f"    [{row.pass_yds}py / {row.rush_yds}ry / {row.rec_yds}recy / {row.ints}int / {row.fum_lost}fum]")
        print(f"\nSeason totals:")
        for yr, grp in bd.groupby("season"):
            bonuses = grp[["pass_bonus_300","rush_bonus_100","rec_bonus_100"]].sum()
            total_bonus = bonuses.sum()
            print(f"  {yr}: {grp.total_dk.sum():.1f} pts  (milestone bonus: +{total_bonus:.0f} pts across {len(grp)} games)")
        return

    # Summary mode — milestone bonus impact by position + top single-game performances
    print(f"\n── Milestone bonus impact by position ({'all seasons' if not args.season else args.season}) ──")
    for pos in ["QB", "WR", "RB", "TE"]:
        g = bd[bd["position"] == pos]
        if g.empty:
            continue
        n_300  = (g["pass_bonus_300"] > 0).sum()
        n_100r = (g["rush_bonus_100"] > 0).sum()
        n_100c = (g["rec_bonus_100"]  > 0).sum()
        tot_bonus = g[["pass_bonus_300","rush_bonus_100","rec_bonus_100"]].sum().sum()
        n_fum  = (g["fumble_penalty"] < 0).sum()
        fum_pts = g["fumble_penalty"].sum()
        print(f"  {pos}: +{tot_bonus:.0f} bonus pts  "
              f"({n_300} 300+pass  {n_100r} 100+rush  {n_100c} 100+rec)  "
              f"  {n_fum} fumbles ({fum_pts:.0f} pts)")

    print(f"\n── Top {args.top} single-game scores ──")
    top = bd.nlargest(args.top, "total_dk")
    print(f"{'Name':<25} {'Pos':>3} {'Yr':>4} {'Wk':>3}  {'Score':>7}  Bonus breakdown")
    print("-" * 80)
    for _, row in top.iterrows():
        bonuses = []
        if row.pass_bonus_300: bonuses.append("300+pass")
        if row.rush_bonus_100: bonuses.append("100+rush")
        if row.rec_bonus_100:  bonuses.append("100+rec")
        bonus_str = "  +" + "/".join(bonuses) if bonuses else ""
        print(f"{row.player_display_name:<25} {row.position:>3} {int(row.season):>4} {int(row.week):>3}  "
              f"{row.total_dk:>7.1f}{bonus_str}")

    print(f"\n── Fumble penalty weeks (spot check) ──")
    fum_weeks = bd[bd["fumble_penalty"] < 0].sort_values("fumble_penalty").head(10)
    for _, row in fum_weeks.iterrows():
        print(f"  {row.player_display_name:<25} {row.position:>3}  "
              f"Wk{int(row.week)} {int(row.season)}  {row.fumble_penalty:.0f} pts  "
              f"({row.fum_lost} fumble(s) lost)")


if __name__ == "__main__":
    main()
