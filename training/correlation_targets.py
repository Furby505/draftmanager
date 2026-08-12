"""
Measure real same-team and bring-back fantasy correlations from history.

These are the TARGETS the structural correlation model must reproduce, so the
sim's stack/bring-back value comes from reality, not invented noise. Used by
both calibration and the correlation audit.

Pairings (weekly DK scores, depth ranked by season targets within team-season):
  QB1-WR1, QB1-TE1, WR1-WR2, WR1-WR3, WR2-WR3, WR1-TE1, WR1-RBrec,
  bring-back: opposing pass-catchers in the same game (overall + high-total).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from scoring import recalc_weekly_df

WEEKLY = Path(__file__).resolve().parent.parent / "data" / "raw" / "player_stats_weekly.csv"
MIN_SEASON = 2018


def _corr(pairs: list[tuple[float, float]]) -> tuple[float, int]:
    if len(pairs) < 30:
        return float("nan"), len(pairs)
    a = np.array(pairs)
    return float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), len(pairs)


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    df = pd.read_csv(WEEKLY, low_memory=False)
    if "season_type" in df.columns:
        df = df[df["season_type"].fillna("REG") == "REG"]
    df = df[(df["season"] >= MIN_SEASON) &
            df["position"].isin(("QB", "RB", "WR", "TE"))].copy()
    df = recalc_weekly_df(df, platform="draftkings")
    df["targets"] = df.get("targets", 0)
    # depth rank within (season, team, position) by season targets (QB by attempts)
    rank_stat = np.where(df["position"] == "QB",
                         df.get("attempts", 0), df["targets"])
    df["_rank_stat"] = rank_stat
    season_use = (df.groupby(["season", "recent_team", "position", "player_id"])
                    ["_rank_stat"].sum().reset_index())
    season_use["depth"] = (season_use.groupby(["season", "recent_team", "position"])
                           ["_rank_stat"].rank(ascending=False, method="first"))
    df = df.merge(season_use[["season", "recent_team", "position", "player_id", "depth"]],
                  on=["season", "recent_team", "position", "player_id"], how="left")
    return df


def _pivot(df, position, depth):
    sub = df[(df["position"] == position) & (df["depth"] == depth)]
    return sub.set_index(["season", "week", "recent_team"])["fantasy_points_ppr"]


def measure() -> dict:
    df = _load()
    out = {}

    def pair_corr(p1, d1, p2, d2, label):
        s1, s2 = _pivot(df, p1, d1), _pivot(df, p2, d2)
        j = pd.concat([s1.rename("a"), s2.rename("b")], axis=1, join="inner").dropna()
        out[label] = _corr(list(zip(j["a"], j["b"])))

    pair_corr("QB", 1, "WR", 1, "QB1-WR1")
    pair_corr("QB", 1, "TE", 1, "QB1-TE1")
    pair_corr("QB", 1, "WR", 2, "QB1-WR2")
    pair_corr("WR", 1, "WR", 2, "WR1-WR2")
    pair_corr("WR", 1, "WR", 3, "WR1-WR3")
    pair_corr("WR", 2, "WR", 3, "WR2-WR3")
    pair_corr("WR", 1, "TE", 1, "WR1-TE1")
    pair_corr("WR", 1, "RB", 1, "WR1-RB1")

    # Bring-back: opposing pass-catchers (WR1/WR2/TE1) in the same game.
    pc = df[((df["position"] == "WR") & (df["depth"] <= 2)) |
            ((df["position"] == "TE") & (df["depth"] == 1))].copy()
    # game key: unordered team pair per (season, week)
    pc["game"] = pc.apply(lambda r: (r["season"], r["week"],
                                     *sorted([r["recent_team"], r["opponent_team"]])), axis=1)
    # combined pass-catcher fantasy per game = "shootout" proxy
    game_tot = pc.groupby("game")["fantasy_points_ppr"].sum()
    hi = set(game_tot[game_tot >= game_tot.quantile(0.66)].index)
    bb_all, bb_hi = [], []
    for gkey, g in pc.groupby("game"):
        teams = g["recent_team"].unique()
        if len(teams) != 2:
            continue
        a = g[g["recent_team"] == teams[0]]["fantasy_points_ppr"].to_numpy()
        b = g[g["recent_team"] == teams[1]]["fantasy_points_ppr"].to_numpy()
        for x in a:
            for y in b:
                bb_all.append((x, y))
                if gkey in hi:
                    bb_hi.append((x, y))
    out["bringback_all"] = _corr(bb_all)
    out["bringback_hi_total"] = _corr(bb_hi)
    return out


if __name__ == "__main__":
    print(f"Historical fantasy correlations (DK, {MIN_SEASON}+):\n")
    for k, (c, n) in measure().items():
        print(f"  {k:20s} corr={c:+.3f}  (n={n})")
