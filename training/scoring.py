"""
Platform-specific fantasy scoring rules for DraftKings, Underdog, etc.

Import recalc_weekly_df() in any script that loads player_stats_weekly.csv
to convert nflverse's default PPR scores to the target platform's scoring.
"""

import numpy as np

PLATFORMS: dict = {
    "draftkings": {
        "name":           "DraftKings Best Ball",
        "pass_td":        4.0,
        "pass_yd_per":    0.04,
        "pass_bonus_300": 3.0,    # 300+ passing yard game bonus
        "int":            -1.0,   # DK uses -1, not -2
        "rush_td":        6.0,
        "rush_yd_per":    0.1,
        "rush_bonus_100": 3.0,    # 100+ rushing yard game bonus
        "rec_td":         6.0,
        "rec_yd_per":     0.1,
        "rec_bonus_100":  3.0,    # 100+ receiving yard game bonus
        "reception":      1.0,
        "fumble_lost":    -1.0,
    },
    "underdog": {
        "name":           "Underdog Best Ball",
        "pass_td":        4.0,
        "pass_yd_per":    0.04,
        "pass_bonus_300": 0.0,
        "int":            -1.0,
        "rush_td":        6.0,
        "rush_yd_per":    0.1,
        "rush_bonus_100": 0.0,
        "rec_td":         6.0,
        "rec_yd_per":     0.1,
        "rec_bonus_100":  0.0,
        "reception":      0.5,
        "fumble_lost":    -1.0,
    },
}

DEFAULT_PLATFORM = "draftkings"


def calc_weekly_score(
    *,
    passing_yards: float  = 0,
    passing_tds: float    = 0,
    interceptions: float  = 0,
    rushing_yards: float  = 0,
    rushing_tds: float    = 0,
    receiving_yards: float = 0,
    receiving_tds: float  = 0,
    receptions: float     = 0,
    fumbles_lost: float   = 0,
    platform: str         = DEFAULT_PLATFORM,
) -> float:
    """Score a single player-week using the given platform's rules."""
    s = PLATFORMS[platform]
    pts = (
        passing_yards   * s["pass_yd_per"]
        + passing_tds   * s["pass_td"]
        + interceptions * s["int"]
        + rushing_yards * s["rush_yd_per"]
        + rushing_tds   * s["rush_td"]
        + receiving_yards * s["rec_yd_per"]
        + receiving_tds * s["rec_td"]
        + receptions    * s["reception"]
        + fumbles_lost  * s["fumble_lost"]
    )
    if passing_yards  >= 300: pts += s["pass_bonus_300"]
    if rushing_yards  >= 100: pts += s["rush_bonus_100"]
    if receiving_yards >= 100: pts += s["rec_bonus_100"]
    return float(pts)


def recalc_weekly_df(df, platform: str = DEFAULT_PLATFORM):
    """
    Recalculate fantasy_points_ppr in a weekly player stats DataFrame.

    Overwrites the existing fantasy_points_ppr column so all downstream
    boom rate / ppg calculations use the correct platform scoring.

    Works with both nflverse player_stats_weekly.csv rows and the 2025
    PBP-derived rows.

    Fumble columns tried (sum of all found):
      rushing_fumbles_lost, receiving_fumbles_lost, sack_fumbles_lost, fumbles_lost
    """
    import pandas as pd

    def _col(name, default=0.0):
        if name in df.columns:
            return df[name].fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    # Aggregate all fumble-lost columns that exist
    fumbles = (
        _col("rushing_fumbles_lost")
        + _col("receiving_fumbles_lost")
        + _col("sack_fumbles_lost")
        + _col("fumbles_lost")        # 2025 PBP-derived column
    )
    # Avoid double-counting if nflverse already sums them as fumbles_lost
    # by capping at a reasonable max per game
    fumbles = fumbles.clip(0, 4)

    pass_yds = _col("passing_yards")
    rush_yds = _col("rushing_yards")
    rec_yds  = _col("receiving_yards")
    s = PLATFORMS[platform]

    pts = (
        pass_yds                 * s["pass_yd_per"]
        + _col("passing_tds")    * s["pass_td"]
        + _col("interceptions")  * s["int"]
        + rush_yds               * s["rush_yd_per"]
        + _col("rushing_tds")    * s["rush_td"]
        + rec_yds                * s["rec_yd_per"]
        + _col("receiving_tds")  * s["rec_td"]
        + _col("receptions")     * s["reception"]
        + fumbles                * s["fumble_lost"]
        + np.where(pass_yds >= 300, s["pass_bonus_300"], 0)
        + np.where(rush_yds >= 100, s["rush_bonus_100"], 0)
        + np.where(rec_yds  >= 100, s["rec_bonus_100"],  0)
    )

    df["fantasy_points_ppr"] = pts
    df["fantasy_points"]     = pts - _col("receptions") * s["reception"]
    return df
