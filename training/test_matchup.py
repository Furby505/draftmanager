"""Tests for the playoff-matchup layer (defense-vs-position weekly modulation)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outcome_model import CorrelatedOutcomeModel, RosterContext, load_market_model  # noqa: E402
from season_sim import simulate_roster                                              # noqa: E402

_fails = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global _fails
    if not cond:
        _fails += 1
    print(("[PASS] " if cond else "[FAIL] ") + name + (f" - {extra}" if extra else ""))


def _ctx() -> RosterContext:
    return RosterContext(
        player_ids=["p"], names=["P"], positions=["RB"], teams=["AAA"],
        team_idx=np.array([0]), loadings=np.zeros(1), qgrids=np.zeros((1, 1)),
        sources=["own"], n_teams=1, q_minor=np.zeros(1), p_major=np.zeros(1),
        anchor=np.array([10.0]), handcuff_groups=None,
    )


def test_mechanism() -> None:
    m = CorrelatedOutcomeModel.__new__(CorrelatedOutcomeModel)
    m.matchup_model = True
    m._def_ratings = {"SOFT": {"RB": 1.3}, "AVG": {"RB": 1.0}, "TOUGH": {"RB": 0.7}}
    ctx = _ctx()

    # Mixed slate: soft / avg / tough. Raw [1.3,1.0,0.7] already has mean 1.0.
    m._opp_by_week = {1: {"AAA": "SOFT"}, 2: {"AAA": "AVG"}, 3: {"AAA": "TOUGH"}}
    M = m._matchup_multipliers(ctx, 3)
    check("soft week boosted, tough week faded", M[0, 0] > 1.0 > M[0, 2], str(M[0]))
    check("matchup multipliers are season-neutral (mean 1.0)", np.isclose(M[0].mean(), 1.0), str(M[0].mean()))

    # Uniformly soft slate -> normalized flat (a soft SEASON is already in the anchor;
    # only week-to-week SPREAD is the new signal). No within-season redistribution.
    m._opp_by_week = {1: {"AAA": "SOFT"}, 2: {"AAA": "SOFT"}, 3: {"AAA": "SOFT"}}
    M2 = m._matchup_multipliers(ctx, 3)
    check("uniform slate -> flat (season-neutral, no fake leverage)", np.allclose(M2[0], 1.0), str(M2[0]))

    # Bye/unknown week -> neutral 1.0 for that week.
    m._opp_by_week = {1: {"AAA": "SOFT"}, 2: {}, 3: {"AAA": "TOUGH"}}
    M3 = m._matchup_multipliers(ctx, 3)
    check("unknown-opponent week is neutral", M3[0, 1] != 0 and np.isclose(M3[0].mean(), 1.0), str(M3[0]))


def test_integration() -> None:
    """End-to-end with the real 2026 ratings + schedule: the layer runs, keeps the
    roster's season total close (per-player season-neutral), and DOES move the
    single-week playoff rounds (redistribution happened)."""
    model = load_market_model()
    check("real defense ratings loaded", bool(model._def_ratings) and len(model._def_ratings) >= 30,
          f"{len(model._def_ratings)} teams")
    check("real opponent schedule loaded", bool(model._opp_by_week), f"{len(model._opp_by_week)} weeks")

    roster = [
        {"name": "Bijan Robinson", "position": "RB", "team": "ATL", "proj_points": 290},
        {"name": "Saquon Barkley", "position": "RB", "team": "PHI", "proj_points": 280},
        {"name": "Ja'Marr Chase", "position": "WR", "team": "CIN", "proj_points": 290},
        {"name": "Puka Nacua", "position": "WR", "team": "LA", "proj_points": 260},
        {"name": "CeeDee Lamb", "position": "WR", "team": "DAL", "proj_points": 270},
        {"name": "Josh Allen", "position": "QB", "team": "BUF", "proj_points": 380},
        {"name": "Trey McBride", "position": "TE", "team": "ARI", "proj_points": 210},
    ]
    on = simulate_roster(roster, n_seasons=5000, rng=np.random.default_rng(11), matchup=True)
    off = simulate_roster(roster, n_seasons=5000, rng=np.random.default_rng(11), matchup=False)

    tot_on, tot_off = float(on.total_points.mean()), float(off.total_points.mean())
    rel = abs(tot_on - tot_off) / max(tot_off, 1e-6)
    check("season total stays anchored (within 8%)", rel < 0.08, f"on={tot_on:.0f} off={tot_off:.0f} ({rel:.1%})")

    # Playoff rounds R2/R3/R4 are columns 1/2/3; they should differ between ON/OFF.
    play_on = on.round_scores[:, 1:].mean()
    play_off = off.round_scores[:, 1:].mean()
    check("playoff-round scores shift with matchups on", not np.isclose(play_on, play_off),
          f"on={play_on:.2f} off={play_off:.2f}")


if __name__ == "__main__":
    test_mechanism()
    test_integration()
    print("\n" + ("ALL CHECKS PASSED" if _fails == 0 else f"{_fails} CHECK(S) FAILED"))
    sys.exit(1 if _fails else 0)
