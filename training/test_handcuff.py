"""Unit + integration tests for the handcuff workload-inheritance layer."""
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


def _ctx(groups, anchor) -> RosterContext:
    n = len(anchor)
    return RosterContext(
        player_ids=[f"p{i}" for i in range(n)], names=[f"P{i}" for i in range(n)],
        positions=["RB"] * n, teams=["AAA"] * n, team_idx=np.zeros(n, dtype=int),
        loadings=np.zeros(n), qgrids=np.zeros((n, 1)), sources=["own"] * n, n_teams=1,
        q_minor=np.zeros(n), p_major=np.zeros(n),
        anchor=np.asarray(anchor, float), handcuff_groups=groups,
    )


def test_mechanism() -> None:
    m = CorrelatedOutcomeModel.__new__(CorrelatedOutcomeModel)
    m._handcuff_transfer = {"RB": 0.65, "QB": 0.70, "WR": 0.45, "TE": 0.45}

    # Starter anchor 10, backup anchor 4, same team. 4 weeks; baseline score = own anchor.
    ctx = _ctx([np.array([0, 1])], [10.0, 4.0])
    base = np.array([[[10., 10., 10., 10.]], [[4., 4., 4., 4.]]])   # [2,1,4]
    active = np.array([[[True, False, False, True]],                # starter out wk2,3
                       [[True, True, True, True]]])                 # backup always in
    out = m.apply_workload_transfer(base.copy(), active, ctx)

    # backup 4 -> 4 + 0.65*(10-4) = 7.9 exactly on the weeks the starter is out
    check("backup inherits when starter out",
          np.allclose(out[1, 0], [4.0, 7.9, 7.9, 4.0]), str(out[1, 0]))
    check("backup unchanged when starter plays",
          out[1, 0, 0] == 4.0 and out[1, 0, 3] == 4.0)
    check("starter score untouched by transfer",
          np.allclose(out[0, 0], [10., 10., 10., 10.]), str(out[0, 0]))

    # No handcuff group => no-op.
    ctx0 = _ctx(None, [10.0, 4.0])
    out0 = m.apply_workload_transfer(base.copy(), active, ctx0)
    check("no transfer without a handcuff pair", np.allclose(out0, base))

    # A lone backup (everyone above inactive) caps at the starter's role, not beyond.
    ctx3 = _ctx([np.array([0, 1, 2])], [12.0, 6.0, 3.0])
    base3 = np.array([[[12.]], [[6.]], [[3.]]])
    active3 = np.array([[[False]], [[False]], [[True]]])           # only the 3rd is active
    out3 = m.apply_workload_transfer(base3.copy(), active3, ctx3)
    # p2 takes role 0 (top active): 3 + 0.65*(12-3) = 8.85
    check("deepest active back inherits the top vacated role",
          np.isclose(out3[2, 0, 0], 8.85), str(out3[2, 0, 0]))


def test_integration() -> None:
    """End-to-end: owning a starter's handcuff should raise roster EV when the
    handcuff layer is ON vs OFF (same seed), because the backup spikes on the
    starter's missed weeks."""
    model = load_market_model()
    # A real injury-prone-ish lead back + a same-team backup, padded to a legal-ish core.
    roster = [
        {"name": "Saquon Barkley", "position": "RB", "team": "PHI", "proj_points": 280},
        {"name": "Will Shipley", "position": "RB", "team": "PHI", "proj_points": 70},
        {"name": "Jalen Hurts", "position": "QB", "team": "PHI", "proj_points": 360},
        {"name": "A.J. Brown", "position": "WR", "team": "PHI", "proj_points": 250},
        {"name": "CeeDee Lamb", "position": "WR", "team": "DAL", "proj_points": 270},
        {"name": "Puka Nacua", "position": "WR", "team": "LA", "proj_points": 260},
        {"name": "Trey McBride", "position": "TE", "team": "ARI", "proj_points": 200},
    ]
    on = simulate_roster(roster, n_seasons=4000, rng=np.random.default_rng(7), handcuff=True)
    off = simulate_roster(roster, n_seasons=4000, rng=np.random.default_rng(7), handcuff=False)
    d = float(on.total_points.mean() - off.total_points.mean())
    check("handcuff ON raises roster EV vs OFF", d > 0, f"delta={d:+.2f} pts/season")


if __name__ == "__main__":
    test_mechanism()
    test_integration()
    print("\n" + ("ALL CHECKS PASSED" if _fails == 0 else f"{_fails} CHECK(S) FAILED"))
    sys.exit(1 if _fails else 0)
