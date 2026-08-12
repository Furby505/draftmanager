"""
Tests for the availability / injury layer.

Run: python training/test_availability.py   (no pytest needed)

Asserts:
  1. Expected games missed matches the calibration target (q_minor*17 + major).
  2. Position fragility ordering: RB misses more than QB.
  3. Major injuries are CLUSTERED (consecutive), not independent weeks.
  4. Playoff weeks (15-17) have lower availability than Week 1 (cumulative hazard).
  5. availability_model=False reproduces the iron-man baseline (A/B), and ON
     reduces season points but does NOT delete elite players.
  6. Historical durability shows through: an iron-man vet misses less than the
     position prior; a fragile vet misses more.
"""

import sys
from collections import Counter

import numpy as np

from outcome_model import (CorrelatedOutcomeModel, FULL_SEASON, POS_MISS_RATE,
                           load_default_model)
from season_sim import simulate_roster


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        check.failed += 1
check.failed = 0


def main():
    rng = np.random.default_rng(3)
    model = load_default_model()

    # One generic player per position (no history -> position prior).
    roster = [{"name": f"{p} Guy", "position": p, "team": f"T{p}"} for p in
              ("QB", "RB", "WR", "TE")]
    ctx = model.prepare_roster(roster)
    W = FULL_SEASON
    active = model.sample_availability(ctx, 40000, W, rng)       # [4, S, W]
    missed = (~active).sum(axis=2).mean(axis=1)                  # [4] mean games missed

    # ---- 1. Calibration: expected missed ~ position prior * 17 ------------
    for i, pos in enumerate(("QB", "RB", "WR", "TE")):
        target = POS_MISS_RATE[pos] * W
        check(f"{pos} expected games missed ~ target",
              abs(missed[i] - target) < 0.4,
              f"sim {missed[i]:.2f} vs target {target:.2f}")

    # ---- 2. Fragility ordering -------------------------------------------
    qb, rb = missed[0], missed[1]
    check("RBs miss more games than QBs", rb > qb + 0.5,
          f"RB {rb:.2f} vs QB {qb:.2f}")

    # ---- 3. Major injuries are clustered ---------------------------------
    # For a player, measure the longest consecutive inactive run per season and
    # confirm clustered absences occur far more than independent minor misses
    # would produce.
    rb_active = active[1]                                        # [S, W]
    def max_run(row):
        best = run = 0
        for v in row:
            run = 0 if v else run + 1
            best = max(best, run)
        return best
    runs = np.array([max_run(rb_active[s]) for s in range(2000)])
    p_minor = ctx.q_minor[1]
    # Under independent minor misses only, P(run>=3) ~ tiny; clustering makes it common.
    indep_run3 = p_minor ** 3
    check("major injuries create clustered multi-week absences",
          np.mean(runs >= 3) > 5 * indep_run3 and np.mean(runs >= 3) > 0.05,
          f"P(max run>=3)={np.mean(runs >= 3):.3f} vs indep~{indep_run3:.4f}")

    # ---- 4. Playoff weeks less available than Week 1 ----------------------
    wk1 = active[:, :, 0].mean()
    wk15_17 = active[:, :, 14:17].mean()
    check("Wk15-17 availability < Week 1 (cumulative hazard)",
          wk15_17 < wk1 - 0.005, f"wk1={wk1:.3f} wk15-17={wk15_17:.3f}")

    # ---- 5. A/B + elite players reduced, not deleted ---------------------
    elite = [{"name": "Ja'Marr Chase", "position": "WR", "team": "CIN"},
             {"name": "Christian McCaffrey", "position": "RB", "team": "SF"}]
    on = simulate_roster(elite, 4000, np.random.default_rng(1), model, availability=True)
    off = simulate_roster(elite, 4000, np.random.default_rng(1), model, availability=False)
    check("availability ON reduces season points vs iron-man",
          on.total_points.mean() < off.total_points.mean(),
          f"on {on.total_points.mean():.0f} < off {off.total_points.mean():.0f}")
    check("elite roster reduced modestly, NOT deleted (>80% retained)",
          on.total_points.mean() > 0.80 * off.total_points.mean(),
          f"retained {on.total_points.mean()/off.total_points.mean():.1%}")
    check("availability_model flag honored",
          model.availability_model is True
          and CorrelatedOutcomeModel(availability_model=False).availability_model is False)

    # ---- 6. Historical durability shows through ---------------------------
    # Compare an iron-man vet (low historical miss) vs a fragile vet (high) at
    # the same position; durable one should have lower p_major/q_minor.
    same_pos = "RB"
    cand = [(pid, model._miss_rate[pid]) for pid in model._miss_rate
            if model._pos_of.get(pid) == same_pos and model._miss_seasons.get(pid, 0) >= 3]
    if len(cand) >= 2:
        durable = min(cand, key=lambda x: x[1])[0]
        fragile = max(cand, key=lambda x: x[1])[0]
        cd = model.prepare_roster([{"player_id": durable, "position": same_pos}])
        cf = model.prepare_roster([{"player_id": fragile, "position": same_pos}])
        check("durable vet less injury-prone than fragile vet",
              cd.q_minor[0] + cd.p_major[0] < cf.q_minor[0] + cf.p_major[0],
              f"{model._display.get(durable)} vs {model._display.get(fragile)}")

    print()
    if check.failed:
        print(f"{check.failed} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
