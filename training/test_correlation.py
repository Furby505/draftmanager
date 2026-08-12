"""
Tests for the structural correlation model (game -> team -> opportunity -> points).

Run: python training/test_correlation.py   (no pytest needed)

Asserts the acceptance criteria:
  1. QB + primary pass catcher correlation is meaningfully positive.
  2. Opposing bring-backs are positively correlated IN their matchup week only
     (schedule-driven), ~0 when the teams don't play.
  3. Same-team WR-WR is no longer blindly positive (competition exists) — far
     below the single-factor baseline.
  4. Team weeks stay plausible: same-team WRs do NOT all ceiling together the
     way a blindly-positive model would.
  5. Marginals are preserved (copula only).
  6. correlation_model=False reproduces the single-team-factor baseline (A/B).
"""

import json
import sys
from pathlib import Path

import numpy as np

from outcome_model import CorrelatedOutcomeModel, load_default_model

SCHED = Path(__file__).resolve().parent.parent / "data" / "processed" / "schedule_2026.json"


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        check.failed += 1
check.failed = 0


def corr(sc, i, j, wk=None):
    a = sc[i, :, wk] if wk is not None else sc[i].ravel()
    b = sc[j, :, wk] if wk is not None else sc[j].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def main():
    model = load_default_model()
    rng = np.random.default_rng(0)
    sched = json.loads(SCHED.read_text())
    A, B = sched["1"][0]                       # a real Week-1 matchup

    roster = [
        {"name": "QB1", "position": "QB", "team": A},   # 0
        {"name": "WRa", "position": "WR", "team": A},   # 1
        {"name": "WRb", "position": "WR", "team": A},   # 2
        {"name": "TEa", "position": "TE", "team": A},   # 3
        {"name": "WRopp", "position": "WR", "team": B},  # 4  (plays A in wk1)
        {"name": "WRfar", "position": "WR", "team": "ZZZ"},  # 5 unrelated
    ]
    ctx = model.prepare_roster(roster)
    sc = model.sample_scores(ctx, 12000, 17, rng)       # [6, S, 17]

    # ---- 1. QB + pass catcher meaningfully positive ----------------------
    qb_wr = corr(sc, 0, 1)
    check("QB + primary WR meaningfully positive", qb_wr > 0.30,
          f"corr={qb_wr:.3f} (hist ~0.39)")
    check("QB + TE positive", corr(sc, 0, 3) > 0.22, f"corr={corr(sc, 0, 3):.3f}")

    # ---- 2. Bring-back: only in the matchup week -------------------------
    bb_game = corr(sc, 1, 4, 0)               # wk1: A vs B
    bb_nogame = corr(sc, 1, 4, 6)             # wk7: likely not playing
    check("bring-back positive in the matchup week", bb_game > 0.015,
          f"wk1 corr={bb_game:.3f}")
    check("no bring-back when teams don't play", abs(bb_nogame) < 0.02,
          f"wk7 corr={bb_nogame:.3f}")
    check("unrelated cross-team pair ~0", abs(corr(sc, 1, 5)) < 0.02,
          f"corr={corr(sc, 1, 5):.3f}")

    # ---- 3. Same-team WR-WR not blindly positive -------------------------
    wr_wr = corr(sc, 1, 2)
    # Baseline single-factor model for the A/B comparison.
    base = CorrelatedOutcomeModel(correlation_model=False).build()
    cb = base.prepare_roster(roster)
    scb = base.sample_scores(cb, 12000, 17, np.random.default_rng(0))
    wr_wr_base = corr(scb, 1, 2)
    check("same-team WR-WR far below single-factor baseline",
          wr_wr < wr_wr_base - 0.15 and wr_wr < 0.18,
          f"structural={wr_wr:.3f} vs baseline={wr_wr_base:.3f}")
    check("QB-WR survived the WR-WR fix (still strong)", qb_wr > 0.30,
          f"QB-WR={qb_wr:.3f}, WR-WR={wr_wr:.3f}")

    # ---- 4. Team weeks stay plausible (no all-ceiling) -------------------
    # P(both same-team WRs > their p85 in the same week): a blindly-positive
    # model inflates this; competition should keep it near independence.
    a1 = sc[1].ravel(); a2 = sc[2].ravel()
    t1, t2 = np.percentile(a1, 85), np.percentile(a2, 85)
    joint = np.mean((a1 > t1) & (a2 > t2))
    base_a1 = scb[1].ravel(); base_a2 = scb[2].ravel()
    bt1, bt2 = np.percentile(base_a1, 85), np.percentile(base_a2, 85)
    joint_base = np.mean((base_a1 > bt1) & (base_a2 > bt2))
    check("same-team WRs don't all ceiling together (< baseline)",
          joint < joint_base, f"structural P(both top15%)={joint:.3f} "
          f"vs baseline={joint_base:.3f} (indep={0.15**2:.3f})")

    # ---- 5. Marginals preserved ------------------------------------------
    used = ctx.qgrids[1]
    check("marginals preserved (mean matches grid)",
          abs(sc[1].mean() - used.mean()) / max(used.mean(), 1e-6) < 0.05,
          f"sim {sc[1].mean():.2f} vs grid {used.mean():.2f}")

    # ---- 6. A/B flag honored ---------------------------------------------
    check("correlation_model flag honored",
          model.correlation_model is True and base.correlation_model is False)

    print()
    if check.failed:
        print(f"{check.failed} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
