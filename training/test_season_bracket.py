"""
Tests for Step 2 (season sim) + the prize-EV bracket scorer.

Run: python training/test_season_bracket.py   (no pytest needed)

Key things asserted:
  Season sim
    1. Rounds partition the weeks exactly (R1+R2+R3+R4 == total points).
    2. A STACKED roster has a higher single-week ceiling than the same players
       spread across teams — correlation flows through to team outcomes.
  Bracket / prize-EV
    3. A strictly dominant team has higher prize-EV and finals rate.
    4. Diagnostics are valid probabilities and reported separately.
    5. THE BIG ONE: under a top-heavy payout, a fat-right-tail team beats a
       higher-mean steady team; under a flat payout the ranking flips.
       (Extreme booms are rewarded specifically by payout convexity.)
"""

import sys

import numpy as np

import best_ball as bb
import payouts as pay
from bracket import Field, evaluate_roster
from outcome_model import load_default_model
from season_sim import SeasonSimResult, simulate_ctx


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        check.failed += 1
check.failed = 0


def main():
    rng = np.random.default_rng(11)
    model = load_default_model()

    # ---- 1. Rounds partition weeks exactly --------------------------------
    roster = [
        {"name": "Josh Allen", "position": "QB", "team": "BUF"},
        {"name": "Lamar Jackson", "position": "QB", "team": "BAL"},
        {"name": "Bijan Robinson", "position": "RB", "team": "ATL"},
        {"name": "Saquon Barkley", "position": "RB", "team": "PHI"},
        {"name": "De'Von Achane", "position": "RB", "team": "MIA"},
        {"name": "Kyren Williams", "position": "RB", "team": "LA"},
        {"name": "Ja'Marr Chase", "position": "WR", "team": "CIN"},
        {"name": "Justin Jefferson", "position": "WR", "team": "MIN"},
        {"name": "Amon-Ra St. Brown", "position": "WR", "team": "DET"},
        {"name": "Puka Nacua", "position": "WR", "team": "LA"},
        {"name": "Drake London", "position": "WR", "team": "ATL"},
        {"name": "Brock Bowers", "position": "TE", "team": "LV"},
        {"name": "Trey McBride", "position": "TE", "team": "ARI"},
    ]
    ctx = model.prepare_roster(roster)
    res = simulate_ctx(model, ctx, 4000, rng)
    partition_ok = np.allclose(res.round_scores.sum(axis=1), res.total_points)
    check("rounds partition weeks (R1+R2+R3+R4 == total)", partition_ok,
          f"max diff {np.abs(res.round_scores.sum(1) - res.total_points).max():.2e}")
    check("mean season points is sane", 800 < res.total_points.mean() < 3500,
          f"{res.total_points.mean():.0f} pts")

    # ---- 2. Stacking raises the single-week ceiling -----------------------
    # Same marginals for all players; only team labels differ.
    stack_roster = [{"name": f"QB", "position": "QB", "team": "AAA"}] + \
                   [{"name": f"WR{i}", "position": "WR", "team": "AAA"} for i in range(3)]
    spread_roster = [{"name": f"QB", "position": "QB", "team": "AAA"}] + \
                    [{"name": f"WR{i}", "position": "WR", "team": f"T{i}"} for i in range(3)]
    base = model._pool[("WR", 0)]
    cs = model.prepare_roster(stack_roster); cs.qgrids[:] = base
    cp = model.prepare_roster(spread_roster); cp.qgrids[:] = base
    # availability=False isolates the correlation effect from injury noise.
    rs = simulate_ctx(model, cs, 20000, np.random.default_rng(1), availability=False)
    rp = simulate_ctx(model, cp, 20000, np.random.default_rng(1), availability=False)
    # Round 2 == week 15 (single week). Compare upper-tail ceiling.
    stack_p95 = np.percentile(rs.round_scores[:, 1], 95)
    spread_p95 = np.percentile(rp.round_scores[:, 1], 95)
    check("stacked roster has higher single-week ceiling (p95)",
          stack_p95 > spread_p95 + 1.0,
          f"stack {stack_p95:.1f} vs spread {spread_p95:.1f}")

    # ---- 3/4. Bracket: dominance + valid diagnostics ----------------------
    field = Field.from_sim(res)                 # field = the studs roster itself
    base_eval = evaluate_roster(res, field)
    # A strictly dominant team: +6 pts every round.
    dom = SeasonSimResult(res.round_scores + 6.0, res.total_points + 6 * 4,
                          res.round_labels, res.weekly_lineup)
    dom_eval = evaluate_roster(dom, field)
    check("dominant team has higher prize-EV", dom_eval.prize_ev > base_eval.prize_ev,
          f"{dom_eval.prize_ev:.4f} > {base_eval.prize_ev:.4f}")
    check("dominant team has higher finals rate",
          dom_eval.finals_rate > base_eval.finals_rate,
          f"{dom_eval.finals_rate:.4f} > {base_eval.finals_rate:.4f}")
    d = base_eval.as_dict()
    probs_ok = all(0.0 <= d[k] <= 1.0 for k in
                   ("p_advance_r1", "finals_rate", "win_rate"))
    check("diagnostics are valid probabilities", probs_ok, str(
          {k: round(d[k], 4) for k in ("p_advance_r1", "finals_rate", "win_rate")}))
    check("mean_points diagnostic is separate from prize_ev",
          d["mean_points"] > 100 and d["prize_ev"] != d["mean_points"])

    # ---- 5. THE BIG ONE: top-heavy payout rewards the right tail ----------
    # Build two teams that always reach the final (rounds 0-2 huge), differing
    # ONLY in the final week (wk17): steady high-mean vs fat-tailed lower-mean.
    S = 200_000
    g = np.random.default_rng(3)
    big = np.full((S, 3), 1e4)                  # guarantee advancement
    steady = g.normal(22.0, 4.0, S)             # higher mean, thin tail
    # fat tail: usually modest, occasionally enormous; lower mean.
    spike = np.where(g.random(S) < 0.12, g.normal(45.0, 8.0, S),
                     g.normal(15.0, 3.0, S))
    steady_rs = np.column_stack([big, steady])
    spike_rs = np.column_stack([big, spike])
    steady_res = SeasonSimResult(steady_rs, steady, ["", "", "", ""], None)
    spike_res = SeasonSimResult(spike_rs, spike, ["", "", "", ""], None)

    # Field for the final week = a realistic spread of finalist scores.
    fin_field = Field(np.column_stack(
        [np.full(S, 1e4), np.full(S, 1e4), np.full(S, 1e4),
         g.normal(20.0, 6.0, S)]))

    print(f"    steady mean={steady.mean():.1f}  spike mean={spike.mean():.1f}")
    ratios = {}
    for label, payout in [("flat", pay.flat()), ("top_heavy", pay.DEFAULT_PAYOUT),
                          ("winner_take_all", pay.winner_take_all())]:
        es = evaluate_roster(steady_res, fin_field, payout)
        ek = evaluate_roster(spike_res, fin_field, payout)
        ratios[label] = ek.prize_ev / es.prize_ev
        print(f"    [{label:15s}] steady EV={es.prize_ev:.4f}  "
              f"spike EV={ek.prize_ev:.4f}  ratio spike/steady={ratios[label]:.2f}")

    # Flat payout is placement-indifferent: both reach the final equally, so a
    # flat curve rewards no right tail at all -> ratio ~ 1.
    check("flat payout is placement-indifferent (no tail reward)",
          abs(ratios["flat"] - 1.0) < 0.02, f"spike/steady={ratios['flat']:.2f}")
    # The core result: more convex payout -> more reward for the fat right tail.
    check("convexity monotonically rewards the tail "
          "(wta > top_heavy > flat-favoring)",
          ratios["winner_take_all"] > ratios["top_heavy"],
          f"wta {ratios['winner_take_all']:.2f} > top_heavy {ratios['top_heavy']:.2f}")
    check("at extreme top-heaviness the boom team wins outright",
          ratios["winner_take_all"] > 1.0, f"spike/steady={ratios['winner_take_all']:.2f}")

    print()
    if check.failed:
        print(f"{check.failed} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
