"""
Tests for the correlated outcome model (step 1).

Run: python training/test_outcome_model.py
Exits non-zero on first failure. No pytest dependency required.

What we assert (correctness over sophistication):
  1. Marginals: simulated scores reproduce each player's real DK distribution.
  2. Cross-team pairs are ~independent.
  3. Teammates are positively correlated, and QB+WR > QB+RB (the hierarchy).
  4. Correlation strength tracks the team_corr knob (0 -> independent).
  5. Stacks boom together more than equivalent non-stacked pairs.
  6. Fallback (position,tier) pool produces sane, position-appropriate scores.
"""

import sys

import numpy as np

from outcome_model import (CorrelatedOutcomeModel, load_default_model,
                           normalize_name)


def _spearman(a, b):
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        check.failed += 1
check.failed = 0


def main():
    rng = np.random.default_rng(7)
    model = load_default_model()

    # Find a high-volume player (lots of own history) for marginal testing.
    own_ids = [pid for pid, g in model._own.items()]
    assert own_ids, "no own-history players built"
    # Use the player whose grid has the widest spread (clearly a real starter).
    pid = max(own_ids, key=lambda p: model._own[p].mean())

    # ---- 1. Marginals reproduce the (possibly anchored) grid actually used --
    solo = [{"player_id": pid, "position": "QB", "team": "T1"}]
    ctx = model.prepare_roster(solo)
    used_grid = ctx.qgrids[0]                       # may be projection-anchored
    used_mean, used_std = used_grid.mean(), used_grid.std()
    sim = model.sample_weeks(ctx, 40000, rng)[0]
    mean_err = abs(sim.mean() - used_mean) / max(used_mean, 1e-6)
    std_err = abs(sim.std() - used_std) / max(used_std, 1e-6)
    check("marginal mean matches the grid used", mean_err < 0.05,
          f"sim {sim.mean():.2f} vs grid {used_mean:.2f} ({mean_err:.1%})")
    check("marginal std matches the grid used", std_err < 0.08,
          f"sim {sim.std():.2f} vs grid {used_std:.2f} ({std_err:.1%})")

    # ---- 2/3/4. Correlation structure ------------------------------------
    # Two real QBs and a real WR/RB; we relabel team/pos to isolate effects.
    def grid_for(position, tier=0):
        # pick any pooled grid so both "players" have identical marginals,
        # which isolates the copula's correlation from marginal differences.
        return model._pool.get((position, tier))

    # Build a roster of 4 players that all share ONE empirical grid but differ
    # in team/position, so correlation differences come purely from the copula.
    base_grid = grid_for("WR", 0)
    assert base_grid is not None

    class _Patched(CorrelatedOutcomeModel):
        pass

    # Same team -> should be correlated; different team -> independent.
    roster = [
        {"name": "QB_A", "position": "QB", "team": "AAA"},   # 0
        {"name": "WR_A", "position": "WR", "team": "AAA"},   # 1 same team as QB_A
        {"name": "RB_A", "position": "RB", "team": "AAA"},   # 2 same team, RB
        {"name": "WR_B", "position": "WR", "team": "BBB"},   # 3 different team
    ]
    ctx = model.prepare_roster(roster)
    # Force identical marginals on all four so only the copula differs.
    ctx.qgrids[:] = base_grid
    sims = model.sample_weeks(ctx, 60000, rng)
    c_qb_wr = _spearman(sims[0], sims[1])   # same team, strong loading
    c_qb_rb = _spearman(sims[0], sims[2])   # same team, RB weak loading
    c_cross = _spearman(sims[0], sims[3])   # different team

    check("cross-team pair ~independent", abs(c_cross) < 0.05,
          f"spearman={c_cross:.3f}")
    check("QB+WR teammates positively correlated", c_qb_wr > 0.15,
          f"spearman={c_qb_wr:.3f}")
    check("QB+WR correlation exceeds QB+RB (hierarchy)", c_qb_wr > c_qb_rb + 0.05,
          f"qb_wr={c_qb_wr:.3f} qb_rb={c_qb_rb:.3f}")

    # team_corr = 0 -> teammates independent
    flat = CorrelatedOutcomeModel(team_corr=0.0).build()
    ctxf = flat.prepare_roster(roster)
    ctxf.qgrids[:] = base_grid
    simf = flat.sample_weeks(ctxf, 60000, rng)
    check("team_corr=0 -> teammates independent",
          abs(_spearman(simf[0], simf[1])) < 0.05,
          f"spearman={_spearman(simf[0], simf[1]):.3f}")

    # ---- 5. Stacks boom together -----------------------------------------
    # P(both boom) for a stack should exceed independent expectation.
    boom = 18.0  # DK "good week" threshold for a starter
    both_stack = np.mean((sims[0] > boom) & (sims[1] > boom))   # same team
    p0, p1 = np.mean(sims[0] > boom), np.mean(sims[1] > boom)
    indep = p0 * p1
    check("stack co-boom exceeds independence", both_stack > indep * 1.15,
          f"joint={both_stack:.3f} vs indep={indep:.3f}")

    # ---- 6. Fallback pool path -------------------------------------------
    rookie = [{"name": "Totally Unknown Rookie", "position": "WR",
               "team": "ZZZ"}]
    ctxr = model.prepare_roster(rookie)
    check("unknown player uses pool fallback",
          ctxr.sources[0].startswith("pool:WR"), ctxr.sources[0])
    simr = model.sample_week(ctxr, rng)
    check("fallback score is finite & non-trivial",
          np.isfinite(simr[0]) and ctxr.qgrids[0].mean() > 1.0,
          f"pool mean={ctxr.qgrids[0].mean():.2f}")

    # ---- 7. Projection anchoring -----------------------------------------
    # Shape from history, level from the 2026 projection. proj_points already
    # includes expected missed games, so the anchor grosses the per-ACTIVE-week
    # mean up to a healthy level (proj_ppg / active_frac); after the availability
    # layer removes ~(1-active_frac) of weeks, the expected season returns to
    # proj_points. We confirm that availability-adjusted level tracks the
    # projection while the shape (CV) is preserved.
    hist_model = CorrelatedOutcomeModel(anchor_to_projection=False).build()
    cands = []
    for p, grid in hist_model._own.items():
        ppg = model._proj_ppg.get(p)
        if ppg and grid.mean() > 6:
            ratio = grid.mean() / ppg            # >1 means projection downgrades him
            if 1.3 <= ratio <= 2.8:              # clearly downgraded, not clipped
                cands.append((abs(ratio - 1.8), p, grid.mean(), ppg))
    assert cands, "no suitable downgrade candidate found"
    _, dp, hist_mean, ppg = min(cands)

    one = [{"player_id": dp, "position": "WR", "team": "X"}]
    ca = model.prepare_roster(one)              # anchored (default)
    cu = hist_model.prepare_roster(one)         # pure historical
    sa = model.sample_weeks(ca, 40000, rng)[0]
    su = hist_model.sample_weeks(cu, 40000, rng)[0]
    active_frac = model._expected_active_fraction(one[0])
    check("availability-adjusted level tracks the 2026 projection",
          abs(sa.mean() * active_frac - ppg) / ppg < 0.05,
          f"anchored(healthy) {sa.mean():.2f} * active {active_frac:.3f} "
          f"= {sa.mean() * active_frac:.2f} vs proj_ppg {ppg:.2f}")
    check("anchoring removes stale historical value (downgrade applied)",
          sa.mean() < su.mean() * 0.9,
          f"anchored {sa.mean():.2f} < historical {su.mean():.2f}")
    check("anchoring preserves boom/bust shape (CV unchanged)",
          abs(sa.std() / sa.mean() - su.std() / su.mean()) < 0.03,
          f"CV anc {sa.std()/sa.mean():.3f} vs hist {su.std()/su.mean():.3f}")
    check("pure-historical mode is available for A/B (flag honored)",
          hist_model.anchor is False and model.anchor is True)

    # ---- 8. Tail shrinkage -----------------------------------------------
    # Compare raw vs shrunk grids (both stored, both at historical level).
    def pctl(grid, p):
        return float(grid[int(round(p / 100.0 * (len(grid) - 1)))])

    # (a) Small-sample absurd ceilings get cut. Pick the n<20 player with the
    #     most extreme normalized p99 and confirm the fake tail shrinks.
    small = [p for p in model._raw_own if model._n_weeks[p] < 20]
    worst = max(small, key=lambda p: pctl(model._raw_own[p], 99)
                / max(model._raw_own[p].mean(), 1e-6))
    raw_r = pctl(model._raw_own[worst], 99) / max(model._raw_own[worst].mean(), 1e-6)
    shr_r = pctl(model._own[worst], 99) / max(model._own[worst].mean(), 1e-6)
    check("small-sample fake p99 ceiling is cut",
          shr_r < raw_r * 0.85,
          f"{model._display[worst]} (n={model._n_weeks[worst]}): "
          f"{raw_r:.1f}x -> {shr_r:.1f}x mean")

    # (b) Established studs keep their real ceiling.
    big = [p for p in model._raw_own if model._n_weeks[p] >= 45]
    stud = max(big, key=lambda p: model._raw_own[p].mean())
    drop95 = (pctl(model._raw_own[stud], 95) - pctl(model._own[stud], 95)) \
        / pctl(model._raw_own[stud], 95)
    check("established stud keeps p95 ceiling (not flattened)",
          abs(drop95) < 0.12,
          f"{model._display[stud]} (n={model._n_weeks[stud]}): p95 drop {drop95:.1%}")

    # (c) Shrinkage only reshapes — it must not move the LEVEL (mean).
    check("shrinkage preserves level (mean unchanged)",
          abs(model._own[stud].mean() - model._raw_own[stud].mean())
          / model._raw_own[stud].mean() < 0.005
          and abs(model._own[worst].mean() - model._raw_own[worst].mean())
          / max(model._raw_own[worst].mean(), 1e-6) < 0.005)

    # (d) Pure-no-shrink mode available for A/B; default model is shrunk.
    check("tail-shrink is an A/B flag (default on, raw kept)",
          model.tail_shrink is True
          and CorrelatedOutcomeModel(tail_shrink=False).tail_shrink is False
          and not np.array_equal(model._own[worst], model._raw_own[worst]))

    # ---- name normalization sanity ---------------------------------------
    check("name normalizer strips punctuation/suffix",
          normalize_name("Ja'Marr Chase Jr.") == "jamarr chase",
          normalize_name("Ja'Marr Chase Jr."))

    print()
    if check.failed:
        print(f"{check.failed} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
