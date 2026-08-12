"""
Structural correlation audit.

Run: python training/audit_correlation.py
Writes: data/processed/correlation_audit.md

Compares three models on the same rosters:
  structural  = game/team/competition model (correlation_model=True)
  single      = old single-team-factor baseline (correlation_model=False)
  independent = no correlation (correlation_model=False, team_corr=0)
"""

import json
from pathlib import Path

import numpy as np

from correlation_targets import measure as measure_history
from outcome_model import CorrelatedOutcomeModel, load_default_model
from season_sim import simulate_roster

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "correlation_audit.md"
SCHED = json.loads((ROOT / "data" / "processed" / "schedule_2026.json").read_text())


def corr(sc, i, j, wk=None):
    a = sc[i, :, wk] if wk is not None else sc[i].ravel()
    b = sc[j, :, wk] if wk is not None else sc[j].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def matchup_week(team_a, team_b):
    """First NFL week (1-17) where a and b play; None if not in 1-17."""
    for w in range(1, 18):
        for h, vis in SCHED.get(str(w), []):
            if {h, vis} == {team_a, team_b}:
                return w
    return None


def playoff_matchup_week(team_a, team_b):
    """First playoff week (15-17) where a and b play; None if absent."""
    for w in range(15, 18):
        for h, vis in SCHED.get(str(w), []):
            if {h, vis} == {team_a, team_b}:
                return w
    return None


def main():
    struct = load_default_model()
    single = CorrelatedOutcomeModel(correlation_model=False).build()
    indep = CorrelatedOutcomeModel(correlation_model=False, team_corr=0.0).build()
    rng = lambda s=0: np.random.default_rng(s)

    lines = []

    def emit(x=""):
        lines.append(x)
        print(x)

    emit("# Structural Correlation Audit")
    emit("_game -> team volume -> target competition -> points. Marginals "
         "unchanged; only the copula changes._\n")

    # Representative roster for correlation measurement.
    A, B = SCHED["1"][0]
    wk = 0
    roster = [
        {"name": "QB", "position": "QB", "team": A},
        {"name": "WR1", "position": "WR", "team": A},
        {"name": "WR2", "position": "WR", "team": A},
        {"name": "TE1", "position": "TE", "team": A},
        {"name": "RBrec", "position": "RB", "team": A},
        {"name": "oppWR", "position": "WR", "team": B},
        {"name": "farWR", "position": "WR", "team": "ZZZ"},
    ]
    sS = struct.sample_scores(struct.prepare_roster(roster), 15000, 17, rng(1))
    sB = single.sample_scores(single.prepare_roster(roster), 15000, 17, rng(1))

    hist = measure_history()
    emit("## 1. Structural model vs historical targets")
    emit("| Pair | historical | structural model |")
    emit("|---|---|---|")
    rows = [
        ("QB1-WR1", corr(sS, 0, 1)),
        ("QB1-TE1", corr(sS, 0, 3)),
        ("WR1-WR2", corr(sS, 1, 2)),
        ("WR1-TE1", corr(sS, 1, 3)),
        ("WR1-RB1", corr(sS, 1, 4)),
        ("bringback (matchup wk)", corr(sS, 1, 5, wk)),
    ]
    hk = {
        "QB1-WR1": "QB1-WR1",
        "QB1-TE1": "QB1-TE1",
        "WR1-WR2": "WR1-WR2",
        "WR1-TE1": "WR1-TE1",
        "WR1-RB1": "WR1-RB1",
        "bringback (matchup wk)": "bringback_all",
    }
    for label, mv in rows:
        h = hist.get(hk[label], (float("nan"), 0))[0]
        emit(f"| {label} | {h:+.3f} | {mv:+.3f} |")
    emit("\n_WR-WR/WR-TE are calibrated for the common 2-catcher case. "
         "With 3+ same-team catchers the zero-sum competition weakens, but the "
         "result remains far below the old single-factor WR-WR correlation._")

    emit("\n## 2. Same-team WR-WR / WR-TE: single-factor (before) vs structural (after)")
    emit("| Pair | single-factor (before) | structural (after) | target competition? |")
    emit("|---|---|---|---|")
    for label, i, j in [("WR1-WR2", 1, 2), ("WR1-TE1", 1, 3)]:
        emit(f"| {label} | {corr(sB, i, j):+.3f} | {corr(sS, i, j):+.3f} | yes |")

    emit("\n## 3. Bring-back (opposing pass catchers): before vs after")
    emit("| In matchup week | single-factor | structural |")
    emit("|---|---|---|")
    emit(f"| {A} WR vs {B} WR (wk{wk + 1}) | {corr(sB, 1, 5, wk):+.3f} | "
         f"{corr(sS, 1, 5, wk):+.3f} |")
    emit(f"| same pair, a non-matchup week | {corr(sB, 1, 5, 8):+.3f} | "
         f"{corr(sS, 1, 5, 8):+.3f} |")
    emit("\n_Structural bring-back fires only when the teams actually play; "
         "single-factor has no game layer._")

    emit("\n## 4. Game-stack ceiling rates (same week, matchup)")
    emit("P(QB + same-team WR + opposing WR all > their p80 in the matchup week):")

    def joint_top(sc, idxs, wk, p=80):
        masks = [sc[i, :, wk] > np.percentile(sc[i].ravel(), p) for i in idxs]
        return float(np.mean(np.logical_and.reduce(masks)))

    js = joint_top(sS, [0, 1, 5], wk)
    ji = joint_top(
        indep.sample_scores(indep.prepare_roster(roster), 15000, 17, rng(1)),
        [0, 1, 5],
        wk,
    )
    emit(f"- structural: **{js:.4f}**  |  independent: {ji:.4f}  |  "
         f"chance (0.2^3): {0.2 ** 3:.4f}")
    emit(f"- the game stack hits its joint ceiling **{js / max(ji, 1e-9):.1f}x** "
         "more often than independent - this is the tournament upside.")

    emit("\n## 5. Where stacking pays: single-week playoff TAIL")
    emit("Stacking is a ceiling/leverage play, not a mean booster. The right "
         "lens is a single playoff week. For favorite-style rosters, structural "
         "correlation can leave p95 flat or slightly lower because booms clump; "
         "the value shows up in the far right tail when a scheduled game stack "
         "hits together.")
    stacked, diffuse, stack_week, stack_label = _stack_rosters()
    playoff_col = stack_week - 14
    emit(f"Primary example: {stack_label}, Week {stack_week}.")
    emit("| Roster | mean wk | p95 indep -> structural | p99 indep -> structural | p99 lift |")
    emit("|---|---|---|---|---|")
    for label, ros in [
        ("QB+WR+WR stack + bring-back", stacked),
        ("diffuse (no stacks)", diffuse),
    ]:
        ri = simulate_roster(ros, 6000, rng(6), model=indep, availability=False)
        rs = simulate_roster(ros, 6000, rng(6), model=struct, availability=False)
        p95i = np.percentile(ri.round_scores[:, playoff_col], 95)
        p95s = np.percentile(rs.round_scores[:, playoff_col], 95)
        p99i = np.percentile(ri.round_scores[:, playoff_col], 99)
        p99s = np.percentile(rs.round_scores[:, playoff_col], 99)
        emit(f"| {label} | {rs.round_scores[:, playoff_col].mean():.0f} | "
             f"{p95i:.0f} -> {p95s:.0f} | {p99i:.0f} -> {p99s:.0f} | "
             f"{(p99s - p99i):+.0f} |")
    emit("\n_This is the honest stack trade-off: structural correlation creates "
         "more shared-nuke paths, but it can reduce consistency metrics because "
         "points arrive in clumps. That is what a playoff best-ball tournament "
         "model should expose instead of assuming stacks are free._")

    emit("\n## 6. What the fixes change vs the single-factor baseline")
    emit(f"Single-week Week {stack_week} p99 under single-factor (old) vs structural (new):")
    emit("| Construction | single-factor p99 | structural p99 | why |")
    emit("|---|---|---|---|")
    for label, ros, why in [
        ("3 same-team WR, NO QB", _same_team_no_qb(),
         "old inflated WR-WR -> ceiling falls (real competition)"),
        ("QB + WR + bring-back stack", stacked,
         "structural adds schedule-driven opponent game environment"),
    ]:
        pb = np.percentile(
            simulate_roster(ros, 6000, rng(6), model=single, availability=False)
            .round_scores[:, playoff_col],
            99,
        )
        ps = np.percentile(
            simulate_roster(ros, 6000, rng(6), model=struct, availability=False)
            .round_scores[:, playoff_col],
            99,
        )
        emit(f"| {label} | {pb:.0f} | {ps:.0f} | {why} |")

    emit("\n## 7. Sanity - common stack correlations (structural)")
    emit(f"- QB + WR1: {corr(sS, 0, 1):+.3f}  (want ~0.39)")
    emit(f"- QB + TE1: {corr(sS, 0, 3):+.3f}  (want ~0.31)")
    emit(f"- WR1 + WR2 same team: {corr(sS, 1, 2):+.3f}  (want ~0.04-0.12, not 0.35)")
    emit(f"- QB + pass-catching RB: {corr(sS, 0, 4):+.3f}  (small)")

    emit("\n## 8. Week 15-17 stack behavior (playoff bring-back)")
    emit("Bring-back only helps the tournament if the teams play in Wk15/16/17. "
         "Structural model prices this exactly:")
    emit("| Stack | matchup week | bring-back corr that week |")
    emit("|---|---|---|")
    pairs = [
        ("LA", "DAL"),
        ("KC", "SF"),
        ("CIN", "BAL"),
        ("LAC", "KC"),
        ("KC", "BUF"),
        ("PHI", "DAL"),
    ]
    for a, b in pairs:
        mw = playoff_matchup_week(a, b) or matchup_week(a, b)
        ros2 = [
            {"name": "q", "position": "QB", "team": a},
            {"name": "w", "position": "WR", "team": a},
            {"name": "o", "position": "WR", "team": b},
        ]
        s2 = struct.sample_scores(struct.prepare_roster(ros2), 8000, 17, rng(2))
        if mw and mw <= 17:
            c = corr(s2, 1, 2, mw - 1)
            tag = "PLAYOFF" if mw >= 15 else "regular season"
            emit(f"| {a} WR + {b} WR | wk{mw} ({tag}) | {c:+.3f} |")
        else:
            emit(f"| {a} WR + {b} WR | no wk1-17 matchup | n/a |")
    emit("\n_A game stack that meets in Wk15/16/17 carries playoff bring-back upside; "
         "one that only meets in the regular-season cumulative round does not get "
         "that single-week playoff tail boost. The model now distinguishes them._")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")


def _fill(extra):
    """Pad to a valid 20-man roster with assorted real players."""
    base = [
        {"name": "Bijan Robinson", "position": "RB", "team": "ATL"},
        {"name": "Saquon Barkley", "position": "RB", "team": "PHI"},
        {"name": "Kyren Williams", "position": "RB", "team": "LA"},
        {"name": "Chase Brown", "position": "RB", "team": "CIN"},
        {"name": "Tony Pollard", "position": "RB", "team": "TEN"},
        {"name": "Travis Etienne", "position": "RB", "team": "JAX"},
        {"name": "Justin Jefferson", "position": "WR", "team": "MIN"},
        {"name": "Drake London", "position": "WR", "team": "ATL"},
        {"name": "Nico Collins", "position": "WR", "team": "HOU"},
        {"name": "Jaylen Waddle", "position": "WR", "team": "MIA"},
        {"name": "Trey McBride", "position": "TE", "team": "ARI"},
        {"name": "David Njoku", "position": "TE", "team": "CLE"},
        {"name": "Bo Nix", "position": "QB", "team": "DEN"},
    ]
    return extra + base


def _stack_rosters():
    stack_week = playoff_matchup_week("CIN", "BAL")
    if stack_week not in (15, 16, 17):
        raise RuntimeError("Expected CIN-BAL to be a playoff-week stack example")
    stacked = _fill([
        {"name": "Joe Burrow", "position": "QB", "team": "CIN"},
        {"name": "Ja'Marr Chase", "position": "WR", "team": "CIN"},
        {"name": "Tee Higgins", "position": "WR", "team": "CIN"},
        {"name": "Zay Flowers", "position": "WR", "team": "BAL"},
        {"name": "Mark Andrews", "position": "TE", "team": "BAL"},
        {"name": "Brock Bowers", "position": "TE", "team": "LV"},
        {"name": "DK Metcalf", "position": "WR", "team": "PIT"},
    ])
    diffuse = _fill([
        {"name": "Lamar Jackson", "position": "QB", "team": "BAL"},
        {"name": "Puka Nacua", "position": "WR", "team": "LA"},
        {"name": "Amon-Ra St. Brown", "position": "WR", "team": "DET"},
        {"name": "Garrett Wilson", "position": "WR", "team": "NYJ"},
        {"name": "Brock Bowers", "position": "TE", "team": "LV"},
        {"name": "Tee Higgins", "position": "WR", "team": "CIN"},
        {"name": "Courtland Sutton", "position": "WR", "team": "DEN"},
    ])
    return stacked, diffuse, stack_week, "CIN QB+WR+WR with BAL WR+TE bring-back"


def _same_team_no_qb():
    return _fill([
        {"name": "Ja'Marr Chase", "position": "WR", "team": "CIN"},
        {"name": "Tee Higgins", "position": "WR", "team": "CIN"},
        {"name": "Andrei Iosivas", "position": "WR", "team": "CIN"},
        {"name": "Lamar Jackson", "position": "QB", "team": "BAL"},
        {"name": "Brock Bowers", "position": "TE", "team": "LV"},
        {"name": "Garrett Wilson", "position": "WR", "team": "NYJ"},
    ])


if __name__ == "__main__":
    main()
