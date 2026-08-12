"""
Availability / injury layer audit — manual-inspection report.

Run: python training/audit_availability.py
Writes: data/processed/availability_audit.md

Reports (per the spec):
  - average games missed by position
  - games-missed distribution by position
  - top players most affected in expected season points
  - top players most affected in Week 15-17 availability
  - fragile players whose tournament EV (playoff points) drops most
  - advance-rate / roster-score comparison before vs after (iron-man vs availability)
  - sanity: elite-but-fragile players are reduced, NOT deleted
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from bracket import evaluate_roster
from opponent_field import build_field, draft_field
from outcome_model import FULL_SEASON, load_default_model
from season_sim import simulate_roster

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "availability_audit.md"
PROJECTIONS = ROOT / "server" / "models" / "projections.json"


def main():
    model = load_default_model()
    proj = json.loads(PROJECTIONS.read_text())
    roster = [{"player_id": p["player_id"], "name": p["player_display_name"],
               "position": p["position"], "team": p["recent_team"]} for p in proj]
    ppg = {p["player_id"]: p["proj_points"] / FULL_SEASON for p in proj}
    ovr = {p["player_id"]: p.get("overall_rank", 9999) for p in proj}

    ctx = model.prepare_roster(roster)
    rng = np.random.default_rng(0)
    K = 2000
    active = model.sample_availability(ctx, K, FULL_SEASON, rng)     # [N,K,17]
    missed = (~active).sum(axis=2)                                   # [N,K]
    exp_missed = missed.mean(axis=1)                                 # [N]
    playoff_rate = active[:, :, 14:17].mean(axis=(1, 2))            # [N] wk15-17 active
    names = ctx.names
    pos = np.array(ctx.positions)
    ppg_arr = np.array([ppg.get(pid, 0.0) for pid in ctx.player_ids])
    ovr_arr = np.array([ovr.get(pid, 9999) for pid in ctx.player_ids])
    pts_lost = ppg_arr * exp_missed
    playoff_pts_lost = ppg_arr * 3 * (1 - playoff_rate)             # EV-relevant hit

    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit("# Availability / Injury Audit")
    emit(f"_{len(roster)} projected players, {K} simulated seasons each. "
         "Iron-man baseline = availability off._\n")

    # ---- avg games missed by position ------------------------------------
    emit("## Average games missed by position")
    emit("| Pos | players | mean games missed | season-long active rate | Wk15-17 active rate |")
    emit("|---|---|---|---|---|")
    for p in ("QB", "RB", "WR", "TE"):
        m = pos == p
        emit(f"| {p} | {m.sum()} | {exp_missed[m].mean():.2f} | "
             f"{active[m].mean():.3f} | {playoff_rate[m].mean():.3f} |")

    # ---- games-missed distribution by position ---------------------------
    emit("\n## Games-missed distribution by position (share of player-seasons)")
    emit("| Pos | 0 | 1-2 | 3-5 | 6-9 | 10+ |")
    emit("|---|---|---|---|---|---|")
    for p in ("QB", "RB", "WR", "TE"):
        mm = missed[pos == p].ravel()
        tot = len(mm)
        b = [np.mean(mm == 0), np.mean((mm >= 1) & (mm <= 2)),
             np.mean((mm >= 3) & (mm <= 5)), np.mean((mm >= 6) & (mm <= 9)),
             np.mean(mm >= 10)]
        emit(f"| {p} | {b[0]:.0%} | {b[1]:.0%} | {b[2]:.0%} | {b[3]:.0%} | {b[4]:.0%} |")

    # ---- top affected by expected season points --------------------------
    emit("\n## Top 15 — most expected SEASON points lost to injury")
    emit("| Player | Pos | proj ppg | exp games missed | exp pts lost |")
    emit("|---|---|---|---|---|")
    for i in np.argsort(-pts_lost)[:15]:
        emit(f"| {names[i]} | {pos[i]} | {ppg_arr[i]:.1f} | "
             f"{exp_missed[i]:.2f} | {pts_lost[i]:.1f} |")

    # ---- top affected in playoff availability (notable players) ----------
    notable = ovr_arr <= 150
    emit("\n## Top 15 — lowest Week 15-17 availability (among top-150 ADP)")
    emit("| Player | Pos | proj ppg | Wk15-17 active rate | p_major |")
    emit("|---|---|---|---|---|")
    idx = np.where(notable)[0]
    for i in idx[np.argsort(playoff_rate[idx])][:15]:
        emit(f"| {names[i]} | {pos[i]} | {ppg_arr[i]:.1f} | "
             f"{playoff_rate[i]:.3f} | {ctx.p_major[i]:.2f} |")

    # ---- fragile players whose tournament EV drops most ------------------
    emit("\n## Top 15 — biggest PLAYOFF EV hit (proj ppg x expected Wk15-17 misses)")
    emit("These are the players whose injury risk most erodes tournament equity.")
    emit("| Player | Pos | proj ppg | Wk15-17 active | playoff pts at risk |")
    emit("|---|---|---|---|---|")
    for i in np.argsort(-playoff_pts_lost)[:15]:
        emit(f"| {names[i]} | {pos[i]} | {ppg_arr[i]:.1f} | "
             f"{playoff_rate[i]:.3f} | {playoff_pts_lost[i]:.1f} |")

    # ---- before/after advance rates (full 20-man drafted rosters) --------
    emit("\n## Advance-rate / score comparison: iron-man vs availability")
    emit("Full 20-man drafted rosters scored against the field, both A/B-matched "
         "(field also iron-man vs availability).")
    fr = draft_field(8, np.random.default_rng(4))                   # 96 opponents -> field
    samples = draft_field(1, np.random.default_rng(40))[:3]         # 3 full rosters to score
    field_off = build_field(fr, n_seasons=150, rng=np.random.default_rng(5),
                            availability=False)
    field_on = build_field(fr, n_seasons=150, rng=np.random.default_rng(5),
                           availability=True)
    emit("| Roster | mean pts (iron-man -> avail) | finals rate (iron-man -> avail) |")
    emit("|---|---|---|")
    for k, ros in enumerate(samples):
        before = evaluate_roster(simulate_roster(ros, 4000, np.random.default_rng(6),
                                                 availability=False), field_off)
        after = evaluate_roster(simulate_roster(ros, 4000, np.random.default_rng(6),
                                                availability=True), field_on)
        emit(f"| drafted roster #{k+1} | {before.mean_points:.0f} -> "
             f"{after.mean_points:.0f} | {before.finals_rate:.4f} -> "
             f"{after.finals_rate:.4f} |")

    # ---- sanity: elite-but-fragile reduced, not deleted ------------------
    emit("\n## Sanity: elite players are reduced, NOT deleted")
    emit("| Player | Pos | iron-man season pts | avail season pts | retained |")
    emit("|---|---|---|---|---|")
    for nm in ["Christian McCaffrey", "Bijan Robinson", "Ja'Marr Chase",
               "Josh Allen", "Saquon Barkley"]:
        ros = [{"name": nm, "position": _pos(nm, proj), "team": _team(nm, proj)}]
        on = simulate_roster(ros, 4000, np.random.default_rng(7), availability=True)
        off = simulate_roster(ros, 4000, np.random.default_rng(7), availability=False)
        emit(f"| {nm} | {_pos(nm, proj)} | {off.total_points.mean():.0f} | "
             f"{on.total_points.mean():.0f} | "
             f"{on.total_points.mean()/max(off.total_points.mean(),1):.0%} |")

    emit("\n_Overcorrection check: elite retained shares should sit ~80-90%, not "
         "near zero. Position means should price RB > WR/TE > QB fragility._")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")


def _pos(name, proj):
    from outcome_model import normalize_name
    nm = normalize_name(name)
    for p in proj:
        if normalize_name(p["player_display_name"]) == nm:
            return p["position"]
    return "WR"


def _team(name, proj):
    from outcome_model import normalize_name
    nm = normalize_name(name)
    for p in proj:
        if normalize_name(p["player_display_name"]) == nm:
            return p["recent_team"]
    return ""


if __name__ == "__main__":
    main()
