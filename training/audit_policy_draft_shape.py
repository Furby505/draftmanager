"""Behavior audit for the served DK EV policy.

Profiles what positions and players the current policy actually drafts across
many DK-style simulated rooms, compared with pure ADP and ADP-noise controls.

Run:
    python training/audit_policy_draft_shape.py --drafts 300 --seats 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_policy import legal_candidate_set, make_adp_agent, make_policy_agent  # noqa: E402
from opponent_field import DEFAULT_SIGMA, _choose, _sample_team_caps, load_market_board  # noqa: E402
from train_policy import DraftState, _apply_pick, snake_team, state_candidate_features  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "server" / "models"
OUT = ROOT / "data" / "processed"
POS = ("QB", "RB", "WR", "TE")
POS_NAME = {0: "QB", 1: "RB", 2: "WR", 3: "TE"}
N_TEAMS = 12
N_ROUNDS = 20
BASE_AGENT_ORDER = ("policy", "decay_light", "decay_medium", "decay_heavy", "pure_adp", "adp_noise")


def agent_name_key(name: str) -> str:
    return name.lower().replace(".", "").replace("'", "").replace("-", " ").strip()


def run_room(board, agent, seat: int, rng: np.random.Generator, sigma: float) -> list[dict]:
    avail = np.ones(len(board), dtype=bool)
    counts = np.zeros((N_TEAMS, len(POS)), dtype=int)
    rosters = [[] for _ in range(N_TEAMS)]
    team_caps = [_sample_team_caps(rng) for _ in range(N_TEAMS)]
    history: list[tuple[int, int]] = []
    rows: list[dict] = []

    for gp in range(N_TEAMS * N_ROUNDS):
        team = snake_team(gp, N_TEAMS)
        if team == seat:
            state = DraftState(board, gp, seat, rosters, counts, avail, team_caps, sigma, "adp_noise", history)
            idx = int(agent(state, rng))
            av = np.where(avail)[0]
            took_rank = int((board.adp[av] < board.adp[idx]).sum())
            round_no = gp // N_TEAMS + 1
            pos = POS_NAME[int(board.pos_code[idx])]
            rows.append({
                "seat": seat + 1,
                "overall_pick": gp + 1,
                "round": round_no,
                "pick_in_roster": len(rosters[seat]) + 1,
                "player": str(board.name[idx]),
                "position": pos,
                "adp": float(board.adp[idx]),
                "better_adp_passed": took_rank,
                "qb_before": int(counts[seat, 0]),
                "rb_before": int(counts[seat, 1]),
                "wr_before": int(counts[seat, 2]),
                "te_before": int(counts[seat, 3]),
            })
        else:
            soft_cap, comfort = team_caps[team]
            picks_left = N_ROUNDS - len(rosters[team])
            idx = int(_choose(board, avail, counts[team], soft_cap, comfort,
                              picks_left, gp // N_TEAMS, N_ROUNDS, sigma, "adp_noise", rng))
        _apply_pick(board, avail, counts, rosters, team, idx)
        history.append((team, int(idx)))

    return rows


def make_position_decay_agent(model, feature_cols: list[str], n_candidates: int, scale: float):
    """Audit-only overlay: penalize candidates as your roster gets heavier at their position."""
    def agent(state: DraftState, rng: np.random.Generator) -> int:
        cands = legal_candidate_set(state, n_candidates)
        if not cands:
            av = np.where(state.avail)[0]
            return int(av[np.argmin(state.board.adp[av])])

        rows = [state_candidate_features(state, ci) for ci in cands]
        X = pd.DataFrame(rows).reindex(columns=feature_cols, fill_value=0).fillna(0)
        scores = np.asarray(model.predict(X), dtype=float)
        for i, ci in enumerate(cands):
            pos_i = int(state.board.pos_code[ci])
            scores[i] -= float(scale) * float(state.counts[state.our_team, pos_i])
        return int(cands[int(np.argmax(scores))])
    return agent


def profile_agent(board, name: str, agent, drafts: int, seats: int, seed: int, sigma: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for draft_id in range(1, drafts + 1):
        for seat in range(seats):
            for row in run_room(board, agent, seat, rng, sigma):
                row["agent"] = name
                row["draft_id"] = draft_id
                rows.append(row)
    return pd.DataFrame(rows)


def pct_table(picks: pd.DataFrame) -> pd.DataFrame:
    counts = picks.groupby(["agent", "round", "position"]).size().rename("n").reset_index()
    denom = picks.groupby(["agent", "round"]).size().rename("denom").reset_index()
    out = counts.merge(denom, on=["agent", "round"])
    out["pct"] = out["n"] / out["denom"]
    pivot = out.pivot_table(index=["agent", "round"], columns="position", values="pct", fill_value=0.0)
    for pos in POS:
        if pos not in pivot:
            pivot[pos] = 0.0
    return pivot[list(POS)].reset_index()


def first_position_rounds(picks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (agent, draft_id, seat), g in picks.groupby(["agent", "draft_id", "seat"]):
        for pos in POS:
            sub = g[g["position"] == pos]
            rows.append({
                "agent": agent,
                "draft_id": draft_id,
                "seat": seat,
                "position": pos,
                "first_round": int(sub["round"].min()) if not sub.empty else None,
            })
    return pd.DataFrame(rows)


def roster_counts(picks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (agent, draft_id, seat), g in picks.groupby(["agent", "draft_id", "seat"]):
        row = {"agent": agent, "draft_id": draft_id, "seat": seat}
        for pos in POS:
            row[pos] = int((g["position"] == pos).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def md_pct(v: float) -> str:
    return f"{100 * float(v):.1f}%"


def write_report(path: Path, picks: pd.DataFrame, by_round: pd.DataFrame,
                 firsts: pd.DataFrame, rosters: pd.DataFrame, args) -> None:
    agent_order = [a for a in BASE_AGENT_ORDER if a in set(picks["agent"])]
    lines = [
        "# DK EV Policy Draft-Shape Audit",
        "",
        "Profiles the current served EV policy in DK-style simulated rooms against ADP controls.",
        "",
        "## Setup",
        "",
        f"- Draft rooms per seat: {args.drafts}",
        f"- Seats profiled: {args.seats}",
        f"- Total policy rosters per agent: {args.drafts * args.seats:,}",
        "- Room format: 12 teams, 20 rounds",
        "- Opponents: ADP-noise drafters",
        "- Policy candidate pool: top 8 legal ADP candidates, matching the served artifact cap",
        f"- Position-decay scales tested: {', '.join(str(x) for x in args.decay_scales)}",
        "- Decay formula: `adjusted_score = model_score - scale * current_count_at_candidate_position`",
        "- Decay is audit-only and not used by the live server.",
        "- Caveat: this is behavior in the model's simulator/market board, not proof the shape is optimal in real contests.",
        "",
        "## Average Final Roster Construction",
        "",
        "| Agent | QB | RB | WR | TE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    roster_mean = rosters.groupby("agent")[list(POS)].mean().reindex(agent_order)
    for agent, r in roster_mean.dropna(how="all").iterrows():
        lines.append(f"| {agent} | {r.QB:.2f} | {r.RB:.2f} | {r.WR:.2f} | {r.TE:.2f} |")

    lines.extend(["", "## First Pick Timing By Position", ""])
    lines.append("| Agent | first QB avg | first RB avg | first WR avg | first TE avg | QB after R10 | RB after R6 | TE by R4 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for agent in agent_order:
        g = firsts[firsts["agent"] == agent]
        if g.empty:
            continue
        vals = {}
        for pos in POS:
            s = g[g["position"] == pos]["first_round"].dropna()
            vals[pos] = float(s.mean()) if len(s) else np.nan
        qb_after_r10 = (g[(g["position"] == "QB")]["first_round"].fillna(99) > 10).mean()
        rb_after_r6 = (g[(g["position"] == "RB")]["first_round"].fillna(99) > 6).mean()
        te_by_r4 = (g[(g["position"] == "TE")]["first_round"].fillna(99) <= 4).mean()
        lines.append(
            f"| {agent} | {vals['QB']:.2f} | {vals['RB']:.2f} | {vals['WR']:.2f} | {vals['TE']:.2f} | "
            f"{md_pct(qb_after_r10)} | {md_pct(rb_after_r6)} | {md_pct(te_by_r4)} |"
        )

    lines.extend(["", "## Position Share By Round", ""])
    for agent in agent_order:
        sub = by_round[by_round["agent"] == agent].copy()
        if sub.empty:
            continue
        lines.extend([f"### {agent}", "", "| Round | QB | RB | WR | TE | top |", "| ---: | ---: | ---: | ---: | ---: | --- |"])
        for _, r in sub.iterrows():
            vals = {pos: float(r.get(pos, 0.0)) for pos in POS}
            top = max(POS, key=lambda p: vals[p])
            lines.append(
                f"| {int(r['round'])} | {md_pct(vals['QB'])} | {md_pct(vals['RB'])} | "
                f"{md_pct(vals['WR'])} | {md_pct(vals['TE'])} | {top} |"
            )
        lines.append("")

    lines.extend(["## Named Player Frequency", ""])
    target_names = ["brock bowers", "trey mcbride"]
    lines.append("| Agent | Player | draft rate | avg round when drafted |")
    lines.append("| --- | --- | ---: | ---: |")
    total_rosters = args.drafts * args.seats
    for agent in agent_order:
        ag = picks[picks["agent"] == agent].copy()
        ag["name_key"] = ag["player"].map(agent_name_key)
        for target in target_names:
            sub = ag[ag["name_key"] == target]
            rate = len(sub) / max(total_rosters, 1)
            avg_round = sub["round"].mean() if len(sub) else np.nan
            lines.append(f"| {agent} | {target.title()} | {md_pct(rate)} | {avg_round:.2f} |")

    lines.extend(["", "## Reach / Chalk Behavior", ""])
    lines.append("`better_adp_passed` = how many available players with better ADP were passed over for the pick.")
    lines.append("")
    lines.append("| Agent | mean better ADP passed | median | p90 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for agent in agent_order:
        s = picks[picks["agent"] == agent]["better_adp_passed"]
        if s.empty:
            continue
        lines.append(f"| {agent} | {s.mean():.2f} | {s.median():.1f} | {s.quantile(0.9):.1f} |")

    if any(a.startswith("decay_") for a in agent_order):
        lines.extend([
            "",
            "## Position-Decay Test Read",
            "",
            "The decay agents are reversible audit overlays. They do not retrain the model and do not affect the live server.",
            "",
            "Useful signs to compare against `policy`:",
            "",
            "- First QB round moving earlier means the overlay fights the QB mega-fade.",
            "- TE by R4 moving lower means it fights the Bowers/McBride spike.",
            "- First RB round moving earlier means it fights the late-RB lean.",
            "- Final roster counts moving toward sane ranges without wrecking WR depth means the overlay is plausible.",
            "",
            "This report only checks draft shape. If a decay shape looks better, the next step is a paired EV/backtest run.",
        ])

    lines.extend([
        "",
        "## Findings",
        "",
        "- The report should be read as a draft-shape audit: it identifies repeated behavior, not whether the behavior is profitable.",
        "- If policy QB timing is far later than both ADP controls, that is the model's learned QB-fade showing up in full drafts.",
        "- If TE by round 4 is materially higher than controls, that confirms the Bowers/McBride elite-TE spike behavior.",
        "- If RB after round 6 is high, the model is structurally delaying RB and relying on later RB value/volume.",
        "",
        "## Output Files",
        "",
        "- `data/processed/dk_ev_draft_shape_picks.csv`",
        "- `data/processed/dk_ev_draft_shape_by_round.csv`",
        "- `data/processed/dk_ev_draft_shape_first_positions.csv`",
        "- `data/processed/dk_ev_draft_shape_rosters.csv`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drafts", type=int, default=300)
    p.add_argument("--seats", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--decay-scales", type=float, nargs="*", default=[1e-5, 2.5e-5, 5e-5],
                   help="Audit-only position decay scales. Pass no values to skip decay agents.")
    p.add_argument("--model", type=Path, default=MODELS / "model_dk_ev_policy.joblib")
    p.add_argument("--features", type=Path, default=MODELS / "dk_ev_policy_feature_cols.json")
    p.add_argument("--out", type=Path, default=OUT / "dk_ev_draft_shape_audit.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    board = load_market_board()
    model = joblib.load(args.model)
    feature_cols = json.loads(args.features.read_text())

    agents = {
        "policy": make_policy_agent(model, feature_cols, 8),
    }
    for label, scale in zip(("light", "medium", "heavy"), args.decay_scales):
        agents[f"decay_{label}"] = make_position_decay_agent(model, feature_cols, 8, scale)
    agents["pure_adp"] = make_adp_agent("pure_adp")
    agents["adp_noise"] = make_adp_agent("adp_noise")

    frames = []
    for i, (name, agent) in enumerate(agents.items()):
        print(f"Profiling {name}...")
        frames.append(profile_agent(board, name, agent, args.drafts, args.seats, args.seed + i * 1000, args.sigma))
    picks = pd.concat(frames, ignore_index=True)
    by_round = pct_table(picks)
    firsts = first_position_rounds(picks)
    rosters = roster_counts(picks)

    OUT.mkdir(parents=True, exist_ok=True)
    picks.to_csv(OUT / "dk_ev_draft_shape_picks.csv", index=False)
    by_round.to_csv(OUT / "dk_ev_draft_shape_by_round.csv", index=False)
    firsts.to_csv(OUT / "dk_ev_draft_shape_first_positions.csv", index=False)
    rosters.to_csv(OUT / "dk_ev_draft_shape_rosters.csv", index=False)
    write_report(args.out, picks, by_round, firsts, rosters, args)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
