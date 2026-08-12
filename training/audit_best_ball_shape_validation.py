"""Compare calibrated DK policy draft shapes to real best-ball winning teams.

This is a validation audit, not a trainer. It drafts simulated DK rooms with
the current EV policy and ADP baselines, then compares construction and position
timing against the BBM winner/top-team data extracted from real human drafts.

Run:
    python training/audit_best_ball_shape_validation.py --drafts 300
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

from backtest_policy import (  # noqa: E402
    DEFAULT_COLS,
    DEFAULT_MODEL,
    make_adp_agent,
    make_policy_agent,
    run_draft,
)
from opponent_field import DEFAULT_SIGMA, POS, load_market_board  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
FULL_COHORTS = OUT / "best_ball_winner_data_full_cohorts.csv"
HALF_COHORTS = OUT / "best_ball_winner_data_half_cohorts.csv"
FULL_ROUND_POS = OUT / "best_ball_winner_data_full_round_position.csv"
HALF_ROUND_POS = OUT / "best_ball_winner_data_half_round_position.csv"
FULL_SHAPES = OUT / "best_ball_winner_data_full_shapes.csv"
HALF_SHAPES = OUT / "best_ball_winner_data_half_shapes.csv"


def roster_shape(roster: list[dict]) -> str:
    return "".join(f"{pos}{sum(1 for p in roster if p.get('position') == pos)}" for pos in POS)


def first_pos_rounds(roster: list[dict]) -> dict[str, int | None]:
    out = {}
    for pos in POS:
        rounds = [i + 1 for i, p in enumerate(roster) if p.get("position") == pos]
        out[f"first_{pos}_round"] = min(rounds) if rounds else np.nan
    return out


def round_bucket(round_no: int) -> str:
    r = int(round_no)
    if r <= 3:
        return "R1-3"
    if r <= 6:
        return "R4-6"
    if r <= 9:
        return "R7-9"
    if r <= 12:
        return "R10-12"
    if r <= 15:
        return "R13-15"
    if r <= 18:
        return "R16-18"
    return "R19-20"


def generate_agent_rosters(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    board = load_market_board()
    policy_model = joblib.load(args.model)
    feature_cols = json.loads(Path(args.cols).read_text())
    agents = {
        "policy": make_policy_agent(policy_model, feature_cols, args.candidates),
        "pure_adp": make_adp_agent("pure_adp"),
        "adp_noise": make_adp_agent("adp_noise"),
    }

    roster_rows = []
    pick_rows = []
    for draft_id in range(args.drafts):
        our_team = int(np.random.default_rng(args.seed + 1 + draft_id).integers(12))
        draft_seed = int(np.random.default_rng(args.seed + 7000 + draft_id).integers(1 << 31))
        for agent_name, agent in agents.items():
            roster = run_draft(
                board,
                our_team,
                agent,
                np.random.default_rng(draft_seed),
                args.sigma,
            )
            row = {
                "source": "sim_dk_calibrated",
                "agent": agent_name,
                "draft_id": draft_id,
                "our_team": our_team,
                "roster_shape": roster_shape(roster),
                "roster_shape_18": roster_shape(roster[:18]),
            }
            for pos in POS:
                row[f"n_{pos}"] = sum(1 for p in roster if p.get("position") == pos)
            row.update(first_pos_rounds(roster))
            roster_rows.append(row)

            for pick_no, p in enumerate(roster, start=1):
                pick_rows.append({
                    "source": "sim_dk_calibrated",
                    "agent": agent_name,
                    "draft_id": draft_id,
                    "our_team": our_team,
                    "team_pick_number": pick_no,
                    "round_bucket": round_bucket(pick_no),
                    "player_name": p.get("name", ""),
                    "position_name": p.get("position", ""),
                    "team": p.get("team", ""),
                    "adp": p.get("adp", np.nan),
                })

        if args.progress and (draft_id + 1) % args.progress == 0:
            print(f"[shape-validation] drafted {draft_id + 1:,}/{args.drafts:,}")

    return pd.DataFrame(roster_rows), pd.DataFrame(pick_rows)


def summarize_agent_rosters(rosters: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent, g in rosters.groupby("agent", sort=False):
        row = {"source": "sim_dk_calibrated", "cohort": agent, "rosters": int(len(g))}
        for pos in POS:
            row[f"avg_n_{pos}"] = float(g[f"n_{pos}"].mean())
            row[f"avg_first_{pos}_round"] = float(g[f"first_{pos}_round"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_agent_round_pos(picks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (agent, bucket), g in picks.groupby(["agent", "round_bucket"], sort=False):
        denom = len(g)
        for pos in POS:
            n = int((g["position_name"] == pos).sum())
            rows.append({
                "source": "sim_dk_calibrated",
                "cohort": agent,
                "round_bucket": bucket,
                "position_name": pos,
                "n": n,
                "denom": int(denom),
                "pct": float(n / denom) if denom else 0.0,
            })
    return pd.DataFrame(rows)


def summarize_agent_shapes(rosters: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent, g in rosters.groupby("agent", sort=False):
        for shape_type, col in (("full20", "roster_shape"), ("first18", "roster_shape_18")):
            vc = g[col].value_counts(normalize=True)
            for shape, rate in vc.items():
                rows.append({
                    "source": "sim_dk_calibrated",
                    "cohort": agent,
                    "shape_type": shape_type,
                    "roster_shape": shape,
                    "rate": float(rate),
                    "count": int((g[col] == shape).sum()),
                })
    return pd.DataFrame(rows)


def load_real_cohorts() -> pd.DataFrame:
    frames = []
    for scoring, path in (("full", FULL_COHORTS), ("half", HALF_COHORTS)):
        df = pd.read_csv(path)
        df.insert(0, "source", f"real_bbm_{scoring}")
        frames.append(df.rename(columns={"cohort": "cohort"}))
    return pd.concat(frames, ignore_index=True)


def load_real_round_pos() -> pd.DataFrame:
    frames = []
    for scoring, path in (("full", FULL_ROUND_POS), ("half", HALF_ROUND_POS)):
        df = pd.read_csv(path)
        df.insert(0, "source", f"real_bbm_{scoring}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_real_shapes() -> pd.DataFrame:
    frames = []
    for scoring, path in (("full", FULL_SHAPES), ("half", HALF_SHAPES)):
        df = pd.read_csv(path)
        frames.append(pd.DataFrame({
            "source": f"real_bbm_{scoring}",
            "shape_type": "full18",
            "roster_shape": df["roster_shape"],
            "all_rate": df["all_rate"],
            "pod_winners_rate": df["winner_rate"],
            "pod_top2_rate": df["top2_rate"],
            "winner_lift": df["winner_lift"],
        }))
    return pd.concat(frames, ignore_index=True)


def position_distance(round_pos: pd.DataFrame, sim_agent: str, real_source: str, real_cohort: str) -> float:
    sim = round_pos[
        round_pos["source"].eq("sim_dk_calibrated")
        & round_pos["cohort"].eq(sim_agent)
        & ~round_pos["round_bucket"].eq("R19-20")
    ][["round_bucket", "position_name", "pct"]]
    real = round_pos[
        round_pos["source"].eq(real_source)
        & round_pos["cohort"].eq(real_cohort)
    ][["round_bucket", "position_name", "pct"]]
    merged = sim.merge(real, on=["round_bucket", "position_name"], how="outer", suffixes=("_sim", "_real")).fillna(0)
    return float((merged["pct_sim"] - merged["pct_real"]).abs().mean())


def _fmt(x, nd=2) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{nd}f}"


def md_table(df: pd.DataFrame, cols: list[str], float_cols: set[str] | None = None) -> str:
    if df.empty:
        return "_No rows._"
    float_cols = float_cols or set()
    view = df[cols].copy().fillna("")
    for c in float_cols:
        if c in view:
            view[c] = view[c].map(lambda x: "" if x == "" else _fmt(x, 2))
    headers = [str(c) for c in view.columns]
    rows = [[str(v) for v in row] for row in view.to_numpy()]
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def fmt(row):
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |"

    return "\n".join([
        fmt(headers),
        "| " + " | ".join("-" * w for w in widths) + " |",
        *(fmt(r) for r in rows),
    ])


def write_report(
    path: Path,
    cohorts: pd.DataFrame,
    round_pos: pd.DataFrame,
    shapes: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    sim_cohorts = cohorts[cohorts["source"].eq("sim_dk_calibrated")].copy()
    real_key = cohorts[
        cohorts["source"].eq("real_bbm_full")
        & cohorts["cohort"].isin(["all", "pod_winners", "global_top_1pct"])
    ].copy()

    dist_rows = []
    for agent in ("policy", "pure_adp", "adp_noise"):
        for source, cohort in (
            ("real_bbm_full", "pod_winners"),
            ("real_bbm_full", "global_top_1pct"),
            ("real_bbm_half", "pod_winners"),
        ):
            dist_rows.append({
                "agent": agent,
                "target": f"{source}:{cohort}",
                "round_pos_l1": position_distance(round_pos, agent, source, cohort),
            })
    dist = pd.DataFrame(dist_rows)

    shape_policy = shapes[
        shapes["source"].eq("sim_dk_calibrated")
        & shapes["cohort"].eq("policy")
        & shapes["shape_type"].eq("first18")
    ][["roster_shape", "rate", "count"]].head(12).copy()
    full_shapes = shapes[shapes["source"].eq("real_bbm_full")][
        ["roster_shape", "all_rate", "pod_winners_rate", "winner_lift"]
    ]
    shape_policy = shape_policy.merge(full_shapes, on="roster_shape", how="left")

    key_rounds = round_pos[
        (
            round_pos["source"].eq("sim_dk_calibrated")
            & round_pos["cohort"].eq("policy")
            & round_pos["round_bucket"].isin(["R1-3", "R4-6", "R7-9", "R10-12", "R13-15", "R16-18", "R19-20"])
        )
        | (
            round_pos["source"].eq("real_bbm_full")
            & round_pos["cohort"].eq("pod_winners")
            & round_pos["round_bucket"].isin(["R1-3", "R4-6", "R7-9", "R10-12", "R13-15", "R16-18"])
        )
    ].copy()
    key_rounds["label"] = key_rounds["source"] + ":" + key_rounds["cohort"]

    lines = [
        "# Best-Ball Winner Shape Validation",
        "",
        f"Generated `{args.drafts}` calibrated DK drafts per agent. This compares draft shape only; it does not rescore or retrain the EV model.",
        "",
        "Real BBM data is Underdog 18-round human drafts. DK is 20 rounds, so use this as a structural sanity check, not a hard target.",
        "",
        "## Construction",
        "",
        md_table(
            pd.concat([real_key, sim_cohorts], ignore_index=True, sort=False),
            [
                "source",
                "cohort",
                "rosters",
                "avg_n_QB",
                "avg_first_QB_round",
                "avg_n_RB",
                "avg_first_RB_round",
                "avg_n_WR",
                "avg_first_WR_round",
                "avg_n_TE",
                "avg_first_TE_round",
            ],
            {
                "avg_n_QB",
                "avg_first_QB_round",
                "avg_n_RB",
                "avg_first_RB_round",
                "avg_n_WR",
                "avg_first_WR_round",
                "avg_n_TE",
                "avg_first_TE_round",
            },
        ),
        "",
        "## Round Position Distance",
        "",
        "Lower is closer. This compares position share by round bucket for rounds 1-18.",
        "",
        md_table(dist.sort_values(["target", "round_pos_l1"]), ["agent", "target", "round_pos_l1"], {"round_pos_l1"}),
        "",
        "## Policy Top First-18 Shapes",
        "",
        md_table(
            shape_policy,
            ["roster_shape", "rate", "count", "all_rate", "pod_winners_rate", "winner_lift"],
            {"rate", "all_rate", "pod_winners_rate", "winner_lift"},
        ),
        "",
        "## Policy Vs Real Full-PPR Pod Winners By Round Bucket",
        "",
        md_table(
            key_rounds.sort_values(["round_bucket", "label", "position_name"]),
            ["label", "round_bucket", "position_name", "pct"],
            {"pct"},
        ),
        "",
        "## Readout",
        "",
    ]

    policy = sim_cohorts[sim_cohorts["cohort"].eq("policy")].iloc[0]
    winners = real_key[real_key["cohort"].eq("pod_winners")].iloc[0]
    top1 = real_key[real_key["cohort"].eq("global_top_1pct")].iloc[0]
    lines.extend([
        f"- Policy construction: QB `{policy.avg_n_QB:.2f}`, RB `{policy.avg_n_RB:.2f}`, WR `{policy.avg_n_WR:.2f}`, TE `{policy.avg_n_TE:.2f}`.",
        f"- Real full-PPR pod winners: QB `{winners.avg_n_QB:.2f}`, RB `{winners.avg_n_RB:.2f}`, WR `{winners.avg_n_WR:.2f}`, TE `{winners.avg_n_TE:.2f}`.",
        f"- Real full-PPR global top 1% first RB round is `{top1.avg_first_RB_round:.2f}`; policy first RB round is `{policy.avg_first_RB_round:.2f}`.",
        "- If policy stays much more WR-heavy or much later at RB than winner cohorts after calibration, that is evidence for adding a structural prior or retraining labels, not direct winner-overfit.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--drafts", type=int, default=300)
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--progress", type=int, default=50)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--cols", type=Path, default=DEFAULT_COLS)
    p.add_argument("--out", type=Path, default=OUT / "best_ball_shape_validation_audit.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rosters, picks = generate_agent_rosters(args)

    sim_cohorts = summarize_agent_rosters(rosters)
    sim_round_pos = summarize_agent_round_pos(picks)
    sim_shapes = summarize_agent_shapes(rosters)

    cohorts = pd.concat([load_real_cohorts(), sim_cohorts], ignore_index=True, sort=False)
    round_pos = pd.concat([load_real_round_pos(), sim_round_pos], ignore_index=True, sort=False)
    shapes = pd.concat([load_real_shapes(), sim_shapes], ignore_index=True, sort=False)

    stem = args.out.with_suffix("")
    rosters.to_csv(stem.with_name(stem.name + "_sim_rosters.csv"), index=False)
    picks.to_csv(stem.with_name(stem.name + "_sim_picks.csv"), index=False)
    cohorts.to_csv(stem.with_name(stem.name + "_cohorts.csv"), index=False)
    round_pos.to_csv(stem.with_name(stem.name + "_round_position.csv"), index=False)
    shapes.to_csv(stem.with_name(stem.name + "_shapes.csv"), index=False)
    write_report(args.out, cohorts, round_pos, shapes, args)
    print(f"[shape-validation] wrote {args.out}")


if __name__ == "__main__":
    main()
