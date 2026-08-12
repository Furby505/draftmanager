"""A/B test draft-shape overlays against the calibrated DK EV policy.

The BBM winner-shape audit showed the bare policy is early-TE, late-RB, and
very late-QB versus real winner cohorts. This script tests a reversible overlay
that nudges only the online pick score, leaving the trained model untouched.

Run:
    python training/audit_shape_overlay_ev.py --pods 60
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

import best_ball as bb  # noqa: E402
from backtest_policy import (  # noqa: E402
    DEFAULT_COLS,
    DEFAULT_MODEL,
    N_ROUNDS,
    N_TEAMS,
    _bootstrap_ci,
    legal_candidate_set,
    make_adp_agent,
    make_policy_agent,
    run_draft,
    score_roster,
)
from opponent_field import DEFAULT_SIGMA, build_field, draft_field, load_market_board  # noqa: E402
from outcome_model import load_market_model  # noqa: E402
from train_policy import state_candidate_features  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"


def _candidate_pos(row: dict) -> str:
    if row.get("cand_pos_QB"):
        return "QB"
    if row.get("cand_pos_RB"):
        return "RB"
    if row.get("cand_pos_WR"):
        return "WR"
    return "TE"


def shape_overlay_bonus(row: dict, strength: float) -> float:
    """Return an EV-scale bonus/penalty.

    Positive means "prefer this candidate"; negative means "require clearer model
    edge." The weights are intentionally small relative to the model's holdout
    MAE (~8.9e-5) and are tested before any live use.
    """
    if strength <= 0:
        return 0.0

    unit = 1e-5 * float(strength)
    rnd = int(row.get("round", 0))
    pos = _candidate_pos(row)
    n_qb = int(row.get("n_QB", 0))
    n_rb = int(row.get("n_RB", 0))
    n_te = int(row.get("n_TE", 0))
    n_wr = int(row.get("n_WR", 0))
    bonus = 0.0

    # The real winner data says early RB mattered in 2024 full/half PPR. This
    # does not force RB, but it reduces the model's willingness to wait until
    # round 5+ for the first one.
    if pos == "RB":
        if rnd <= 3 and n_rb == 0:
            bonus += 4.0 * unit
        elif rnd <= 5 and n_rb <= 1:
            bonus += 2.5 * unit
        elif 7 <= rnd <= 12 and n_rb < 5:
            bonus += 1.0 * unit

    # The bare model takes elite TE extremely early. Keep that possible only
    # when the learned EV edge is big enough to overcome this small structural
    # prior.
    if pos == "TE":
        if rnd <= 3:
            bonus -= (5.0 if n_te == 0 else 8.0) * unit
        elif rnd <= 6 and n_te >= 1:
            bonus -= 2.0 * unit

    # The model waits too long at QB. Nudge the first QB into the middle rounds,
    # then make zero-QB past round 11 increasingly uncomfortable.
    if pos == "QB":
        if 7 <= rnd <= 10 and n_qb == 0:
            bonus += 3.0 * unit
        elif 11 <= rnd <= 13 and n_qb == 0:
            bonus += 6.0 * unit
        elif rnd >= 14 and n_qb == 0:
            bonus += 10.0 * unit
        elif rnd >= 15 and n_qb == 1:
            bonus += 3.0 * unit

    # Avoid turning the 20-round DK build into WR/RB hoarding when the roster is
    # already deep enough at one position.
    if pos == "WR" and n_wr >= 8 and rnd <= 15:
        bonus -= 1.0 * unit
    if pos == "RB" and n_rb >= 7 and rnd <= 15:
        bonus -= 1.0 * unit

    return float(bonus)


def make_shape_overlay_agent(model, feature_cols: list[str], n_candidates: int, strength: float):
    def agent(state, rng: np.random.Generator) -> int:
        cands = legal_candidate_set(state, n_candidates)
        if not cands:
            from backtest_policy import _legal_fallback
            return _legal_fallback(state)
        rows = [state_candidate_features(state, ci) for ci in cands]
        X = pd.DataFrame(rows).reindex(columns=feature_cols, fill_value=0).fillna(0)
        pred = np.asarray(model.predict(X), dtype=float)
        bonus = np.asarray([shape_overlay_bonus(r, strength) for r in rows], dtype=float)
        return int(cands[int(np.argmax(pred + bonus))])
    return agent


def roster_shape(roster: list[dict]) -> str:
    return "".join(f"{pos}{sum(1 for p in roster if p.get('position') == pos)}" for pos in ("QB", "RB", "WR", "TE"))


def first_round(roster: list[dict], pos: str) -> float:
    xs = [i + 1 for i, p in enumerate(roster) if p.get("position") == pos]
    return float(min(xs)) if xs else float("nan")


def run_overlay_audit(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    board = load_market_board()
    model_out = load_market_model()

    print(f"[shape-overlay] building field ({args.field_rooms} rooms x {args.field_seasons} seasons)")
    field_rosters = draft_field(args.field_rooms, rng, board=board, sigma=args.sigma)
    field = build_field(field_rosters, model=model_out, n_seasons=args.field_seasons, rng=rng)

    policy_model = joblib.load(args.model)
    feature_cols = json.loads(Path(args.cols).read_text())

    agents = {
        "policy": make_policy_agent(policy_model, feature_cols, args.candidates),
        "pure_adp": make_adp_agent("pure_adp"),
        "adp_noise": make_adp_agent("adp_noise"),
    }
    for strength in args.strengths:
        agents[f"shape_{strength:g}"] = make_shape_overlay_agent(
            policy_model,
            feature_cols,
            args.candidates,
            strength,
        )

    recs = []
    for pod in range(args.pods):
        our_team = int(np.random.default_rng(args.seed + 1 + pod).integers(N_TEAMS))
        draft_seed = int(np.random.default_rng(args.seed + 7000 + pod).integers(1 << 31))
        season_seed = int(np.random.default_rng(args.seed + 9000 + pod).integers(1 << 31))
        for name, agent in agents.items():
            roster = run_draft(board, our_team, agent, np.random.default_rng(draft_seed), args.sigma)
            sc = score_roster(roster, field, model_out, args.eval_seasons, season_seed)
            rec = {
                "pod": pod,
                "our_team": our_team,
                "agent": name,
                **sc,
                "weekly_points": float(sc["mean_points"] / 17.0),
                "roster_shape": roster_shape(roster),
                "first_QB_round": first_round(roster, "QB"),
                "first_RB_round": first_round(roster, "RB"),
                "first_WR_round": first_round(roster, "WR"),
                "first_TE_round": first_round(roster, "TE"),
                "n_QB": sum(1 for p in roster if p.get("position") == "QB"),
                "n_RB": sum(1 for p in roster if p.get("position") == "RB"),
                "n_WR": sum(1 for p in roster if p.get("position") == "WR"),
                "n_TE": sum(1 for p in roster if p.get("position") == "TE"),
            }
            recs.append(rec)
        if args.progress and (pod + 1) % args.progress == 0:
            print(f"[shape-overlay] pod {pod + 1:,}/{args.pods:,}")
    return pd.DataFrame(recs)


def summarize(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    policy = df[df["agent"].eq("policy")].set_index("pod")
    for agent, g in df.groupby("agent", sort=False):
        gp = g.set_index("pod")
        row = {
            "agent": agent,
            "pods": int(len(g)),
            "prize_ev": float(g["prize_ev"].mean()),
            "finals_rate": float(g["finals_rate"].mean()),
            "win_rate": float(g["win_rate"].mean()),
            "weekly_points": float(g["weekly_points"].mean()),
            "avg_n_QB": float(g["n_QB"].mean()),
            "avg_n_RB": float(g["n_RB"].mean()),
            "avg_n_WR": float(g["n_WR"].mean()),
            "avg_n_TE": float(g["n_TE"].mean()),
            "avg_first_QB_round": float(g["first_QB_round"].mean()),
            "avg_first_RB_round": float(g["first_RB_round"].mean()),
            "avg_first_TE_round": float(g["first_TE_round"].mean()),
        }
        if agent != "policy":
            aligned = gp.join(policy, lsuffix="", rsuffix="_policy")
            diff = aligned["prize_ev"] - aligned["prize_ev_policy"]
            lo, hi = _bootstrap_ci(diff.to_numpy(float), args.bootstrap, args.seed)
            row.update({
                "ev_lift_vs_policy": float(diff.mean()),
                "ev_ratio_vs_policy": float(g["prize_ev"].mean() / policy["prize_ev"].mean()),
                "h2h_vs_policy": float((diff > 0).mean()),
                "ci95_lift_lo": lo,
                "ci95_lift_hi": hi,
                "significant": bool(lo > 0 or hi < 0),
            })
        else:
            row.update({
                "ev_lift_vs_policy": 0.0,
                "ev_ratio_vs_policy": 1.0,
                "h2h_vs_policy": float("nan"),
                "ci95_lift_lo": float("nan"),
                "ci95_lift_hi": float("nan"),
                "significant": False,
            })
        rows.append(row)
    summary = pd.DataFrame(rows)

    shape_rows = []
    for agent, g in df.groupby("agent", sort=False):
        vc = g["roster_shape"].value_counts(normalize=True).head(10)
        for shape, rate in vc.items():
            shape_rows.append({
                "agent": agent,
                "roster_shape": shape,
                "rate": float(rate),
                "count": int((g["roster_shape"] == shape).sum()),
            })
    return summary, pd.DataFrame(shape_rows)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    view = df[cols].copy().fillna("")
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: "" if x == "" else f"{float(x):.4g}")
    headers = [str(c) for c in view.columns]
    rows = [[str(v) for v in row] for row in view.to_numpy()]
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def fmt(row):
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |"

    return "\n".join([fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |", *(fmt(r) for r in rows)])


def write_report(path: Path, summary: pd.DataFrame, shapes: pd.DataFrame, args: argparse.Namespace) -> None:
    shape_agents = summary[summary["agent"].str.startswith("shape_")].copy()
    best = shape_agents.sort_values("prize_ev", ascending=False).iloc[0] if not shape_agents.empty else None
    policy = summary[summary["agent"].eq("policy")].iloc[0]

    lines = [
        "# Shape Overlay EV Audit",
        "",
        f"Pods: `{args.pods}`. Eval seasons/roster: `{args.eval_seasons}`. Field: `{args.field_rooms}` rooms x `{args.field_seasons}` seasons.",
        "",
        "This tests a reversible online score overlay on top of the existing calibrated DK EV policy. It does not retrain the model and does not change the server.",
        "",
        "## Summary",
        "",
        md_table(
            summary.sort_values("prize_ev", ascending=False),
            [
                "agent",
                "prize_ev",
                "ev_ratio_vs_policy",
                "h2h_vs_policy",
                "weekly_points",
                "avg_first_RB_round",
                "avg_first_QB_round",
                "avg_first_TE_round",
                "avg_n_RB",
                "avg_n_WR",
                "avg_n_QB",
                "avg_n_TE",
                "ci95_lift_lo",
                "ci95_lift_hi",
                "significant",
            ],
        ),
        "",
        "## Top Shapes",
        "",
        md_table(shapes[shapes["agent"].isin(["policy", best["agent"] if best is not None else ""])], ["agent", "roster_shape", "rate", "count"]),
        "",
        "## Readout",
        "",
    ]
    if best is not None:
        verdict = "PASS" if best["prize_ev"] > policy["prize_ev"] and best["ci95_lift_lo"] > 0 else "NO LIVE CHANGE"
        lines.extend([
            f"- Best overlay: `{best.agent}` at `{best.prize_ev:.3e}` prize-EV vs bare policy `{policy.prize_ev:.3e}`.",
            f"- Weekly points: `{best.weekly_points:.1f}` vs bare policy `{policy.weekly_points:.1f}`.",
            f"- Shape movement: first RB `{best.avg_first_RB_round:.2f}` vs `{policy.avg_first_RB_round:.2f}`, first QB `{best.avg_first_QB_round:.2f}` vs `{policy.avg_first_QB_round:.2f}`, first TE `{best.avg_first_TE_round:.2f}` vs `{policy.avg_first_TE_round:.2f}`.",
            f"- Verdict: **{verdict}**.",
        ])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pods", type=int, default=60)
    p.add_argument("--eval-seasons", type=int, default=400)
    p.add_argument("--field-rooms", type=int, default=24)
    p.add_argument("--field-seasons", type=int, default=200)
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--strengths", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--progress", type=int, default=10)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--cols", type=Path, default=DEFAULT_COLS)
    p.add_argument("--out", type=Path, default=OUT / "shape_overlay_ev_audit.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = run_overlay_audit(args)
    summary, shapes = summarize(df, args)
    stem = args.out.with_suffix("")
    df.to_csv(stem.with_name(stem.name + "_rows.csv"), index=False)
    summary.to_csv(stem.with_name(stem.name + "_summary.csv"), index=False)
    shapes.to_csv(stem.with_name(stem.name + "_shapes.csv"), index=False)
    write_report(args.out, summary, shapes, args)
    print(f"[shape-overlay] wrote {args.out}")


if __name__ == "__main__":
    main()
