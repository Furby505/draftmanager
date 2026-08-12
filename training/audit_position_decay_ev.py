"""Paired EV audit for position-decay policy overlays.

This is audit-only. It does not modify the served model or server scoring.

Run:
    python training/audit_position_decay_ev.py --pods 24 --eval-seasons 150
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
from audit_policy_draft_shape import make_position_decay_agent  # noqa: E402
from backtest_policy import (  # noqa: E402
    DEFAULT_COLS,
    DEFAULT_MODEL,
    N_TEAMS,
    _bootstrap_ci,
    make_policy_agent,
    run_draft,
    score_roster,
)
from opponent_field import DEFAULT_SIGMA, build_field, draft_field, load_market_board  # noqa: E402
from outcome_model import load_market_model  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
POS = ("QB", "RB", "WR", "TE")
BASE_LIGHT_SCALE = 1e-5


def roster_pos(roster: list[dict]) -> str:
    return "".join(f"{p}{sum(1 for x in roster if x.get('position') == p)}" for p in POS)


def decay_agent_name(scale: float) -> str:
    multiplier = scale / BASE_LIGHT_SCALE
    label = f"{multiplier:g}x_light".replace(".", "p").replace("-", "m")
    return f"decay_{label}"


def build_agents(model, feature_cols: list[str], candidates: int, decay_scales: list[float]):
    agents = {"policy": make_policy_agent(model, feature_cols, candidates)}
    for scale in decay_scales:
        agents[decay_agent_name(scale)] = make_position_decay_agent(model, feature_cols, candidates, scale)
    return agents


def run_audit(args) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    board = load_market_board()
    outcome = load_market_model()

    print(f"Building shared field ({args.field_rooms} rooms x {args.field_seasons} seasons)...")
    field_rosters = draft_field(args.field_rooms, rng, board=board, sigma=args.sigma)
    field = build_field(field_rosters, model=outcome, n_seasons=args.field_seasons, rng=rng)

    policy_model = joblib.load(args.model)
    feature_cols = json.loads(args.features.read_text())
    agents = build_agents(policy_model, feature_cols, args.candidates, args.decay_scales)

    rows = []
    for pod in range(args.pods):
        our_team = int(np.random.default_rng(args.seed + 1 + pod).integers(N_TEAMS))
        draft_seed = int(np.random.default_rng(args.seed + 7000 + pod).integers(1 << 31))
        season_seed = int(np.random.default_rng(args.seed + 9000 + pod).integers(1 << 31))
        row = {"pod": pod, "our_team": our_team}

        for name, agent in agents.items():
            roster = run_draft(board, our_team, agent, np.random.default_rng(draft_seed), args.sigma)
            sc = score_roster(roster, field, outcome, args.eval_seasons, season_seed)
            for metric, value in sc.items():
                row[f"{name}_{metric}"] = value
            row[f"{name}_pos"] = roster_pos(roster)

        rows.append(row)
        msg = [f"pod {pod + 1}/{args.pods}", f"seat={our_team}"]
        for name in agents:
            msg.append(f"{name}={row[f'{name}_prize_ev']:.3e}")
        print("  " + " ".join(msg))

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for agent in [c.removesuffix("_prize_ev") for c in df.columns if c.endswith("_prize_ev")]:
        if agent == "policy":
            continue
        for metric in ("prize_ev", "finals_rate", "win_rate", "mean_points"):
            p = df[f"policy_{metric}"].to_numpy(float)
            a = df[f"{agent}_{metric}"].to_numpy(float)
            diff = a - p
            lo, hi = _bootstrap_ci(diff, args.bootstrap, args.seed)
            rows.append({
                "agent": agent,
                "metric": metric,
                "policy_mean": float(p.mean()),
                "agent_mean": float(a.mean()),
                "mean_lift": float(diff.mean()),
                "lift_ratio": float(a.mean() / p.mean()) if p.mean() else float("nan"),
                "h2h_winrate": float((a > p).mean()),
                "ci95_lift_lo": lo,
                "ci95_lift_hi": hi,
                "significant": bool(lo > 0 or hi < 0),
            })
    return pd.DataFrame(rows)


def write_report(path: Path, df: pd.DataFrame, summary: pd.DataFrame, args) -> None:
    lines = [
        "# Position-Decay EV Audit",
        "",
        "Paired simulator-EV comparison of bare policy vs reversible position-decay overlays.",
        "",
        "## Setup",
        "",
        f"- Pods: {len(df)}",
        f"- Candidates per pick: {args.candidates}",
        f"- Eval seasons per roster: {args.eval_seasons}",
        f"- Field: {args.field_rooms} rooms x {args.field_seasons} seasons",
        f"- Decay scales: {', '.join(str(x) for x in args.decay_scales)}",
        "- Formula: `adjusted_score = model_score - scale * current_count_at_candidate_position`",
        "- This is in-simulator EV only. It is not real-money proof.",
        "",
        "## Headline",
        "",
        "| overlay | metric | policy | overlay | lift | ratio | H2H | 95% CI lift | signif |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r.agent} | {r.metric} | {r.policy_mean:.3e} | {r.agent_mean:.3e} | "
            f"{r.mean_lift:.3e} | {r.lift_ratio:.3f}x | {r.h2h_winrate:.1%} | "
            f"[{r.ci95_lift_lo:.2e}, {r.ci95_lift_hi:.2e}] | {'YES' if r.significant else 'no'} |"
        )

    lines.extend([
        "",
        "## Read",
        "",
        "- Primary metric is `prize_ev`.",
        "- Positive lift means the decay overlay beat bare policy in the same pod/seat with the same season seed.",
        "- If the CI crosses 0, treat the result as directional only.",
        "- Do not port an overlay into live drafting unless it improves `prize_ev` robustly, not just roster shape.",
        "",
        f"_Per-pod rows: {len(df)}. Full rows are in `{path.with_suffix('.csv').name}`._",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    df.to_csv(path.with_suffix(".csv"), index=False)
    summary.to_csv(path.with_name(path.stem + "_summary.csv"), index=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pods", type=int, default=24)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--eval-seasons", type=int, default=150)
    p.add_argument("--field-rooms", type=int, default=12)
    p.add_argument("--field-seasons", type=int, default=100)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--decay-scales", type=float, nargs="*", default=[1e-5, 2.5e-5, 5e-5])
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--features", type=Path, default=DEFAULT_COLS)
    p.add_argument("--out", type=Path, default=OUT / "position_decay_ev_audit.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = run_audit(args)
    summary = summarize(df, args)
    write_report(args.out, df, summary, args)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
