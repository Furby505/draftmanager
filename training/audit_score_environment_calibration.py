"""Audit simulator scoring level against real best-ball score baselines.

This is a calibration diagnostic only. It does not train or change the served
EV model. The goal is to explain whether inflated simulated team scores are
coming from the draft room, the DK scoring format, or the outcome-model toggles.

Run:
    python training/audit_score_environment_calibration.py --rooms 12 --seasons 80
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from opponent_field import MARKET_PROJ_SCALE, draft_field, load_market_board, POS  # noqa: E402
from outcome_model import CorrelatedOutcomeModel, load_market_model  # noqa: E402
from season_sim import simulate_roster  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
FULL_REAL = OUT / "best_ball_winner_data_full_rosters.csv"
HALF_REAL = OUT / "best_ball_winner_data_half_rosters.csv"


@dataclass(frozen=True)
class Variant:
    name: str
    model: CorrelatedOutcomeModel
    availability: bool | None = None
    handcuff: bool | None = None
    matchup: bool | None = None
    proj_scale: float = 1.0


def _q(s: pd.Series, p: float) -> float:
    return float(s.quantile(p)) if len(s) else float("nan")


def summarize_real(path: Path, scoring: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    cohorts = [
        ("all_humans", df),
        ("pod_winners", df[df["pod_winner"].eq(1)]),
        ("pod_top2", df[df["pod_top2"].eq(1)]),
        ("global_top_pct", df[df["global_top_pct"].eq(1)]),
        ("global_top_1pct", df[df["global_top_1pct"].eq(1)]),
    ]
    rows = []
    for cohort, g in cohorts:
        if g.empty:
            continue
        rows.append({
            "source": f"real_bbm_{scoring}",
            "variant": cohort,
            "teams": int(len(g)),
            "weekly_avg": float(g["score_weekly_avg"].mean()),
            "weekly_p10": _q(g["score_weekly_avg"], 0.10),
            "weekly_p50": _q(g["score_weekly_avg"], 0.50),
            "weekly_p90": _q(g["score_weekly_avg"], 0.90),
            "wk1_14_total": float(g["score_total"].mean()),
            "wk1_17_total": float("nan"),
        })
    return pd.DataFrame(rows)


def roster_counts(roster: list[dict]) -> dict[str, int]:
    return {f"n_{p}": sum(1 for r in roster if r.get("position") == p) for p in POS}


def simulate_variant(
    rosters: list[list[dict]],
    variant: Variant,
    n_seasons: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for i, roster in enumerate(rosters):
        sim_roster = roster
        if variant.proj_scale != 1.0:
            sim_roster = []
            for p in roster:
                q = dict(p)
                if q.get("proj_points") is not None:
                    q["proj_points"] = float(q["proj_points"]) * variant.proj_scale
                sim_roster.append(q)
        res = simulate_roster(
            sim_roster,
            n_seasons=n_seasons,
            rng=rng,
            model=variant.model,
            availability=variant.availability,
            handcuff=variant.handcuff,
            matchup=variant.matchup,
        )
        weekly = res.weekly_lineup
        proj_points = [
            float(p.get("proj_points", 0.0) or 0.0)
            for p in sim_roster
        ]
        row = {
            "source": "sim_dk_adp_noise",
            "variant": variant.name,
            "room": int(i // 12),
            "seat": int(i % 12) + 1,
            "seasons": int(n_seasons),
            "weekly_avg": float(weekly.mean()),
            "weekly_sd": float(weekly.std()),
            "weekly_p10": float(np.quantile(weekly, 0.10)),
            "weekly_p50": float(np.quantile(weekly, 0.50)),
            "weekly_p90": float(np.quantile(weekly, 0.90)),
            "wk1_14_total": float(weekly[:, :14].sum(axis=1).mean()),
            "wk1_17_total": float(weekly.sum(axis=1).mean()),
            "roster_proj_points": float(np.sum(proj_points)),
            "roster_proj_ppg_sum": float(np.sum(proj_points) / 17.0),
        }
        row.update(roster_counts(roster))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sim(roster_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in roster_rows.groupby("variant", sort=False):
        rows.append({
            "source": "sim_dk_adp_noise",
            "variant": variant,
            "teams": int(len(g)),
            "weekly_avg": float(g["weekly_avg"].mean()),
            "weekly_p10": _q(g["weekly_avg"], 0.10),
            "weekly_p50": _q(g["weekly_avg"], 0.50),
            "weekly_p90": _q(g["weekly_avg"], 0.90),
            "wk1_14_total": float(g["wk1_14_total"].mean()),
            "wk1_17_total": float(g["wk1_17_total"].mean()),
            "roster_proj_points": float(g["roster_proj_points"].mean()),
            "avg_qb": float(g["n_QB"].mean()),
            "avg_rb": float(g["n_RB"].mean()),
            "avg_wr": float(g["n_WR"].mean()),
            "avg_te": float(g["n_TE"].mean()),
        })
    return pd.DataFrame(rows)


def board_projection_summary(board) -> pd.DataFrame:
    if board.proj_points is None:
        return pd.DataFrame()
    rows = []
    buckets = [
        ("1-36", board.adp <= 36),
        ("37-96", (board.adp > 36) & (board.adp <= 96)),
        ("97-168", (board.adp > 96) & (board.adp <= 168)),
        ("169-240", (board.adp > 168) & (board.adp <= 240)),
    ]
    for label, mask in buckets:
        for pos_idx, pos in enumerate(POS):
            m = mask & (board.pos_code == pos_idx)
            if not np.any(m):
                continue
            rows.append({
                "adp_bucket": label,
                "pos": pos,
                "players": int(np.sum(m)),
                "market_ppg": float(np.mean(board.proj_points[m] / 17.0)),
                "adp_min": float(np.min(board.adp[m])),
                "adp_max": float(np.max(board.adp[m])),
            })
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    view = df[cols].copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    view = view.fillna("")
    headers = [str(c) for c in view.columns]
    rows = [[str(v) for v in row] for row in view.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    return "\n".join([fmt(headers), sep, *(fmt(row) for row in rows)])


def write_report(
    out_path: Path,
    summary: pd.DataFrame,
    board_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    real_full_all = summary[
        summary["source"].eq("real_bbm_full") & summary["variant"].eq("all_humans")
    ]
    real_half_all = summary[
        summary["source"].eq("real_bbm_half") & summary["variant"].eq("all_humans")
    ]
    real_full_winners = summary[
        summary["source"].eq("real_bbm_full") & summary["variant"].eq("pod_winners")
    ]
    sim = summary[summary["source"].eq("sim_dk_adp_noise")].copy()
    full_avg = float(real_full_all["weekly_avg"].iloc[0]) if not real_full_all.empty else np.nan
    half_avg = float(real_half_all["weekly_avg"].iloc[0]) if not real_half_all.empty else np.nan
    full_winner_avg = (
        float(real_full_winners["weekly_avg"].iloc[0]) if not real_full_winners.empty else np.nan
    )
    sim["gap_vs_full_all"] = sim["weekly_avg"] - full_avg
    sim["gap_vs_half_all"] = sim["weekly_avg"] - half_avg
    current_avg = float(sim.loc[sim["variant"].eq("current"), "weekly_avg"].iloc[0])
    sim["delta_vs_current"] = sim["weekly_avg"] - current_avg

    lines = [
        "# Score Environment Calibration Audit",
        "",
        f"Run seed: `{args.seed}`. Simulated rooms: `{args.rooms}`. Seasons per roster: `{args.seasons}`.",
        f"Default market projection scale: `{MARKET_PROJ_SCALE:.2f}`.",
        "",
        "This is a scoring-level audit only. It does not change the EV model or autodraft.",
        "",
        "## Real baselines",
        "",
        "Real BBM rows are 2024 human Underdog drafts scored over Weeks 1-14. The full-PPR file is a diagnostic rescore; the half-PPR file matches Underdog scoring more closely. Both use 18-player Underdog rosters, so they are not a perfect DK 20-player baseline.",
        "",
        _md_table(
            summary[summary["source"].str.startswith("real_bbm")],
            ["source", "variant", "teams", "weekly_avg", "weekly_p10", "weekly_p50", "weekly_p90", "wk1_14_total"],
        ),
        "",
        "## Simulated DK ADP-noise room",
        "",
        "The simulator uses DK full-PPR scoring with 300/100-yard bonuses and 20-player rosters. `weekly_avg` is the mean expected best-ball lineup score per roster.",
        "",
        _md_table(
            sim,
            [
                "variant",
                "teams",
                "weekly_avg",
                "delta_vs_current",
                "gap_vs_full_all",
                "weekly_p10",
                "weekly_p50",
                "weekly_p90",
                "wk1_14_total",
                "wk1_17_total",
            ],
        ),
        "",
        "## Market projection curve",
        "",
        "These are the ADP-derived `proj_points` anchors fed into the DK outcome model before best-ball lineup optimization.",
        "",
        _md_table(board_summary, ["adp_bucket", "pos", "players", "market_ppg", "adp_min", "adp_max"]),
        "",
        "## Readout",
        "",
    ]

    no_anchor = sim[sim["variant"].eq("no_anchor")]
    no_avail = sim[sim["variant"].eq("no_availability")]
    no_anchor_delta = (
        float(no_anchor["delta_vs_current"].iloc[0])
        if not no_anchor.empty else float("nan")
    )
    no_avail_delta = (
        float(no_avail["delta_vs_current"].iloc[0])
        if not no_avail.empty else float("nan")
    )
    scale_candidates = sim[
        sim["variant"].eq("current") | sim["variant"].str.startswith("anchor_scale_")
    ]
    closest = (
        scale_candidates.iloc[(scale_candidates["weekly_avg"] - full_winner_avg).abs().argsort().iloc[0]]
        if not scale_candidates.empty else None
    )
    lines.extend([
        f"- Current simulated DK rooms average `{current_avg:.1f}` points/week.",
        f"- Real BBM full-PPR human rooms average `{full_avg:.1f}` points/week; real BBM half-PPR rooms average `{half_avg:.1f}` points/week.",
        f"- Disabling market projection anchoring changes simulated scoring by `{no_anchor_delta:+.1f}` points/week.",
        f"- Disabling availability changes simulated scoring by `{no_avail_delta:+.1f}` points/week.",
    ])
    if closest is not None:
        lines.append(
            f"- Among the default and tested relative anchor scales, `{closest['variant']}` lands closest to the real full-PPR pod-winner diagnostic baseline: `{closest['weekly_avg']:.1f}` vs `{full_winner_avg:.1f}` points/week."
        )
    lines.extend([
        "",
        "If `no_anchor` drops most of the gap, the ADP-implied projection curve is too hot. If it barely moves, the inflation is more likely coming from DK scoring/bonuses, 20-player roster depth, lineup optimization, or the empirical weekly distribution itself.",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rooms", type=int, default=12)
    p.add_argument("--seasons", type=int, default=80)
    p.add_argument("--sigma", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument(
        "--anchor-scales",
        default="0.90,0.95,1.05,1.10",
        help="Comma-separated in-memory multipliers relative to the default roster proj_points anchors.",
    )
    p.add_argument("--out", type=Path, default=OUT / "score_environment_calibration_audit.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    board = load_market_board()
    rosters = draft_field(args.rooms, rng, board=board, sigma=args.sigma, strategy="adp_noise")

    current = load_market_model()
    anchor_scales = [
        float(x.strip())
        for x in str(args.anchor_scales).split(",")
        if x.strip()
    ]
    variants = [
        Variant("current", current),
        *[
            Variant(f"anchor_scale_{scale:.2f}", current, proj_scale=scale)
            for scale in anchor_scales
        ],
        Variant("no_handcuff", current, handcuff=False),
        Variant("no_matchup", current, matchup=False),
        Variant("no_availability", current, availability=False),
        Variant("no_correlation", CorrelatedOutcomeModel(correlation_model=False).build(projections=None)),
        Variant("no_anchor", CorrelatedOutcomeModel(anchor_to_projection=False).build(projections=None)),
    ]

    roster_frames = []
    for n, variant in enumerate(variants):
        print(f"[score-calibration] simulating {variant.name} ({n + 1}/{len(variants)})")
        roster_frames.append(
            simulate_variant(rosters, variant, args.seasons, args.seed + 1000 * (n + 1))
        )
    sim_rosters = pd.concat(roster_frames, ignore_index=True)
    sim_summary = summarize_sim(sim_rosters)

    real_summary = pd.concat([
        summarize_real(FULL_REAL, "full"),
        summarize_real(HALF_REAL, "half"),
    ], ignore_index=True)
    summary = pd.concat([real_summary, sim_summary], ignore_index=True, sort=False)
    board_summary = board_projection_summary(board)

    stem = args.out.with_suffix("")
    sim_rosters.to_csv(stem.with_name(stem.name + "_sim_rosters.csv"), index=False)
    summary.to_csv(stem.with_name(stem.name + "_summary.csv"), index=False)
    board_summary.to_csv(stem.with_name(stem.name + "_market_curve.csv"), index=False)
    write_report(args.out, summary, board_summary, args)

    current_avg = sim_summary.loc[sim_summary["variant"].eq("current"), "weekly_avg"].iloc[0]
    print(f"[score-calibration] wrote {args.out}")
    print(f"[score-calibration] current simulated weekly avg: {current_avg:.1f}")


if __name__ == "__main__":
    main()
