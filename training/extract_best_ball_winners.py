"""Extract real best-ball winner/top-team data from the BBM V sample.

This builds a clean dataset of all-human rosters scored on real weekly results,
then labels pod winners, top-2 teams, and global top-percentile teams. It is a
data-gathering step for learning winner shapes; it does not train or change the
served DK EV policy.

Run:
    python training/extract_best_ball_winners.py --scoring full --drafts 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_vs_humans import REG_WEEKS, build_weekly_lookup, norm  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
DEFAULT_SAMPLE = RAW / "bbm_v_sample.csv"
POS = ("QB", "RB", "WR", "TE")
SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
FLEX = ("RB", "WR", "TE")


def score_roster_weekly(roster: pd.DataFrame, pts: dict[str, dict[int, float]]) -> tuple[np.ndarray, list[str]]:
    """Return weekly best-ball lineup scores and comma-separated starter names."""
    by_pos: dict[str, list[tuple[str, dict[int, float]]]] = {p: [] for p in POS}
    for _, r in roster.iterrows():
        pos = str(r["position_name"])
        if pos in by_pos:
            nn = str(r["name_norm"])
            by_pos[pos].append((str(r["player_name"]), pts.get(nn, {})))

    weekly = []
    starter_names = []
    for wk in REG_WEEKS:
        total = 0.0
        leftovers: list[tuple[float, str]] = []
        starters: list[str] = []
        for pos, n_start in SLOTS.items():
            scored = sorted(
                ((float(p.get(wk, 0.0)), name) for name, p in by_pos[pos]),
                reverse=True,
                key=lambda x: x[0],
            )
            for val, name in scored[:n_start]:
                total += val
                starters.append(name)
            if pos in FLEX:
                leftovers.extend(scored[n_start:])
        if leftovers:
            val, name = max(leftovers, key=lambda x: x[0])
            total += val
            starters.append(name)
        weekly.append(total)
        starter_names.append(", ".join(starters))
    return np.asarray(weekly, dtype=float), starter_names


def roster_shape(roster: pd.DataFrame) -> str:
    return "".join(f"{pos}{int((roster['position_name'] == pos).sum())}" for pos in POS)


def first_pos_rounds(roster: pd.DataFrame) -> dict[str, int | None]:
    out = {}
    for pos in POS:
        sub = roster[roster["position_name"] == pos]
        out[f"first_{pos}_round"] = int(sub["team_pick_number"].min()) if not sub.empty else None
    return out


def draft_phase(round_no: int) -> str:
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
    return "R16-18"


def build_datasets(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pts, team_map = build_weekly_lookup(args.scoring)
    usecols = [
        "draft_id",
        "draft_created_time",
        "draft_completed_time",
        "draft_entry_id",
        "tournament_entry_id",
        "player_name",
        "player_id",
        "position_name",
        "projection_adp",
        "source",
        "pick_order",
        "overall_pick_number",
        "team_pick_number",
        "roster_points",
        "made_playoffs",
    ]
    df = pd.read_csv(args.sample, usecols=usecols, low_memory=False)
    draft_ids = df["draft_id"].drop_duplicates().tolist()[: args.drafts]
    df = df[df["draft_id"].isin(draft_ids)].copy()
    df["name_norm"] = df["player_name"].map(norm)
    df["nfl_team"] = df["name_norm"].map(team_map).fillna("")
    df["round_bucket"] = df["team_pick_number"].map(draft_phase)

    roster_rows: list[dict] = []
    weekly_rows: list[dict] = []
    player_frames: list[pd.DataFrame] = []

    grouped = df.sort_values(["draft_id", "pick_order", "team_pick_number"]).groupby(
        ["draft_id", "pick_order"],
        sort=False,
    )
    total_groups = len(grouped)
    for n, ((draft_id, seat), roster) in enumerate(grouped, start=1):
        weekly, starters = score_roster_weekly(roster, pts)
        row = {
            "draft_id": draft_id,
            "seat": int(seat),
            "draft_entry_id": roster["draft_entry_id"].iloc[0],
            "tournament_entry_id": roster["tournament_entry_id"].iloc[0],
            "draft_completed_time": roster["draft_completed_time"].iloc[0],
            "score_total": float(weekly.sum()),
            "score_weekly_avg": float(weekly.mean()),
            "score_weekly_sd": float(weekly.std()),
            "score_best_week": float(weekly.max()),
            "score_worst_week": float(weekly.min()),
            "ud_roster_points": float(roster["roster_points"].iloc[0]),
            "ud_made_playoffs": int(roster["made_playoffs"].iloc[0]),
            "roster_shape": roster_shape(roster),
        }
        for pos in POS:
            row[f"n_{pos}"] = int((roster["position_name"] == pos).sum())
        row.update(first_pos_rounds(roster))
        roster_rows.append(row)

        for week, score, names in zip(REG_WEEKS, weekly, starters):
            weekly_rows.append({
                "draft_id": draft_id,
                "seat": int(seat),
                "week": int(week),
                "score": float(score),
                "starter_names": names,
            })

        p = roster.copy()
        p["seat"] = int(seat)
        p["score_total"] = float(weekly.sum())
        p["score_weekly_avg"] = float(weekly.mean())
        player_frames.append(p)

        if args.progress and n % args.progress == 0:
            print(f"  scored {n:,}/{total_groups:,} rosters")

    rosters = pd.DataFrame(roster_rows)
    weekly_df = pd.DataFrame(weekly_rows)
    players = pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame()

    rosters["pod_rank"] = rosters.groupby("draft_id")["score_total"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    rosters["pod_winner"] = (rosters["pod_rank"] == 1).astype(int)
    rosters["pod_top2"] = (rosters["pod_rank"] <= 2).astype(int)
    rosters["global_pct_rank"] = rosters["score_total"].rank(pct=True, ascending=False, method="average")
    rosters["global_top_pct"] = (rosters["global_pct_rank"] <= args.top_pct).astype(int)
    rosters["global_top_1pct"] = (rosters["global_pct_rank"] <= 0.01).astype(int)

    labels = rosters[[
        "draft_id",
        "seat",
        "pod_rank",
        "pod_winner",
        "pod_top2",
        "global_pct_rank",
        "global_top_pct",
        "global_top_1pct",
    ]]
    players = players.merge(labels, on=["draft_id", "seat"], how="left")
    weekly_df = weekly_df.merge(labels, on=["draft_id", "seat"], how="left")
    return rosters, players, weekly_df


def cohort_summary(rosters: pd.DataFrame) -> pd.DataFrame:
    cohorts = {
        "all": rosters,
        "pod_winners": rosters[rosters["pod_winner"].eq(1)],
        "pod_top2": rosters[rosters["pod_top2"].eq(1)],
        "global_top_pct": rosters[rosters["global_top_pct"].eq(1)],
        "global_top_1pct": rosters[rosters["global_top_1pct"].eq(1)],
    }
    rows = []
    for name, g in cohorts.items():
        if g.empty:
            continue
        row = {
            "cohort": name,
            "rosters": int(len(g)),
            "score_weekly_avg": float(g["score_weekly_avg"].mean()),
            "score_total_avg": float(g["score_total"].mean()),
            "score_total_p10": float(g["score_total"].quantile(0.10)),
            "score_total_p50": float(g["score_total"].quantile(0.50)),
            "score_total_p90": float(g["score_total"].quantile(0.90)),
        }
        for pos in POS:
            row[f"avg_n_{pos}"] = float(g[f"n_{pos}"].mean())
            row[f"avg_first_{pos}_round"] = float(g[f"first_{pos}_round"].dropna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def shape_summary(rosters: pd.DataFrame) -> pd.DataFrame:
    base = rosters["roster_shape"].value_counts(normalize=True).rename("all_rate")
    winners = rosters[rosters["pod_winner"].eq(1)]["roster_shape"].value_counts(normalize=True).rename("winner_rate")
    top2 = rosters[rosters["pod_top2"].eq(1)]["roster_shape"].value_counts(normalize=True).rename("top2_rate")
    out = pd.concat([base, winners, top2], axis=1).fillna(0.0).reset_index(names="roster_shape")
    out["winner_lift"] = out["winner_rate"] / out["all_rate"].replace(0, np.nan)
    return out.sort_values(["winner_rate", "top2_rate"], ascending=False)


def player_frequency(players: pd.DataFrame, min_winner_rosters: int = 5) -> pd.DataFrame:
    roster_key = ["draft_id", "seat"]
    total_rosters = players[roster_key].drop_duplicates().shape[0]
    winner_rosters = players.loc[players["pod_winner"].eq(1), roster_key].drop_duplicates().shape[0]
    top2_rosters = players.loc[players["pod_top2"].eq(1), roster_key].drop_duplicates().shape[0]

    cols = ["player_id", "player_name", "position_name"]
    all_counts = players.groupby(cols).size().rename("all_count")
    winner_counts = players[players["pod_winner"].eq(1)].groupby(cols).size().rename("winner_count")
    top2_counts = players[players["pod_top2"].eq(1)].groupby(cols).size().rename("top2_count")
    adp = players.groupby(cols)["projection_adp"].median().rename("projection_adp_median")
    out = pd.concat([all_counts, winner_counts, top2_counts], axis=1).fillna(0).reset_index()
    out = out.merge(adp.reset_index(), on=cols, how="left")
    out["all_rate"] = out["all_count"] / max(total_rosters, 1)
    out["winner_rate"] = out["winner_count"] / max(winner_rosters, 1)
    out["top2_rate"] = out["top2_count"] / max(top2_rosters, 1)
    out["winner_lift"] = out["winner_rate"] / out["all_rate"].replace(0, np.nan)
    out["top2_lift"] = out["top2_rate"] / out["all_rate"].replace(0, np.nan)
    return out[out["winner_count"] >= min_winner_rosters].sort_values(
        ["winner_lift", "winner_count"],
        ascending=False,
    )


def round_position_summary(players: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort_name, mask in (
        ("all", pd.Series(True, index=players.index)),
        ("pod_winners", players["pod_winner"].eq(1)),
        ("pod_top2", players["pod_top2"].eq(1)),
        ("global_top_pct", players["global_top_pct"].eq(1)),
    ):
        g = players[mask]
        if g.empty:
            continue
        counts = g.groupby(["round_bucket", "position_name"]).size().rename("n").reset_index()
        denom = g.groupby("round_bucket").size().rename("denom").reset_index()
        out = counts.merge(denom, on="round_bucket")
        out["pct"] = out["n"] / out["denom"]
        out["cohort"] = cohort_name
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_report(
    path: Path,
    rosters: pd.DataFrame,
    cohorts: pd.DataFrame,
    shapes: pd.DataFrame,
    players_freq: pd.DataFrame,
    round_pos: pd.DataFrame,
    args,
) -> None:
    lines = [
        "# Real Best-Ball Winner Data Pull",
        "",
        "All-human Underdog BBM V sample scored on real weekly results. This is a data pull for winner-shape analysis; it does not train the DK EV model.",
        "",
        "## Setup",
        "",
        f"- Drafts: {rosters['draft_id'].nunique():,}",
        f"- Rosters: {len(rosters):,}",
        f"- Scoring: {args.scoring}-PPR",
        "- Format: 12 teams, 18 rounds, best-ball lineup 1QB/2RB/3WR/1TE/1FLEX",
        f"- Weeks scored: {min(REG_WEEKS)}-{max(REG_WEEKS)}",
        f"- Global top-pct label: top {args.top_pct:.1%}",
        "- Caveat: this is Underdog BBM V 2024, not DK 20-round best ball.",
        "",
        "## Cohort Scores And Shape",
        "",
        "| cohort | rosters | weekly avg | total avg | QB | RB | WR | TE | first QB | first RB | first WR | first TE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in cohorts.iterrows():
        lines.append(
            f"| {r.cohort} | {int(r.rosters)} | {r.score_weekly_avg:.1f} | {r.score_total_avg:.1f} | "
            f"{r.avg_n_QB:.2f} | {r.avg_n_RB:.2f} | {r.avg_n_WR:.2f} | {r.avg_n_TE:.2f} | "
            f"{r.avg_first_QB_round:.2f} | {r.avg_first_RB_round:.2f} | "
            f"{r.avg_first_WR_round:.2f} | {r.avg_first_TE_round:.2f} |"
        )

    lines.extend([
        "",
        "## Most Common Winner Builds",
        "",
        "| roster shape | all rate | winner rate | top2 rate | winner lift |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for _, r in shapes.head(15).iterrows():
        lines.append(
            f"| {r.roster_shape} | {r.all_rate:.1%} | {r.winner_rate:.1%} | "
            f"{r.top2_rate:.1%} | {r.winner_lift:.2f}x |"
        )

    lines.extend([
        "",
        "## Highest Winner-Rate Player Lifts",
        "",
        "| player | pos | ADP | all rate | winner rate | winner lift | winner count |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, r in players_freq.head(20).iterrows():
        lines.append(
            f"| {r.player_name} | {r.position_name} | {float(r.projection_adp_median):.1f} | "
            f"{r.all_rate:.1%} | {r.winner_rate:.1%} | {r.winner_lift:.2f}x | {int(r.winner_count)} |"
        )

    lines.extend([
        "",
        "## Position Share By Draft Phase",
        "",
        "| cohort | phase | QB | RB | WR | TE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    if not round_pos.empty:
        pivot = round_pos.pivot_table(
            index=["cohort", "round_bucket"],
            columns="position_name",
            values="pct",
            fill_value=0.0,
        ).reset_index()
        for pos in POS:
            if pos not in pivot:
                pivot[pos] = 0.0
        order = ["R1-3", "R4-6", "R7-9", "R10-12", "R13-15", "R16-18"]
        pivot["phase_order"] = pivot["round_bucket"].map({v: i for i, v in enumerate(order)})
        pivot = pivot.sort_values(["cohort", "phase_order"])
        for _, r in pivot.iterrows():
            lines.append(
                f"| {r.cohort} | {r.round_bucket} | {r.QB:.1%} | {r.RB:.1%} | "
                f"{r.WR:.1%} | {r.TE:.1%} |"
            )

    lines.extend([
        "",
        "## Output Files",
        "",
        f"- `{path.with_name(path.stem + '_rosters.csv').name}`",
        f"- `{path.with_name(path.stem + '_players.csv').name}`",
        f"- `{path.with_name(path.stem + '_weekly.csv').name}`",
        f"- `{path.with_name(path.stem + '_cohorts.csv').name}`",
        f"- `{path.with_name(path.stem + '_player_frequency.csv').name}`",
        f"- `{path.with_name(path.stem + '_round_position.csv').name}`",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    p.add_argument("--drafts", type=int, default=1500)
    p.add_argument("--scoring", choices=["full", "half"], default="full")
    p.add_argument("--top-pct", type=float, default=0.05)
    p.add_argument("--progress", type=int, default=1000)
    p.add_argument("--out", type=Path, default=OUT / "best_ball_winner_data_full.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rosters, players, weekly = build_datasets(args)
    cohorts = cohort_summary(rosters)
    shapes = shape_summary(rosters)
    freq = player_frequency(players)
    round_pos = round_position_summary(players)

    stem = args.out.stem
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rosters.to_csv(args.out.with_name(stem + "_rosters.csv"), index=False)
    players.to_csv(args.out.with_name(stem + "_players.csv"), index=False)
    weekly.to_csv(args.out.with_name(stem + "_weekly.csv"), index=False)
    cohorts.to_csv(args.out.with_name(stem + "_cohorts.csv"), index=False)
    shapes.to_csv(args.out.with_name(stem + "_shapes.csv"), index=False)
    freq.to_csv(args.out.with_name(stem + "_player_frequency.csv"), index=False)
    round_pos.to_csv(args.out.with_name(stem + "_round_position.csv"), index=False)
    write_report(args.out, rosters, cohorts, shapes, freq, round_pos, args)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
