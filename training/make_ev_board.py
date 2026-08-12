"""
Build a diagnostic DK ADP EV summary from rollout labels.

This is not a draft board and should not be surfaced as player rankings. It is
a static summary of how players performed when they appeared as candidate picks
in simulated DK draft states. The main score is within-state EV edge:

    candidate EV - average EV of the other candidates in that same draft state

That contrast is appropriate for the state-conditioned policy model, but it is
not a stable player-level normalization. Use this file only for diagnostics;
use fit_dk_ev_policy.py for draft recommendations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "data" / "processed" / "dk_ev_rollouts.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "dk_ev_board.csv"
DEFAULT_ROUND_OUT = ROOT / "data" / "processed" / "dk_ev_board_by_round.csv"


def round_bucket(r: float) -> str:
    r = int(r)
    if r <= 3:
        return "R1-3"
    if r <= 6:
        return "R4-6"
    if r <= 10:
        return "R7-10"
    if r <= 14:
        return "R11-14"
    return "R15-18"


def build_board(path: Path, min_samples: int = 20) -> pd.DataFrame:
    df = load_scored_rows(path)
    board = summarize_player_board(df, min_samples)
    return board


def load_scored_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {
        "run_seed", "state_id", "pick_no", "candidate_name", "candidate_pos",
        "candidate_team", "cand_adp", "round", "ev_mean",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # State identity must include seed because state_id restarts each chunk.
    df["_state_key"] = (
        df["run_seed"].astype(str) + ":" +
        df["state_id"].astype(str) + ":" +
        df["pick_no"].astype(str)
    )
    df["_state_avg_ev"] = df.groupby("_state_key")["ev_mean"].transform("mean")
    df["_state_max_ev"] = df.groupby("_state_key")["ev_mean"].transform("max")
    df["ev_edge"] = df["ev_mean"] - df["_state_avg_ev"]
    df["state_win"] = (df["ev_mean"] == df["_state_max_ev"]).astype(int)
    df["round_bucket"] = df["round"].map(round_bucket)
    return df


def score_summary(board: pd.DataFrame) -> pd.DataFrame:
    board = board.copy()
    group_cols = ["candidate_name", "candidate_pos", "candidate_team"]

    # Percentile score from edge + state win rate. Kept transparent rather than
    # pretending this is a final learned policy.
    for col in ["avg_edge", "p75_edge", "state_win_rate"]:
        rank = board[col].rank(method="average", pct=True)
        board[f"{col}_pct"] = rank
    board["ev_board_score"] = (
        60.0 * board["avg_edge_pct"] +
        25.0 * board["p75_edge_pct"] +
        15.0 * board["state_win_rate_pct"]
    )
    return board


def summarize_player_board(df: pd.DataFrame, min_samples: int = 20) -> pd.DataFrame:
    group_cols = ["candidate_name", "candidate_pos", "candidate_team"]
    board = (
        df.groupby(group_cols)
        .agg(
            samples=("ev_mean", "size"),
            avg_ev=("ev_mean", "mean"),
            p75_ev=("ev_mean", lambda s: s.quantile(0.75)),
            p90_ev=("ev_mean", lambda s: s.quantile(0.90)),
            max_ev=("ev_mean", "max"),
            avg_edge=("ev_edge", "mean"),
            p75_edge=("ev_edge", lambda s: s.quantile(0.75)),
            state_win_rate=("state_win", "mean"),
            avg_round=("round", "mean"),
            avg_adp=("cand_adp", "mean"),
            avg_pick_no=("pick_no", "mean"),
            earliest_round=("round", "min"),
            latest_round=("round", "max"),
        )
        .reset_index()
    )
    board = board[board["samples"] >= min_samples].copy()
    board["target_zone"] = "R" + board["earliest_round"].astype(int).astype(str) + "-R" + board["latest_round"].astype(int).astype(str)
    board["avg_pick_vs_adp"] = board["avg_pick_no"] - board["avg_adp"]
    board = score_summary(board)

    board = board.sort_values(
        ["ev_board_score", "avg_edge", "avg_ev"],
        ascending=False,
        kind="stable",
    )
    board.insert(0, "ev_rank", np.arange(1, len(board) + 1))

    keep = [
        "ev_rank", "candidate_name", "candidate_pos", "candidate_team",
        "avg_adp", "target_zone", "avg_round", "avg_pick_no", "avg_pick_vs_adp",
        "samples", "ev_board_score",
        "avg_edge", "p75_edge", "state_win_rate",
        "avg_ev", "p75_ev", "p90_ev", "max_ev",
        "earliest_round", "latest_round",
    ]
    return board[keep]


def summarize_round_board(df: pd.DataFrame, min_samples: int = 8) -> pd.DataFrame:
    group_cols = ["round_bucket", "candidate_name", "candidate_pos", "candidate_team"]
    board = (
        df.groupby(group_cols)
        .agg(
            samples=("ev_mean", "size"),
            avg_ev=("ev_mean", "mean"),
            p75_ev=("ev_mean", lambda s: s.quantile(0.75)),
            avg_edge=("ev_edge", "mean"),
            p75_edge=("ev_edge", lambda s: s.quantile(0.75)),
            state_win_rate=("state_win", "mean"),
            avg_round=("round", "mean"),
            avg_adp=("cand_adp", "mean"),
            avg_pick_no=("pick_no", "mean"),
            earliest_round=("round", "min"),
            latest_round=("round", "max"),
        )
        .reset_index()
    )
    board = board[board["samples"] >= min_samples].copy()
    board["avg_pick_vs_adp"] = board["avg_pick_no"] - board["avg_adp"]

    scored = []
    for _, g in board.groupby("round_bucket", sort=False):
        g = score_summary(g)
        g = g.sort_values(
            ["ev_board_score", "avg_edge", "avg_ev"],
            ascending=False,
            kind="stable",
        ).copy()
        g.insert(1, "bucket_rank", np.arange(1, len(g) + 1))
        scored.append(g)
    if not scored:
        return board
    out = pd.concat(scored, ignore_index=True)
    keep = [
        "round_bucket", "bucket_rank", "candidate_name", "candidate_pos",
        "candidate_team", "avg_adp", "avg_round", "avg_pick_no",
        "avg_pick_vs_adp", "samples", "ev_board_score", "avg_edge",
        "p75_edge", "state_win_rate", "avg_ev", "p75_ev",
        "earliest_round", "latest_round",
    ]
    bucket_order = {"R1-3": 0, "R4-6": 1, "R7-10": 2, "R11-14": 3, "R15-18": 4}
    out["_bucket_order"] = out["round_bucket"].map(bucket_order).fillna(99)
    out = out.sort_values(["_bucket_order", "bucket_rank"], kind="stable")
    return out[keep]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--round-out", type=Path, default=DEFAULT_ROUND_OUT)
    p.add_argument("--min-samples", type=int, default=20)
    p.add_argument("--round-min-samples", type=int, default=8)
    args = p.parse_args()

    rows = load_scored_rows(args.input)
    board = summarize_player_board(rows, args.min_samples)
    round_board = summarize_round_board(rows, args.round_min_samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(args.out, index=False)
    args.round_out.parent.mkdir(parents=True, exist_ok=True)
    round_board.to_csv(args.round_out, index=False)

    print("Diagnostic only: this is not a draft board; use fit_dk_ev_policy.py for recommendations.")
    print(f"Wrote {len(board):,} players -> {args.out}")
    print(f"Wrote {len(round_board):,} player/bucket rows -> {args.round_out}")
    print(board.head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
