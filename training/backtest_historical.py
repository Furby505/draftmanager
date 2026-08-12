"""
Real-outcome backtest: does drafting by the DK EV policy beat drafting by ADP,
judged by what actually happened on the field?

This is the circularity-free counterpart to backtest_policy.py. Instead of
scoring rosters with our own outcome simulator, it scores them on REAL historical
weekly DK results. The only synthetic pieces are the draft board (a lookahead-free
ADP proxy) and the ADP-bot opponents.

Pipeline per past season N:
  1. Board: rank season-N-eligible veterans by their prior-season (N-1) real DK
     points -> a consensus-ADP proxy (last year's points ~ this year's draft slot).
  2. Draft a 12-team / 20-round snake. One seat = policy (model_dk_ev_policy),
     the rest = ADP-noise bots. Also run the same pod with ADP bots in our seat
     (paired control), so the only difference is how our seat drafted.
  3. Score every roster on REAL season-N weekly results: auto-start the best-ball
     optimal lineup each week, then R1 = sum(weeks 1-14), R2..R4 = weeks 15..17.
     Real injuries and byes are already baked into the weekly scores (a missed
     week = 0). NO simulator is used here.
  4. Within the pod, the real top-2 by R1 advance. Compare the policy seat's real
     advance / pod-win rate vs. the ADP seats (paired) and vs. the 1/6 baseline,
     pooled over many pods x seasons.

What it proves: if you had drafted by the model in these past seasons, your teams
would have advanced more often than drafting by consensus ADP -- measured on real
results, not our model of football.

v1 limitations (see data/processed/model_improvement_backlog.md): ADP proxy =
prior-year finish (not real historical ADP); opponents are ADP-bots; rookies
excluded (no lookahead-free rookie ADP); single-pod R1 focus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import best_ball as bb
from opponent_field import Board, POS, POS_CODE, market_tier_from_adp
from scoring import recalc_weekly_df
from backtest_policy import make_policy_agent, make_adp_agent
from opponent_field import _choose, _sample_team_caps
from train_policy import DraftState, snake_team, _apply_pick

ROOT = Path(__file__).resolve().parent.parent
WEEKLY = ROOT / "data" / "raw" / "player_stats_weekly.csv"
MODELS = ROOT / "server" / "models"
DEFAULT_OUT = ROOT / "data" / "processed" / "backtest_historical_audit.md"
N_TEAMS = bb.TEAMS_PER_POD
N_ROUNDS = bb.DRAFT_ROUNDS
SCORE_POS = ("QB", "RB", "WR", "TE")


# ── Data: real weekly DK scores + lookahead-free board ────────────────────────
def load_weekly(path: Path = WEEKLY) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "season_type" in df.columns:
        df = df[df["season_type"].fillna("REG") == "REG"]
    df = df[df["position"].isin(SCORE_POS)].copy()
    return recalc_weekly_df(df, platform="draftkings")   # -> fantasy_points_ppr (DK)


def real_weekly_scores(weekly: pd.DataFrame, season: int) -> tuple[dict, dict]:
    """player_id -> np.array(17) of real DK weekly points (0 for weeks not played,
    which is exactly how byes/injuries show up), plus player_id -> position."""
    s = weekly[(weekly["season"] == season) & (weekly["week"].between(1, 17))]
    scores: dict[str, np.ndarray] = {}
    pos: dict[str, str] = {}
    for pid, g in s.groupby("player_id"):
        arr = np.zeros(17)
        for wk, pts in zip(g["week"].astype(int), g["fantasy_points_ppr"].astype(float)):
            arr[wk - 1] = pts
        scores[str(pid)] = arr
        pos[str(pid)] = g["position"].iloc[-1]
    return scores, pos


def build_historical_board(weekly: pd.DataFrame, season: int,
                           max_players: int = 280) -> Board:
    """Rank season-N veterans by prior-season (N-1) real DK total points -> ADP proxy.
    Rookies / no-prior players are excluded (no lookahead-free draft slot for them)."""
    cur = weekly[(weekly["season"] == season) & (weekly["week"].between(1, 17))]
    prior = weekly[(weekly["season"] == season - 1) & (weekly["week"].between(1, 17))]
    prior_pts = prior.groupby("player_id")["fantasy_points_ppr"].sum()

    rows = []
    for pid, g in cur.groupby("player_id"):
        pid = str(pid)
        if pid not in prior_pts.index:           # rookie / no prior -> not draftable in v1
            continue
        rows.append({
            "player_id": pid,
            "name": str(g["player_display_name"].iloc[-1]),
            "team": str(g["recent_team"].iloc[-1] or ""),
            "pos": g["position"].iloc[-1],
            "prior": float(prior_pts.loc[pid]),
        })
    rows.sort(key=lambda r: -r["prior"])          # best last year drafted first
    rows = rows[:max_players]
    if len(rows) < N_TEAMS * N_ROUNDS:
        raise RuntimeError(f"{season}: only {len(rows)} draftable veterans")

    adp = np.arange(1, len(rows) + 1, dtype=float)     # synthetic ADP = market rank
    return Board(
        player_id=np.array([r["player_id"] for r in rows], dtype=object),
        name=np.array([r["name"] for r in rows], dtype=object),
        team=np.array([r["team"] for r in rows], dtype=object),
        pos_code=np.array([POS_CODE[r["pos"]] for r in rows], dtype=int),
        adp=adp,
        has_real_adp=np.ones(len(rows), dtype=bool),
        proj_points=None,                          # not used: real results do the scoring
        market_tier=np.array([market_tier_from_adp(a) for a in adp], dtype=int),
        bye_week=np.zeros(len(rows), dtype=int),   # real weekly 0s already encode byes
    )


# ── Full-pod draft (returns ALL 12 rosters so we can score head-to-head) ──────
def run_draft_full(board: Board, our_team: int, our_agent, rng, sigma: float,
                   opp_strategy: str = "adp_noise") -> list[list[dict]]:
    avail = np.ones(len(board), dtype=bool)
    counts = np.zeros((N_TEAMS, 4), dtype=int)
    rosters: list[list[dict]] = [[] for _ in range(N_TEAMS)]
    team_caps = [_sample_team_caps(rng) for _ in range(N_TEAMS)]
    pick_history: list[tuple[int, int]] = []
    for gp in range(N_TEAMS * N_ROUNDS):
        team = snake_team(gp, N_TEAMS)
        if team == our_team:
            state = DraftState(board, gp, our_team, rosters, counts, avail,
                               team_caps, sigma, opp_strategy, pick_history)
            idx = our_agent(state, rng)
        else:
            soft_cap, comfort = team_caps[team]
            picks_left = N_ROUNDS - len(rosters[team])
            idx = _choose(board, avail, counts[team], soft_cap, comfort,
                          picks_left, gp // N_TEAMS, N_ROUNDS, sigma,
                          opp_strategy, rng)
        _apply_pick(board, avail, counts, rosters, team, idx)
        pick_history.append((team, int(idx)))
    return rosters


# ── Score a roster on REAL weekly results ─────────────────────────────────────
_ZERO17 = np.zeros(17)


def real_round_scores(roster: list[dict], scores: dict, pos_map: dict) -> dict:
    pids = [p["player_id"] for p in roster]
    posn = {pid: pos_map.get(pid, "WR") for pid in pids}
    weekly = np.zeros(17)
    for w in range(17):
        wp = {pid: scores.get(pid, _ZERO17)[w] for pid in pids}
        weekly[w] = bb.optimal_lineup_score(wp, posn)
    return {"r1": float(weekly[:14].sum()), "total": float(weekly.sum())}


def pod_advance(pod_r1: list[float], seat: int, advance: int = 2) -> tuple[bool, bool, int]:
    """(advanced top-`advance`, won pod, rank) for `seat` by real R1 score."""
    order = np.argsort(-np.asarray(pod_r1), kind="stable")
    rank = int(np.where(order == seat)[0][0]) + 1
    return rank <= advance, rank == 1, rank


# ── Backtest driver ───────────────────────────────────────────────────────────
def run(args) -> dict:
    weekly = load_weekly()
    policy_model = joblib.load(args.model)
    feature_cols = json.loads(Path(args.cols).read_text())
    agents = {
        "policy": make_policy_agent(policy_model, feature_cols, args.candidates),
        "pure_adp": make_adp_agent("pure_adp"),
        "adp_noise": make_adp_agent("adp_noise"),
    }
    seasons = [int(s) for s in args.seasons.split(",")]

    recs = []
    for season in seasons:
        board = build_historical_board(weekly, season)
        scores, pos_map = real_weekly_scores(weekly, season)
        print(f"\n=== season {season}: {len(board)} draftable, "
              f"{args.pods} pods ===")
        for pod in range(args.pods):
            our_team = int(np.random.default_rng(args.seed + season * 1000 + pod)
                           .integers(N_TEAMS))
            draft_seed = int(np.random.default_rng(args.seed + season * 7919 + pod)
                             .integers(1 << 31))
            row = {"season": season, "pod": pod, "our_team": our_team}
            for name, agent in agents.items():
                rng = np.random.default_rng(draft_seed)
                rosters = run_draft_full(board, our_team, agent, rng, args.sigma)
                pod_r1 = [real_round_scores(r, scores, pos_map)["r1"] for r in rosters]
                adv, won, rank = pod_advance(pod_r1, our_team)
                row[f"{name}_r1"] = pod_r1[our_team]
                row[f"{name}_advanced"] = int(adv)
                row[f"{name}_won"] = int(won)
                row[f"{name}_rank"] = rank
            recs.append(row)
        df_s = pd.DataFrame([r for r in recs if r["season"] == season])
        print(f"  policy advance={df_s['policy_advanced'].mean():.1%}  "
              f"pure_adp={df_s['pure_adp_advanced'].mean():.1%}  "
              f"adp_noise={df_s['adp_noise_advanced'].mean():.1%}  "
              f"(baseline {2/N_TEAMS:.1%})")

    df = pd.DataFrame(recs)
    summary = _summarize(df, args)
    _write_audit(args.out, summary, df, args, seasons)
    return summary


def _bootstrap_ci(diffs, iters, seed, lo=2.5, hi=97.5):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float)
    if len(d) == 0:
        return float("nan"), float("nan")
    means = d[rng.integers(0, len(d), size=(iters, len(d)))].mean(axis=1)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def _summarize(df: pd.DataFrame, args) -> dict:
    out = {"pods": int(len(df)), "baseline_advance": 2 / N_TEAMS,
           "policy_advance_rate": float(df["policy_advanced"].mean()),
           "policy_pod_win_rate": float(df["policy_won"].mean())}
    for base in ("pure_adp", "adp_noise"):
        for metric, col in (("advance", "advanced"), ("pod_win", "won")):
            p = df[f"policy_{col}"].to_numpy(float)
            b = df[f"{base}_{col}"].to_numpy(float)
            diff = p - b
            lo, hi = _bootstrap_ci(diff, args.bootstrap, args.seed)
            out[f"policy_vs_{base}__{metric}"] = {
                "policy_rate": float(p.mean()),
                "baseline_rate": float(b.mean()),
                "mean_lift": float(diff.mean()),
                "ci95_lo": lo, "ci95_hi": hi,
                "significant": bool(lo > 0 or hi < 0),
            }
    return out


def _write_audit(path, summary, df, args, seasons):
    L = [
        "# DK EV Policy — REAL-OUTCOME Backtest (vs ADP, scored on real results)",
        "",
        "Drafts a past season on a lookahead-free ADP proxy (prior-year DK points), "
        "then scores every roster on **real** weekly DK results (real injuries/byes "
        "included). No simulator in the scoring loop. Opponents are ADP-bots; rookies "
        "excluded; single-pod R1 advance. See model_improvement_backlog.md.",
        "",
        f"- Seasons: {seasons}  |  pods/season: {args.pods}  |  total pods: {summary['pods']}",
        f"- base candidates/state: {args.candidates}  |  model: `{Path(args.model).name}`",
        f"- Random-baseline advance rate (top-2 of 12): {summary['baseline_advance']:.1%}",
        "",
        "## Headline — real R1 advance rate (top-2 of 12)",
        "",
        f"- **Policy advance rate: {summary['policy_advance_rate']:.1%}** "
        f"(vs {summary['baseline_advance']:.1%} random) | pod-win "
        f"{summary['policy_pod_win_rate']:.1%} (vs {1/N_TEAMS:.1%} random)",
        "",
        "| comparison | metric | policy | baseline | mean lift | 95% CI | signif |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for base in ("pure_adp", "adp_noise"):
        for metric in ("advance", "pod_win"):
            s = summary[f"policy_vs_{base}__{metric}"]
            L.append(
                f"| vs {base} | {metric} | {s['policy_rate']:.1%} | "
                f"{s['baseline_rate']:.1%} | {s['mean_lift']:+.1%} | "
                f"[{s['ci95_lo']:+.1%}, {s['ci95_hi']:+.1%}] | "
                f"{'YES' if s['significant'] else 'no'} |")
    L += [
        "",
        "## How to read this",
        "- Advance rate above the 16.7% random baseline = the policy's rosters really "
        "finished top-2 of their pod more than chance, on real results.",
        "- `vs pure_adp` is the tough comparison: beating a disciplined ADP-board "
        "drafter on real outcomes. `signif=YES` means the paired 95% CI excludes 0.",
        "- v1 caveats: ADP proxy = prior-year finish; ADP-bot opponents; no rookies; "
        "R1-only. A real-money-grade version needs real historical ADP + deeper bracket.",
        "",
        f"_Per-pod rows: {len(df)} (full table in the sibling CSV)._",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    df.to_csv(Path(path).with_suffix(".csv"), index=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", type=str, default="2019,2021,2022,2023,2024")
    p.add_argument("--pods", type=int, default=120)
    p.add_argument(
        "--candidates",
        type=int,
        default=12,
        help="base candidates per pick; later rounds expand automatically",
    )
    p.add_argument("--sigma", type=float, default=12.0)
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20260613)
    p.add_argument("--model", type=Path, default=MODELS / "model_dk_ev_policy.joblib")
    p.add_argument("--cols", type=Path, default=MODELS / "dk_ev_policy_feature_cols.json")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print("\n=== SUMMARY (real-outcome) ===")
    print(f"Policy advance rate: {summary['policy_advance_rate']:.1%} "
          f"(random {summary['baseline_advance']:.1%})")
    for base in ("pure_adp", "adp_noise"):
        s = summary[f"policy_vs_{base}__advance"]
        print(f"  vs {base:>9}: {s['policy_rate']:.1%} vs {s['baseline_rate']:.1%}  "
              f"lift {s['mean_lift']:+.1%}  CI[{s['ci95_lo']:+.1%},{s['ci95_hi']:+.1%}]  "
              f"{'SIGNIF' if s['significant'] else 'n.s.'}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
