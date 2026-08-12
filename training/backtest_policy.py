"""
Backtest: does drafting by the DK EV policy beat drafting by ADP?

The fit-time audit (top-1 accuracy / edge_corr) grades the policy on the SAME
EV labels it was trained on — it shows the model learned the simulator's notion
of EV, but it is graded within single pick-states, not over a whole draft. This
script answers the end-to-end question instead:

    Seat one agent in a 12-team pod, let it draft a full 20-man roster, then
    score that roster's prize-EV against a realistic ADP-drafted field. Do this
    for the POLICY agent and for ADP agents sitting in the *same* seat, and
    compare.

Three agents, all facing the same ADP-noise opponents:
  - policy   : builds the draft state at each of our picks, scores the top-N ADP
               candidates with model_dk_ev_policy, drafts the highest predicted
               within-state EV edge (exactly what the server does at top-N).
  - pure_adp : "just follow the ADP board" — always the best ADP still available
               that keeps a legal roster. The thing most casual entrants do.
  - adp_noise: a realistic human drafter (ADP with reaches/falls + need nudges).

Pairing for low variance: within a pod, all three agents are run from the SAME
rng seed (identical opponent build archetypes + noise stream; the rooms only
diverge because our seat picks differently), then each resulting roster is scored
against the SAME shared field with the SAME season seed. We report the mean
prize-EV lift, the head-to-head win rate (fraction of pods where policy's roster
out-EVs the ADP roster), and bootstrap 95% CIs.

IMPORTANT — what this does and does not prove. This is an IN-SIMULATION test: it
shows the policy drafts higher-prize-EV rosters than ADP *inside our world
model*. It is only as trustworthy as the simulator and outcome-model assumptions.
It is NOT a real-money backtest (that needs historical ADP boards scored on real
weekly results).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import best_ball as bb
from bracket import evaluate_roster
from opponent_field import (
    DEFAULT_SIGMA,
    HARD_CAP,
    REALISM_CAP,
    MIN_POS,
    _choose,
    _sample_team_caps,
    build_field,
    draft_field,
    load_market_board,
)
from season_sim import simulate_roster
from outcome_model import load_market_model
from train_policy import (
    DraftState,
    candidate_depth_for_round,
    candidate_indices,
    snake_team,
    state_candidate_features,
    _apply_pick,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "server" / "models"
DEFAULT_MODEL = MODELS_DIR / "model_dk_ev_policy.joblib"
DEFAULT_COLS = MODELS_DIR / "dk_ev_policy_feature_cols.json"
DEFAULT_OUT = ROOT / "data" / "processed" / "backtest_policy_audit.md"

N_TEAMS = bb.TEAMS_PER_POD
N_ROUNDS = bb.DRAFT_ROUNDS


# ── Legal candidate set (mirrors opponent_field._choose's legality) ───────────
def legal_candidate_set(state: DraftState, n: int, max_candidate_rank: int | None = None) -> list[int]:
    """Top-`n` available candidates by ADP that keep a LEGAL best-ball roster —
    the exact legality the ADP opponents obey: hard caps (<=5 QB/TE), a realism
    cap, soft caps (drop over-soft positions unless that empties the pool), and
    the validity guarantee (when remaining picks == unmet 1QB/2RB/3WR/1TE
    minimums, restrict to needed positions). Without this the policy can draft an
    illegal 0-QB roster, which scores 0 at QB all season."""
    board = state.board
    counts = state.counts[state.our_team]
    soft_cap, _ = state.team_caps[state.our_team]
    picks_left = state.n_rounds - len(state.roster)

    under_hard = counts[board.pos_code] < HARD_CAP[board.pos_code]
    legal = state.avail & under_hard
    under_realism = counts[board.pos_code] < REALISM_CAP[board.pos_code]
    if (legal & under_realism).any():
        legal = legal & under_realism

    unmet = np.maximum(0, MIN_POS - counts)
    if picks_left <= int(unmet.sum()) and unmet.sum() > 0:
        needed = np.isin(board.pos_code, np.where(unmet > 0)[0])
        legal = legal & needed
    else:
        under_soft = counts[board.pos_code] < soft_cap[board.pos_code]
        if (legal & under_soft).any():
            legal = legal & under_soft

    idx = np.where(legal)[0]
    if idx.size == 0:
        idx = np.where(state.avail & under_hard)[0]
    order = idx[np.argsort(board.adp[idx], kind="stable")]
    depth = candidate_depth_for_round(state.round + 1, n, max_candidate_rank)
    return [int(i) for i in order[:depth]]


# ── Agents (each returns the board index to draft, given the live state) ──────
def make_policy_agent(
    model,
    feature_cols: list[str],
    n_candidates: int,
    max_candidate_rank: int | None = None,
):
    """Draft the highest predicted within-state EV edge among the top-N ADP
    candidates that keep a LEGAL roster (same legality the ADP bots obey)."""
    def agent(state: DraftState, rng: np.random.Generator) -> int:
        cands = legal_candidate_set(state, n_candidates, max_candidate_rank)
        if not cands:
            return _legal_fallback(state)
        rows = [state_candidate_features(state, ci) for ci in cands]
        X = pd.DataFrame(rows).reindex(columns=feature_cols, fill_value=0).fillna(0)
        pred = model.predict(X)
        return int(cands[int(np.argmax(pred))])
    return agent


def _indifference_band_from_meta(meta_path: Path = MODELS_DIR / "dk_ev_policy_meta.json") -> float:
    """Mirror server _dk_indifference_band(): the EV gap below which two candidates are
    decision-noise ties. Calibrated to the model's own avg regret, then a fraction of MAE."""
    try:
        m = json.loads(Path(meta_path).read_text()).get("metrics", {})
        for key, scale in (("model_avg_regret", 1.0), ("edge_mae", 0.5)):
            v = m.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return float(v) * scale
    except Exception:
        pass
    return 5e-5


def make_tiebreak_policy_agent(model, feature_cols: list[str], n_candidates: int,
                               band: float | None = None, max_band: int = 6,
                               max_candidate_rank: int | None = None):
    """Same candidate scoring as make_policy_agent, but within the model's indifference
    band rerank the near-tied top candidates on practical draft logic — a faithful port
    of the server's apply_dk_ev_tiebreaker (survival to next pick, roster need / legality
    pressure, stack quality, bye clash, ADP-reach guard). Used to measure whether the live
    tiebreaker actually beats the bare-model argmax."""
    if band is None:
        band = _indifference_band_from_meta()

    def agent(state: DraftState, rng: np.random.Generator) -> int:
        cands = legal_candidate_set(state, n_candidates, max_candidate_rank)
        if not cands:
            return _legal_fallback(state)
        rows = [state_candidate_features(state, ci) for ci in cands]
        X = pd.DataFrame(rows).reindex(columns=feature_cols, fill_value=0).fillna(0)
        pred = np.asarray(model.predict(X), dtype=float)
        order = np.argsort(-pred, kind="stable")            # candidate positions, best first
        top = float(pred[order[0]])

        band_pos: list[int] = []
        for p in order:
            if top - float(pred[p]) > band:
                break
            band_pos.append(int(p))
            if len(band_pos) >= max_band:
                break
        if len(band_pos) < 2:
            return int(cands[order[0]])

        picks_left = state.n_rounds - len(state.roster)
        best_p, best_bonus = band_pos[0], None
        for p in band_pos:
            r = rows[p]
            bonus = 0.0
            sgap = float(r.get("cand_adp_minus_pick_window", 0.0))   # adp - your next pick
            if sgap < 0:
                bonus += min(3.0, -sgap * 0.15)
            elif sgap > N_TEAMS:
                bonus -= 0.5
            pos = ("QB" if r.get("cand_pos_QB") else "RB" if r.get("cand_pos_RB")
                   else "WR" if r.get("cand_pos_WR") else "TE")
            need = int(r.get(f"need_{pos}", 0))
            if need > 0:
                bonus += 1.0 + (2.0 if picks_left <= need + 1 else 0.0)
            if r.get("cand_stack"):
                bonus += 1.5
            if r.get("qb_bye_overlap"):
                bonus -= 2.0
            if r.get("te_bye_overlap"):
                bonus -= 1.0
            if float(r.get("cand_adp_minus_pick", 0.0)) > 2 * N_TEAMS:
                bonus -= 0.5
            if best_bonus is None or bonus > best_bonus + 1e-9:   # ties keep model order
                best_bonus, best_p = bonus, p
        return int(cands[best_p])
    return agent


def make_adp_agent(strategy: str):
    """ADP drafter sitting in our seat. pure_adp = strict best-ADP-available;
    adp_noise = realistic reaches/falls + need nudges (same model as opponents)."""
    def agent(state: DraftState, rng: np.random.Generator) -> int:
        soft_cap, comfort = state.team_caps[state.our_team]
        picks_left = state.n_rounds - len(state.roster)
        return int(_choose(
            state.board, state.avail, state.counts[state.our_team],
            soft_cap, comfort, picks_left, state.round, state.n_rounds,
            state.sigma, strategy, rng,
        ))
    return agent


def _legal_fallback(state: DraftState) -> int:
    """Any available player under hard caps (board should never truly empty)."""
    from opponent_field import HARD_CAP
    under_hard = state.counts[state.our_team][state.board.pos_code] < HARD_CAP[state.board.pos_code]
    idx = np.where(state.avail & under_hard)[0]
    if idx.size == 0:
        idx = np.where(state.avail)[0]
    return int(idx[np.argmin(state.board.adp[idx])])


# ── One full draft with `our_agent` in `our_team`'s seat ──────────────────────
def run_draft(board, our_team: int, our_agent, rng: np.random.Generator,
              sigma: float, opp_strategy: str = "adp_noise") -> list[dict]:
    """Snake draft: opponents always draft `opp_strategy`; our seat uses
    `our_agent`. Returns our finished 20-man roster."""
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
    return rosters[our_team]


# ── Score a roster's prize-EV against the shared field ────────────────────────
def score_roster(roster, field, model, eval_seasons: int, season_seed: int) -> dict:
    sim = simulate_roster(roster, n_seasons=eval_seasons,
                          rng=np.random.default_rng(season_seed), model=model)
    ev = evaluate_roster(sim, field)
    return {"prize_ev": ev.prize_ev, "finals_rate": ev.finals_rate,
            "win_rate": ev.win_rate, "mean_points": ev.mean_points}


def _bootstrap_ci(diffs: np.ndarray, iters: int, seed: int, lo=2.5, hi=97.5):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    means = diffs[rng.integers(0, n, size=(iters, n))].mean(axis=1)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


# ── Backtest driver ───────────────────────────────────────────────────────────
def run_backtest(args) -> dict:
    rng = np.random.default_rng(args.seed)
    board = load_market_board()
    model_out = load_market_model()                       # outcome sim (no proj board)

    print(f"Building shared opponent field ({args.field_rooms} rooms x "
          f"{args.field_seasons} seasons)...")
    field_rosters = draft_field(args.field_rooms, rng, board=board, sigma=args.sigma)
    field = build_field(field_rosters, model=model_out,
                        n_seasons=args.field_seasons, rng=rng)

    policy_model = joblib.load(args.model)
    feature_cols = json.loads(Path(args.cols).read_text())
    agents = {
        "policy": make_policy_agent(policy_model, feature_cols, args.candidates),
        "pure_adp": make_adp_agent("pure_adp"),
        "adp_noise": make_adp_agent("adp_noise"),
    }

    recs = []
    for pod in range(args.pods):
        our_team = int(np.random.default_rng(args.seed + 1 + pod).integers(N_TEAMS))
        # Per-pod seeds: a DRAFT seed (re-used so all 3 agents face the identical
        # opponent stream) and a SEASON seed (re-used so all 3 rosters are scored
        # on the same weeks -> paired, low-variance EV comparison).
        draft_seed = int(np.random.default_rng(args.seed + 7000 + pod).integers(1 << 31))
        season_seed = int(np.random.default_rng(args.seed + 9000 + pod).integers(1 << 31))

        row = {"pod": pod, "our_team": our_team}
        for name, agent in agents.items():
            roster = run_draft(board, our_team, agent,
                               np.random.default_rng(draft_seed), args.sigma)
            sc = score_roster(roster, field, model_out, args.eval_seasons, season_seed)
            for k, v in sc.items():
                row[f"{name}_{k}"] = v
            row[f"{name}_pos"] = "".join(
                f"{p}{sum(1 for x in roster if x['position'] == p)}"
                for p in ("QB", "RB", "WR", "TE"))
        recs.append(row)
        print(f"  pod {pod + 1}/{args.pods} seat={our_team:>2} "
              f"EV policy={row['policy_prize_ev']:.3e} "
              f"pure_adp={row['pure_adp_prize_ev']:.3e} "
              f"adp_noise={row['adp_noise_prize_ev']:.3e}")

    df = pd.DataFrame(recs)
    summary = _summarize(df, args)
    _write_audit(args.out, summary, df, args)
    return summary


def _summarize(df: pd.DataFrame, args) -> dict:
    out = {"pods": int(len(df))}
    for base in ("pure_adp", "adp_noise"):
        for metric in ("prize_ev", "finals_rate", "win_rate"):
            p = df[f"policy_{metric}"].to_numpy(float)
            b = df[f"{base}_{metric}"].to_numpy(float)
            diff = p - b
            lo, hi = _bootstrap_ci(diff, args.bootstrap, args.seed)
            out[f"policy_vs_{base}__{metric}"] = {
                "policy_mean": float(p.mean()),
                "baseline_mean": float(b.mean()),
                "mean_lift": float(diff.mean()),
                "lift_ratio": float(p.mean() / b.mean()) if b.mean() else float("nan"),
                "head_to_head_winrate": float((p > b).mean()),
                "ci95_lift_lo": lo,
                "ci95_lift_hi": hi,
                "significant": bool(lo > 0 or hi < 0),
            }
    return out


def _write_audit(path: Path, summary: dict, df: pd.DataFrame, args) -> None:
    L = [
        "# DK EV Policy Backtest — drafting by the model vs. by ADP",
        "",
        "**In-simulation** end-to-end test: each agent drafts a full 20-man roster "
        "against a shared ADP-noise field; rosters are scored by prize-EV. This is "
        "only as valid as the simulator and outcome-model assumptions. "
        "It is NOT a real-money backtest (that needs historical ADP + real results).",
        "",
        f"- Pods: {summary['pods']}  |  base candidates/state: {args.candidates}  |  "
        f"eval-seasons/roster: {args.eval_seasons}",
        f"- Field: {args.field_rooms} rooms x {args.field_seasons} seasons  |  "
        f"model: `{args.model.name}`",
        "",
        "## Headline",
        "",
        "| comparison | metric | policy | baseline | lift ratio | H2H winrate | "
        "95% CI on mean lift | signif |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for base in ("pure_adp", "adp_noise"):
        for metric in ("prize_ev", "finals_rate", "win_rate"):
            s = summary[f"policy_vs_{base}__{metric}"]
            L.append(
                f"| vs {base} | {metric} | {s['policy_mean']:.3e} | "
                f"{s['baseline_mean']:.3e} | {s['lift_ratio']:.2f}x | "
                f"{s['head_to_head_winrate']:.0%} | "
                f"[{s['ci95_lift_lo']:.2e}, {s['ci95_lift_hi']:.2e}] | "
                f"{'YES' if s['significant'] else 'no'} |")
    L += [
        "",
        "## How to read this",
        "- **lift ratio > 1** and **H2H winrate > 50%**: the policy builds better "
        "rosters than that baseline.",
        "- **signif = YES**: the 95% bootstrap CI on the mean per-pod lift excludes "
        "0 (the edge is unlikely to be noise at this pod count).",
        "- `prize_ev` is the primary objective; `finals_rate`/`win_rate` are "
        "diagnostics. Absolute EV is a relative score (see payouts.py), so trust "
        "the ratios and the paired sign, not the raw magnitude.",
        "",
        f"_Per-pod rows: {len(df)} (full table in the sibling CSV)._",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    df.to_csv(path.with_suffix(".csv"), index=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pods", type=int, default=60)
    p.add_argument(
        "--candidates",
        type=int,
        default=12,
        help="base candidates per pick; later rounds expand automatically",
    )
    p.add_argument("--eval-seasons", type=int, default=400)
    p.add_argument("--field-rooms", type=int, default=24)
    p.add_argument("--field-seasons", type=int, default=200)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260613)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--cols", type=Path, default=DEFAULT_COLS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = run_backtest(args)
    print("\n=== SUMMARY ===")
    for base in ("pure_adp", "adp_noise"):
        s = summary[f"policy_vs_{base}__prize_ev"]
        print(f"policy vs {base:>9}: prize-EV {s['lift_ratio']:.2f}x  "
              f"H2H {s['head_to_head_winrate']:.0%}  "
              f"CI[{s['ci95_lift_lo']:.2e},{s['ci95_lift_hi']:.2e}]  "
              f"{'SIGNIF' if s['significant'] else 'n.s.'}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
