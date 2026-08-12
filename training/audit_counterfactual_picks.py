"""Counterfactual pick audit for the served DK EV policy.

This is audit-only. It does not retrain the model or change live draft logic.

For states the current policy actually reaches, force each top candidate,
complete the rest of the draft with the current policy, and compare final
roster prize EV. This answers whether repeated live decisions are locally
wrong after full-draft consequences, and which positions are being passed up.

Run:
    python training/audit_counterfactual_picks.py --states 24 --rollouts 2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
    legal_candidate_set,
    make_policy_agent,
)
from opponent_field import (  # noqa: E402
    DEFAULT_SIGMA,
    _choose,
    _sample_team_caps,
    build_field,
    draft_field,
    load_market_board,
)
from outcome_model import load_market_model  # noqa: E402
from train_policy import (  # noqa: E402
    DraftState,
    _apply_pick,
    complete_with_candidate,
    snake_team,
    state_candidate_features,
)
from season_sim import simulate_roster  # noqa: E402
from bracket import evaluate_roster  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
POS = ("QB", "RB", "WR", "TE")
POS_NAME = {0: "QB", 1: "RB", 2: "WR", 3: "TE"}


@dataclass
class PolicyState:
    state_id: int
    pod: int
    seat: int
    state: DraftState
    candidates: list[int]
    model_scores: np.ndarray
    policy_idx: int


def clone_state(state: DraftState) -> DraftState:
    return DraftState(
        board=state.board,
        gp=int(state.gp),
        our_team=int(state.our_team),
        rosters=[[dict(p) for p in roster] for roster in state.rosters],
        counts=state.counts.copy(),
        avail=state.avail.copy(),
        team_caps=[(soft.copy(), comfort.copy()) for soft, comfort in state.team_caps],
        sigma=float(state.sigma),
        strategy=str(state.strategy),
        pick_history=list(state.pick_history),
    )


def score_candidates(model, feature_cols: list[str], state: DraftState, candidates: list[int]) -> np.ndarray:
    rows = [state_candidate_features(state, ci) for ci in candidates]
    X = pd.DataFrame(rows).reindex(columns=feature_cols, fill_value=0).fillna(0)
    return np.asarray(model.predict(X), dtype=float)


def collect_policy_states(
    board,
    model,
    feature_cols: list[str],
    args,
    rng: np.random.Generator,
) -> list[PolicyState]:
    states: list[PolicyState] = []
    state_id = 0

    for pod in range(args.pods):
        our_team = int(np.random.default_rng(args.seed + 1 + pod).integers(N_TEAMS))
        draft_seed = int(np.random.default_rng(args.seed + 7000 + pod).integers(1 << 31))
        room_rng = np.random.default_rng(draft_seed)

        avail = np.ones(len(board), dtype=bool)
        counts = np.zeros((N_TEAMS, len(POS)), dtype=int)
        rosters: list[list[dict]] = [[] for _ in range(N_TEAMS)]
        team_caps = [_sample_team_caps(room_rng) for _ in range(N_TEAMS)]
        pick_history: list[tuple[int, int]] = []

        for gp in range(N_TEAMS * N_ROUNDS):
            team = snake_team(gp, N_TEAMS)
            rnd = gp // N_TEAMS
            if team == our_team:
                live_state = DraftState(
                    board, gp, our_team, rosters, counts, avail,
                    team_caps, args.sigma, "adp_noise", pick_history,
                )
                candidates = legal_candidate_set(live_state, args.candidates)
                if not candidates:
                    raise RuntimeError(f"no legal candidates at pod={pod} pick={gp + 1}")
                scores = score_candidates(model, feature_cols, live_state, candidates)
                pick_pos = int(np.argmax(scores))
                idx = int(candidates[pick_pos])

                round_no = rnd + 1
                if args.min_round <= round_no <= args.max_round:
                    states.append(PolicyState(
                        state_id=state_id,
                        pod=pod,
                        seat=our_team,
                        state=clone_state(live_state),
                        candidates=list(candidates),
                        model_scores=scores.copy(),
                        policy_idx=idx,
                    ))
                    state_id += 1
            else:
                soft_cap, comfort = team_caps[team]
                picks_left = N_ROUNDS - len(rosters[team])
                idx = int(_choose(
                    board, avail, counts[team], soft_cap, comfort,
                    picks_left, rnd, N_ROUNDS, args.sigma, "adp_noise", room_rng,
                ))

            _apply_pick(board, avail, counts, rosters, team, idx)
            pick_history.append((team, int(idx)))

    if len(states) <= args.states:
        return states

    keep = np.sort(rng.choice(len(states), size=args.states, replace=False))
    return [states[int(i)] for i in keep]


def roster_pos(roster: list[dict]) -> str:
    return "".join(f"{p}{sum(1 for x in roster if x.get('position') == p)}" for p in POS)


def evaluate_forced_candidate(
    ps: PolicyState,
    candidate_idx: int,
    completion_agent,
    field,
    outcome_model,
    args,
    rollout_seeds: list[tuple[int, int]],
) -> dict:
    evs, finals, wins, points = [], [], [], []
    final_pos = []
    for complete_seed, season_seed in rollout_seeds:
        roster = complete_with_candidate(
            ps.state,
            candidate_idx,
            np.random.default_rng(complete_seed),
            our_agent=completion_agent,
        )
        sim = simulate_roster(
            roster,
            n_seasons=args.eval_seasons,
            rng=np.random.default_rng(season_seed),
            model=outcome_model,
        )
        ev = evaluate_roster(sim, field)
        evs.append(ev.prize_ev)
        finals.append(ev.finals_rate)
        wins.append(ev.win_rate)
        points.append(ev.mean_points)
        final_pos.append(roster_pos(roster))

    return {
        "rollout_ev": float(np.mean(evs)),
        "rollout_ev_std": float(np.std(evs)),
        "rollout_finals": float(np.mean(finals)),
        "rollout_win_rate": float(np.mean(wins)),
        "rollout_mean_points": float(np.mean(points)),
        "final_pos_mode": pd.Series(final_pos).mode().iat[0] if final_pos else "",
    }


def run_audit(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    board = load_market_board()
    outcome = load_market_model()

    print(f"Building shared field ({args.field_rooms} rooms x {args.field_seasons} seasons)...")
    field_rosters = draft_field(args.field_rooms, rng, board=board, sigma=args.sigma)
    field = build_field(field_rosters, model=outcome, n_seasons=args.field_seasons, rng=rng)

    policy_model = joblib.load(args.model)
    feature_cols = json.loads(args.features.read_text())
    completion_agent = make_policy_agent(policy_model, feature_cols, args.candidates)

    states = collect_policy_states(board, policy_model, feature_cols, args, rng)
    if not states:
        raise RuntimeError("no states collected; widen --pods or round range")

    rows: list[dict] = []
    state_rows: list[dict] = []
    pos_rows: list[dict] = []

    for ordinal, ps in enumerate(states, start=1):
        print(
            f"State {ordinal}/{len(states)} pod={ps.pod} seat={ps.seat} "
            f"pick={ps.state.gp + 1} round={ps.state.round + 1} cands={len(ps.candidates)}"
        )
        seed_arr = rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=(args.rollouts, 2),
            dtype=np.uint32,
        )
        rollout_seeds = [(int(a), int(b)) for a, b in seed_arr]

        cand_rows = []
        for rank, candidate_idx in enumerate(ps.candidates, start=1):
            cand = ps.state.board.player_dict(candidate_idx)
            features = state_candidate_features(ps.state, candidate_idx)
            row = {
                "state_id": ps.state_id,
                "state_key": f"{args.seed}:{ps.state_id}:{ps.state.gp + 1}",
                "run_seed": args.seed,
                "pod": ps.pod,
                "seat": ps.seat + 1,
                "overall_pick": ps.state.gp + 1,
                "round": ps.state.round + 1,
                "candidate_rank": rank,
                "candidate_rank_in_state": rank,
                "candidate_idx": int(candidate_idx),
                "candidate_name": cand["name"],
                "candidate_pos": cand["position"],
                "candidate_team": cand["team"],
                "candidate_adp": cand["adp"],
                "model_score": float(ps.model_scores[rank - 1]),
                "is_policy_pick": int(candidate_idx == ps.policy_idx),
            }
            row.update(features)
            row.update(evaluate_forced_candidate(
                ps,
                candidate_idx,
                completion_agent,
                field,
                outcome,
                args,
                rollout_seeds,
            ))
            rows.append(row)
            cand_rows.append(row)
            print(
                f"  {rank:>2}. {cand['name']:<24} {cand['position']:<2} "
                f"model={row['model_score']:.3e} ev={row['rollout_ev']:.3e}"
                f"{' POLICY' if row['is_policy_pick'] else ''}"
            )

        cdf = pd.DataFrame(cand_rows)
        best = cdf.loc[cdf["rollout_ev"].idxmax()]
        policy = cdf[cdf["is_policy_pick"] == 1].iloc[0]
        state_rows.append({
            "state_id": ps.state_id,
            "pod": ps.pod,
            "seat": ps.seat + 1,
            "overall_pick": ps.state.gp + 1,
            "round": ps.state.round + 1,
            "policy_name": policy["candidate_name"],
            "policy_pos": policy["candidate_pos"],
            "policy_ev": float(policy["rollout_ev"]),
            "policy_model_score": float(policy["model_score"]),
            "best_name": best["candidate_name"],
            "best_pos": best["candidate_pos"],
            "best_ev": float(best["rollout_ev"]),
            "best_model_score": float(best["model_score"]),
            "ev_regret": float(best["rollout_ev"] - policy["rollout_ev"]),
            "policy_is_rollout_best": bool(best["candidate_idx"] == policy["candidate_idx"]),
            "model_rank_of_rollout_best": int(best["candidate_rank"]),
        })

        for pos in POS:
            sub = cdf[cdf["candidate_pos"] == pos]
            if sub.empty:
                continue
            pos_best = sub.loc[sub["rollout_ev"].idxmax()]
            pos_rows.append({
                "state_id": ps.state_id,
                "round": ps.state.round + 1,
                "position": pos,
                "policy_pos": policy["candidate_pos"],
                "policy_ev": float(policy["rollout_ev"]),
                "best_pos_name": pos_best["candidate_name"],
                "best_pos_ev": float(pos_best["rollout_ev"]),
                "pos_lift_vs_policy": float(pos_best["rollout_ev"] - policy["rollout_ev"]),
                "pos_beats_policy": bool(pos_best["rollout_ev"] > policy["rollout_ev"]),
            })

    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates["_state_avg_rollout_ev"] = candidates.groupby("state_key")["rollout_ev"].transform("mean")
        candidates["_state_max_rollout_ev"] = candidates.groupby("state_key")["rollout_ev"].transform("max")
        candidates["_state_policy_rollout_ev"] = candidates.groupby("state_key")["rollout_ev"].transform(
            lambda s: float(candidates.loc[s.index, "rollout_ev"][candidates.loc[s.index, "is_policy_pick"].eq(1)].iloc[0])
        )
        candidates["rollout_ev_edge"] = candidates["rollout_ev"] - candidates["_state_avg_rollout_ev"]
        candidates["rollout_lift_vs_policy"] = candidates["rollout_ev"] - candidates["_state_policy_rollout_ev"]
        candidates["rollout_best"] = (candidates["rollout_ev"] == candidates["_state_max_rollout_ev"]).astype(int)

    return candidates, pd.DataFrame(state_rows), pd.DataFrame(pos_rows)


def summarize(states: pd.DataFrame, positions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if states.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_round = (
        states.assign(round_bucket=lambda d: pd.cut(
            d["round"],
            bins=[0, 4, 8, 12, 16, 20],
            labels=["R1-4", "R5-8", "R9-12", "R13-16", "R17-20"],
        ))
        .groupby("round_bucket", observed=False)
        .agg(
            states=("state_id", "count"),
            policy_best_rate=("policy_is_rollout_best", "mean"),
            avg_regret=("ev_regret", "mean"),
            median_regret=("ev_regret", "median"),
            p90_regret=("ev_regret", lambda s: float(s.quantile(0.9))),
        )
        .reset_index()
    )

    by_pos = (
        positions.groupby("position")
        .agg(
            states_with_pos=("state_id", "count"),
            avg_lift_vs_policy=("pos_lift_vs_policy", "mean"),
            median_lift_vs_policy=("pos_lift_vs_policy", "median"),
            beats_policy_rate=("pos_beats_policy", "mean"),
        )
        .reset_index()
    )
    return by_round, by_pos


def write_report(
    path: Path,
    candidates: pd.DataFrame,
    states: pd.DataFrame,
    positions: pd.DataFrame,
    by_round: pd.DataFrame,
    by_pos: pd.DataFrame,
    args,
) -> None:
    overall_best_rate = float(states["policy_is_rollout_best"].mean()) if len(states) else float("nan")
    avg_regret = float(states["ev_regret"].mean()) if len(states) else float("nan")
    positive_regret = float((states["ev_regret"] > 0).mean()) if len(states) else float("nan")

    lines = [
        "# Counterfactual Pick Audit",
        "",
        "Audit-only rollout comparison for states reached by the current DK EV policy.",
        "",
        "## Setup",
        "",
        f"- States audited: {len(states)}",
        f"- Pods sampled: {args.pods}",
        f"- Rounds: {args.min_round}-{args.max_round}",
        f"- Candidates per state: top {args.candidates} legal ADP candidates, expanded by round schedule",
        f"- Rollouts per candidate: {args.rollouts}",
        f"- Eval seasons per rollout: {args.eval_seasons}",
        f"- Field: {args.field_rooms} rooms x {args.field_seasons} seasons",
        "- Opponents: ADP-noise drafters",
        "- Future picks after a forced counterfactual: current policy",
        "- This is in-simulator EV only. It is not real-money proof.",
        "",
        "## Headline",
        "",
        f"- Policy pick was rollout-best in {overall_best_rate:.1%} of audited states.",
        f"- Average rollout regret when comparing to the best candidate in the set: {avg_regret:.3e}.",
        f"- A different candidate beat the policy pick in {positive_regret:.1%} of states.",
        "",
        "## By Round Bucket",
        "",
        "| rounds | states | policy best | avg regret | median regret | p90 regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in by_round.iterrows():
        if int(r["states"]) == 0:
            continue
        lines.append(
            f"| {r['round_bucket']} | {int(r['states'])} | {float(r['policy_best_rate']):.1%} | "
            f"{float(r['avg_regret']):.3e} | {float(r['median_regret']):.3e} | {float(r['p90_regret']):.3e} |"
        )

    lines.extend([
        "",
        "## Best Available Position Counterfactual",
        "",
        "For each state, this compares the best candidate at each position to the policy pick.",
        "",
        "| position | states with pos | avg lift vs policy | median lift | beats policy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for _, r in by_pos.iterrows():
        lines.append(
            f"| {r['position']} | {int(r['states_with_pos'])} | {float(r['avg_lift_vs_policy']):.3e} | "
            f"{float(r['median_lift_vs_policy']):.3e} | {float(r['beats_policy_rate']):.1%} |"
        )

    misses = states[~states["policy_is_rollout_best"]].sort_values("ev_regret", ascending=False).head(12)
    lines.extend([
        "",
        "## Biggest Misses",
        "",
        "| pick | round | policy | best | regret | model rank of best |",
        "| ---: | ---: | --- | --- | ---: | ---: |",
    ])
    for _, r in misses.iterrows():
        lines.append(
            f"| {int(r['overall_pick'])} | {int(r['round'])} | {r['policy_name']} ({r['policy_pos']}) | "
            f"{r['best_name']} ({r['best_pos']}) | {float(r['ev_regret']):.3e} | "
            f"{int(r['model_rank_of_rollout_best'])} |"
        )

    lines.extend([
        "",
        "## Read",
        "",
        "- If one position's best counterfactual repeatedly beats the policy pick, that is a candidate signal for a second-stage reranker.",
        "- If misses cluster by round or roster shape, train the reranker on those state features instead of hardcoding position rules.",
        "- Treat small regrets cautiously; rollout noise remains even with common random numbers.",
        "",
        "## Output Files",
        "",
        f"- `{path.with_suffix('.csv').name}`",
        f"- `{path.with_name(path.stem + '_states.csv').name}`",
        f"- `{path.with_name(path.stem + '_positions.csv').name}`",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    candidates.to_csv(path.with_suffix(".csv"), index=False)
    states.to_csv(path.with_name(path.stem + "_states.csv"), index=False)
    positions.to_csv(path.with_name(path.stem + "_positions.csv"), index=False)
    by_round.to_csv(path.with_name(path.stem + "_by_round.csv"), index=False)
    by_pos.to_csv(path.with_name(path.stem + "_by_position.csv"), index=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=int, default=24)
    p.add_argument("--pods", type=int, default=12)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--rollouts", type=int, default=2)
    p.add_argument("--eval-seasons", type=int, default=180)
    p.add_argument("--field-rooms", type=int, default=12)
    p.add_argument("--field-seasons", type=int, default=100)
    p.add_argument("--min-round", type=int, default=2)
    p.add_argument("--max-round", type=int, default=16)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--features", type=Path, default=DEFAULT_COLS)
    p.add_argument("--out", type=Path, default=OUT / "counterfactual_pick_audit.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    candidates, states, positions = run_audit(args)
    by_round, by_pos = summarize(states, positions)
    write_report(args.out, candidates, states, positions, by_round, by_pos, args)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
