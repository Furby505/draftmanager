"""
Interactive DraftKings pod lab.

You draft one seat against 11 copies of the DK EV policy model. On your picks,
the runner shows model-ranked candidates and accepts:

  - Enter: take the model's top recommendation
  - 1..N: take that displayed candidate
  - player name text: take the best matching available legal player

Example:
  python training/draft_pod.py --seat 7
  python training/draft_pod.py --seat 7 --auto-user
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import best_ball as bb
from opponent_field import DEFAULT_SIGMA, MIN_POS, POS, _sample_team_caps, load_market_board
from train_policy import DraftState, legal_candidate_set, snake_team, state_candidate_features, _apply_pick


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "server" / "models"
PROCESSED = ROOT / "data" / "processed"
DEFAULT_MODEL = MODELS_DIR / "model_dk_ev_policy.joblib"
DEFAULT_COLS = MODELS_DIR / "dk_ev_policy_feature_cols.json"


def roster_counts(roster: list[dict]) -> Counter:
    return Counter(p.get("position", "?") for p in roster)


def fmt_player(p: dict) -> str:
    adp = p.get("adp")
    adp_s = f"{float(adp):.1f}" if adp is not None else "?"
    return f"{p['name']} {p['position']} {p.get('team') or ''} ADP {adp_s}".strip()


def make_state(board, gp, team, rosters, counts, avail, team_caps, sigma, pick_history) -> DraftState:
    return DraftState(
        board=board,
        gp=gp,
        our_team=team,
        rosters=rosters,
        counts=counts,
        avail=avail,
        team_caps=team_caps,
        sigma=sigma,
        strategy="model_pod",
        pick_history=pick_history,
    )


def score_candidates(state: DraftState, model, feature_cols: list[str], n: int) -> list[tuple[int, float]]:
    cands = legal_candidate_set(state, n)
    if not cands:
        return []
    rows = [state_candidate_features(state, ci) for ci in cands]
    x = pd.DataFrame(rows).reindex(columns=feature_cols, fill_value=0).fillna(0)
    pred = model.predict(x)
    return sorted(
        [(int(ci), float(score)) for ci, score in zip(cands, pred)],
        key=lambda item: item[1],
        reverse=True,
    )


def fallback_pick(state: DraftState) -> int:
    cands = legal_candidate_set(state, 1)
    if cands:
        return int(cands[0])
    idx = np.where(state.avail)[0]
    if idx.size == 0:
        raise RuntimeError("draft board exhausted")
    return int(idx[np.argmin(state.board.adp[idx])])


def model_pick(state: DraftState, model, feature_cols: list[str], n_candidates: int) -> int:
    scored = score_candidates(state, model, feature_cols, n_candidates)
    return int(scored[0][0]) if scored else fallback_pick(state)


def print_roster(roster: list[dict], label: str = "Roster") -> None:
    counts = roster_counts(roster)
    picks = "  ".join(f"{pos}{counts.get(pos, 0)}" for pos in POS)
    print(f"{label}: {len(roster)}/{bb.ROSTER_SIZE}  {picks}")
    if roster:
        print("  " + " | ".join(f"{p['name']}({p['position']})" for p in roster[-6:]))


def choose_user_pick(state: DraftState, model, feature_cols: list[str], n_candidates: int) -> int:
    scored = score_candidates(state, model, feature_cols, n_candidates)
    if not scored:
        return fallback_pick(state)

    print()
    print("=" * 88)
    print(f"YOUR PICK #{state.gp + 1}  Round {state.round + 1}  Seat {state.our_team + 1}")
    print_roster(state.roster, "Your roster")
    print("-" * 88)
    for i, (idx, score) in enumerate(scored, 1):
        p = state.board.player_dict(idx)
        delta = float(state.board.adp[idx]) - float(state.gp + 1)
        print(f"{i:>2}. {fmt_player(p):<42} score {score: .5f}  ADP-pick {delta: .1f}")

    while True:
        raw = input("Pick (Enter top, number, or player name): ").strip()
        if not raw:
            return int(scored[0][0])
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(scored):
                return int(scored[n - 1][0])
        needle = raw.lower()
        legal = legal_candidate_set(state, 240)
        matches = [
            idx for idx in legal
            if needle in str(state.board.name[idx]).lower()
        ]
        if matches:
            matches.sort(key=lambda idx: float(state.board.adp[idx]))
            if len(matches) > 1:
                print("Matches: " + ", ".join(str(state.board.name[i]) for i in matches[:8]))
            return int(matches[0])
        print("No legal available match. Try a number from the list or a clearer name.")


def score_rosters(rosters: list[list[dict]], args, rng: np.random.Generator) -> list[dict] | None:
    if not args.score:
        return None

    from bracket import evaluate_roster
    from opponent_field import build_field, draft_field
    from outcome_model import load_market_model
    from season_sim import simulate_roster

    print()
    print(f"Scoring rosters: field {args.field_rooms} rooms x {args.field_seasons} seasons; "
          f"eval {args.eval_seasons} seasons/roster...")
    model_out = load_market_model()
    board = load_market_board()
    field_rosters = draft_field(args.field_rooms, rng, board=board, sigma=args.sigma)
    field = build_field(field_rosters, model=model_out, n_seasons=args.field_seasons, rng=rng)

    scored = []
    for team, roster in enumerate(rosters):
        sim = simulate_roster(
            roster,
            n_seasons=args.eval_seasons,
            rng=np.random.default_rng(args.seed + 10000 + team),
            model=model_out,
        )
        ev = evaluate_roster(sim, field)
        row = {
            "team": team + 1,
            "prize_ev": ev.prize_ev,
            "finals_rate": ev.finals_rate,
            "win_rate": ev.win_rate,
            "mean_points": ev.mean_points,
        }
        scored.append(row)
        print(f"  T{team + 1:<2} EV={ev.prize_ev:.3e} finals={ev.finals_rate:.1%} "
              f"win={ev.win_rate:.2%} pts={ev.mean_points:.1f}")
    return scored


def write_outputs(args, board, pick_log: list[dict], rosters: list[list[dict]], user_team: int,
                  scores: list[dict] | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROCESSED / f"draft_pod_{stamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(pick_log).to_csv(out.with_suffix(".csv"), index=False)

    lines = [
        "# Draft Pod",
        "",
        f"- Seat: {user_team + 1}",
        f"- Opponents: 11 DK EV policy model seats",
        f"- Candidates scored per pick: {args.candidates}",
        f"- Seed: {args.seed}",
        "",
        "## Rosters",
        "",
    ]
    if scores:
        lines += [
            "## Sim Score",
            "",
            "| team | seat | prize_ev | finals | win | mean_points |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        score_by_team = {s["team"]: s for s in scores}
        for team in range(len(rosters)):
            s = score_by_team[team + 1]
            tag = "YOU" if team == user_team else "MODEL"
            lines.append(
                f"| {team + 1} | {tag} | {s['prize_ev']:.3e} | "
                f"{s['finals_rate']:.1%} | {s['win_rate']:.2%} | {s['mean_points']:.1f} |"
            )
        lines.append("")

    for team, roster in enumerate(rosters):
        counts = roster_counts(roster)
        tag = "YOU" if team == user_team else "MODEL"
        lines.append(f"### Team {team + 1} ({tag}) - " + " ".join(f"{p}{counts.get(p,0)}" for p in POS))
        for i, p in enumerate(roster, 1):
            lines.append(f"{i:>2}. {p['name']} ({p['position']} {p.get('team') or ''}) ADP {float(p.get('adp', 0)):.1f}")
        lines.append("")

    lines += ["## Pick Log", ""]
    for row in pick_log:
        lines.append(
            f"{row['pick']:>3}. T{row['team']} {row['name']} "
            f"({row['position']} {row['nfl_team']}) [{row['agent']}]"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run(args) -> Path:
    rng = np.random.default_rng(args.seed)
    board = load_market_board()
    model = joblib.load(args.model)
    feature_cols = json.loads(Path(args.cols).read_text(encoding="utf-8"))

    user_team = int(args.seat) - 1 if args.seat else int(rng.integers(bb.TEAMS_PER_POD))
    if not 0 <= user_team < bb.TEAMS_PER_POD:
        raise ValueError("--seat must be 1..12")

    avail = np.ones(len(board), dtype=bool)
    counts = np.zeros((bb.TEAMS_PER_POD, len(POS)), dtype=int)
    rosters: list[list[dict]] = [[] for _ in range(bb.TEAMS_PER_POD)]
    team_caps = [_sample_team_caps(rng) for _ in range(bb.TEAMS_PER_POD)]
    pick_history: list[tuple[int, int]] = []
    pick_log: list[dict] = []

    print(f"Draft pod started. Your seat: {user_team + 1}. Rounds: {bb.DRAFT_ROUNDS}.")
    print("On your pick, press Enter to accept the model's top rec.")

    for gp in range(bb.TEAMS_PER_POD * bb.DRAFT_ROUNDS):
        team = snake_team(gp, bb.TEAMS_PER_POD)
        state = make_state(board, gp, team, rosters, counts, avail, team_caps, args.sigma, pick_history)
        if team == user_team and not args.auto_user:
            idx = choose_user_pick(state, model, feature_cols, args.candidates)
            agent = "YOU"
        else:
            idx = model_pick(state, model, feature_cols, args.candidates)
            agent = "MODEL" if team != user_team else "AUTO_USER"

        _apply_pick(board, avail, counts, rosters, team, idx)
        pick_history.append((team, int(idx)))
        p = board.player_dict(idx)
        pick_log.append({
            "pick": gp + 1,
            "round": gp // bb.TEAMS_PER_POD + 1,
            "team": team + 1,
            "agent": agent,
            "name": p["name"],
            "position": p["position"],
            "nfl_team": p.get("team") or "",
            "adp": p.get("adp"),
        })
        if team != user_team or args.auto_user:
            print(f"{gp + 1:>3}. T{team + 1:<2} {p['name']} ({p['position']} {p.get('team') or ''}) [{agent}]")

    print()
    print_roster(rosters[user_team], "Final user roster")
    scores = score_rosters(rosters, args, rng)
    out = write_outputs(args, board, pick_log, rosters, user_team, scores)
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.csv')}")
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seat", type=int, default=1, help="Your draft slot, 1..12.")
    parser.add_argument("--auto-user", action="store_true", help="Let the model draft your seat too.")
    parser.add_argument(
        "--candidates",
        type=int,
        default=12,
        help="Base ADP/legal candidates the model scores; later rounds expand automatically.",
    )
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cols", type=Path, default=DEFAULT_COLS)
    parser.add_argument("--score", action="store_true", help="Score all 12 rosters with the tournament EV simulator.")
    parser.add_argument("--eval-seasons", type=int, default=200)
    parser.add_argument("--field-rooms", type=int, default=8)
    parser.add_argument("--field-seasons", type=int, default=80)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
