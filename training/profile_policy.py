"""What does the served DK EV policy actually prioritize?

Runs the policy (and a pure-ADP control) through many REAL Underdog BBM boards in
the same harness as validate_vs_humans, then reports:
  - position drafted by round (draft-capital allocation / archetype)
  - final roster construction (avg QB/RB/WR/TE)
  - reach/wait behavior vs the board (mean ADP-rank of pick minus best available)

Behavioral, not a feature-importance hack: it shows how the model drafts, and
where it diverges from chalk. New standalone file; only reads existing modules.

    python training/profile_policy.py --drafts 300 --seat 5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_policy import DraftState, snake_team, _apply_pick          # noqa: E402
from backtest_policy import make_policy_agent, make_adp_agent         # noqa: E402
from validate_vs_humans import build_draft_board, build_weekly_lookup, resolve_candidate_depth_config, _caps, N_TEAMS, N_ROUNDS, _POS_NAME  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "server" / "models"
SAMPLE = ROOT / "data" / "raw" / "bbm_v_sample.csv"
POS = ("QB", "RB", "WR", "TE")


def run_seat(board, prefs, seat, agent, rng):
    """Draft `seat` with `agent`, opponents replay real picks. Return list of
    (round, position, adp_rank_of_pick, best_available_adp_rank) for our picks."""
    avail = np.ones(len(board), dtype=bool)
    counts = np.zeros((N_TEAMS, 4), dtype=int)
    rosters = [[] for _ in range(N_TEAMS)]
    ptr = {s: 0 for s in range(N_TEAMS)}
    hist = []
    log = []
    for gp in range(N_TEAMS * N_ROUNDS):
        team = snake_team(gp, N_TEAMS)
        if team == seat:
            state = DraftState(board, gp, seat, rosters, counts, avail,
                               [_caps()] * N_TEAMS, 0.0, "adp_noise", hist,
                               N_TEAMS, N_ROUNDS, N_ROUNDS)
            idx = agent(state, rng)
            rnd = gp // N_TEAMS + 1
            av = np.where(avail)[0]
            best_rank = int(np.argmin(board.adp[av]))            # 0 = took the literal best ADP
            took_rank = int((board.adp[av] < board.adp[idx]).sum())  # how many better-ADP players passed over
            log.append((rnd, _POS_NAME[board.pos_code[idx]], took_rank))
        else:
            idx = -1
            pl = prefs[team]
            while ptr[team] < len(pl):
                cand = pl[ptr[team]]; ptr[team] += 1
                if avail[cand]:
                    idx = cand; break
            if idx < 0:
                av = np.where(avail)[0]
                idx = int(av[np.argmin(board.adp[av])])
        _apply_pick(board, avail, counts, rosters, team, idx)
        hist.append((team, int(idx)))
    return log


def profile(agent, draft_ids, df, team_map, seat, rng):
    by_round = defaultdict(lambda: defaultdict(int))   # round -> pos -> count
    totals = defaultdict(int)
    reaches = []                                       # players-better-passed-over per pick
    for did in draft_ids:
        g = df[df["draft_id"] == did]
        board, _nn, prefs = build_draft_board(g, team_map)
        for rnd, pos, took_rank in run_seat(board, prefs, seat, agent, rng):
            by_round[rnd][pos] += 1
            totals[pos] += 1
            reaches.append(took_rank)
    return by_round, totals, np.array(reaches, float)


def show(name, by_round, totals, reaches, n_drafts):
    print(f"\n===== {name} =====")
    print("Round  QB   RB   WR   TE   (most-drafted position in CAPS)")
    for rnd in range(1, N_ROUNDS + 1):
        row = by_round[rnd]
        cells = []
        top = max(POS, key=lambda p: row.get(p, 0))
        for p in POS:
            pct = 100 * row.get(p, 0) / max(n_drafts, 1)
            cells.append(f"{pct:4.0f}" + ("*" if p == top else " "))
        print(f"  R{rnd:<3} " + " ".join(cells))
    tot = sum(totals.values())
    print("Avg roster:", "  ".join(f"{p} {totals[p]/max(n_drafts,1):.1f}" for p in POS),
          f"(of {tot/max(n_drafts,1):.0f} picks)")
    print(f"Reach behavior: mean {reaches.mean():.1f} better-ADP players passed over per pick "
          f"(0 = always took the best ADP available; higher = reaches for upside/need/stack)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", type=int, default=300)
    ap.add_argument("--seat", type=int, default=5, help="0-11 snake seat to profile")
    ap.add_argument("--model", type=Path, default=MODELS / "model_dk_ev_policy.joblib")
    ap.add_argument("--cols", type=Path, default=MODELS / "dk_ev_policy_feature_cols.json")
    ap.add_argument("--meta", type=Path, default=None,
                    help="optional policy meta JSON for auto candidate-depth detection")
    ap.add_argument("--candidates", type=int, default=None,
                    help="base candidate depth for the policy agent; default reads policy meta when available")
    ap.add_argument("--max-candidate-rank", type=int, default=None,
                    help="cap policy candidates to the model's trained max rank")
    args = ap.parse_args()

    _pts, team_map = build_weekly_lookup("half")
    df = pd.read_csv(SAMPLE)
    draft_ids = df["draft_id"].drop_duplicates().tolist()[: args.drafts]

    model = joblib.load(args.model)
    cols = json.loads(Path(args.cols).read_text())
    candidate_depth, max_candidate_rank = resolve_candidate_depth_config(args)
    policy = make_policy_agent(model, cols, candidate_depth, max_candidate_rank)
    policy_label = f"POLICY ({args.model.stem})"
    pure_adp = make_adp_agent("pure_adp")
    adp_noise = make_adp_agent("adp_noise")

    print(f"Profiling seat {args.seat} across {len(draft_ids)} real BBM boards "
          f"(% = share of drafts taking that position in that round)")
    for name, agent in (
        (policy_label, policy),
        ("PURE-ADP CONTROL", pure_adp),
        ("ADP-NOISE CONTROL", adp_noise),
    ):
        rng = np.random.default_rng(7)
        br, tot, reaches = profile(agent, draft_ids, df, team_map, args.seat, rng)
        show(name, br, tot, reaches, len(draft_ids))


if __name__ == "__main__":
    main()
