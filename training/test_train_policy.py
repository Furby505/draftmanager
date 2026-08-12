"""
Smoke tests for train_policy.py.

These avoid the expensive season/bracket evaluation and validate the draft-state
and completion mechanics that the rollout trainer depends on.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

import best_ball as bb
from backtest_policy import legal_candidate_set as served_policy_candidate_set
from opponent_field import Board, COMFORT, POS_CODE, SOFT_CAP, load_market_board
from train_policy import (
    DraftState,
    candidate_indices,
    candidate_depth_for_round,
    complete_with_candidate,
    main as train_policy_main,
    next_best_adp_gap_same_pos,
    sample_state,
    state_candidate_features,
)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        check.failed += 1
check.failed = 0


def main():
    check("default candidate depth starts at 12", candidate_depth_for_round(1) == 12)
    check("candidate depth expands by late rounds", candidate_depth_for_round(15) == 32)
    check("small smoke candidate bases scale instead of forcing 12",
          candidate_depth_for_round(15, 3) == 8)
    check("candidate depth can be capped to trained max rank",
          candidate_depth_for_round(15, 8, max_rank=8) == 8)

    board = load_market_board()
    rng = np.random.default_rng(123)
    state = sample_state(board, rng, min_round=2, max_round=8)

    check("state is stopped at our team's pick", state.current_team == state.our_team,
          f"current={state.current_team} ours={state.our_team}")
    check("partial roster is not full yet", 0 < len(state.roster) < bb.ROSTER_SIZE,
          f"size={len(state.roster)}")

    slot_state = sample_state(
        board,
        np.random.default_rng(456),
        min_round=2,
        max_round=8,
        target_slot=10,
    )
    check("targeted state sampler honors slot_in_round",
          slot_state.slot_in_round == 10,
          f"slot={slot_state.slot_in_round}")
    check("targeted state sampler keeps current team consistent",
          slot_state.current_team == slot_state.our_team,
          f"current={slot_state.current_team} ours={slot_state.our_team}")

    cands = candidate_indices(state, 5)
    check("candidate list is non-empty", len(cands) > 0, f"{len(cands)} candidates")
    check("candidates are available", all(state.avail[i] for i in cands))
    capped_cands = candidate_indices(state, 5, max_candidate_rank=3)
    check("candidate list honors max candidate rank cap",
          len(capped_cands) <= 3, f"{len(capped_cands)} candidates")
    gaps = [next_best_adp_gap_same_pos(state, i) for i in cands]
    check("same-position replacement gaps are non-negative",
          all(g >= 0 for g in gaps), ",".join(f"{g:.1f}" for g in gaps))

    cand = cands[0]
    features = state_candidate_features(state, cand)
    check("features include EV model columns",
          all(k in features for k in ("round", "cand_adp", "cand_pos_QB", "board_wr_left")))
    check("candidate metadata present", bool(features["candidate_name"]),
          features["candidate_name"])

    roster = complete_with_candidate(state, cand, rng)
    ids = [p["player_id"] or p["name"] for p in roster]
    positions = [p["position"] for p in roster]
    check("completion produces a full roster", len(roster) == bb.ROSTER_SIZE,
          f"size={len(roster)}")
    check("completion keeps roster unique", len(ids) == len(set(ids)))
    check("forced candidate is on completed roster",
          board.player_id[cand] in ids or board.name[cand] in ids,
          str(board.name[cand]))
    check("completed roster is DK-valid", bb.is_valid_roster(positions),
          ",".join(positions))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rollouts.csv"
        original = "run_seed,state_id,pick_no,candidate_name,candidate_pos,candidate_team,cand_adp,round,candidate_rank_in_state,ev_mean\n1,1,1,A,WR,KC,10,1,1,0.1\n"
        path.write_text(original, encoding="utf-8")
        train_policy_main(["--states", "0", "--out", str(path), "--no-model"])
        check("--states 0 does not overwrite existing rollout labels",
              path.read_text(encoding="utf-8") == original)

    mini = Board(
        player_id=np.array(["qb", "wr"], dtype=object),
        name=np.array(["Quarterback", "Wide Receiver"], dtype=object),
        team=np.array(["KC", "KC"], dtype=object),
        pos_code=np.array([POS_CODE["QB"], POS_CODE["WR"]], dtype=int),
        adp=np.array([1.0, 10.0], dtype=float),
        has_real_adp=np.ones(2, dtype=bool),
        market_tier=np.zeros(2, dtype=int),
        bye_week=np.zeros(2, dtype=int),
    )
    counts = np.zeros((12, 4), dtype=int)
    counts[0] = np.array([1, 2, 2, 1], dtype=int)
    rosters = [[{"position": "QB"}] * 17] + [[] for _ in range(11)]
    eighteen_round_state = DraftState(
        mini, 17, 0, rosters, counts, np.ones(len(mini), dtype=bool),
        [(SOFT_CAP, COMFORT)] * 12, 0.0, "pure_adp", [],
        12, 18, 18,
    )
    served_cands = served_policy_candidate_set(eighteen_round_state, 2)
    check("served policy legality honors 18-round DraftState deadline",
          served_cands == [1], f"candidates={served_cands}")

    print()
    if check.failed:
        print(f"{check.failed} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
