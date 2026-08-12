"""
Tests for Step 3 — opponent field (ADP-with-noise draft rooms).

Run: python training/test_opponent_field.py   (no pytest needed)

Asserts:
  1. Board loads, is ADP-sorted, and is deep enough to fill a draft.
  2. A drafted room: 12 rosters x 20 players, no duplicate players, every
     roster valid (1QB/2RB/3WR/1TE) and within caps (<=5 QB / <=5 TE).
  3. Position distributions are realistic (no degenerate builds).
  4. pure_adp is deterministic and front-loads top-ADP players;
     adp_noise varies across seeds but still tracks ADP.
  5. The field feeds the bracket: a strong roster beats a weak one in
     finals rate / prize-EV against the generated field.
"""

import sys
from collections import Counter

import numpy as np

import best_ball as bb
from bracket import evaluate_roster
from opponent_field import (POS, draft_field, load_market_board, simulate_draft)
from season_sim import simulate_roster


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        check.failed += 1
check.failed = 0


def _pos_counts(roster):
    c = Counter(p["position"] for p in roster)
    return {k: c.get(k, 0) for k in POS}


def main():
    board = load_market_board()

    # ---- 1. Board sanity --------------------------------------------------
    check("board deep enough for a full draft", len(board) >= 12 * 20 + 20,
          f"{len(board)} players")
    check("board is ADP-sorted", np.all(np.diff(board.adp) >= 0))
    check("top of board is a real player", board.name[0] != "",
          f"#1 = {board.name[0]} ({POS[board.pos_code[0]]})")

    # ---- 2/3. A drafted room is legal and realistic -----------------------
    rng = np.random.default_rng(5)
    room = simulate_draft(board, rng)
    sizes = [len(r) for r in room]
    all_ids = [p["player_id"] or p["name"] for r in room for p in r]
    check("12 rosters of 20 players", len(room) == 12 and set(sizes) == {20},
          f"sizes={set(sizes)}")
    check("no player drafted twice", len(all_ids) == len(set(all_ids)),
          f"{len(all_ids)} picks, {len(set(all_ids))} unique")

    valid = all(bb.is_valid_roster([p["position"] for p in r]) for r in room)
    check("every roster is valid (1QB/2RB/3WR/1TE & caps)", valid)

    counts = [_pos_counts(r) for r in room]
    qb = [c["QB"] for c in counts]; te = [c["TE"] for c in counts]
    rb = [c["RB"] for c in counts]; wr = [c["WR"] for c in counts]
    check("QB/TE within hard cap of 5", max(qb) <= 5 and max(te) <= 5,
          f"max QB={max(qb)} TE={max(te)}")
    check("position distributions realistic",
          1 <= min(qb) and max(qb) <= 4 and 1 <= min(te) and max(te) <= 4
          and 3 <= min(rb) and 4 <= min(wr),
          f"QB{min(qb)}-{max(qb)} RB{min(rb)}-{max(rb)} "
          f"WR{min(wr)}-{max(wr)} TE{min(te)}-{max(te)}")
    print(f"    mean per team: QB={np.mean(qb):.1f} RB={np.mean(rb):.1f} "
          f"WR={np.mean(wr):.1f} TE={np.mean(te):.1f}")
    builds = {(c["QB"], c["RB"], c["WR"], c["TE"]) for c in counts}
    check("field has construction diversity (not one uniform build)",
          len(builds) >= 4 and (max(qb) > min(qb) or max(rb) > min(rb)),
          f"{len(builds)} distinct builds; QB {min(qb)}-{max(qb)} RB {min(rb)}-{max(rb)}")

    # ---- 4. Strategy behavior --------------------------------------------
    r1 = simulate_draft(board, np.random.default_rng(0), strategy="pure_adp")
    r2 = simulate_draft(board, np.random.default_rng(999), strategy="pure_adp")
    ids1 = [p["player_id"] for r in r1 for p in r]
    ids2 = [p["player_id"] for r in r2 for p in r]
    check("pure_adp is deterministic (seed-independent)", ids1 == ids2)

    # First 12 picks of pure_adp should be (close to) the 12 best ADP players.
    first12 = {r1[t][0]["player_id"] for t in range(12)}
    board_top12 = set(board.player_id[:12])
    overlap = len(first12 & board_top12)
    check("pure_adp front-loads top-ADP players", overlap >= 10,
          f"{overlap}/12 of round 1 are board top-12")

    n1 = simulate_draft(board, np.random.default_rng(1))
    n2 = simulate_draft(board, np.random.default_rng(2))
    diff = sum(a["player_id"] != b["player_id"]
               for ra, rb_ in zip(n1, n2) for a, b in zip(ra, rb_))
    check("adp_noise varies across seeds", diff > 20, f"{diff} differing picks")

    # ADP adherence: average draft slot should rise with board ADP rank.
    slot_of = {}
    for room_ in (n1,):
        for t, r in enumerate(room_):
            for rnd, p in enumerate(r):
                slot_of.setdefault(p["player_id"], rnd * 12 + t)
    ranks = [(board.adp[i], slot_of.get(str(board.player_id[i])))
             for i in range(len(board))]
    ranks = [(a, s) for a, s in ranks if s is not None]
    a_arr = np.array([a for a, _ in ranks]); s_arr = np.array([s for _, s in ranks])
    corr = np.corrcoef(np.argsort(np.argsort(a_arr)),
                       np.argsort(np.argsort(s_arr)))[0, 1]
    check("draft order tracks ADP (noisy but strong)", corr > 0.7,
          f"spearman={corr:.3f}")

    # ---- 5. Field feeds the bracket --------------------------------------
    field_rosters = draft_field(6, np.random.default_rng(7))   # 72 opponent teams
    # Avoid the circular import at module top; build_field lives in opponent_field.
    from opponent_field import build_field
    field = build_field(field_rosters, n_seasons=120, rng=np.random.default_rng(8))

    strong = [
        {"name": "Ja'Marr Chase", "position": "WR", "team": "CIN"},
        {"name": "Justin Jefferson", "position": "WR", "team": "MIN"},
        {"name": "Puka Nacua", "position": "WR", "team": "LA"},
        {"name": "Bijan Robinson", "position": "RB", "team": "ATL"},
        {"name": "Saquon Barkley", "position": "RB", "team": "PHI"},
        {"name": "Josh Allen", "position": "QB", "team": "BUF"},
        {"name": "Lamar Jackson", "position": "QB", "team": "BAL"},
        {"name": "Brock Bowers", "position": "TE", "team": "LV"},
        {"name": "Trey McBride", "position": "TE", "team": "ARI"},
    ]
    # Weak roster: the deepest, lowest-ADP players on the board.
    weak = [board.player_dict(int(i)) for i in np.argsort(-board.adp)[:13]]

    rng2 = np.random.default_rng(9)
    es = evaluate_roster(simulate_roster(strong, 2000, rng2), field)
    ew = evaluate_roster(simulate_roster(weak, 2000, rng2), field)
    print(f"    strong: finals={es.finals_rate:.4f} prizeEV={es.prize_ev:.5f}")
    print(f"    weak:   finals={ew.finals_rate:.4f} prizeEV={ew.prize_ev:.5f}")
    check("strong roster beats weak roster vs the field",
          es.finals_rate > ew.finals_rate and es.prize_ev > ew.prize_ev)
    check("finals rates are valid probabilities",
          0 <= es.finals_rate <= 1 and 0 <= ew.finals_rate <= 1)

    print()
    if check.failed:
        print(f"{check.failed} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
