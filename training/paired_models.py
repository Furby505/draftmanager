"""PAIRED head-to-head between two DK EV policy model files.

Same seat / same draft / same field for both; the only difference is the model.
Used to isolate one training change (e.g. the handcuff layer): build a model with
the change ON and one with it OFF from matched rollouts, then measure the paired
advance/win/H2H delta on real weekly results.

    python training/paired_models.py --a model_ON.joblib --a-cols cols_ON.json \
        --b model_OFF.joblib --b-cols cols_OFF.json --label-a handcuff --label-b control
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_policy import make_policy_agent                          # noqa: E402
from validate_vs_humans import (                                       # noqa: E402
    build_draft_board, build_weekly_lookup, run_counterfactual,
    score_roster, N_TEAMS,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "server" / "models"
SAMPLE = ROOT / "data" / "raw" / "bbm_v_sample.csv"


def advances(board, board_nn, prefs, seat, agent, rng, pts):
    rosters = run_counterfactual(board, board_nn, prefs, seat, agent, rng)
    scores = np.array([score_roster(r, pts) for r in rosters])
    order = scores.argsort()[::-1]
    return seat in set(order[:2].tolist()), int(order[0]) == seat, float(scores[seat])


def _ci(d, iters=4000):
    d = np.asarray(d, float)
    rng = np.random.default_rng(1)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(iters)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", type=int, default=1500)
    ap.add_argument("--seats", type=int, default=12)
    ap.add_argument("--scoring", choices=["half", "full"], default="full")
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--a-max-candidate-rank", type=int, default=None)
    ap.add_argument("--b-max-candidate-rank", type=int, default=None)
    ap.add_argument("--a", type=Path, required=True, help="model A (the change, e.g. handcuff ON)")
    ap.add_argument("--a-cols", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True, help="model B (control, e.g. handcuff OFF)")
    ap.add_argument("--b-cols", type=Path, required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "paired_models_audit.md")
    args = ap.parse_args()

    print(f"Loading 2024 weekly ({args.scoring}-PPR) + BBM sample ...")
    pts, team_map = build_weekly_lookup(args.scoring)
    df = pd.read_csv(SAMPLE)
    draft_ids = df["draft_id"].drop_duplicates().tolist()[: args.drafts]

    a = make_policy_agent(
        joblib.load(args.a),
        json.loads(Path(args.a_cols).read_text()),
        args.candidates,
        args.a_max_candidate_rank,
    )
    b = make_policy_agent(
        joblib.load(args.b),
        json.loads(Path(args.b_cols).read_text()),
        args.candidates,
        args.b_max_candidate_rank,
    )
    seats = list(range(N_TEAMS))[: args.seats]

    a_adv, b_adv, a_win, b_win = [], [], [], []
    d_adv, d_win, score_diff, head2head = [], [], [], []

    for k, did in enumerate(draft_ids):
        g = df[df["draft_id"] == did]
        board, board_nn, prefs = build_draft_board(g, team_map)
        for s in seats:
            aa, aw, as_ = advances(board, board_nn, prefs, s, a, np.random.default_rng(1000 + s), pts)
            ba, bw, bs = advances(board, board_nn, prefs, s, b, np.random.default_rng(1000 + s), pts)
            a_adv.append(aa); b_adv.append(ba); a_win.append(aw); b_win.append(bw)
            d_adv.append(int(aa) - int(ba)); d_win.append(int(aw) - int(bw))
            score_diff.append(as_ - bs); head2head.append(int(as_ > bs))
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(draft_ids)} | {args.label_a} adv {np.mean(a_adv):.1%} "
                  f"vs {args.label_b} adv {np.mean(b_adv):.1%} | A-better {np.mean(head2head):.1%}")

    da, lo_a, hi_a = _ci(d_adv)
    dw, lo_w, hi_w = _ci(d_win)
    n = len(d_adv)
    sig_a = "YES" if (lo_a > 0 or hi_a < 0) else "no"
    changed = int(np.sum(np.asarray(score_diff) != 0))

    lines = [
        f"# PAIRED Head-to-Head — {args.label_a} vs {args.label_b} (matched rollouts, only the model differs)",
        "",
        f"- Scoring: **{args.scoring}-PPR**  |  drafts: {len(draft_ids)}  |  seats: {len(seats)}  "
        f"|  paired comparisons: {n}",
        f"- Rosters that differed: **{changed}/{n} ({changed/max(n,1):.1%})**",
        "",
        f"## Headline (paired {args.label_a} - {args.label_b})",
        "",
        f"- **Advance (top-2 of 12): {args.label_a} {np.mean(a_adv):.1%} vs {args.label_b} {np.mean(b_adv):.1%}** "
        f"-> paired lift {da:+.2%} (95% CI [{lo_a:+.2%}, {hi_a:+.2%}], signif={sig_a})",
        f"- **Pod-win (1st of 12): {args.label_a} {np.mean(a_win):.1%} vs {args.label_b} {np.mean(b_win):.1%}** "
        f"-> paired lift {dw:+.2%} (95% CI [{lo_w:+.2%}, {hi_w:+.2%}])",
        f"- **Roster-score H2H: {args.label_a} out-scores {args.label_b} {np.mean(head2head):.1%} of the time.**",
        "",
        "Caveat: 2024 only, Underdog field. signif=YES on advance means the change helps (or hurts).",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[5:11]))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
