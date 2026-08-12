"""PAIRED head-to-head: bare DK EV model vs model + indifference-band tiebreaker.

Isolates the live tiebreaker's contribution. Both agents are the SAME model on the
SAME candidate set in the SAME seat of the SAME draft against the SAME field; the only
difference is that the tiebreak agent reranks near-tied candidates on practical draft
logic (the server's apply_dk_ev_tiebreaker). So the paired delta is the tiebreaker's
real edge, with a bootstrap CI.

    python training/paired_tiebreaker.py --drafts 1500 --scoring full
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
from backtest_policy import (                                       # noqa: E402
    make_policy_agent, make_tiebreak_policy_agent, _indifference_band_from_meta,
)
from validate_vs_humans import (                                    # noqa: E402
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
    ap.add_argument("--model", type=Path, default=MODELS / "model_dk_ev_policy.joblib")
    ap.add_argument("--cols", type=Path, default=MODELS / "dk_ev_policy_feature_cols.json")
    ap.add_argument("--candidates", type=int, default=8,
                    help="candidate depth the agents score (match the served max_candidate_rank)")
    ap.add_argument("--max-candidate-rank", type=int, default=None,
                    help="cap both agents to the model's trained max rank")
    ap.add_argument("--band", type=float, default=None,
                    help="override the indifference band (EV gap). Default = model_avg_regret from meta.")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "processed" / "paired_tiebreaker_audit.md")
    args = ap.parse_args()

    band = args.band if args.band is not None else _indifference_band_from_meta(args.cols.parent / "dk_ev_policy_meta.json")
    print(f"Loading 2024 weekly ({args.scoring}-PPR) + BBM sample ... band={band:.2e}")
    pts, team_map = build_weekly_lookup(args.scoring)
    df = pd.read_csv(SAMPLE)
    draft_ids = df["draft_id"].drop_duplicates().tolist()[: args.drafts]

    model = joblib.load(args.model)
    cols = json.loads(Path(args.cols).read_text())
    bare = make_policy_agent(model, cols, args.candidates, args.max_candidate_rank)
    tb = make_tiebreak_policy_agent(
        model,
        cols,
        args.candidates,
        band=band,
        max_candidate_rank=args.max_candidate_rank,
    )
    seats = list(range(N_TEAMS))[: args.seats]

    b_adv, t_adv, b_win, t_win = [], [], [], []
    d_adv, d_win, score_diff, head2head = [], [], [], []

    for k, did in enumerate(draft_ids):
        g = df[df["draft_id"] == did]
        board, board_nn, prefs = build_draft_board(g, team_map)
        for s in seats:
            ba, bw, bs = advances(board, board_nn, prefs, s, bare,
                                  np.random.default_rng(1000 + s), pts)
            ta, tw, ts = advances(board, board_nn, prefs, s, tb,
                                  np.random.default_rng(1000 + s), pts)
            b_adv.append(ba); t_adv.append(ta); b_win.append(bw); t_win.append(tw)
            d_adv.append(int(ta) - int(ba)); d_win.append(int(tw) - int(bw))
            score_diff.append(ts - bs); head2head.append(int(ts > bs))
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(draft_ids)} | tb adv {np.mean(t_adv):.1%} "
                  f"vs bare adv {np.mean(b_adv):.1%} | tb-better-roster {np.mean(head2head):.1%}")

    t_a, t_w = float(np.mean(t_adv)), float(np.mean(t_win))
    b_a, b_w = float(np.mean(b_adv)), float(np.mean(b_win))
    da, lo_a, hi_a = _ci(d_adv)
    dw, lo_w, hi_w = _ci(d_win)
    n = len(d_adv)
    sig_a = "YES" if (lo_a > 0 or hi_a < 0) else "no"
    changed = int(np.sum(np.asarray(score_diff) != 0))

    lines = [
        "# PAIRED Head-to-Head — DK EV model + tiebreaker vs bare model",
        "",
        "Same model, same candidate set, same seat/draft/field for both agents. The only "
        "difference is that the tiebreak agent reranks candidates within the model's "
        "indifference band (apply_dk_ev_tiebreaker logic). The paired delta is the "
        "tiebreaker's real contribution on top of the served model.",
        "",
        f"- Scoring: **{args.scoring}-PPR**  |  drafts: {len(draft_ids)}  |  seats: {len(seats)}  "
        f"|  paired comparisons: {n}  |  band: {band:.2e}",
        f"- Rosters that actually differed (tiebreaker changed a pick somewhere): "
        f"**{changed}/{n} ({changed/max(n,1):.1%})**",
        "",
        "## Headline (paired tiebreaker - bare model)",
        "",
        f"- **Advance (top-2 of 12): tiebreaker {t_a:.1%} vs bare {b_a:.1%}** "
        f"-> paired lift {da:+.2%} (95% CI [{lo_a:+.2%}, {hi_a:+.2%}], signif={sig_a})",
        f"- **Pod-win (1st of 12): tiebreaker {t_w:.1%} vs bare {b_w:.1%}** "
        f"-> paired lift {dw:+.2%} (95% CI [{lo_w:+.2%}, {hi_w:+.2%}])",
        f"- **Roster-score head-to-head: tiebreaker out-scores bare {np.mean(head2head):.1%} "
        f"of the time** (50% = tie; most rosters are identical so this hugs 50%).",
        "",
        "## How to read this",
        "- Most picks are NOT near-tied, so most rosters are identical -> small overall delta is expected.",
        "- What matters: among the rosters that DID change, did advance rate go UP? signif=YES on advance "
        "means the tiebreaker helps; signif=YES the other way means it hurts and should be reverted.",
        "- Caveat: 2024 only, Underdog field, snake legality; same caveats as paired_vs_adp.",
        "",
        f"_Per-comparison rows: {n}._",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[7:14]))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
