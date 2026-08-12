"""Direct PAIRED head-to-head: DK EV policy vs a realistic ADP-noise drafter.

The validate_vs_humans run compares each agent against the human-in-seat
separately, so "model 42.5% vs adp_noise 41.6%" is two unpaired numbers. This
runs BOTH agents in the SAME seat of the SAME draft against the SAME real-human
field, then compares who advances. The harness flattery (weak non-reactive
field + steal asymmetry) is identical for both, so it cancels: the paired delta
is the real model-over-ADP edge, with a bootstrap CI.

    python training/paired_vs_adp.py --drafts 1500 --scoring half
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
from backtest_policy import make_policy_agent, make_adp_agent          # noqa: E402
from validate_vs_humans import (                                       # noqa: E402
    build_draft_board, build_weekly_lookup, run_counterfactual,
    resolve_candidate_depth_config, score_roster, N_TEAMS,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "server" / "models"
SAMPLE = ROOT / "data" / "raw" / "bbm_v_sample.csv"


def advances(board, board_nn, prefs, seat, agent, rng, pts):
    """Run one counterfactual with `agent` in `seat`; return outcome plus roster shape."""
    rosters = run_counterfactual(board, board_nn, prefs, seat, agent, rng)
    scores = np.array([score_roster(r, pts) for r in rosters])
    order = scores.argsort()[::-1]
    shape = {pos: sum(1 for p in rosters[seat] if p["position"] == pos)
             for pos in ("QB", "RB", "WR", "TE")}
    return seat in set(order[:2].tolist()), int(order[0]) == seat, float(scores[seat]), shape


def _ci(d, iters=4000):
    d = np.asarray(d, float)
    rng = np.random.default_rng(1)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(iters)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", type=int, default=1500)
    ap.add_argument("--seats", type=int, default=12)
    ap.add_argument("--scoring", choices=["half", "full"], default="half")
    ap.add_argument("--model", type=Path, default=MODELS / "model_dk_ev_policy.joblib")
    ap.add_argument("--cols", type=Path, default=MODELS / "dk_ev_policy_feature_cols.json")
    ap.add_argument("--meta", type=Path, default=None,
                    help="optional policy meta JSON for auto candidate-depth detection")
    ap.add_argument("--candidates", type=int, default=None,
                    help="base candidate depth for the policy agent; default reads policy meta when available")
    ap.add_argument("--max-candidate-rank", type=int, default=None,
                    help="cap policy candidates to the model's trained max rank")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "paired_vs_adp_audit.md")
    args = ap.parse_args()

    print(f"Loading 2024 weekly ({args.scoring}-PPR) + BBM sample ...")
    pts, team_map = build_weekly_lookup(args.scoring)
    df = pd.read_csv(SAMPLE)
    draft_ids = df["draft_id"].drop_duplicates().tolist()[: args.drafts]

    model = joblib.load(args.model)
    cols = json.loads(Path(args.cols).read_text())
    candidate_depth, max_candidate_rank = resolve_candidate_depth_config(args)
    policy = make_policy_agent(model, cols, candidate_depth, max_candidate_rank)
    policy_label = f"policy ({args.model.stem})"
    noise = make_adp_agent("adp_noise")
    seats = list(range(N_TEAMS))[: args.seats]

    p_adv, n_adv, p_win, n_win = [], [], [], []
    d_adv, d_win, score_diff = [], [], []   # paired model - noise
    head2head = []                          # 1 if model roster > noise roster (same seat/draft)
    rows = []

    for k, did in enumerate(draft_ids):
        g = df[df["draft_id"] == did]
        board, board_nn, prefs = build_draft_board(g, team_map)
        for s in seats:
            # fresh, identically-seeded rng per (draft,seat) so noise stream matches across agents
            pa, pw, ps, p_shape = advances(board, board_nn, prefs, s, policy,
                                            np.random.default_rng(1000 + s), pts)
            na, nw, ns, n_shape = advances(board, board_nn, prefs, s, noise,
                                           np.random.default_rng(1000 + s), pts)
            p_adv.append(pa); n_adv.append(na); p_win.append(pw); n_win.append(nw)
            d_adv.append(int(pa) - int(na)); d_win.append(int(pw) - int(nw))
            score_diff.append(ps - ns); head2head.append(int(ps > ns))
            row = {
                "draft_id": did,
                "seat": s,
                "policy_adv": int(pa),
                "noise_adv": int(na),
                "policy_win": int(pw),
                "noise_win": int(nw),
                "policy_score": ps,
                "noise_score": ns,
                "score_diff": ps - ns,
                "policy_score_gt_noise": int(ps > ns),
            }
            for pos in ("QB", "RB", "WR", "TE"):
                row[f"policy_{pos}"] = p_shape[pos]
                row[f"noise_{pos}"] = n_shape[pos]
                row[f"delta_{pos}"] = p_shape[pos] - n_shape[pos]
            rows.append(row)
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(draft_ids)} | model adv {np.mean(p_adv):.1%} "
                  f"vs noise adv {np.mean(n_adv):.1%} | H2H {np.mean(head2head):.1%}")

    m_adv, m_win = float(np.mean(p_adv)), float(np.mean(p_win))
    a_adv, a_win = float(np.mean(n_adv)), float(np.mean(n_win))
    da, lo_a, hi_a = _ci(d_adv)
    dw, lo_w, hi_w = _ci(d_win)
    h2h, lo_h, hi_h = _ci(head2head)
    n = len(d_adv)
    sig_a = "YES" if (lo_a > 0 or hi_a < 0) else "no"
    sig_h = "YES" if (lo_h > 0.5 or hi_h < 0.5) else "no"
    detail = pd.DataFrame(rows)
    by_seat = (
        detail.groupby("seat")
        .agg(
            n=("seat", "size"),
            policy_adv=("policy_adv", "mean"),
            noise_adv=("noise_adv", "mean"),
            policy_win=("policy_win", "mean"),
            noise_win=("noise_win", "mean"),
            h2h=("policy_score_gt_noise", "mean"),
            score_diff=("score_diff", "mean"),
            policy_QB=("policy_QB", "mean"),
            policy_RB=("policy_RB", "mean"),
            policy_WR=("policy_WR", "mean"),
            policy_TE=("policy_TE", "mean"),
            noise_QB=("noise_QB", "mean"),
            noise_RB=("noise_RB", "mean"),
            noise_WR=("noise_WR", "mean"),
            noise_TE=("noise_TE", "mean"),
        )
        .reset_index()
    )
    by_seat["adv_lift"] = by_seat["policy_adv"] - by_seat["noise_adv"]
    by_seat["win_lift"] = by_seat["policy_win"] - by_seat["noise_win"]

    lines = [
        f"# PAIRED Head-to-Head - {policy_label} vs realistic ADP-noise drafter",
        "",
        "Both agents drafted the SAME seat of the SAME real Underdog BBM V (2024) draft "
        "against the SAME real-human field, then scored on real 2024 weekly results. Paired "
        "per (draft, seat): the only difference is the agent, so the harness flattery (weak "
        "non-reactive field + steal asymmetry) cancels. The paired delta is the real edge.",
        "",
        f"- Scoring: **{args.scoring}-PPR**  |  drafts: {len(draft_ids)}  |  seats: {len(seats)}  |  paired comparisons: {n}",
        f"- Policy artifact: `{args.model}`",
        f"- Policy base candidates: {candidate_depth}",
        f"- Policy max candidate rank: {max_candidate_rank}",
        "",
        "## Headline (paired model - adp_noise)",
        "",
        f"- **Advance (top-2 of 12): model {m_adv:.1%} vs noise {a_adv:.1%}** "
        f"-> paired lift {da:+.2%} (95% CI [{lo_a:+.2%}, {hi_a:+.2%}], signif={sig_a})",
        f"- **Pod-win (1st of 12): model {m_win:.1%} vs noise {a_win:.1%}** "
        f"-> paired lift {dw:+.2%} (95% CI [{lo_w:+.2%}, {hi_w:+.2%}])",
        f"- **Roster-score head-to-head: model out-scores noise {h2h:.1%} of the time** "
        f"(95% CI [{lo_h:.1%}, {hi_h:.1%}], 50% = tie, signif={sig_h})",
        "",
        "## How to read this",
        "- This is the honest model-vs-ADP comparison: same field for both, flattery cancelled.",
        "- Advance signif=YES means the paired 95% CI on (model - noise) excludes 0.",
        "- H2H signif=YES means the model beats adp_noise's roster more/less than a coin flip.",
        "- Caveat: UD half-PPR / 18 rounds, one season (2024). adp_noise = ADP + reaches/falls + need nudges.",
        "",
        "## By Seat",
        "",
        "| seat | n | policy adv | noise adv | adv lift | policy win | noise win | win lift | H2H | score diff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {int(r.seat)} | {int(r.n)} | {r.policy_adv:.1%} | {r.noise_adv:.1%} | "
            f"{r.adv_lift:+.1%} | {r.policy_win:.1%} | {r.noise_win:.1%} | "
            f"{r.win_lift:+.1%} | {r.h2h:.1%} | {r.score_diff:.1f} |"
            for r in by_seat.itertuples(index=False)
        ],
        "",
        "## By Seat Roster Shape",
        "",
        "| seat | policy QB/RB/WR/TE | noise QB/RB/WR/TE | delta QB/RB/WR/TE |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| {int(r.seat)} | "
            f"{r.policy_QB:.1f}/{r.policy_RB:.1f}/{r.policy_WR:.1f}/{r.policy_TE:.1f} | "
            f"{r.noise_QB:.1f}/{r.noise_RB:.1f}/{r.noise_WR:.1f}/{r.noise_TE:.1f} | "
            f"{r.policy_QB - r.noise_QB:+.1f}/{r.policy_RB - r.noise_RB:+.1f}/"
            f"{r.policy_WR - r.noise_WR:+.1f}/{r.policy_TE - r.noise_TE:+.1f} |"
            for r in by_seat.itertuples(index=False)
        ],
        "",
        f"_Per-comparison rows: {n}._",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    detail.to_csv(args.out.with_suffix(".csv"), index=False)
    by_seat.to_csv(args.out.with_name(args.out.stem + "_by_seat.csv"), index=False)
    print("\n" + "\n".join(lines[6:14]))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
