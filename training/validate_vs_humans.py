"""REAL-HUMAN validation: drop the DK EV policy into a real Underdog Best Ball
Mania V draft seat and see if it beats the actual humans who drafted that pod.

- Opponents are the 11 REAL humans (preference replay of their real picks; they
  only deviate if the model steals one of their players, then fall to next real pick).
- The model drafts one seat off the REAL board/ADP using its policy.
- All 12 rosters scored identically on REAL 2024 weekly results (best-ball optimal
  lineup, weeks 1-14 = UD qualifier window), top-2 of 12 advances.

No simulator, no ADP bots. Usage:
    python training/validate_vs_humans.py --drafts 1500 --seats 12
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from opponent_field import Board, POS_CODE, market_tier_from_adp  # noqa: E402
from scoring import recalc_weekly_df                              # noqa: E402
from train_policy import DraftState, snake_team, _apply_pick      # noqa: E402
from backtest_policy import make_policy_agent, make_adp_agent     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "server" / "models"
SAMPLE = ROOT / "data" / "raw" / "bbm_v_sample.csv"
WEEKLY = ROOT / "data" / "raw" / "player_stats_weekly.csv"
N_TEAMS, N_ROUNDS = 12, 18
REG_WEEKS = range(1, 15)   # UD BBM Round-1 qualifier = weeks 1-14

# High-frequency BBM name variants -> nflverse display name.
ALIASES = {
    "joshua palmer": "josh palmer",
    "hollywood brown": "marquise brown",
    "chig okonkwo": "chigoziem okonkwo",
    "donovan peoplesjones": "donovan peoples-jones",
    "gabe davis": "gabriel davis",
    "cam akers": "cameron akers",
    "tank dell": "nathaniel dell",
}


def norm(n: str) -> str:
    n = str(n).lower().strip()
    n = re.sub(r"[.'’\-]", "", n)          # strip periods/apostrophes AND hyphens
    n = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", n)
    n = re.sub(r"[^a-z ]", "", n)          # hyphens already gone; keep letters + space
    n = re.sub(r"\s+", " ", n).strip()
    return ALIASES.get(n, n)


def resolve_candidate_depth(args) -> int:
    """Use explicit --candidates, else read base candidate depth from meta when possible."""
    return resolve_candidate_depth_config(args)[0]


def resolve_candidate_depth_config(args) -> tuple[int, int | None]:
    """Return (base candidate depth, trained max candidate rank)."""
    explicit = getattr(args, "candidates", None)
    if explicit is not None:
        max_rank = getattr(args, "max_candidate_rank", None)
        return int(explicit), int(max_rank) if max_rank is not None else None

    model = Path(getattr(args, "model", MODELS / "model_dk_ev_policy.joblib"))
    cols = Path(getattr(args, "cols", MODELS / "dk_ev_policy_feature_cols.json"))
    candidates = []
    meta = getattr(args, "meta", None)
    if meta is not None:
        candidates.append(Path(meta))
    candidates.append(model.with_name(model.name.replace("model_", "").replace(".joblib", "_meta.json")))
    candidates.append(cols.with_name(cols.name.replace("feature_cols", "meta")))
    candidates.append(MODELS / "dk_ev_policy_meta.json")

    for path in candidates:
        try:
            meta = json.loads(Path(path).read_text())
            depth = int(meta.get("base_candidate_depth") or meta.get("max_candidate_rank") or 0)
            max_rank = int(meta.get("max_candidate_rank") or depth or 0)
            if depth > 0:
                return depth, max_rank if max_rank > 0 else None
        except Exception:
            pass
    return 8, 8


def build_weekly_lookup(scoring: str = "half") -> tuple[dict, dict]:
    """name_norm -> {week: pts} for 2024 weeks 1-14, and name_norm -> NFL team.
    scoring: 'half' = Underdog rules, what humans drafted for | 'full' = DK rules."""
    w = pd.read_csv(WEEKLY, low_memory=False)
    w = w[(w.season == 2024) & (w.week.isin(REG_WEEKS))]
    platform = "underdog" if scoring == "half" else "draftkings"
    w = recalc_weekly_df(w.copy(), platform=platform)
    pts: dict[str, dict[int, float]] = {}
    team: dict[str, str] = {}
    for _, r in w.iterrows():
        nm = norm(r["player_display_name"])
        val = float(r["fantasy_points_ppr"] or 0.0)
        pts.setdefault(nm, {})[int(r["week"])] = val
        t = r.get("recent_team")
        if isinstance(t, str) and t:
            team[nm] = t
    return pts, team


# ── best-ball optimal-lineup scoring (1QB/2RB/3WR/1TE/1FLEX) ───────────────────
SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
FLEX = ("RB", "WR", "TE")


def score_roster(roster: list[dict], pts: dict) -> float:
    """Sum optimal weekly best-ball lineup over weeks 1-14 (real 2024 results)."""
    by_pos = {p: [] for p in ("QB", "RB", "WR", "TE")}
    for pk in roster:
        by_pos[pk["position"]].append(pts.get(pk["nn"], {}))
    total = 0.0
    for wk in REG_WEEKS:
        used = []
        for pos, n in SLOTS.items():
            wkpts = sorted((d.get(wk, 0.0) for d in by_pos[pos]), reverse=True)
            total += sum(wkpts[:n])
            used.append((pos, wkpts[n:]))
        # FLEX: best remaining among RB/WR/TE
        leftover = []
        for pos, rem in used:
            if pos in FLEX:
                leftover.extend(rem)
        if leftover:
            total += max(leftover)
    return total


# ── build a Board for one real draft ──────────────────────────────────────────
def build_draft_board(g: pd.DataFrame, team_map: dict) -> tuple[Board, dict, list]:
    """g = all 216 rows of one draft. Returns Board (ordered by real ADP),
    a player_id->board_index map, and per-seat ordered real pick board-indices."""
    players = g.drop_duplicates("player_id")
    players = players.sort_values("projection_adp")
    pid = players["player_id"].to_numpy(object)
    idx_of = {p: i for i, p in enumerate(pid)}
    names = players["player_name"].to_numpy(object)
    nn = np.array([norm(x) for x in names], dtype=object)
    pos = players["position_name"].to_numpy(object)
    adp = players["projection_adp"].to_numpy(float)
    teams = np.array([team_map.get(n, "") for n in nn], dtype=object)
    bye = players["player_name"].map(lambda _: 0).to_numpy(int)  # byes already in weekly 0s

    board = Board(
        player_id=pid, name=names, team=teams,
        pos_code=np.array([POS_CODE[p] for p in pos], dtype=int),
        adp=adp, has_real_adp=np.ones(len(pid), dtype=bool), proj_points=None,
        market_tier=np.array([market_tier_from_adp(a) for a in adp], dtype=int),
        bye_week=bye,
    )
    # nn lookup attached for scoring
    board_nn = nn

    # per-seat ordered real picks (by overall_pick_number) -> board indices
    prefs = {s: [] for s in range(N_TEAMS)}
    for _, r in g.sort_values("overall_pick_number").iterrows():
        seat = int(r["pick_order"]) - 1
        prefs[seat].append(idx_of[r["player_id"]])
    return board, board_nn, prefs


def run_counterfactual(board, board_nn, prefs, model_seat, agent, rng):
    """Snake draft: model_seat uses `agent`; others replay real picks (preference
    replay). Returns all 12 final rosters (lists of pick dicts with nn+position)."""
    avail = np.ones(len(board), dtype=bool)
    counts = np.zeros((N_TEAMS, 4), dtype=int)
    rosters = [[] for _ in range(N_TEAMS)]
    team_caps = [None] * N_TEAMS
    pick_history = []
    ptr = {s: 0 for s in range(N_TEAMS)}     # pointer into each human's pref list

    for gp in range(N_TEAMS * N_ROUNDS):
        team = snake_team(gp, N_TEAMS)
        if team == model_seat:
            state = DraftState(board, gp, model_seat, rosters, counts, avail,
                               [_caps()] * N_TEAMS, 0.0, "adp_noise", pick_history,
                               N_TEAMS, N_ROUNDS, N_ROUNDS)
            idx = agent(state, rng)
        else:
            # take first real pick still available; else best available by ADP
            idx = -1
            pl = prefs[team]
            while ptr[team] < len(pl):
                cand = pl[ptr[team]]
                ptr[team] += 1
                if avail[cand]:
                    idx = cand
                    break
            if idx < 0:
                av = np.where(avail)[0]
                idx = int(av[np.argmin(board.adp[av])])
        _apply_pick(board, avail, counts, rosters, team, idx)
        pick_history.append((team, int(idx)))

    # attach nn + position to each pick for scoring
    out = []
    for s in range(N_TEAMS):
        roster = []
        for pk in rosters[s]:
            bi = _board_index(board, pk)
            roster.append({"nn": board_nn[bi],
                           "position": _POS_NAME[board.pos_code[bi]]})
        out.append(roster)
    return out


_POS_NAME = {v: k for k, v in POS_CODE.items()}


def _board_index(board, pick) -> int:
    return int(np.where(board.player_id == pick["player_id"])[0][0])


def _caps():
    from opponent_field import SOFT_CAP, COMFORT
    return (SOFT_CAP, COMFORT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", type=int, default=1500)
    ap.add_argument("--seats", type=int, default=12, help="model seats per draft (1-12)")
    ap.add_argument("--model", type=Path, default=MODELS / "model_dk_ev_policy.joblib")
    ap.add_argument("--cols", type=Path, default=MODELS / "dk_ev_policy_feature_cols.json")
    ap.add_argument("--meta", type=Path, default=None,
                    help="optional policy meta JSON for auto candidate-depth detection")
    ap.add_argument("--scoring", choices=["half", "full"], default="half",
                    help="half = Underdog rules (fair to humans); full = DK rules (model's objective)")
    ap.add_argument("--agent", choices=["policy", "pure_adp", "adp_noise"], default="policy",
                    help="policy = DK EV model; pure_adp/adp_noise = control bots in the SAME harness")
    ap.add_argument("--candidates", type=int, default=None,
                    help="base candidate depth for policy agents; default reads policy meta when available")
    ap.add_argument("--max-candidate-rank", type=int, default=None,
                    help="cap policy candidates to the model's trained max rank")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "validate_vs_humans_audit.md")
    args = ap.parse_args()

    print(f"Loading 2024 weekly results ({args.scoring}-PPR) + BBM sample ...")
    pts, team_map = build_weekly_lookup(args.scoring)
    df = pd.read_csv(SAMPLE)
    draft_ids = df["draft_id"].drop_duplicates().tolist()[: args.drafts]
    candidate_depth, max_candidate_rank = resolve_candidate_depth_config(args)

    if args.agent == "policy":
        model = joblib.load(args.model)
        cols = json.loads(Path(args.cols).read_text())
        agent = make_policy_agent(model, cols, candidate_depth, max_candidate_rank)
        agent_label = f"policy ({args.model.stem})"
        agent_source = f"- Model artifact: `{args.model}`"
    else:
        agent = make_adp_agent(args.agent)
        agent_label = args.agent
        agent_source = f"- Control agent: `{args.agent}`"
    print(f"Seat agent: {agent_label}")
    rng = np.random.default_rng(20260613)

    seats = list(range(N_TEAMS))[: args.seats]
    # paired per (draft, seat): model-in-seat vs the real human-in-seat
    pair_model_adv, pair_human_adv = [], []
    pair_model_win, pair_human_win = [], []
    xcheck_mine, xcheck_ud = [], []   # my score vs UD roster_points (sanity)

    for k, did in enumerate(draft_ids):
        g = df[df["draft_id"] == did]
        board, board_nn, prefs = build_draft_board(g, team_map)

        # (a) pure all-human replay -> measured human baseline under MY scoring
        human_rosters = run_counterfactual(board, board_nn, prefs, -1, agent, rng)
        hscores = np.array([score_roster(r, pts) for r in human_rosters])
        horder = hscores.argsort()[::-1]
        htop2 = set(horder[:2].tolist())
        hwin = int(horder[0])
        # UD roster_points cross-check (per seat)
        ud = (g.sort_values("pick_order").drop_duplicates("pick_order")
              .set_index(g.sort_values("pick_order").drop_duplicates("pick_order")["pick_order"] - 1)["roster_points"])
        for s in range(N_TEAMS):
            if s in ud.index:
                xcheck_mine.append(hscores[s]); xcheck_ud.append(float(ud[s]))

        # (b) model in each seat
        for ms in seats:
            rosters = run_counterfactual(board, board_nn, prefs, ms, agent, rng)
            scores = np.array([score_roster(r, pts) for r in rosters])
            order = scores.argsort()[::-1]
            top2 = set(order[:2].tolist())
            pair_model_adv.append(ms in top2)
            pair_human_adv.append(ms in htop2)
            pair_model_win.append(int(order[0]) == ms)
            pair_human_win.append(hwin == ms)
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(draft_ids)} | model adv {np.mean(pair_model_adv):.1%} "
                  f"vs human adv {np.mean(pair_human_adv):.1%}")

    def _ci(a, b, iters=2000):
        a, b = np.array(a, float), np.array(b, float)
        d = a - b
        rng2 = np.random.default_rng(1)
        bs = [d[rng2.integers(0, len(d), len(d))].mean() for _ in range(iters)]
        return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    adv = float(np.mean(pair_model_adv)); hadv = float(np.mean(pair_human_adv))
    win = float(np.mean(pair_model_win)); hwinr = float(np.mean(pair_human_win))
    d_adv, lo_a, hi_a = _ci(pair_model_adv, pair_human_adv)
    d_win, lo_w, hi_w = _ci(pair_model_win, pair_human_win)
    xcorr = float(np.corrcoef(xcheck_mine, xcheck_ud)[0, 1]) if len(xcheck_mine) > 2 else float("nan")

    n = len(pair_model_adv)
    sig_a = "YES" if lo_a > 0 else "no"
    sig_w = "YES" if lo_w > 0 else "no"
    scoring_label = (
        "Underdog rules - fair to the humans"
        if args.scoring == "half"
        else "DK rules - the model objective"
    )

    lines = [
        f"# REAL-HUMAN Validation - {agent_label} vs actual Underdog BBM V drafters",
        "",
        "Drops the selected seat agent into a real Underdog Best Ball Mania V (2024) draft seat. The "
        "other 11 seats are the REAL humans (preference replay of their actual picks; they "
        "only deviate if the model steals one of their players). All 12 rosters scored "
        "identically on REAL 2024 weekly results (best-ball optimal lineup, weeks 1-14, "
        "top-2 of 12 advance). No simulator, no ADP bots. Paired: seat-agent-in-seat vs the "
        "real human who held that seat (measured, not assumed).",
        "",
        f"- Scoring: **{args.scoring}-PPR** ({scoring_label})",
        f"- Real drafts: {len(draft_ids)}  |  model seats/draft: {len(seats)}  |  "
        f"paired comparisons: {n}",
        f"- Seat agent: **{agent_label}**",
        f"- Policy base candidates: {candidate_depth}" if args.agent == "policy" else None,
        f"- Policy max candidate rank: {max_candidate_rank}" if args.agent == "policy" else None,
        agent_source,
        f"- Scoring cross-check vs UD roster_points: corr = {xcorr:.3f} "
        f"(n={len(xcheck_mine)}; high = my engine agrees with Underdog's own scoring)",
        "",
        "## Headline",
        "",
        f"- **Seat-agent advance (top-2 of 12): {adv:.1%}**  vs  real-human {hadv:.1%}  "
        f"(lift {d_adv:+.1%}, 95% CI [{lo_a:+.1%}, {hi_a:+.1%}], signif={sig_a})",
        f"- **Seat-agent pod-win (1st of 12): {win:.1%}**  vs  real-human {hwinr:.1%}  "
        f"(lift {d_win:+.1%}, 95% CI [{lo_w:+.1%}, {hi_w:+.1%}], signif={sig_w})",
        "",
        "## How to read this",
        "- Every other seat is a real person's real draft. The baseline is the REAL human who "
        "actually held the agent's seat, scored by the same engine (so any scoring quirk hits "
        "both sides equally).",
        "- signif=YES means the paired 95% bootstrap CI on (agent - human) excludes 0.",
        "- Caveats: UD is half-PPR / 18 rounds. This is one season (2024), weeks 1-14.",
        "",
        f"_Per-comparison rows: {n}._",
    ]
    args.out.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")
    print("\n".join(lines[-8:]))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
