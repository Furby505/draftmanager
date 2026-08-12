"""
Rollout-based draft policy trainer for the DK tournament simulator.

This is the first bridge from the calibrated environment
(ADP draft rooms -> correlated season sim -> bracket prize-EV) to an actual
pick policy. It samples realistic in-draft states, evaluates candidate picks by
forcing each candidate, completing the rest of the room with ADP/noise drafters,
and scoring the completed roster by expected prize equity.

The output CSV is intentionally plain training data:

    state features + candidate features -> ev_mean

A later pass can distill this into a fast online model for the server. By
default this script only generates EV rollout labels; fitting the served policy
is opt-in.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import best_ball as bb
from bracket import evaluate_roster
from opponent_field import (
    DEFAULT_SIGMA,
    HARD_CAP,
    REALISM_CAP,
    MIN_POS,
    POS,
    Board,
    _choose,
    _sample_team_caps,
    build_field,
    draft_field,
    load_bbm_v_board,
    load_adp_board,
    load_market_board,
)
from season_sim import simulate_roster
from outcome_model import (
    A_GAME, load_market_model, load_matchup_tables, playoff_matchup_rating,
)

# Defense-vs-position matchup tables, loaded once for the policy feature.
_MATCHUP_RATINGS, _MATCHUP_OPP = load_matchup_tables()


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "server" / "models"

DEFAULT_OUT = PROCESSED / "policy_training_rollouts.csv"
DEFAULT_MODEL_OUT = MODELS_DIR / "model_dk_ev_policy.joblib"
DEFAULT_FEATURES_OUT = MODELS_DIR / "dk_ev_policy_feature_cols.json"

DEFAULT_CANDIDATES = 12
CANDIDATE_DEPTH_SCHEDULE = (
    (1, 12),
    (3, 14),
    (7, 18),
    (11, 24),
    (15, 32),
)

FEATURE_COLS = [
    "pick_no",
    "round",
    "slot_in_round",
    "picks_made_by_team",
    "picks_left",
    "n_QB",
    "n_RB",
    "n_WR",
    "n_TE",
    "need_QB",
    "need_RB",
    "need_WR",
    "need_TE",
    "cand_adp",
    "cand_adp_minus_pick",
    "cand_board_rank",
    "cand_has_real_adp",
    "cand_bye_week",
    "cand_market_tier",
    "cand_pos_QB",
    "cand_pos_RB",
    "cand_pos_WR",
    "cand_pos_TE",
    "cand_stack",
    "cand_bringback",
    "cand_same_team_roster_count",
    "cand_stack_depth_after_pick",
    "cand_bye_overlap_count",
    "qb_bye_overlap",
    "te_bye_overlap",
    "rb_to_comfort",
    "wr_to_comfort",
    "qb_to_comfort",
    "te_to_comfort",
    "roster_wr_rb_ratio",
    "board_qb_left",
    "board_rb_left",
    "board_wr_left",
    "board_te_left",
    "best_qb_adp_left",
    "best_rb_adp_left",
    "best_wr_adp_left",
    "best_te_adp_left",
    "picks_until_next_pick",
    "teams_until_next_pick",
    "opp_min_need_QB_before_next",
    "opp_min_need_RB_before_next",
    "opp_min_need_WR_before_next",
    "opp_min_need_TE_before_next",
    "opp_comfort_need_QB_before_next",
    "opp_comfort_need_RB_before_next",
    "opp_comfort_need_WR_before_next",
    "opp_comfort_need_TE_before_next",
    "last6_qb_taken",
    "last6_rb_taken",
    "last6_wr_taken",
    "last6_te_taken",
    "last12_qb_taken",
    "last12_rb_taken",
    "last12_wr_taken",
    "last12_te_taken",
    "cand_pos_min_need_before_next",
    "cand_pos_comfort_need_before_next",
    "cand_pos_last6_taken",
    "cand_pos_last12_taken",
    "cand_pos_next_best_adp_gap",
    "pick_window_adp_end",
    "cand_adp_minus_pick_window",
    "overall_players_adp_before_next",
    "cand_pos_players_adp_before_next",
    "cand_tier_players_left",
    "cand_tier_pos_players_left",
    "cand_handcuff_depth",
    "cand_playoff_matchup",
]


def contest_for_training(platform: str, draft_rounds: int) -> bb.ContestConfig:
    """Contest config for rollout scoring while preserving DK defaults."""
    base = bb.UNDERDOG_TOURNAMENT if platform == "underdog" else bb.TOURNAMENT
    return bb.ContestConfig(
        contest_type=base.contest_type,
        pod_size=base.pod_size,
        draft_rounds=int(draft_rounds),
        roster_size=int(draft_rounds),
        rounds=base.rounds,
        advance_per_round=base.advance_per_round,
        platform=platform,
    )


def candidate_depth_for_round(
    round_num: int,
    base: int = DEFAULT_CANDIDATES,
    max_rank: int | None = None,
) -> int:
    """Round-aware candidate depth.

    Early picks should stay close to market. Late picks need a wider set because
    small ADP differences are less meaningful once the board is mostly thin.
    """
    base_depth = max(1, int(base))
    depth = base_depth
    round_num = max(1, int(round_num))
    for start_round, scheduled_depth in CANDIDATE_DEPTH_SCHEDULE:
        if round_num >= start_round:
            scaled_depth = (scheduled_depth * base_depth + DEFAULT_CANDIDATES - 1) // DEFAULT_CANDIDATES
            depth = max(depth, scaled_depth)
    if max_rank is not None:
        depth = min(depth, max(1, int(max_rank)))
    return depth


def candidate_depth_for_pick(
    pick_no: int,
    total_teams: int = bb.TEAMS_PER_POD,
    base: int = DEFAULT_CANDIDATES,
    max_rank: int | None = None,
) -> int:
    teams = max(1, int(total_teams))
    round_num = ((max(1, int(pick_no)) - 1) // teams) + 1
    return candidate_depth_for_round(round_num, base, max_rank)


def infer_policy_max_candidate_rank(model_path: Path, features_path: Path) -> int | None:
    """Best-effort metadata lookup for the candidate rank a policy was trained on."""
    candidates = [
        model_path.with_name(model_path.name.replace("model_", "").replace(".joblib", "_meta.json")),
        features_path.with_name(features_path.name.replace("feature_cols", "meta")),
        MODELS_DIR / "dk_ev_policy_meta.json",
    ]
    for path in candidates:
        try:
            meta = json.loads(Path(path).read_text())
            depth = int(meta.get("max_candidate_rank") or meta.get("base_candidate_depth") or 0)
            if depth > 0:
                return depth
        except Exception:
            pass
    return None


@dataclass
class DraftState:
    """A draft room stopped immediately before our team's pick."""

    board: Board
    gp: int
    our_team: int
    rosters: list[list[dict]]
    counts: np.ndarray
    avail: np.ndarray
    team_caps: list[tuple[np.ndarray, np.ndarray]]
    sigma: float
    strategy: str
    pick_history: list[tuple[int, int]]  # (team_idx, board_idx), picks before gp
    n_teams: int = bb.TEAMS_PER_POD
    n_rounds: int = bb.DRAFT_ROUNDS
    roster_size: int = bb.ROSTER_SIZE

    @property
    def round(self) -> int:
        return self.gp // self.n_teams

    @property
    def slot_in_round(self) -> int:
        return self.gp % self.n_teams

    @property
    def current_team(self) -> int:
        rnd, slot = divmod(self.gp, self.n_teams)
        return slot if rnd % 2 == 0 else self.n_teams - 1 - slot

    @property
    def roster(self) -> list[dict]:
        return self.rosters[self.our_team]


def snake_team(global_pick: int, n_teams: int = bb.TEAMS_PER_POD) -> int:
    rnd, slot = divmod(global_pick, n_teams)
    return slot if rnd % 2 == 0 else n_teams - 1 - slot


def roster_counts(roster: list[dict]) -> dict[str, int]:
    return {p: sum(1 for r in roster if r.get("position") == p) for p in POS}


def sample_state(
    board: Board,
    rng: np.random.Generator,
    min_round: int = 2,
    max_round: int = 14,
    sigma: float = DEFAULT_SIGMA,
    strategy: str = "adp_noise",
    faller_prob: float = 0.0,
    faller_top_adp: int = 24,
    n_teams: int = bb.TEAMS_PER_POD,
    n_rounds: int = bb.DRAFT_ROUNDS,
    target_slot: int | None = None,
    target_team: int | None = None,
) -> DraftState:
    """Draft up to one of our future picks and return the stopped room state.

    Rounds are 0-indexed internally. Defaults focus training on rounds 3-15,
    where pick context matters most and there are enough remaining picks to
    repair roster construction after forced candidate choices.

    faller_prob: with this probability, HOLD an elite player (ADP <= faller_top_adp)
    out of the pre-draft so he "falls" to our pick — injects the rare but crucial
    "elite available far past ADP" states the model otherwise never trains on.
    """
    min_round = int(np.clip(min_round, 0, n_rounds - 1))
    max_round = int(np.clip(max_round, min_round, n_rounds - 1))

    if target_slot is not None:
        target_slot = int(target_slot)
        if not 0 <= target_slot < n_teams:
            raise ValueError(f"target_slot must be 0..{n_teams - 1}")
    if target_team is not None:
        target_team = int(target_team)
        if not 0 <= target_team < n_teams:
            raise ValueError(f"target_team must be 0..{n_teams - 1}")

    our_team = int(target_team) if target_team is not None else int(rng.integers(n_teams))
    pick_options = []
    for gp in range(min_round * n_teams, (max_round + 1) * n_teams):
        if target_slot is not None and gp % n_teams != target_slot:
            continue
        team = snake_team(gp, n_teams)
        if target_team is not None and team != target_team:
            continue
        if target_team is None or team == our_team:
            pick_options.append(gp)

    if not pick_options:
        raise ValueError("round range produced no picks for sampled team")
    target_gp = int(rng.choice(pick_options))
    our_team = snake_team(target_gp, n_teams)

    avail = np.ones(len(board), dtype=bool)
    counts = np.zeros((n_teams, len(POS)), dtype=int)
    rosters: list[list[dict]] = [[] for _ in range(n_teams)]
    team_caps = [_sample_team_caps(rng) for _ in range(n_teams)]
    pick_history: list[tuple[int, int]] = []

    # Faller injection: hold an elite player out so nobody drafts him, then
    # release him as available at our pick (he "fell" far below his ADP).
    faller_idx = None
    if faller_prob > 0.0 and rng.random() < faller_prob:
        elite = np.where(board.adp <= faller_top_adp)[0]
        # Only inject players whose ADP is well above our pick (a real fall).
        elite = elite[board.adp[elite] < target_gp - 6]
        if len(elite):
            faller_idx = int(rng.choice(elite))
            avail[faller_idx] = False        # held out of the pre-draft

    for gp in range(target_gp):
        team = snake_team(gp, n_teams)
        rnd = gp // n_teams
        picks_left = n_rounds - len(rosters[team])
        soft_cap, comfort = team_caps[team]
        idx = _choose(
            board, avail, counts[team], soft_cap, comfort,
            picks_left, rnd, n_rounds, sigma, strategy, rng,
        )
        _apply_pick(board, avail, counts, rosters, team, idx)
        pick_history.append((team, idx))

    if faller_idx is not None:
        avail[faller_idx] = True              # release the faller at our pick

    state = DraftState(
        board=board,
        gp=target_gp,
        our_team=our_team,
        rosters=rosters,
        counts=counts,
        avail=avail,
        team_caps=team_caps,
        sigma=sigma,
        strategy=strategy,
        pick_history=pick_history,
        n_teams=n_teams,
        n_rounds=n_rounds,
        roster_size=n_rounds,
    )
    if state.current_team != our_team:
        raise AssertionError("sampled state is not at our team's pick")
    return state


def candidate_indices(
    state: DraftState,
    n_candidates: int,
    max_candidate_rank: int | None = None,
) -> list[int]:
    """Top available, hard-legal candidates by ADP, widened in later rounds."""
    team_counts = state.counts[state.our_team]
    under_hard = team_counts[state.board.pos_code] < HARD_CAP[state.board.pos_code]
    idx = np.where(state.avail & under_hard)[0]
    if idx.size == 0:
        return []
    order = idx[np.argsort(state.board.adp[idx], kind="stable")]
    depth = candidate_depth_for_round(state.round + 1, n_candidates, max_candidate_rank)
    return [int(i) for i in order[:depth]]


def legal_candidate_set(state: DraftState, n: int, max_candidate_rank: int | None = None) -> list[int]:
    """Top-`n` available candidates by ADP that keep a LEGAL best-ball roster —
    the same legality the ADP opponents obey (hard/realism/soft caps + the
    1QB/2RB/3WR/1TE validity guarantee). Used by the self-play completion so the
    policy finishes its OWN roster instead of being bailed out by ADP bots."""
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


def make_completion_policy_agent(
    model,
    feature_cols: list[str],
    n_candidates: int,
    max_candidate_rank: int | None = None,
):
    """A draft agent that finishes our roster the way the SERVED policy ACTUALLY
    behaves: score the top-N available candidates (hard-cap legal only — NO
    validity guarantee) and take the best edge. Using `candidate_indices`, not
    `legal_candidate_set`, is deliberate: the completion must be allowed to run
    off the end with 0 QB just like the live model does, so that skipping a
    needed position carries its real cost in the rollout EV. A legality-forced
    completion would silently rescue the roster and teach nothing."""
    def agent(state: DraftState, rng: np.random.Generator) -> int | None:
        cands = candidate_indices(state, n_candidates, max_candidate_rank)
        if not cands:
            return None
        rows = [state_candidate_features(state, ci) for ci in cands]
        X = pd.DataFrame(rows).reindex(columns=feature_cols, fill_value=0).fillna(0)
        pred = model.predict(X)
        return int(cands[int(np.argmax(pred))])
    return agent


def complete_with_candidate(
    state: DraftState,
    candidate_idx: int,
    rng: np.random.Generator,
    our_agent=None,
) -> list[dict]:
    """Force one candidate now, then finish the room. Opponents always draft
    ADP/noise. OUR remaining picks are made by `our_agent` (self-play policy) when
    given, else by the ADP bot — the latter is the old behavior that bailed the
    roster out of its own positional mistakes."""
    board = state.board
    n_teams = state.n_teams
    n_rounds = state.n_rounds
    total_picks = n_teams * n_rounds

    avail = state.avail.copy()
    counts = state.counts.copy()
    rosters = [[dict(p) for p in team] for team in state.rosters]
    pick_history = list(state.pick_history)

    _apply_pick(board, avail, counts, rosters, state.our_team, candidate_idx)
    pick_history.append((state.our_team, int(candidate_idx)))

    for gp in range(state.gp + 1, total_picks):
        team = snake_team(gp, n_teams)
        rnd = gp // n_teams
        soft_cap, comfort = state.team_caps[team]
        idx = None
        if our_agent is not None and team == state.our_team:
            snap = DraftState(board, gp, state.our_team, rosters, counts, avail,
                              state.team_caps, state.sigma, state.strategy, pick_history,
                              state.n_teams, state.n_rounds, state.roster_size)
            idx = our_agent(snap, rng)
        if idx is None:
            picks_left = n_rounds - len(rosters[team])
            idx = _choose(
                board, avail, counts[team], soft_cap, comfort,
                picks_left, rnd, n_rounds, state.sigma, state.strategy, rng,
            )
        _apply_pick(board, avail, counts, rosters, team, idx)
        pick_history.append((team, int(idx)))

    return rosters[state.our_team]


def _apply_pick(
    board: Board,
    avail: np.ndarray,
    counts: np.ndarray,
    rosters: list[list[dict]],
    team: int,
    idx: int,
) -> None:
    if not avail[idx]:
        raise ValueError(f"player index {idx} is not available")
    avail[idx] = False
    counts[team][board.pos_code[idx]] += 1
    rosters[team].append(board.player_dict(idx))


def next_pick_window(state: DraftState) -> list[int]:
    """Global pick numbers after this pick and before our next pick."""
    total_picks = state.n_teams * state.n_rounds
    window = []
    for gp in range(state.gp + 1, total_picks):
        team = snake_team(gp, state.n_teams)
        if team == state.our_team:
            break
        window.append(gp)
    return window


def opponent_window_features(state: DraftState) -> dict:
    """Demand features from opponents who pick before our next turn."""
    window = next_pick_window(state)
    teams = [snake_team(gp, state.n_teams) for gp in window]
    unique_teams = sorted(set(teams))

    min_need = {pos: 0 for pos in POS}
    comfort_need = {pos: 0 for pos in POS}
    for team in unique_teams:
        counts = state.counts[team]
        soft_cap, comfort = state.team_caps[team]
        for pos_i, pos in enumerate(POS):
            if counts[pos_i] < bb.MIN_PER_POS.get(pos, 0):
                min_need[pos] += 1
            if counts[pos_i] < comfort[pos_i]:
                comfort_need[pos] += 1

    recent6 = {pos: 0 for pos in POS}
    recent12 = {pos: 0 for pos in POS}
    for _, idx in state.pick_history[-6:]:
        recent6[POS[int(state.board.pos_code[idx])]] += 1
    for _, idx in state.pick_history[-12:]:
        recent12[POS[int(state.board.pos_code[idx])]] += 1

    return {
        "picks_until_next_pick": len(window),
        "teams_until_next_pick": len(unique_teams),
        "opp_min_need_QB_before_next": min_need["QB"],
        "opp_min_need_RB_before_next": min_need["RB"],
        "opp_min_need_WR_before_next": min_need["WR"],
        "opp_min_need_TE_before_next": min_need["TE"],
        "opp_comfort_need_QB_before_next": comfort_need["QB"],
        "opp_comfort_need_RB_before_next": comfort_need["RB"],
        "opp_comfort_need_WR_before_next": comfort_need["WR"],
        "opp_comfort_need_TE_before_next": comfort_need["TE"],
        "last6_qb_taken": recent6["QB"],
        "last6_rb_taken": recent6["RB"],
        "last6_wr_taken": recent6["WR"],
        "last6_te_taken": recent6["TE"],
        "last12_qb_taken": recent12["QB"],
        "last12_rb_taken": recent12["RB"],
        "last12_wr_taken": recent12["WR"],
        "last12_te_taken": recent12["TE"],
    }


def next_best_adp_gap_same_pos(state: DraftState, candidate_idx: int) -> float:
    pos_i = int(state.board.pos_code[candidate_idx])
    same = np.where(state.avail & (state.board.pos_code == pos_i))[0]
    order = np.lexsort((same, state.board.adp[same]))
    same = [int(i) for i in same[order]]
    try:
        pos = same.index(int(candidate_idx))
    except ValueError:
        return 9999.0
    if pos + 1 >= len(same):
        return 9999.0
    return float(state.board.adp[same[pos + 1]] - state.board.adp[candidate_idx])


def adp_survival_features(state: DraftState, candidate_idx: int) -> dict:
    """Crude market-survival signals for what may be gone by our next pick."""
    cand_pos_i = int(state.board.pos_code[candidate_idx])
    cand_tier = int(state.board.market_tier[candidate_idx]) if state.board.market_tier is not None else -1
    window_end = float(state.gp + 1 + len(next_pick_window(state)))
    avail_idx = np.where(state.avail)[0]
    before_next = avail_idx[state.board.adp[avail_idx] <= window_end]
    same_before = before_next[state.board.pos_code[before_next] == cand_pos_i]
    tier_left = 0
    tier_pos_left = 0
    if state.board.market_tier is not None and cand_tier >= 0:
        tier_left = int(np.sum(state.avail & (state.board.market_tier == cand_tier)))
        tier_pos_left = int(np.sum(
            state.avail
            & (state.board.market_tier == cand_tier)
            & (state.board.pos_code == cand_pos_i)
        ))
    return {
        "pick_window_adp_end": window_end,
        "cand_adp_minus_pick_window": float(state.board.adp[candidate_idx] - window_end),
        "overall_players_adp_before_next": int(len(before_next)),
        "cand_pos_players_adp_before_next": int(len(same_before)),
        "cand_tier_players_left": tier_left,
        "cand_tier_pos_players_left": tier_pos_left,
    }


def state_candidate_features(state: DraftState, candidate_idx: int) -> dict:
    board = state.board
    roster = state.roster
    counts = roster_counts(roster)
    cand = board.player_dict(candidate_idx)
    cand_pos = cand["position"]
    cand_team = cand.get("team") or ""
    cand_bye = int(cand.get("bye_week") or 0)
    cand_tier = int(cand.get("market_tier", -1))

    qbs = {p.get("team") for p in roster if p.get("position") == "QB" and p.get("team")}
    pass_catcher_teams = {
        p.get("team") for p in roster
        if p.get("position") in ("WR", "TE") and p.get("team")
    }
    same_team_roster = [p for p in roster if p.get("team") == cand_team and cand_team]
    # Handcuff signal: same-team same-position players already rostered. The outcome
    # sim now transfers an injured starter's workload to his backup, so this is the
    # position-aware feature that lets the policy condition on "this is my starter's
    # backup" (vs the generic cand_same_team_roster_count).
    same_team_same_pos = [
        p for p in roster
        if cand_team and p.get("team") == cand_team and p.get("position") == cand_pos
    ]
    same_team_qbs = [p for p in roster if p.get("position") == "QB" and p.get("team") == cand_team]
    same_team_catchers = [
        p for p in roster
        if p.get("position") in ("WR", "TE") and p.get("team") == cand_team
    ]
    roster_byes = [int(p.get("bye_week") or 0) for p in roster]
    bye_overlap = sum(1 for b in roster_byes if cand_bye and b == cand_bye)
    pos_bye_overlap = sum(
        1 for p in roster
        if cand_bye and p.get("position") == cand_pos and int(p.get("bye_week") or 0) == cand_bye
    )

    avail_idx = np.where(state.avail)[0]
    left_by_pos = {}
    best_by_pos = {}
    for pos_i, pos in enumerate(POS):
        pi = avail_idx[board.pos_code[avail_idx] == pos_i]
        left_by_pos[pos] = int(len(pi))
        best_by_pos[pos] = float(np.min(board.adp[pi])) if len(pi) else 9999.0

    needs = {
        pos: max(0, bb.MIN_PER_POS.get(pos, 0) - counts.get(pos, 0))
        for pos in POS
    }
    opp = opponent_window_features(state)
    cand_need_key = f"opp_min_need_{cand_pos}_before_next"
    cand_comfort_key = f"opp_comfort_need_{cand_pos}_before_next"
    survival = adp_survival_features(state, candidate_idx)
    soft_cap, comfort = state.team_caps[state.our_team]
    to_comfort = {
        pos: max(0, int(comfort[pos_i]) - counts.get(pos, 0))
        for pos_i, pos in enumerate(POS)
    }
    stack_depth_after = 0
    if cand_pos in ("WR", "TE"):
        stack_depth_after = len(same_team_qbs)
    elif cand_pos == "QB":
        stack_depth_after = len(same_team_catchers)

    row = {
        "pick_no": state.gp + 1,
        "round": state.round + 1,
        "slot_in_round": state.slot_in_round + 1,
        "picks_made_by_team": len(roster),
        "picks_left": state.n_rounds - len(roster) - 1,
        "n_QB": counts["QB"],
        "n_RB": counts["RB"],
        "n_WR": counts["WR"],
        "n_TE": counts["TE"],
        "need_QB": needs["QB"],
        "need_RB": needs["RB"],
        "need_WR": needs["WR"],
        "need_TE": needs["TE"],
        "cand_adp": float(board.adp[candidate_idx]),
        # How far the player has fallen past his ADP at THIS pick (negative = a
        # faller available below his market price -> a value signal).
        "cand_adp_minus_pick": float(board.adp[candidate_idx] - (state.gp + 1)),
        "cand_board_rank": int(candidate_idx + 1),
        "cand_has_real_adp": int(bool(board.has_real_adp[candidate_idx])),
        "cand_bye_week": cand_bye,
        "cand_market_tier": cand_tier,
        "cand_pos_QB": int(cand_pos == "QB"),
        "cand_pos_RB": int(cand_pos == "RB"),
        "cand_pos_WR": int(cand_pos == "WR"),
        "cand_pos_TE": int(cand_pos == "TE"),
        "cand_stack": int(cand_pos in ("WR", "TE") and cand_team in qbs),
        "cand_bringback": int(cand_pos == "QB" and cand_team in pass_catcher_teams),
        "cand_same_team_roster_count": len(same_team_roster),
        "cand_handcuff_depth": len(same_team_same_pos),
        # Soft (>1) / tough (<1) playoff-week (15/16/17) matchup for this candidate's
        # team+position — the single-week bracket rounds where ceiling decides advancement.
        "cand_playoff_matchup": playoff_matchup_rating(cand_team, cand_pos, _MATCHUP_RATINGS, _MATCHUP_OPP),
        "cand_stack_depth_after_pick": stack_depth_after,
        "cand_bye_overlap_count": bye_overlap,
        "qb_bye_overlap": int(cand_pos == "QB" and pos_bye_overlap > 0),
        "te_bye_overlap": int(cand_pos == "TE" and pos_bye_overlap > 0),
        "rb_to_comfort": to_comfort["RB"],
        "wr_to_comfort": to_comfort["WR"],
        "qb_to_comfort": to_comfort["QB"],
        "te_to_comfort": to_comfort["TE"],
        "roster_wr_rb_ratio": counts["WR"] / max(counts["RB"], 1),
        "board_qb_left": left_by_pos["QB"],
        "board_rb_left": left_by_pos["RB"],
        "board_wr_left": left_by_pos["WR"],
        "board_te_left": left_by_pos["TE"],
        "best_qb_adp_left": best_by_pos["QB"],
        "best_rb_adp_left": best_by_pos["RB"],
        "best_wr_adp_left": best_by_pos["WR"],
        "best_te_adp_left": best_by_pos["TE"],
        **opp,
        "cand_pos_min_need_before_next": opp[cand_need_key],
        "cand_pos_comfort_need_before_next": opp[cand_comfort_key],
        "cand_pos_last6_taken": opp[f"last6_{cand_pos.lower()}_taken"],
        "cand_pos_last12_taken": opp[f"last12_{cand_pos.lower()}_taken"],
        "cand_pos_next_best_adp_gap": next_best_adp_gap_same_pos(state, candidate_idx),
        **survival,
        "candidate_player_id": cand["player_id"],
        "candidate_name": cand["name"],
        "candidate_pos": cand_pos,
        "candidate_team": cand_team,
    }
    return row


def evaluate_candidate(
    state: DraftState,
    candidate_idx: int,
    field,
    model,
    rollouts: int,
    eval_seasons: int,
    rng: np.random.Generator,
    rollout_seeds: list[tuple[int, int]] | None = None,
    our_agent=None,
    conditional_field: bool = False,
    contest: bb.ContestConfig = bb.DEFAULT_CONTEST,
) -> dict:
    evs, finals, means = [], [], []
    valid = 0

    if rollout_seeds is None:
        seed_arr = rng.integers(0, np.iinfo(np.uint32).max, size=(rollouts, 2), dtype=np.uint32)
        rollout_seeds = [(int(a), int(b)) for a, b in seed_arr]

    for complete_seed, season_seed in rollout_seeds[:rollouts]:
        complete_rng = np.random.default_rng(complete_seed)
        season_rng = np.random.default_rng(season_seed)
        roster = complete_with_candidate(state, candidate_idx, complete_rng, our_agent)
        # Score full-size rosters even if ILLEGAL (e.g. 0 QB): an invalid roster
        # scores 0 at the position it can't fill every week, which is exactly the
        # disaster the policy must learn to avoid. Skipping it (the old behavior)
        # hid that consequence and is why the policy never valued taking a QB.
        if len(roster) != state.roster_size:
            continue
        valid += 1
        sim = simulate_roster(roster, n_seasons=eval_seasons, rng=season_rng,
                              model=model, contest=contest)
        ev = evaluate_roster(sim, field, contest=contest,
                             conditional_field=conditional_field)
        evs.append(ev.prize_ev)
        finals.append(ev.finals_rate)
        means.append(ev.mean_points)

    if not evs:
        return {
            "valid_rollouts": valid,
            "ev_mean": np.nan,
            "ev_std": np.nan,
            "finals_mean": np.nan,
            "mean_points": np.nan,
        }

    return {
        "valid_rollouts": valid,
        "ev_mean": float(np.mean(evs)),
        "ev_std": float(np.std(evs)),
        "finals_mean": float(np.mean(finals)),
        "mean_points": float(np.mean(means)),
    }


def build_policy_rows(args) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    contest = contest_for_training(args.platform, args.draft_rounds)
    if args.board_source == "bbm_v":
        if getattr(args, "field_bias", False):
            print("[warn] --field-bias is only implemented for the DK board; ignoring for BBM V")
        board = load_bbm_v_board()
        print("Board source: historical Underdog BBM V projection_adp median")
    elif args.board_source == "underdog":
        if getattr(args, "field_bias", False):
            print("[warn] --field-bias is only implemented for the DK board; ignoring for Underdog ADP")
        board = load_adp_board(adp_json=MODELS_DIR / "underdog_adp.json")
        print("Board source: current Underdog ADP")
    else:
        board = load_market_board(field_bias=getattr(args, "field_bias", False))
        print("Board source: DraftKings ADP")
    if getattr(args, "field_bias", False):
        print("Asymmetric opponent field bias ON (reach hype / fade vets)")
    if getattr(args, "conditional_field", False):
        print("Conditional (survivorship) bracket field ON")

    print("Building outcome model...")
    model = load_market_model(platform=args.platform)
    if getattr(args, "no_handcuff", False):
        model.handcuff_model = False
        print("Handcuff workload-inheritance OFF (control arm)")
    else:
        print("Handcuff workload-inheritance ON")
    if getattr(args, "no_matchup", False):
        model.matchup_model = False
        print("Playoff-matchup modulation OFF (control arm)")
    else:
        print("Playoff-matchup modulation ON")

    # Self-play completion: finish OUR roster with the served policy instead of
    # ADP bots, so deferring a needed position actually costs the rollout EV.
    our_agent = None
    if getattr(args, "completion", "adp") == "self":
        import joblib
        cpath = Path(args.completion_model)
        fpath = Path(args.completion_features)
        if cpath.exists() and fpath.exists():
            cmodel = joblib.load(cpath)
            cfeats = json.loads(fpath.read_text())
            completion_max = args.completion_max_candidates
            if completion_max is None:
                completion_max = infer_policy_max_candidate_rank(cpath, fpath)
            our_agent = make_completion_policy_agent(
                cmodel,
                cfeats,
                args.candidates,
                completion_max,
            )
            suffix = f", max_rank={completion_max}" if completion_max else ""
            print(f"Self-play completion ON (policy={cpath.name}{suffix})")
        else:
            print(f"[warn] completion=self but model not found ({cpath}); "
                  f"falling back to ADP completion")

    print("Building opponent field...")
    field_rosters = draft_field(args.field_rooms, rng, board=board,
                                sigma=args.sigma, n_rounds=args.draft_rounds)
    field = build_field(
        field_rosters,
        model=model,
        n_seasons=args.field_seasons,
        rng=rng,
        contest=contest,
    )

    rows = []
    started = time.time()
    sampling_mode = getattr(args, "state_sampling", "random")
    if sampling_mode == "balanced_slots":
        print("State sampling: balanced by slot_in_round")
    else:
        print("State sampling: random")

    for state_id in range(args.states):
        target_slot = None
        if sampling_mode == "balanced_slots":
            target_slot = state_id % bb.TEAMS_PER_POD
        state = sample_state(
            board,
            rng,
            min_round=args.min_round - 1,
            max_round=args.max_round - 1,
            sigma=args.sigma,
            faller_prob=getattr(args, "faller_prob", 0.0),
            n_rounds=args.draft_rounds,
            target_slot=target_slot,
        )
        cands = candidate_indices(state, args.candidates, args.max_candidate_rank)
        print(
            f"State {state_id + 1}/{args.states}: pick {state.gp + 1}, "
            f"round {state.round + 1}, candidates={len(cands)}"
        )
        rollout_seed_arr = rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=(args.rollouts, 2),
            dtype=np.uint32,
        )
        rollout_seeds = [(int(a), int(b)) for a, b in rollout_seed_arr]

        for rank, cand_idx in enumerate(cands, start=1):
            row = state_candidate_features(state, cand_idx)
            row["state_id"] = state_id
            row["candidate_rank_in_state"] = rank
            row.update(
                evaluate_candidate(
                    state,
                    cand_idx,
                    field,
                    model,
                    args.rollouts,
                    args.eval_seasons,
                    rng,
                    rollout_seeds=rollout_seeds,
                    our_agent=our_agent,
                    conditional_field=getattr(args, "conditional_field", False),
                    contest=contest,
                )
            )
            rows.append(row)
            print(
                f"  {rank:>2}. {row['candidate_name']:<24} "
                f"{row['candidate_pos']:<2} EV={row['ev_mean']:.6f} "
                f"valid={row['valid_rollouts']}/{args.rollouts}"
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["run_seed"] = args.seed
        df["field_rooms"] = args.field_rooms
        df["field_seasons"] = args.field_seasons
        df["eval_seasons"] = args.eval_seasons
        df["rollouts"] = args.rollouts
        df["crn_enabled"] = True
        df["a_game"] = A_GAME
        df["completion"] = getattr(args, "completion", "adp")
        df["faller_prob"] = getattr(args, "faller_prob", 0.0)
        df["field_bias"] = bool(getattr(args, "field_bias", False))
        df["conditional_field"] = bool(getattr(args, "conditional_field", False))
        df["platform"] = args.platform
        df["draft_rounds"] = int(args.draft_rounds)
        df["board_source"] = args.board_source
        df["base_candidates"] = int(args.candidates)
        if args.max_candidate_rank is not None:
            df["max_candidate_rank"] = int(args.max_candidate_rank)
        df["state_sampling"] = sampling_mode

    print(f"Built {len(df):,} candidate rows in {(time.time() - started) / 60:.1f}m")
    return df


def fit_dk_ev_policy_model(input_csv: Path, model_out: Path, features_out: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "training" / "fit_dk_ev_policy.py"),
        "--input",
        str(input_csv),
        "--model-out",
        str(model_out),
        "--features-out",
        str(features_out),
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"DK EV policy fit failed with exit code {result.returncode}")



def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=int, default=8, help="draft states to sample")
    p.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATES,
        help="base candidates per state; later rounds expand automatically",
    )
    p.add_argument("--rollouts", type=int, default=3, help="draft completions per candidate")
    p.add_argument("--eval-seasons", type=int, default=600, help="season sims per completed roster")
    p.add_argument("--field-rooms", type=int, default=6, help="opponent draft rooms for field")
    p.add_argument("--field-seasons", type=int, default=120, help="season sims per field roster")
    p.add_argument("--min-round", type=int, default=3, help="first sampled round, 1-indexed")
    p.add_argument("--max-round", type=int, default=15, help="last sampled round, 1-indexed")
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="ADP draft noise")
    p.add_argument("--state-sampling", choices=["random", "balanced_slots"], default="random",
                   help="draft-state sampler; balanced_slots cycles slot_in_round 1-12")
    p.add_argument("--platform", choices=["draftkings", "underdog"], default="draftkings",
                   help="scoring platform for the outcome model")
    p.add_argument("--draft-rounds", type=int, default=bb.DRAFT_ROUNDS,
                   help="number of draft rounds/roster spots to simulate")
    p.add_argument("--board-source", choices=["dk", "bbm_v", "underdog"], default="dk",
                   help="market board source: current DK ADP, current Underdog ADP, or historical BBM V projection_adp")
    p.add_argument("--completion", choices=["adp", "self"], default="adp",
                   help="how OUR roster is finished in each rollout: 'adp' (legacy "
                        "ADP-bot crutch) or 'self' (self-play with the policy model)")
    p.add_argument("--completion-model", type=Path,
                   default=MODELS_DIR / "model_dk_ev_policy.joblib",
                   help="policy used for --completion self")
    p.add_argument("--completion-features", type=Path,
                   default=MODELS_DIR / "dk_ev_policy_feature_cols.json")
    p.add_argument("--completion-max-candidates", type=int, default=None,
                   help="max candidate rank for --completion self; default reads policy meta")
    p.add_argument("--max-candidate-rank", type=int, default=None,
                   help="cap generated candidate rows to a fixed trained rank")
    p.add_argument("--faller-prob", type=float, default=0.0,
                   help="fraction of states that inject an elite ADP faller candidate")
    p.add_argument("--no-handcuff", action="store_true",
                   help="disable the sim's handcuff workload-inheritance (control arm for A/B)")
    p.add_argument("--no-matchup", action="store_true",
                   help="disable the sim's playoff-matchup modulation (control arm for A/B)")
    p.add_argument("--field-bias", action="store_true",
                   help="opponent rooms reach for hyped rookies / recent producers and "
                        "let vets fall (asymmetric field bias; default symmetric noise)")
    p.add_argument("--conditional-field", action="store_true",
                   help="score later bracket rounds against survivors only (survivorship "
                        "field; rewards ceiling). Default unconditional.")
    p.add_argument("--seed", type=int, default=20260611)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES_OUT)
    p.add_argument("--append", action="store_true", help="append to an existing rollout CSV")
    p.add_argument("--fit-model", action="store_true",
                   help="fit the server DK EV policy from the rollout labels")
    p.add_argument("--no-model", action="store_true",
                   help=argparse.SUPPRESS)  # backward-compatible alias for older smoke commands
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.states < 0:
        raise SystemExit("--states must be >= 0")

    if args.states == 0:
        if not args.out.exists():
            raise SystemExit(f"--states 0 requires an existing rollout CSV: {args.out}")
        df = pd.read_csv(args.out)
        print(f"Loaded existing rollout labels -> {args.out} ({len(df):,} rows)")
    else:
        new_df = build_policy_rows(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.append and args.out.exists():
            old_df = pd.read_csv(args.out)
            df = pd.concat([old_df, new_df], ignore_index=True)
            print(f"Appended {len(new_df):,} new rows to {len(old_df):,} existing rows.")
        else:
            df = new_df
        df.to_csv(args.out, index=False)
        print(f"Saved rollout labels -> {args.out}")

    if args.fit_model and not args.no_model:
        fit_dk_ev_policy_model(args.out, args.model_out, args.features_out)
    else:
        print("Skipped policy fit. Use --fit-model after collecting enough EV labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
