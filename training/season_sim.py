"""
Step 2 — Monte Carlo season simulator.

For a 20-man roster, simulate many seasons of CORRELATED weekly outcomes, auto-
start the best-ball optimal lineup each week, and reduce to the per-round scores
the bracket needs:  R1 = sum(weeks 1-14), R2 = wk15, R3 = wk16, R4 = wk17, plus
total season points as a diagnostic.

The correlation comes entirely from outcome_model (same-team passing-game
factor). This module's only job is: weeks -> optimal weekly lineup -> rounds.
Weeks are independent across the season (documented v1 simplification).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import best_ball as bb
from outcome_model import CorrelatedOutcomeModel, RosterContext, load_default_model


def optimal_lineup_scores(scores: np.ndarray, positions) -> np.ndarray:
    """
    Vectorized best-ball lineup score per column.

    scores:    [n_players, W] weekly DK scores
    positions: length-n list/array of "QB"/"RB"/"WR"/"TE"
    returns:   [W] optimal weekly lineup totals (1QB/2RB/3WR/1TE/1FLEX)

    Same greedy logic as best_ball.optimal_lineup_score: fill fixed slots with
    the top scorers per position, then the single FLEX takes the best RB/WR/TE
    leftover. QB extras never reach FLEX.
    """
    pos = np.asarray(positions)
    W = scores.shape[1]
    total = np.zeros(W)
    flex_pool = []
    for p, n_start in bb.STARTERS.items():
        rows = scores[pos == p]
        if rows.shape[0] == 0:
            continue
        s = -np.sort(-rows, axis=0)          # descending per column
        total += s[:n_start].sum(axis=0)
        if p in bb.FLEX_ELIGIBLE and s.shape[0] > n_start:
            flex_pool.append(s[n_start:])
    if bb.FLEX_COUNT and flex_pool:
        pool = np.concatenate(flex_pool, axis=0)
        total += pool.max(axis=0)            # best single leftover -> FLEX
    return total


@dataclass
class SeasonSimResult:
    round_scores: np.ndarray   # [S, n_rounds] per-round totals
    total_points: np.ndarray   # [S] full-season optimal-lineup points (diagnostic)
    round_labels: list[str]
    weekly_lineup: np.ndarray  # [S, n_weeks] per-week optimal lineup (diagnostic)


def _round_week_indices(contest: bb.ContestConfig) -> list[np.ndarray]:
    """Map each tournament round to 0-based week indices (wk N -> idx N-1)."""
    return [np.array([w - 1 for w in r.weeks], dtype=int) for r in contest.rounds]


def simulate_ctx(model: CorrelatedOutcomeModel, ctx: RosterContext,
                 n_seasons: int, rng: np.random.Generator,
                 contest: bb.ContestConfig = bb.DEFAULT_CONTEST,
                 availability: bool | None = None,
                 handcuff: bool | None = None,
                 matchup: bool | None = None) -> SeasonSimResult:
    """Simulate from a prepared RosterContext (lets callers override marginals).

    availability: None -> use model.availability_model; True/False -> force the
    A/B (iron-man) toggle for this run.
    handcuff: None -> use model.handcuff_model; True/False -> force the A/B toggle
    for workload inheritance (only fires when availability is on).
    matchup: None -> use model.matchup_model; True/False -> force the A/B toggle for
    per-week opponent defense-vs-position modulation.
    """
    week_idx = _round_week_indices(contest)
    n_weeks = max(int(w.max()) for w in week_idx) + 1   # cover the latest week

    # One big correlated draw with week-aware game structure: [n, S, Wk].
    scores = model.sample_scores(ctx, n_seasons, n_weeks, rng, matchup=matchup)

    # Zero each player's BYE week: he scores 0 that week (he doesn't play). This
    # is what makes overlapping byes costly — if too many same-position players
    # share a bye, the optimal lineup can't be filled that week and the team
    # loses points, so the EV correctly penalizes bye stacking.
    byes = getattr(ctx, "bye_weeks", None)
    if byes is not None:
        on = np.where((byes >= 1) & (byes <= n_weeks))[0]
        if len(on):
            scores[on, :, (byes[on] - 1).astype(int)] = 0.0

    use_avail = model.availability_model if availability is None else availability
    if use_avail:
        active = model.sample_availability(ctx, n_seasons, n_weeks, rng)  # [n,S,Wk]
        use_hc = model.handcuff_model if handcuff is None else handcuff
        if use_hc:
            # Backups inherit a vacated starter's workload BEFORE the starter is zeroed.
            scores = model.apply_workload_transfer(scores, active, ctx)
        scores = scores * active                                   # zero inactive weeks

    flat = scores.reshape(len(ctx.player_ids), n_seasons * n_weeks)
    weekly = optimal_lineup_scores(flat, ctx.positions)            # [S*Wk]
    weekly = weekly.reshape(n_seasons, n_weeks)                    # [S, Wk]

    round_scores = np.stack(
        [weekly[:, idx].sum(axis=1) for idx in week_idx], axis=1)  # [S, R]
    total_points = weekly.sum(axis=1)                             # [S]
    labels = [r.label for r in contest.rounds]
    return SeasonSimResult(round_scores, total_points, labels, weekly)


def simulate_roster(roster: list[dict], n_seasons: int = 5000,
                    rng: np.random.Generator | None = None,
                    model: CorrelatedOutcomeModel | None = None,
                    contest: bb.ContestConfig = bb.DEFAULT_CONTEST,
                    availability: bool | None = None,
                    handcuff: bool | None = None,
                    matchup: bool | None = None) -> SeasonSimResult:
    """
    Simulate `n_seasons` correlated seasons for `roster`.

    roster: list of {name|player_id, position, team} dicts (any length; a real
            entry is 20 players, but the sim works for partial rosters too).
    """
    rng = rng or np.random.default_rng()
    model = model or load_default_model()
    ctx = model.prepare_roster(roster)
    return simulate_ctx(model, ctx, n_seasons, rng, contest, availability, handcuff, matchup)
