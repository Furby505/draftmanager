"""
DraftKings Best Ball — canonical contest rules (single source of truth).

Everything downstream (simulator, opponent field, bracket scorer, draft policy)
must read roster/lineup/tournament structure from here so the rules live in
exactly one place. Scoring point values live in scoring.py; this module imports
them and adds the roster, weekly-lineup, and tournament-bracket structure that
the points-only scoring module does not cover.

Verified against the official DraftKings "NFL Best Ball (Season Long)" rules
(2026 season).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Roster / draft ──────────────────────────────────────────────────────────
ROSTER_SIZE   = 20      # DK: 20-round snake draft, 20 players
DRAFT_ROUNDS  = 20
UNDERDOG_ROSTER_SIZE  = 18
UNDERDOG_DRAFT_ROUNDS = 18
TEAMS_PER_POD = 12      # standard pod size (advancement is relative to this field)

# ── Weekly lineup (8 starters / 12 bench) ───────────────────────────────────
# Highest-scoring eligible players auto-start at each slot every NFL Week.
STARTERS      = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}   # fixed-position slots
FLEX_COUNT    = 1
FLEX_ELIGIBLE = ("RB", "WR", "TE")                     # NOTE: no QB in FLEX
N_STARTERS    = sum(STARTERS.values()) + FLEX_COUNT    # = 8
N_BENCH       = ROSTER_SIZE - N_STARTERS               # = 12

# ── Position caps / validity ────────────────────────────────────────────────
# A roster may hold up to 5 QB and 5 TE. To be a *valid* (scoring) entry it
# must be able to field 1 QB / 2 RB / 3 WR / 1 TE every week.
MAX_PER_POS = {"QB": 5, "TE": 5}                       # RB/WR effectively uncapped
MIN_PER_POS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}     # else entry is invalid (0 pts)

# Auto-draft guardrails (only apply before the user edits rankings/queue/picks):
AUTODRAFT_CAPS = {"QB": 3, "RB": 6, "WR": 8, "TE": 3}


def is_valid_roster(positions) -> bool:
    """True if the roster can field a legal lineup (and respects QB/TE caps)."""
    counts = {p: 0 for p in ("QB", "RB", "WR", "TE")}
    for p in positions:
        if p in counts:
            counts[p] += 1
    if any(counts.get(p, 0) > cap for p, cap in MAX_PER_POS.items()):
        return False
    return all(counts[p] >= n for p, n in MIN_PER_POS.items())


def optimal_lineup_score(week_points: dict[str, float],
                         positions: dict[str, str]) -> float:
    """
    Best-ball optimal lineup score for a single NFL Week.

    week_points: {player_id: fantasy_points_this_week}
    positions:   {player_id: "QB"|"RB"|"WR"|"TE"}

    Fills each fixed slot with the top scorers at that position, then fills the
    single FLEX with the best remaining RB/WR/TE leftover. Greedy is optimal
    here because slots are position-exclusive and FLEX takes one leftover.
    """
    by_pos: dict[str, list[float]] = {"QB": [], "RB": [], "WR": [], "TE": []}
    for pid, pos in positions.items():
        if pos in by_pos:
            by_pos[pos].append(week_points.get(pid, 0.0))
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    total = 0.0
    flex_pool: list[float] = []
    for pos, n_start in STARTERS.items():
        scores = by_pos[pos]
        total += sum(scores[:n_start])
        if pos in FLEX_ELIGIBLE:                       # QB extras never reach FLEX
            flex_pool.extend(scores[n_start:])

    if FLEX_COUNT and flex_pool:
        flex_pool.sort(reverse=True)
        total += sum(flex_pool[:FLEX_COUNT])
    return total


# ── Tournament bracket ──────────────────────────────────────────────────────
# Season-long Best Ball tournaments run 4 rounds. R1 is a 14-week accumulation;
# R2/R3/R4 are SINGLE NFL Weeks. Advancement each round is relative to your pod.
@dataclass(frozen=True)
class Round:
    number: int
    weeks: tuple[int, ...]
    label: str

    @property
    def is_single_week(self) -> bool:
        return len(self.weeks) == 1


TOURNAMENT_ROUNDS: tuple[Round, ...] = (
    Round(1, tuple(range(1, 15)), "R1: Weeks 1-14 (cumulative)"),
    Round(2, (15,),               "R2: Week 15 (single week)"),
    Round(3, (16,),               "R3: Week 16 (single week)"),
    Round(4, (17,),               "R4: Week 17 (final, single week)"),
)

# Tiebreaker order for advancement (per official rules), for reference in the
# bracket scorer: highest single week in the round, then next-highest week, ...,
# then highest single player, ... then overall cumulative score, then draft order.
TIEBREAKERS = ("best_week", "next_best_weeks", "best_player",
               "next_best_players", "cumulative_total", "drafted_first")


# ── Contest configuration (tournament-first; Sit & Go scaffolding) ──────────
@dataclass(frozen=True)
class ContestConfig:
    contest_type: str                       # "tournament" | "sit_n_go"
    pod_size: int = TEAMS_PER_POD
    draft_rounds: int = DRAFT_ROUNDS
    roster_size: int = ROSTER_SIZE
    rounds: tuple[Round, ...] = TOURNAMENT_ROUNDS
    # Fraction of each pod that advances per round (tournament only). Verified DK
    # Best Ball structure (2026-06): Round 1 (Weeks 1-14 cumulative) advances the
    # TOP 2 of each 12-team pod; Round 2 (Wk15) and Round 3 (Wk16) advance the top
    # 1; Round 4 (Wk17) is the placement-paid championship. Configurable for the
    # field model.
    advance_per_round: tuple[float, ...] = (2 / 12, 1 / 12, 1 / 12, 0.0)
    platform: str = "draftkings"

    @property
    def is_tournament(self) -> bool:
        return self.contest_type == "tournament"


# Sit & Go: no bracket — cumulative points over the scoring window, flatter
# payout, smaller pod. Objective shifts toward expected finish/EV (less
# variance-seeking). Engine reads this flag to switch objectives.
SIT_N_GO = ContestConfig(
    contest_type="sit_n_go",
    rounds=(Round(1, tuple(range(1, 18)), "Cumulative Weeks 1-17"),),
    advance_per_round=(0.0,),
)

TOURNAMENT = ContestConfig(contest_type="tournament")

# Underdog Best Ball Mania style validation config: same 12-team pod and
# 1QB/2RB/3WR/1TE/1FLEX lineup, but half-PPR scoring and an 18-round roster.
UNDERDOG_TOURNAMENT = ContestConfig(
    contest_type="tournament",
    draft_rounds=UNDERDOG_DRAFT_ROUNDS,
    roster_size=UNDERDOG_ROSTER_SIZE,
    platform="underdog",
)

DEFAULT_CONTEST = TOURNAMENT
