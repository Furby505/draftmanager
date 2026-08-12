"""
Step 4 (scaffolded early) — bracket / prize-EV scorer.

Turns a roster's simulated per-round score distribution into EXPECTED PRIZE
EQUITY, the optimizer's objective, plus separate diagnostics:
  - prize_ev     : expected prize equity under the payout table  (PRIMARY)
  - p_advance_r1 : probability of surviving Round 1
  - finals_rate  : probability of reaching the final round
  - win_rate     : probability of winning the final
  - mean_points  : mean season points (pure diagnostic, not the objective)

Why prize-EV and not P(advance): a top-heavy payout is convex, so it pays
specifically for the extreme right tail. Optimizing P(advance) alone would
under-value the nuke teams that actually win top-heavy tournaments.

Field model (v1 placeholder, replaceable in Step 3):
  The opponent field is supplied as per-round score distributions (a `Field`).
  Advancement uses a binomial pod model: in a pod of `pod` with `advance`
  spots, you advance a round if at most (advance-1) of your (pod-1) random
  opponents beat your score that round. Final placement pays by your rank
  within the final pod.

Survivorship (`conditional_field=True`): the real R2/R3/final field are not the
average entry — they are the rosters that already advanced, i.e. the strong
ones. With the conditional model on, later rounds are scored against only the
field entries that cleared the earlier-round advance thresholds, so the bar
rises each round the way it really does. This removes the inflation of
absolute finals/win rates AND, more importantly, restores the premium on the
extreme right tail (the high-ceiling "nuke" rosters that actually win top-heavy
tournaments). Default OFF preserves the legacy unconditional behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

import best_ball as bb
import payouts as pay
from season_sim import SeasonSimResult


# ── Field model ──────────────────────────────────────────────────────────────
# Don't tighten the survivor pool below this many entries: a single-week round
# advancing ~1/12 would otherwise starve a small field down to a handful of
# noisy samples. Below the floor we freeze the pool (keep the last stable one),
# so big production fields compound round-by-round while smoke fields degrade to
# a stable "R1-gate" survivor set instead of breaking.
MIN_SURVIVORS = 64


class Field:
    """Opponent per-round score distributions as empirical CDFs."""

    def __init__(self, round_scores: np.ndarray):
        # round_scores: [F, n_rounds]
        self._raw = np.asarray(round_scores, dtype=float)
        self._sorted = [np.sort(self._raw[:, r])
                        for r in range(self._raw.shape[1])]
        self.n_rounds = self._raw.shape[1]

    @classmethod
    def from_sim(cls, result: SeasonSimResult) -> "Field":
        return cls(result.round_scores)

    @staticmethod
    def _exceed(sorted_scores: np.ndarray, x: np.ndarray) -> np.ndarray:
        """P(a field entry's score > x) from a pre-sorted score array."""
        # fraction strictly greater = 1 - P(field <= x)
        le = np.searchsorted(sorted_scores, x, side="right") / len(sorted_scores)
        return np.clip(1.0 - le, 0.0, 1.0)

    def prob_opponent_exceeds(self, r: int, x: np.ndarray) -> np.ndarray:
        """P(a random field entry's round-r score > x), elementwise over x."""
        return self._exceed(self._sorted[r], x)

    def conditional_sorted(self, advance_fracs) -> list[np.ndarray]:
        """Per-round sorted score arrays restricted to *survivors*.

        Round 0 is the full field. For each later round, the comparison pool is
        narrowed to the field entries that placed in the top `advance_fracs[r-1]`
        of the still-alive pool in every prior round — the survivorship the real
        bracket imposes. Tightening stops once the pool would fall below
        `MIN_SURVIVORS` (see that constant), so the result is always a stable
        sample. With a large production field this compounds round-by-round; with
        a tiny smoke field it freezes at the round-1 survivor set.
        """
        raw = self._raw
        F, R = raw.shape
        survivor = np.ones(F, dtype=bool)
        out: list[np.ndarray] = []
        for r in range(R):
            out.append(np.sort(raw[survivor, r]))
            if r >= R - 1:
                continue
            frac = float(advance_fracs[r]) if r < len(advance_fracs) else 1.0
            alive = int(survivor.sum())
            if not (0.0 < frac < 1.0) or alive * frac < MIN_SURVIVORS:
                continue  # freeze: don't starve the pool
            thr = np.quantile(raw[survivor, r], 1.0 - frac)
            nxt = survivor & (raw[:, r] >= thr)
            if int(nxt.sum()) >= MIN_SURVIVORS:
                survivor = nxt
        return out


# ── Evaluation result ────────────────────────────────────────────────────────
@dataclass
class RosterEvaluation:
    prize_ev: float          # PRIMARY objective (expected prize equity)
    p_advance_r1: float
    finals_rate: float
    win_rate: float
    mean_points: float
    per_round_advance: list[float]   # mean advance prob, each non-final round

    def as_dict(self) -> dict:
        return {
            "prize_ev": self.prize_ev,
            "p_advance_r1": self.p_advance_r1,
            "finals_rate": self.finals_rate,
            "win_rate": self.win_rate,
            "mean_points": self.mean_points,
            "per_round_advance": self.per_round_advance,
        }


# ── Helpers ──────────────────────────────────────────────────────────────────
def _binom_cdf(k: int, n: int, q: np.ndarray) -> np.ndarray:
    """P(X <= k) for X ~ Binomial(n, q), elementwise over q. Small n only."""
    out = np.zeros_like(q, dtype=float)
    for j in range(k + 1):
        out += comb(n, j) * q ** j * (1.0 - q) ** (n - j)
    return out


def _advance_prob(q_exceed: np.ndarray, pod: int, advance: int) -> np.ndarray:
    """P(advance): at most (advance-1) of (pod-1) opponents beat me."""
    return _binom_cdf(advance - 1, pod - 1, q_exceed)


# ── Scorer ───────────────────────────────────────────────────────────────────
def evaluate_roster(result: SeasonSimResult, field: Field,
                    payout: np.ndarray = pay.DEFAULT_PAYOUT,
                    contest: bb.ContestConfig = bb.DEFAULT_CONTEST,
                    conditional_field: bool = False
                    ) -> RosterEvaluation:
    """
    Score a roster's simulated season against a field under a payout table.

    result: my SeasonSimResult ([S, n_rounds] round scores + total points)
    field:  opponent per-round distributions
    payout: length-`pod` array, prize equity by final rank (rank-1 indexed)
    conditional_field: if True, later rounds are scored against only the field
        entries that would have advanced the earlier rounds (survivorship), so
        the bar rises each round. Default False keeps the legacy unconditional
        field. See the module docstring / `Field.conditional_sorted`.
    """
    rs = result.round_scores                  # [S, R]
    S, R = rs.shape
    pod = contest.pod_size
    advance_counts = [max(1, round(f * pod)) for f in contest.advance_per_round]

    if conditional_field:
        advance_fracs = [advance_counts[r] / pod for r in range(R - 1)]
        sorted_by_round = field.conditional_sorted(advance_fracs)
    else:
        sorted_by_round = field._sorted

    # Advance through every round EXCEPT the last (the last is placement-paid).
    reach = np.ones(S)
    per_round_advance = []
    for r in range(R - 1):
        q = Field._exceed(sorted_by_round[r], rs[:, r])
        p_adv = _advance_prob(q, pod, advance_counts[r])
        per_round_advance.append(float(p_adv.mean()))
        if r == 0:
            p_advance_r1 = float(p_adv.mean())
        reach = reach * p_adv

    finals_rate = float(reach.mean())

    # Final round: placement payout by rank within the final pod.
    q_final = Field._exceed(sorted_by_round[R - 1], rs[:, R - 1])   # P(opp beats me)
    n_opp = pod - 1
    exp_payout = np.zeros(S)
    win_prob = np.zeros(S)
    for j in range(pod):                       # j opponents beat me -> rank j+1
        prob_j = comb(n_opp, j) * q_final ** j * (1.0 - q_final) ** (n_opp - j)
        exp_payout += prob_j * payout[j]
        if j == 0:
            win_prob = prob_j                  # rank 1

    prize_ev = float((reach * exp_payout).mean())
    win_rate = float((reach * win_prob).mean())

    # Single-round contests (e.g., Sit & Go): no advancement, placement only.
    if R == 1:
        p_advance_r1 = 1.0

    return RosterEvaluation(
        prize_ev=prize_ev,
        p_advance_r1=p_advance_r1 if R > 1 else 1.0,
        finals_rate=finals_rate,
        win_rate=win_rate,
        mean_points=float(result.total_points.mean()),
        per_round_advance=per_round_advance,
    )
