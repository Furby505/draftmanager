"""
Tests for the two field-realism levers (both default OFF):

  1. Asymmetric opponent field bias  (opponent_field.py)
     - rookies/recent producers get reached for, vets fall
     - a biased field drafts the hyped set earlier than a symmetric-noise field
  2. Conditional (survivorship) bracket field  (bracket.py)
     - round 0 pool is the full field; later rounds narrow to survivors
     - survivors are tougher, so finals/win rates drop vs the unconditional field

Run: python training/test_field_realism.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import opponent_field as of
from bracket import Field, MIN_SURVIVORS, evaluate_roster
from season_sim import SeasonSimResult
import best_ball as bb


# ── 1. Field bias — unit ─────────────────────────────────────────────────────
def test_player_bias_directions():
    p = of.DEFAULT_FIELD_BIAS
    rookie = of._player_field_bias({"is_rookie": True, "age": 22.0, "prior_pts": 0.0}, p)
    star = of._player_field_bias({"is_rookie": False, "age": 25.0, "prior_pts": 300.0}, p)
    vet = of._player_field_bias({"is_rookie": False, "age": 33.0, "prior_pts": 0.0}, p)
    neutral = of._player_field_bias({"is_rookie": False, "age": 26.0, "prior_pts": 0.0}, p)
    assert rookie < 0, f"rookie should be reached for (negative), got {rookie}"
    assert star < 0, f"last-year star should be reached for, got {star}"
    assert vet > 0, f"old vet should fall (positive), got {vet}"
    assert neutral == 0.0, f"plain mid player should be unbiased, got {neutral}"
    # recency is capped at recency_max
    capped = of._player_field_bias({"is_rookie": False, "age": 25.0, "prior_pts": 9999.0}, p)
    assert capped >= -p.recency_max - 1e-9
    print("  ok: per-player bias directions (rookie/star reach, vet fall, mid neutral)")


def test_board_carries_bias():
    plain = of.load_market_board(field_bias=False)
    biased = of.load_market_board(field_bias=True)
    assert plain.field_bias is None
    assert biased.field_bias is not None
    assert len(biased.field_bias) == len(biased)
    assert np.any(biased.field_bias != 0.0), "expected some players to be biased"
    # board identity is unchanged; only the bias array is added
    assert np.array_equal(plain.adp, biased.adp)
    print(f"  ok: biased board carries {(biased.field_bias != 0).sum()} non-zero shifts")


def test_biased_field_reaches_hyped_earlier():
    """A biased field should draft reached-for players in earlier rounds.

    Measured on MID-ADP reached players (ADP 50-170), where there is room for
    the draft slot to actually move — elite players are floored at pick 1 and
    deep rookies at the final round, so they can't shift either way.
    """
    biased = of.load_market_board(field_bias=True)
    plain = of.load_market_board(field_bias=False)
    mask = (biased.field_bias <= -3.0) & (biased.adp >= 50) & (biased.adp <= 170)
    hyped_ids = {str(biased.player_id[i]) for i in np.where(mask)[0]}
    assert len(hyped_ids) >= 10, "need a reasonable mid-ADP reached set to test"

    def mean_draft_round(board, seed, n_rooms=150):
        rng = np.random.default_rng(seed)
        rounds, hits = 0, 0
        for roster in of.draft_field(n_rooms, rng, board=board):
            for rnd, pl in enumerate(roster):
                if pl["player_id"] in hyped_ids:
                    rounds += rnd
                    hits += 1
        return rounds / max(1, hits)

    biased_round = mean_draft_round(biased, 7)
    plain_round = mean_draft_round(plain, 7)
    assert biased_round < plain_round, (
        f"reached players should go earlier with bias on: "
        f"biased={biased_round:.3f} vs plain={plain_round:.3f}"
    )
    print(f"  ok: mid-ADP reached set mean draft round {biased_round:.3f} (biased) "
          f"< {plain_round:.3f} (plain)")


# ── 2. Conditional bracket field ─────────────────────────────────────────────
def _correlated_field(n=4000, R=4, seed=1) -> np.ndarray:
    """Field round scores with a latent per-roster strength, so strong rosters
    score high across rounds (survivors are genuinely tougher)."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1, size=n)[:, None]
    noise = rng.normal(0, 1, size=(n, R))
    return 100.0 + 12.0 * strength + 8.0 * noise


def test_conditional_round0_is_full_field():
    raw = _correlated_field()
    field = Field(raw)
    fracs = [2 / 12, 1 / 12, 1 / 12]
    cond = field.conditional_sorted(fracs)
    assert np.array_equal(cond[0], field._sorted[0]), "round 0 must be the full field"
    # later rounds restrict to a subset (survivors)
    assert len(cond[1]) < len(cond[0]), "round 1 pool should be narrowed to survivors"
    assert all(len(c) >= MIN_SURVIVORS for c in cond), "pool must never starve below floor"
    print(f"  ok: round pools {[len(c) for c in cond]} (full -> survivors, >= floor)")


def test_conditional_survivors_are_tougher():
    raw = _correlated_field()
    field = Field(raw)
    fracs = [2 / 12, 1 / 12, 1 / 12]
    cond = field.conditional_sorted(fracs)
    # survivor pool in the last round should have a higher mean than the full field
    assert cond[-1].mean() > field._sorted[-1].mean(), "survivors should be stronger"
    print(f"  ok: final-round survivor mean {cond[-1].mean():.1f} "
          f"> full-field mean {field._sorted[-1].mean():.1f}")


def test_conditional_lowers_advance_rates():
    raw = _correlated_field()
    field = Field(raw)
    R = raw.shape[1]
    # an average-ish roster across the simulated seasons
    rng = np.random.default_rng(99)
    rs = 100.0 + 12.0 * rng.normal(0, 1, size=(500, 1)) + 8.0 * rng.normal(0, 1, size=(500, R))
    result = SeasonSimResult(round_scores=rs, total_points=rs.sum(1),
                             round_labels=["R1", "R2", "R3", "R4"],
                             weekly_lineup=rs)
    base = evaluate_roster(result, field, conditional_field=False)
    cond = evaluate_roster(result, field, conditional_field=True)
    assert cond.finals_rate <= base.finals_rate + 1e-9, (
        f"survivorship field should not inflate finals: "
        f"cond={cond.finals_rate:.4f} base={base.finals_rate:.4f}"
    )
    assert cond.finals_rate < base.finals_rate, "tougher late field should lower finals rate"
    print(f"  ok: finals_rate {base.finals_rate:.4f} (uncond) -> "
          f"{cond.finals_rate:.4f} (conditional)")


def main() -> int:
    tests = [
        test_player_bias_directions,
        test_board_carries_bias,
        test_biased_field_reaches_hyped_earlier,
        test_conditional_round0_is_full_field,
        test_conditional_survivors_are_tougher,
        test_conditional_lowers_advance_rates,
    ]
    print("Field-realism tests")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\nAll {len(tests)} field-realism tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
