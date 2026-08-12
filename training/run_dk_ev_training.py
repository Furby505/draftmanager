"""
Canonical DK EV training entrypoint.

This is the runner for the pivoted project:

    real DraftKings ADP -> DK draft-room simulation -> correlated season sims
    -> DK tournament bracket -> prize-EV labels

The legacy marginal-points self-play path has been removed from the runtime.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from fetch_dk_adp import _read_rows, _resolve_default_source


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "processed" / "dk_ev_rollouts.csv"


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def assert_real_dk_adp() -> None:
    source = _resolve_default_source()
    if source is None or not source.exists():
        raise SystemExit(
            "DK EV training requires a real DraftKings ADP export at "
            "data/raw/dk_adp.csv or draftkings_best_ball_adp_latest.csv."
        )
    rows = _read_rows(source)
    draftable = [
        r for r in rows
        if r.get("pos") in {"QB", "RB", "WR", "TE"} and r.get("team") and float(r.get("adp", 0)) > 0
    ]
    print(f"DK ADP OK: {source.name} rows={len(rows)} draftable={len(draftable)}")
    if len(draftable) < 300:
        raise SystemExit(f"DK ADP source looks too small for EV training: {len(draftable)} draftable rows")


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backfill", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--skip-tests", action="store_true",
                   help="skip DK EV smoke tests before rollout generation")
    p.add_argument("--states", type=int, default=40)
    p.add_argument(
        "--candidates",
        type=int,
        default=12,
        help="base candidates per state; train_policy expands later rounds automatically",
    )
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--eval-seasons", type=int, default=1200)
    p.add_argument("--field-rooms", type=int, default=16)
    p.add_argument("--field-seasons", type=int, default=220)
    p.add_argument("--min-round", type=int, default=2)
    p.add_argument("--max-round", type=int, default=18)
    p.add_argument("--seed", type=int, default=20260611)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--append", action="store_true")
    p.add_argument("--fit-model", action="store_true",
                   help="fit the served DK EV policy after writing rollout labels")
    p.add_argument("--field-bias", action="store_true",
                   help="opponent rooms reach for hyped rookies / recent producers and "
                        "let vets fall (asymmetric field realism)")
    p.add_argument("--conditional-field", action="store_true",
                   help="score later bracket rounds against survivors only "
                        "(survivorship field; rewards ceiling)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    py = sys.executable

    if args.backfill:
        raise SystemExit("--backfill belongs to the old projection-board lane, not DK EV training.")

    assert_real_dk_adp()

    if not args.skip_tests:
        run_step("Outcome model test", [py, "training/test_outcome_model.py"])
        run_step("Season/bracket test", [py, "training/test_season_bracket.py"])
        run_step("Opponent field test", [py, "training/test_opponent_field.py"])
        run_step("Policy mechanics test", [py, "training/test_train_policy.py"])

    cmd = [
        py, "training/train_policy.py",
        "--states", str(args.states),
        "--candidates", str(args.candidates),
        "--rollouts", str(args.rollouts),
        "--eval-seasons", str(args.eval_seasons),
        "--field-rooms", str(args.field_rooms),
        "--field-seasons", str(args.field_seasons),
        "--min-round", str(args.min_round),
        "--max-round", str(args.max_round),
        "--seed", str(args.seed),
        "--out", str(args.out),
    ]
    if args.append:
        cmd.append("--append")
    if args.fit_model:
        cmd.append("--fit-model")
    if args.field_bias:
        cmd.append("--field-bias")
    if args.conditional_field:
        cmd.append("--conditional-field")

    run_step("Generate DK EV rollout labels", cmd)
    print("\nDK EV training batch complete.")
    print(f"Rollout labels: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
