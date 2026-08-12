"""
Chunked DK EV rollout training.

Use this for real training runs. It repeatedly calls `train_policy.py` with
different seeds and `--append`, so labels are saved after every chunk instead
of only at the end of one long process.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

from fetch_dk_adp import _read_rows, _resolve_default_source


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "processed" / "dk_ev_rollouts.csv"


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def assert_market_board() -> None:
    source = _resolve_default_source()
    if source is None or not source.exists():
        raise SystemExit("No raw DraftKings ADP source found.")
    rows = _read_rows(source)
    draftable = [
        r for r in rows
        if r.get("pos") in {"QB", "RB", "WR", "TE"} and r.get("team") and float(r.get("adp", 0)) > 0
    ]
    print(f"DK ADP source: {source.name} rows={len(rows)} draftable={len(draftable)}")
    if len(draftable) < 300:
        raise SystemExit(f"DK ADP source too small: {len(draftable)} draftable rows")


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", type=int, default=40)
    p.add_argument("--states-per-chunk", type=int, default=250)
    p.add_argument(
        "--candidates",
        type=int,
        default=12,
        help="base candidates per state; train_policy expands later rounds automatically",
    )
    p.add_argument("--rollouts", type=int, default=6)
    p.add_argument("--eval-seasons", type=int, default=1500)
    p.add_argument("--field-rooms", type=int, default=32)
    p.add_argument("--field-seasons", type=int, default=350)
    p.add_argument("--min-round", type=int, default=2)
    p.add_argument("--max-round", type=int, default=18)
    p.add_argument("--completion", choices=["adp", "self"], default="adp",
                   help="how our roster is finished in each rollout")
    p.add_argument("--completion-model", type=Path,
                   default=ROOT / "server" / "models" / "model_dk_ev_policy.joblib")
    p.add_argument("--completion-features", type=Path,
                   default=ROOT / "server" / "models" / "dk_ev_policy_feature_cols.json")
    p.add_argument("--faller-prob", type=float, default=0.0,
                   help="fraction of states that inject an elite ADP faller")
    p.add_argument("--seed", type=int, default=20260611)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--fit-model-at-end", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    py = sys.executable
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    assert_market_board()

    start_rows = count_rows(out)
    print(f"Starting rows in {out}: {start_rows:,}")
    t0 = time.time()

    for chunk in range(args.chunks):
        seed = args.seed + chunk
        before = count_rows(out)
        print(f"\n=== DK EV chunk {chunk + 1}/{args.chunks} seed={seed} rows_before={before:,} ===")
        cmd = [
            py, "training/train_policy.py",
            "--states", str(args.states_per_chunk),
            "--candidates", str(args.candidates),
            "--rollouts", str(args.rollouts),
            "--eval-seasons", str(args.eval_seasons),
            "--field-rooms", str(args.field_rooms),
            "--field-seasons", str(args.field_seasons),
            "--min-round", str(args.min_round),
            "--max-round", str(args.max_round),
            "--completion", str(args.completion),
            "--completion-model", str(args.completion_model),
            "--completion-features", str(args.completion_features),
            "--faller-prob", str(args.faller_prob),
            "--seed", str(seed),
            "--out", str(out),
            "--append",
        ]
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(f"Chunk {chunk + 1} failed with exit code {result.returncode}")
        after = count_rows(out)
        elapsed = time.time() - t0
        done = chunk + 1
        avg = elapsed / done
        eta = avg * (args.chunks - done)
        print(
            f"Chunk {done}/{args.chunks} complete: +{after - before:,} rows "
            f"total={after:,} elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
        )

    if args.fit_model_at_end:
        print("\n=== Fitting served DK EV policy from accumulated labels ===")
        cmd = [
            py, "training/train_policy.py",
            "--states", "0",
            "--candidates", str(args.candidates),
            "--out", str(out),
            "--append",
            "--fit-model",
        ]
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(f"DK EV policy fit failed with exit code {result.returncode}")

    print(f"\nDK EV batch run complete. Rows: {count_rows(out):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
