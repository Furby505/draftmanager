"""Status/finalizer for the fixed-gap DK EV selfplay rollout run.

Default mode is read-only: report chunks, row count, stderr, and same-position
gap sanity. With --fit-if-complete, fit a side model and write its manifest once
the expected chunk count is complete.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "server" / "models"

DEFAULT_ROLLOUTS = PROCESSED / "dk_ev_rollouts_selfplay_fixedgap_full_20260617.csv"
DEFAULT_STDOUT = PROCESSED / "dk_ev_fixedgap_full_20260617.out.log"
DEFAULT_STDERR = PROCESSED / "dk_ev_fixedgap_full_20260617.err.log"
DEFAULT_MODEL = MODELS / "model_dk_ev_policy_fixedgap_full_20260617.joblib"
DEFAULT_FEATURES = MODELS / "dk_ev_policy_fixedgap_full_20260617_feature_cols.json"
DEFAULT_META = MODELS / "dk_ev_policy_fixedgap_full_20260617_meta.json"
DEFAULT_AUDIT = PROCESSED / "dk_ev_policy_fixedgap_full_20260617_audit.md"
DEFAULT_PRED = PROCESSED / "dk_ev_policy_fixedgap_full_20260617_holdout_predictions.csv"
DEFAULT_MANIFEST = MODELS / "dk_ev_policy_fixedgap_full_20260617_manifest.json"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", errors="ignore") as f:
        return max(0, sum(1 for _ in f) - 1)


def parse_log(path: Path) -> tuple[int, int | None]:
    if not path.exists():
        return 0, None
    text = path.read_text(encoding="utf-8", errors="ignore")
    completed = [int(m.group(1)) for m in re.finditer(r"Chunk\s+(\d+)/(\d+)\s+complete", text)]
    totals = [int(m.group(2)) for m in re.finditer(r"Chunk\s+(\d+)/(\d+)\s+complete", text)]
    started_totals = [int(m.group(1)) for m in re.finditer(r"DK EV chunk\s+\d+/(\d+)", text)]
    total = totals[-1] if totals else (started_totals[-1] if started_totals else None)
    return (completed[-1] if completed else 0), total


def gap_sanity(path: Path) -> tuple[int | None, float | None, float | None]:
    if not path.exists() or count_rows(path) == 0:
        return None, None, None
    df = pd.read_csv(path, usecols=["cand_pos_next_best_adp_gap"])
    gaps = df["cand_pos_next_best_adp_gap"]
    return int((gaps < 0).sum()), float(gaps.min()), float(gaps.max())


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    p.add_argument("--stdout", type=Path, default=DEFAULT_STDOUT)
    p.add_argument("--stderr", type=Path, default=DEFAULT_STDERR)
    p.add_argument("--expected-chunks", type=int, default=90)
    p.add_argument("--fit-if-complete", action="store_true")
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--meta-out", type=Path, default=DEFAULT_META)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--pred-out", type=Path, default=DEFAULT_PRED)
    p.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rollouts = args.rollouts if args.rollouts.is_absolute() else ROOT / args.rollouts
    stdout = args.stdout if args.stdout.is_absolute() else ROOT / args.stdout
    stderr = args.stderr if args.stderr.is_absolute() else ROOT / args.stderr

    rows = count_rows(rollouts)
    done, total = parse_log(stdout)
    neg_gaps, min_gap, max_gap = gap_sanity(rollouts)
    err_bytes = stderr.stat().st_size if stderr.exists() else 0
    complete = done >= args.expected_chunks

    print("Fixed-gap rollout status")
    print(f"  rollouts: {rel(rollouts)}")
    print(f"  rows: {rows:,}")
    print(f"  chunks: {done}/{total or args.expected_chunks}")
    print(f"  stderr bytes: {err_bytes}")
    if neg_gaps is not None:
        print(f"  same-pos gap: negatives={neg_gaps} min={min_gap:.3f} max={max_gap:.3f}")
    print(f"  complete: {'YES' if complete else 'no'}")

    if err_bytes:
        print(f"\nStderr tail ({rel(stderr)}):")
        lines = stderr.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines[-20:]:
            print(line)

    if not args.fit_if_complete:
        return 0
    if not complete:
        print("Not fitting: run is not complete.")
        return 2
    if neg_gaps:
        print("Not fitting: negative same-position gaps found.")
        return 3

    run([
        sys.executable,
        "training/fit_dk_ev_policy.py",
        "--input", str(rollouts),
        "--model-out", str(args.model_out),
        "--features-out", str(args.features_out),
        "--meta-out", str(args.meta_out),
        "--audit-out", str(args.audit_out),
        "--pred-out", str(args.pred_out),
    ])
    run([
        sys.executable,
        "training/write_policy_manifest.py",
        "--model", str(args.model_out),
        "--features", str(args.features_out),
        "--meta", str(args.meta_out),
        "--rollouts", str(rollouts),
        "--audit", str(args.audit_out),
        "--predictions", str(args.pred_out),
        "--validation", "data/processed/paired_vs_adp_full_fixed.md",
        "--validation", "data/processed/paired_vs_adp_half.md",
        "--training-command",
        "python training/finalize_fixedgap_run.py --fit-if-complete",
        "--notes",
        "Fixed-gap side model. Not promoted unless validation beats current served model.",
        "--out", str(args.manifest_out),
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
