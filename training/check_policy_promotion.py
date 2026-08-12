"""Compare the served DK EV policy with a candidate side policy.

This script is read-only. It checks manifests, compares fit metrics, optionally
extracts paired-validation headlines from report markdown, and prints a
conservative promotion recommendation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "server" / "models"
PROCESSED = ROOT / "data" / "processed"

CURRENT_MANIFEST = MODELS / "dk_ev_policy_manifest.json"
CURRENT_META = MODELS / "dk_ev_policy_meta.json"
CANDIDATE_MANIFEST = MODELS / "dk_ev_policy_fixedgap_full_20260617_manifest.json"
CANDIDATE_META = MODELS / "dk_ev_policy_fixedgap_full_20260617_meta.json"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    p = path if path.is_absolute() else ROOT / path
    if not p.exists():
        return {}
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return blob if isinstance(blob, dict) else {}


def manifest_ok(path: Path) -> bool:
    p = path if path.is_absolute() else ROOT / path
    if not p.exists():
        return False
    result = subprocess.run(
        [sys.executable, "training/write_policy_manifest.py", "--check", "--out", str(p)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def metrics_from_meta(path: Path) -> dict[str, float]:
    meta = read_json(path)
    metrics = meta.get("metrics", {})
    out = {}
    for key in (
        "edge_corr",
        "model_top1_accuracy",
        "adp_top1_accuracy",
        "model_avg_regret",
        "adp_avg_regret",
        "model_avg_chosen_edge",
        "adp_avg_chosen_edge",
    ):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            out[key] = float(val)
    for key in ("rows", "states", "max_candidate_rank"):
        val = meta.get(key)
        if isinstance(val, (int, float)):
            out[key] = float(val)
    return out


def validation_headlines(path: Path) -> list[str]:
    p = path if path.is_absolute() else ROOT / path
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in text.splitlines():
        if line.startswith("- **"):
            lines.append(line)
    return lines


def pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.1%}"


def num(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.8f}" if abs(x) < 0.01 else f"{x:.3f}"


def compare_metric(label: str, cur: dict[str, float], cand: dict[str, float],
                   higher_better: bool = True) -> tuple[str, bool | None]:
    c = cur.get(label)
    n = cand.get(label)
    if c is None or n is None:
        return f"{label:24s} current=n/a candidate=n/a delta=n/a", None
    delta = n - c
    better = delta > 0 if higher_better else delta < 0
    return (
        f"{label:24s} current={num(c)} candidate={num(n)} delta={num(delta)} "
        f"{'OK' if better else 'worse'}",
        better,
    )


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current-manifest", type=Path, default=CURRENT_MANIFEST)
    p.add_argument("--current-meta", type=Path, default=CURRENT_META)
    p.add_argument("--candidate-manifest", type=Path, default=CANDIDATE_MANIFEST)
    p.add_argument("--candidate-meta", type=Path, default=CANDIDATE_META)
    p.add_argument("--candidate-validation", type=Path, action="append", default=[],
                   help="candidate validation markdown; may be repeated")
    p.add_argument("--min-corr-lift", type=float, default=0.0)
    p.add_argument("--min-top1-lift", type=float, default=0.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    current_meta = args.current_meta if args.current_meta.is_absolute() else ROOT / args.current_meta
    candidate_meta = args.candidate_meta if args.candidate_meta.is_absolute() else ROOT / args.candidate_meta
    current_manifest = args.current_manifest if args.current_manifest.is_absolute() else ROOT / args.current_manifest
    candidate_manifest = args.candidate_manifest if args.candidate_manifest.is_absolute() else ROOT / args.candidate_manifest

    print("DK EV Policy Promotion Check")
    print(f"  current meta:      {rel(current_meta)}")
    print(f"  candidate meta:    {rel(candidate_meta)}")
    print(f"  current manifest:  {rel(current_manifest)}")
    print(f"  candidate manifest:{rel(candidate_manifest)}")

    current_exists = current_meta.exists()
    candidate_exists = candidate_meta.exists()
    print(f"\nExistence")
    print(f"  current meta:   {'OK' if current_exists else 'MISSING'}")
    print(f"  candidate meta: {'OK' if candidate_exists else 'MISSING'}")
    if not current_exists or not candidate_exists:
        print("\nRecommendation: DO NOT PROMOTE - missing metadata.")
        return 2

    current_manifest_ok = manifest_ok(current_manifest)
    candidate_manifest_ok = manifest_ok(candidate_manifest)
    print(f"\nManifest")
    print(f"  current:   {'OK' if current_manifest_ok else 'MISSING/STALE'}")
    print(f"  candidate: {'OK' if candidate_manifest_ok else 'MISSING/STALE'}")

    cur = metrics_from_meta(current_meta)
    cand = metrics_from_meta(candidate_meta)
    print("\nFit Metrics")
    checks: list[bool] = []
    for key, higher in [
        ("rows", True),
        ("states", True),
        ("edge_corr", True),
        ("model_top1_accuracy", True),
        ("model_avg_regret", False),
        ("model_avg_chosen_edge", True),
    ]:
        line, better = compare_metric(key, cur, cand, higher_better=higher)
        print("  " + line)
        if key in {"edge_corr", "model_top1_accuracy", "model_avg_regret"} and better is not None:
            checks.append(better)

    corr_lift = cand.get("edge_corr", float("-inf")) - cur.get("edge_corr", float("inf"))
    top1_lift = cand.get("model_top1_accuracy", float("-inf")) - cur.get("model_top1_accuracy", float("inf"))
    gates = [
        current_manifest_ok,
        candidate_manifest_ok,
        corr_lift >= args.min_corr_lift,
        top1_lift >= args.min_top1_lift,
        cand.get("model_avg_regret", float("inf")) <= cur.get("model_avg_regret", float("-inf")),
    ]

    if args.candidate_validation:
        print("\nCandidate Validation Headlines")
        for path in args.candidate_validation:
            print(f"  {rel(path)}")
            for line in validation_headlines(path):
                print(f"    {line}")

    promote = all(gates)
    print("\nRecommendation: " + ("PROMOTE CANDIDATE" if promote else "DO NOT PROMOTE"))
    if not promote:
        print("Reason: one or more gates failed. Require fresh manifest, non-worse corr/top1/regret, and validation review.")
    return 0 if promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
