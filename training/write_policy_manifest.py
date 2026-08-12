"""Write a reproducibility manifest for a DK EV policy artifact.

The manifest binds a model to the exact files and metrics used to judge it:
model/features/meta hashes, rollout labels, audit reports, validation reports,
and relevant code hashes. It is intentionally side-effect free except for the
JSON file it writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "server" / "models"
PROCESSED = ROOT / "data" / "processed"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path) -> int | None:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".csv":
        return None
    with path.open("rb") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def file_record(path: Path) -> dict[str, Any]:
    p = path if path.is_absolute() else ROOT / path
    exists = p.exists()
    rec: dict[str, Any] = {
        "path": rel(p),
        "exists": exists,
    }
    if exists and p.is_file():
        st = p.stat()
        rec.update({
            "bytes": st.st_size,
            "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "sha256": sha256(p),
        })
        rows = count_rows(p)
        if rows is not None:
            rec["rows"] = rows
    return rec


def read_json(path: Path) -> dict[str, Any]:
    p = path if path.is_absolute() else ROOT / path
    if not p.exists():
        return {}
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return blob if isinstance(blob, dict) else {}


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=MODELS / "model_dk_ev_policy.joblib")
    p.add_argument("--features", type=Path, default=MODELS / "dk_ev_policy_feature_cols.json")
    p.add_argument("--meta", type=Path, default=MODELS / "dk_ev_policy_meta.json")
    p.add_argument("--rollouts", type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--validation", type=Path, action="append", default=[],
                   help="validation/audit report to include; may be repeated")
    p.add_argument("--code-file", type=Path, action="append", default=[
        Path("server/server.py"),
        Path("training/train_policy.py"),
        Path("training/fit_dk_ev_policy.py"),
        Path("training/backtest_policy.py"),
        Path("training/paired_vs_adp.py"),
        Path("training/validate_vs_humans.py"),
    ])
    p.add_argument("--training-command", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--out", type=Path, default=MODELS / "dk_ev_policy_manifest.json")
    p.add_argument("--check", action="store_true",
                   help="validate an existing manifest instead of writing one")
    return p.parse_args(argv)


def iter_file_records(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        if "path" in obj and "sha256" in obj:
            yield prefix.rstrip("."), obj
            return
        for key, val in obj.items():
            yield from iter_file_records(val, f"{prefix}{key}.")
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            yield from iter_file_records(val, f"{prefix}{i}.")


def check_manifest(path: Path) -> int:
    p = path if path.is_absolute() else ROOT / path
    manifest = read_json(p)
    if not manifest:
        print(f"Manifest not found or not valid JSON: {rel(p)}")
        return 1

    failures = 0
    checked = 0
    for label, rec in iter_file_records(manifest):
        expected_hash = rec.get("sha256")
        expected_exists = bool(rec.get("exists"))
        fpath = ROOT / rec.get("path", "")
        exists = fpath.exists()
        if expected_exists and not exists:
            print(f"[FAIL] {label}: missing {rec.get('path')}")
            failures += 1
            continue
        if not expected_exists:
            continue
        actual_hash = sha256(fpath)
        checked += 1
        if actual_hash != expected_hash:
            print(
                f"[FAIL] {label}: hash mismatch {rec.get('path')} "
                f"expected={str(expected_hash)[:12]} actual={str(actual_hash)[:12]}"
            )
            failures += 1

    if failures:
        print(f"Manifest check failed: {failures} failure(s), {checked} hashed file(s) checked.")
        return 1
    print(f"Manifest check passed: {checked} hashed file(s) checked.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        return check_manifest(args.out)

    meta = read_json(args.meta)

    audit = args.audit or Path(meta.get("audit_out", ""))
    predictions = args.predictions or Path(meta.get("pred_out", ""))

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "DraftKings Best Ball tournament EV policy",
        "training_command": args.training_command,
        "notes": args.notes,
        "artifacts": {
            "model": file_record(args.model),
            "features": file_record(args.features),
            "meta": file_record(args.meta),
        },
        "data": {},
        "reports": {},
        "code": [file_record(path) for path in args.code_file],
        "meta": meta,
        "metrics": meta.get("metrics", {}),
    }

    if args.rollouts:
        manifest["data"]["rollouts"] = file_record(args.rollouts)
    if audit and str(audit):
        manifest["reports"]["fit_audit"] = file_record(audit)
    if predictions and str(predictions):
        manifest["reports"]["holdout_predictions"] = file_record(predictions)
    if args.validation:
        manifest["reports"]["validation"] = [file_record(path) for path in args.validation]

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
