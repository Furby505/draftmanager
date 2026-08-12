"""Quick status check for the current DraftManager DK EV runtime."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "server" / "models"


def section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def show_file(path: Path, label: str) -> None:
    if path.exists():
        size = path.stat().st_size / 1024
        print(f"  {label:32s} OK   {size:,.1f} KB")
    else:
        print(f"  {label:32s} MISSING")


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_runtime_artifacts() -> None:
    section("Server Artifacts")
    show_file(MODELS / "projections.json", "projections.json")
    show_file(MODELS / "dk_adp.json", "dk_adp.json")
    show_file(MODELS / "model_dk_ev_policy.joblib", "DK EV policy model")
    show_file(MODELS / "dk_ev_policy_feature_cols.json", "DK EV policy features")
    show_file(MODELS / "dk_ev_policy_meta.json", "DK EV policy metadata")
    show_file(MODELS / "dk_ev_policy_manifest.json", "DK EV policy manifest")


def check_policy_meta() -> None:
    section("DK EV Policy")
    meta_path = MODELS / "dk_ev_policy_meta.json"
    if not meta_path.exists():
        print("  No policy metadata found. Fit with: python training/fit_dk_ev_policy.py")
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics = meta.get("metrics", {})
    print(f"  Model kind: {meta.get('model_kind', '?')}")
    print(f"  Target:     {meta.get('target', '?')}")
    print(f"  Rows:       {meta.get('rows', 0):,}")
    print(f"  Max candidate rank trained: {meta.get('max_candidate_rank', '?')}")
    if metrics:
        print(
            "  Holdout:   "
            f"corr={metrics.get('edge_corr', 0):.3f} "
            f"top1={metrics.get('model_top1_accuracy', 0):.1%} "
            f"regret={metrics.get('model_avg_regret', 0):.8f}"
        )


def check_policy_manifest() -> None:
    section("DK EV Manifest")
    path = MODELS / "dk_ev_policy_manifest.json"
    if not path.exists():
        print("  No manifest found. Generate with: python training/write_policy_manifest.py")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    model = manifest.get("artifacts", {}).get("model", {})
    features = manifest.get("artifacts", {}).get("features", {})
    meta = manifest.get("artifacts", {}).get("meta", {})
    checks = [
        ("model", MODELS / "model_dk_ev_policy.joblib", model.get("sha256")),
        ("features", MODELS / "dk_ev_policy_feature_cols.json", features.get("sha256")),
        ("metadata", MODELS / "dk_ev_policy_meta.json", meta.get("sha256")),
    ]
    ok = True
    for label, fpath, expected in checks:
        actual = sha256(fpath)
        same = bool(expected) and actual == expected
        ok = ok and same
        print(f"  {label:10s} {'MATCH' if same else 'MISMATCH'} "
              f"{(actual or '')[:12]}")
    rows = manifest.get("data", {}).get("rollouts", {}).get("rows")
    if rows is not None:
        print(f"  rollout rows recorded: {rows:,}")
    print(f"  manifest status: {'OK' if ok else 'STALE'}")


def check_rollouts() -> None:
    section("Rollout Labels")
    path = PROCESSED / "dk_ev_rollouts.csv"
    if not path.exists():
        print("  No rollout CSV found. Generate with: python training/run_dk_ev_training.py --append")
        return
    with path.open(encoding="utf-8", errors="replace") as f:
        rows = max(0, sum(1 for _ in f) - 1)
    print(f"  dk_ev_rollouts.csv rows: {rows:,}")


def check_projections() -> None:
    section("Projection Metadata")
    path = MODELS / "projections.json"
    if not path.exists():
        print("  projections.json not found.")
        return
    players = json.loads(path.read_text(encoding="utf-8"))
    total = len(players)
    rookies = sum(1 for p in players if p.get("is_rookie"))
    with_bye = sum(1 for p in players if p.get("bye_week"))
    print(f"  Total players: {total}  rookies: {rookies}  with bye week: {with_bye}")


def check_processes() -> None:
    section("Running Python Processes")
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return
    raw = result.stdout.strip()
    if not raw:
        print("  No python.exe processes running.")
        return
    print(raw)


if __name__ == "__main__":
    print("\nDraftManager - DK EV Status")
    check_runtime_artifacts()
    check_policy_meta()
    check_policy_manifest()
    check_rollouts()
    check_projections()
    check_processes()
    print()
