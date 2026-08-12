"""Show whether running Python jobs are DK EV training or the server."""

from __future__ import annotations

import json
import subprocess
import sys


def classify(cmd: str) -> str:
    low = cmd.lower()
    if (
        "run_dk_ev_training.py" in low
        or "run_dk_ev_batches.py" in low
        or "train_policy.py" in low
        or "fit_dk_ev_policy.py" in low
    ):
        return "DK_EV"
    if "server.py" in low:
        return "SERVER"
    return "OTHER"


def main() -> int:
    ps = (
        "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return result.returncode

    raw = result.stdout.strip()
    if not raw:
        print("No python.exe processes found.")
        return 0
    data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]

    print("PID      KIND              COMMAND")
    print("-" * 90)
    for row in data:
        cmd = row.get("CommandLine") or ""
        print(f"{row.get('ProcessId'):<8} {classify(cmd):<17} {cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
