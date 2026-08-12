"""
Daily DK Best Ball ADP refresh.

Pulls current DraftKings best-ball ADP from the occupyfantasy API, overwrites the
canonical source export, and regenerates server/models/dk_adp.json via
fetch_dk_adp.build(). This is the ONLY reproducible way to keep the live board's
ADP current -- the server reads dk_adp.json once at startup, so run this (via the
Windows scheduled task, or by hand before a draft) and then start/restart the
server.

    python training/refresh_dk_adp.py

The API needs no auth. On any network failure the existing dk_adp.json is left
untouched (a stale board beats an empty one), and the script exits non-zero so a
scheduler surfaces the failure.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import fetch_dk_adp

ROOT       = Path(__file__).resolve().parent.parent
SOURCE_URL = "https://www.occupyfantasyapi.com/best_ball/adps?site=draftkings&contest=all"
JSON_OUT   = ROOT / "draftkings_best_ball_adp_latest.json"
CSV_OUT    = ROOT / "draftkings_best_ball_adp_latest.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (DraftManager ADP refresh)"}


def fetch_rows() -> list[dict]:
    """GET the live ADP feed and return its list of player rows."""
    r = requests.get(SOURCE_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("adps") if isinstance(payload, dict) else payload
    if not rows:
        raise ValueError(f"ADP feed returned no rows (payload keys: "
                         f"{list(payload) if isinstance(payload, dict) else type(payload)})")
    return rows


def write_source(rows: list[dict]) -> None:
    """Persist the raw feed as JSON (with envelope) + CSV, matching prior exports."""
    teams = sorted({str(r.get("team", "")) for r in rows if r.get("team")})
    envelope = {
        "source": SOURCE_URL,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "teams": teams,
        "adps": rows,
    }
    JSON_OUT.write_text(json.dumps(envelope, indent=0))
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)


def main() -> int:
    try:
        rows = fetch_rows()
    except Exception as e:  # network / parse failure -> keep the existing board
        print(f"[refresh_dk_adp] FETCH FAILED, leaving dk_adp.json untouched: {e}")
        return 1

    write_source(rows)
    print(f"[refresh_dk_adp] pulled {len(rows)} players from {SOURCE_URL}")

    # Regenerate the canonical server board from the CSV we just wrote.
    meta, matched, unmatched = fetch_dk_adp.build(source=CSV_OUT)
    print(f"[refresh_dk_adp] rebuilt dk_adp.json  generated={meta['generated']}  "
          f"matched={meta['n_matched']}  unmatched={meta['n_unmatched']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
