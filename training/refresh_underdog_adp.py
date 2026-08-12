"""
Refresh current Underdog best-ball ADP.

Primary source is The Fantasy Fellowship's public Underdog ADP table. It exposes
a full Top 300 with decimal ADP and is parseable without browser automation.

Output:
  server/models/underdog_adp.json
  underdog_best_ball_adp_latest.{json,csv}
"""

from __future__ import annotations

import csv
import html
import json
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "server" / "models"
SOURCE_URL = "https://thefantasyfellowship.com/underdog-adp/"
JSON_OUT = ROOT / "underdog_best_ball_adp_latest.json"
CSV_OUT = ROOT / "underdog_best_ball_adp_latest.csv"
CANONICAL_OUT = MODELS / "underdog_adp.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (DraftManager Underdog ADP refresh)",
    "Accept": "text/html,application/xhtml+xml",
}


def normalize(name: str) -> str:
    n = html.unescape(name).replace("’", "'")
    n = n.lower()
    n = re.sub(r"[.,'\-]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    n = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", n)
    aliases = {
        "hollywood brown": "marquise brown",
        "kenny gainwell": "kenneth gainwell",
    }
    return aliases.get(n, n)


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_target = False
        self.in_row = False
        self.in_cell = False
        self.rows: list[list[str]] = []
        self.cur_row: list[str] = []
        self.cur_cell = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = {k: v or "" for k, v in attrs}
        if tag == "table" and attrs_d.get("id") == "tablepress-43":
            self.in_target = True
        elif self.in_target and tag == "tr":
            self.in_row = True
            self.cur_row = []
        elif self.in_target and tag in {"td", "th"}:
            self.in_cell = True
            self.cur_cell = ""

    def handle_endtag(self, tag: str) -> None:
        if self.in_target and tag in {"td", "th"} and self.in_cell:
            self.cur_row.append(html.unescape(self.cur_cell).strip())
            self.cur_cell = ""
            self.in_cell = False
        elif self.in_target and tag == "tr" and self.in_row:
            if self.cur_row:
                self.rows.append(self.cur_row[:])
            self.cur_row = []
            self.in_row = False
        elif self.in_target and tag == "table":
            self.in_target = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cur_cell += data


def fetch_rows() -> list[dict]:
    r = requests.get(SOURCE_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()

    parser = TableParser()
    parser.feed(r.text)

    rows: list[dict] = []
    for row in parser.rows:
        if len(row) < 7 or row[0].lower() == "rank":
            continue
        try:
            rank = int(row[0])
            adp = float(row[6])
        except ValueError:
            continue
        first = row[2].replace("’", "'").strip()
        last = row[3].replace("’", "'").strip()
        name = f"{first} {last}".strip()
        if not name:
            continue
        rows.append({
            "rank": rank,
            "name": name,
            "position": row[1].strip(),
            "team": row[4].strip(),
            "pos_rank": row[5].strip(),
            "adp": adp,
            "projection": row[7].strip() if len(row) > 7 else "",
            "salary": row[8].strip() if len(row) > 8 else "",
        })

    if len(rows) < 200:
        raise RuntimeError(f"parsed only {len(rows)} Underdog ADP rows")
    return rows


def write_outputs(rows: list[dict]) -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    envelope = {
        "source": SOURCE_URL,
        "fetched_at_utc": fetched_at,
        "row_count": len(rows),
        "adps": rows,
    }
    JSON_OUT.write_text(json.dumps(envelope, indent=2))
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    adp = {normalize(r["name"]): float(r["adp"]) for r in rows}
    canonical = {
        "_meta": {
            "source": SOURCE_URL,
            "generated": fetched_at,
            "is_real_underdog_adp": True,
            "n_rows": len(rows),
        },
        "adp": dict(sorted(adp.items(), key=lambda kv: kv[1])),
    }
    CANONICAL_OUT.write_text(json.dumps(canonical, indent=2))


def main() -> int:
    try:
        rows = fetch_rows()
    except Exception as e:
        print(f"[refresh_underdog_adp] FETCH FAILED, leaving existing board untouched: {e}")
        return 1
    write_outputs(rows)
    print(f"[refresh_underdog_adp] rebuilt {CANONICAL_OUT} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
