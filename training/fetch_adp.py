"""
Fetch best ball ADP from FantasyPros and save to server/models/adp.json.
Run before a draft or periodically to keep ADP fresh.

Usage:
  python training/fetch_adp.py

Output: server/models/adp.json  — {normalized_name: adp_rank}
"""

import json
import re
import time
import requests
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "server" / "models"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fantasypros.com/",
}

ADP_URLS = [
    # Best ball overall (consensus across DK, Underdog, etc.)
    "https://www.fantasypros.com/nfl/adp/best-ball-overall.php",
    # Fallback: standard best ball
    "https://www.fantasypros.com/nfl/adp/best-ball.php",
]


def normalize(name: str) -> str:
    n = name.lower()
    n = re.sub(r"[.,'\-]", "", n)  # include comma for "Last, First" format
    n = re.sub(r"\s+", " ", n).strip()
    n = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", n)
    return n


def scrape_fantasypros_adp(url: str) -> dict[str, float]:
    """
    Scrape FantasyPros ADP table and return {normalized_name: adp}.
    Tries both HTML table parsing and embedded JSON data.
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return {}

    from html.parser import HTMLParser

    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_table = False
            self.in_row   = False
            self.in_cell  = False
            self.rows     = []
            self.cur_row  = []
            self.cur_cell = ""
            self.depth    = 0

        def handle_starttag(self, tag, attrs):
            attrs_d = dict(attrs)
            if tag == "table":
                self.in_table = True
                self.depth += 1
            elif self.in_table and tag == "tr":
                self.in_row = True
                self.cur_row = []
            elif self.in_table and tag in ("td", "th"):
                self.in_cell = True
                self.cur_cell = ""
            elif tag == "a" and self.in_cell:
                pass  # will capture text inside <a> normally

        def handle_endtag(self, tag):
            if tag == "table":
                self.depth -= 1
                if self.depth == 0:
                    self.in_table = False
            elif self.in_table and tag == "tr":
                if self.cur_row:
                    self.rows.append(self.cur_row[:])
                self.in_row = False
                self.cur_row = []
            elif self.in_table and tag in ("td", "th"):
                self.cur_row.append(self.cur_cell.strip())
                self.in_cell = False

        def handle_data(self, data):
            if self.in_cell:
                self.cur_cell += data

    parser = TableParser()
    parser.feed(r.text)

    results: dict[str, float] = {}
    for row in parser.rows:
        if len(row) < 3:
            continue
        # Typical columns: Rank | Player (Name Team BYE) | POS | AVG | ...
        # The player name cell usually contains name + team + position info
        name_cell = row[1] if len(row) > 1 else ""
        avg_cell  = ""

        # Try to find the numeric ADP column (labeled AVG or ADP)
        for cell in row[2:]:
            try:
                val = float(cell.replace(",", ""))
                if 1 <= val <= 300:
                    avg_cell = cell
                    break
            except ValueError:
                continue

        if not avg_cell:
            continue

        # Extract player name (first part before team abbreviation)
        # FantasyPros usually formats as "Name Team BYE" in one cell
        # or as separate cells. Try to grab the name part.
        name_parts = name_cell.strip().split()
        if len(name_parts) < 2:
            continue

        # Heuristic: NFL team abbreviations are 2-3 uppercase letters
        # Find where the team abbr starts and take everything before it as name
        name_tokens = []
        for tok in name_parts:
            if re.match(r'^[A-Z]{2,4}$', tok):  # looks like a team abbr
                break
            name_tokens.append(tok)

        if len(name_tokens) < 2:
            name_tokens = name_parts[:2]  # fallback: first two words

        player_name = " ".join(name_tokens)
        norm_name   = normalize(player_name)
        adp_val     = float(avg_cell)

        if norm_name and adp_val > 0:
            results[norm_name] = adp_val

    return results


def fetch_adp() -> dict[str, float]:
    for url in ADP_URLS:
        print(f"  Fetching: {url}")
        data = scrape_fantasypros_adp(url)
        if len(data) >= 100:
            print(f"  Got {len(data)} players")
            return data
        print(f"  Only {len(data)} players — trying next URL")
        time.sleep(1)

    print("  WARNING: Could not scrape enough ADP data from any source.")
    return {}


def main():
    print("=" * 55)
    print("DraftManager -- Fetch Best Ball ADP")
    print("=" * 55)

    adp = fetch_adp()

    if not adp:
        print("No ADP data fetched. server/models/adp.json not updated.")
        return

    out_path = MODELS_DIR / "adp.json"
    with open(out_path, "w") as f:
        json.dump(adp, f, indent=2, sort_keys=True)

    print(f"\nSaved {len(adp)} player ADPs to {out_path}")

    # Show sample
    top10 = sorted(adp.items(), key=lambda x: x[1])[:10]
    print("\nTop 10 by ADP:")
    for name, adp_val in top10:
        print(f"  {adp_val:6.1f}  {name}")


if __name__ == "__main__":
    main()
