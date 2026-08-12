"""
Tests for DK ADP ingestion (fetch_dk_adp).

Run: python training/test_dk_adp.py   (no pytest needed)
Does NOT clobber the canonical server/models/dk_adp.json (writes to a temp out).
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

import fetch_dk_adp as fa


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        check.failed += 1
check.failed = 0


def main():
    # ---- 1. Ingest the real DK export -------------------------------------
    src = fa._resolve_default_source()
    check("a real DK ADP source is present", src is not None and "draftkings" in str(src).lower()
          or (src and src.name == "dk_adp.csv"), str(src))

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "dk_adp.json"
        meta, matched, unmatched = fa.build(source=src, out=out)
        adp, _ = fa.load_dk_adp(out)
        check("ingested real DK ADP (flagged real)", meta["is_real_dk_adp"] is True)
        check("matched a full board to projections", meta["n_matched"] > 300,
              f"{meta['n_matched']}/{meta['n_source']} matched, "
              f"{meta['n_alias_matched']} via alias")
        check("curr_adp parsed (Gibbs ~1.2 at the top)",
              abs(adp.get("jahmyr gibbs", 99) - 1.2) < 0.5,
              f"gibbs adp={adp.get('jahmyr gibbs')}")
        check("DK ADP differs from FantasyPros (Gibbs over Bijan)",
              adp.get("jahmyr gibbs", 99) < adp.get("bijan robinson", 99))

    # ---- 2. Flexible column detection on an odd schema --------------------
    with tempfile.TemporaryDirectory() as d:
        odd = Path(d) / "weird.csv"
        pd.DataFrame({"Player": ["Some Guy", "Other Guy"],
                      "Average": [10.5, 22.0]}).to_csv(odd, index=False)
        m = fa.load_source(odd)
        check("auto-detects name/adp columns (Player/Average)",
              abs(m.get("some guy", 0) - 10.5) < 1e-6 and "other guy" in m,
              str(m))

    # ---- 3. occupyfantasy-style JSON ({"adps":[...]}) --------------------
    with tempfile.TemporaryDirectory() as d:
        jf = Path(d) / "x.json"
        jf.write_text('{"adps":[{"player_name":"Aa Bb","pos":"WR","curr_adp":4.4}]}')
        m = fa.load_source(jf)
        check("parses occupyfantasy adps[] JSON", abs(m.get("aa bb", 0) - 4.4) < 1e-6, str(m))

    print()
    if check.failed:
        print(f"{check.failed} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
