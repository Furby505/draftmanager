"""
DK Best Ball ADP ingestion + matching to the projection universe.

The opponent field's edge is measured relative to the *market it actually drafts
against*, so the board must use real DraftKings Best Ball ADP. This module
normalizes a DK ADP export and MATCHES each DK player to the projection universe
(the board can only contain players we project), writing one canonical file:

    server/models/dk_adp.json  ->  { "_meta": {...}, "adp": { proj_norm_name: adp } }

Matching: exact normalized name first, then surname + position + team (with
team-abbreviation normalization) to recover name variants like
"Nick Singleton"->"Nicholas Singleton", "Joshua Palmer"->"Josh Palmer",
"Hollywood Brown"->"Marquise Brown". Players with no projection (FAs, uncovered
rookies) stay UNMATCHED and are surfaced by audit_dk_adp_match.py.

Usage
-----
    python training/fetch_dk_adp.py [--source PATH]
Default source: data/raw/dk_adp.csv, else draftkings_best_ball_adp_latest.{csv,json},
else the FantasyPros adp.json STAND-IN (clearly labeled; not real DK ADP).
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from outcome_model import normalize_name

ROOT          = Path(__file__).resolve().parent.parent
DK_ADP_JSON   = ROOT / "server" / "models" / "dk_adp.json"
FP_ADP_JSON   = ROOT / "server" / "models" / "adp.json"          # FantasyPros stand-in
PROJECTIONS   = ROOT / "server" / "models" / "projections.json"

SOURCE_CANDIDATES = (
    ROOT / "data" / "raw" / "dk_adp.csv",
    ROOT / "draftkings_best_ball_adp_latest.csv",
    ROOT / "draftkings_best_ball_adp_latest.json",
)

NAME_COLS = ("player_name", "player", "name", "playername", "full_name")
ADP_COLS  = ("curr_adp", "adp", "avg", "average", "avg_pick",
             "averagedraftposition", "bestball_adp", "adp_ppr", "overall", "rank")
POS_COLS  = ("pos", "position")
TEAM_COLS = ("team", "tm", "recent_team")
BYE_COLS  = ("bye_week", "bye", "byeweek")

# DK -> projection team-abbreviation convention.
TEAM_ALIASES = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "LVR": "LV",
                "OAK": "LV", "SD": "LAC", "STL": "LA", "ARZ": "ARI"}


def _pick_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def _norm_team(t) -> str:
    t = str(t or "").strip().upper()
    return TEAM_ALIASES.get(t, t)


def _surname(nm: str) -> str:
    toks = nm.split()
    return toks[-1] if toks else nm


def _read_rows(source: Path) -> list[dict]:
    """Read a DK ADP source into rows: {nm, pos, team, adp}. pos/team optional."""
    source = Path(source)
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text())
        if isinstance(data, dict) and "adps" in data:
            df = pd.DataFrame(data["adps"])
        elif isinstance(data, dict):                       # {name: adp}
            return [{"nm": normalize_name(k), "pos": None, "team": None,
                     "adp": float(v)} for k, v in data.items()
                    if str(v).replace(".", "", 1).replace("-", "").isdigit()]
        else:
            df = pd.DataFrame(data)
    else:
        df = pd.read_csv(source)

    name_col = _pick_col(df.columns, NAME_COLS)
    adp_col = _pick_col(df.columns, ADP_COLS)
    pos_col = _pick_col(df.columns, POS_COLS)
    team_col = _pick_col(df.columns, TEAM_COLS)
    bye_col = _pick_col(df.columns, BYE_COLS)
    if name_col is None or adp_col is None:
        raise ValueError(f"Need name+adp columns. Saw {list(df.columns)}.")
    rows = []
    for _, r in df.iterrows():
        try:
            adp = float(r[adp_col])
        except (TypeError, ValueError):
            continue
        nm = normalize_name(str(r[name_col]))
        if nm and adp > 0:
            try:
                bye_week = int(r[bye_col]) if bye_col else 0
            except (TypeError, ValueError):
                bye_week = 0
            rows.append({"nm": nm, "adp": adp,
                         "pos": (str(r[pos_col]).upper() if pos_col else None),
                         "team": (_norm_team(r[team_col]) if team_col else None),
                         "bye_week": bye_week})
    return rows


def _index_projections(proj: list[dict]):
    by_name, by_spt = {}, {}
    for rec in proj:
        nm = normalize_name(rec.get("player_display_name", ""))
        by_name[nm] = rec
        key = (_surname(nm), str(rec.get("position", "")).upper(),
               _norm_team(rec.get("recent_team", "")))
        by_spt.setdefault(key, []).append(rec)
    return by_name, by_spt


def match_to_projections(rows: list[dict], proj: list[dict]):
    """Return (mapping{proj_norm: adp}, matched[], unmatched[])."""
    by_name, by_spt = _index_projections(proj)
    mapping, matched, unmatched = {}, [], []
    for r in rows:
        rec, how = by_name.get(r["nm"]), "exact"
        if rec is None and r["pos"] and r["team"]:
            cand = by_spt.get((_surname(r["nm"]), r["pos"], r["team"]), [])
            if len(cand) == 1:
                rec, how = cand[0], "alias(surname+pos+team)"
        if rec is not None:
            pn = normalize_name(rec["player_display_name"])
            mapping[pn] = r["adp"]                          # last write wins (sorted by adp later N/A)
            matched.append({**r, "proj_name": rec["player_display_name"], "how": how})
        else:
            unmatched.append(r)
    return mapping, matched, unmatched


def build(source: Path | None = None, out: Path = DK_ADP_JSON):
    """Build canonical dk_adp.json. Returns (meta, matched, unmatched)."""
    source = Path(source) if source else _resolve_default_source()
    proj = json.loads(PROJECTIONS.read_text())
    if source and source.exists():
        rows = _read_rows(source)
        src_label, is_real = str(source.name), True
    else:
        rows = [{"nm": normalize_name(k), "pos": None, "team": None, "adp": float(v)}
                for k, v in json.loads(FP_ADP_JSON.read_text()).items()]
        src_label, is_real = "FALLBACK:fantasypros(adp.json) — NOT real DK ADP", False

    mapping, matched, unmatched = match_to_projections(rows, proj)
    n_alias = sum(1 for m in matched if m["how"].startswith("alias"))
    meta = {"source": src_label, "is_real_dk_adp": is_real,
            "generated": date.today().isoformat(),
            "n_source": len(rows), "n_matched": len(matched),
            "n_alias_matched": n_alias, "n_unmatched": len(unmatched)}
    out.write_text(json.dumps({"_meta": meta, "adp": mapping}, indent=0))
    print(f"[dk_adp] source={src_label}  matched={len(matched)}/{len(rows)} "
          f"(alias={n_alias})  unmatched={len(unmatched)}")
    if not is_real:
        print(f"[dk_adp] WARNING: stand-in. Drop a real DK export at "
              f"{SOURCE_CANDIDATES[0]} (or --source) and re-run.")
    return meta, matched, unmatched


def _resolve_default_source() -> Path | None:
    for c in SOURCE_CANDIDATES:
        if c.exists():
            return c
    return None


def load_source(source: Path) -> dict[str, float]:
    """{norm_name: adp} (no projection matching). Used by tests/fallbacks."""
    return {r["nm"]: r["adp"] for r in _read_rows(source)}


def load_dk_adp(path: Path = DK_ADP_JSON) -> tuple[dict[str, float], dict]:
    """Return ({proj_norm_name: adp}, meta). Auto-builds the stand-in if missing."""
    path = Path(path)
    if not path.exists():
        build()
    blob = json.loads(path.read_text())
    if isinstance(blob, dict) and "adp" in blob:
        return {k: float(v) for k, v in blob["adp"].items()}, blob.get("_meta", {})
    return {k: float(v) for k, v in blob.items()}, {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=None)
    build(ap.parse_args().source)
