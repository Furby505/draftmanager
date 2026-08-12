"""
Draft day recommendation server.
Keep this running during your draft: python server/server.py
Serves on http://localhost:8765
"""

import json
import math
import re
from collections import defaultdict
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

MODELS_DIR = Path(__file__).parent / "models"

# Format-derived replacement ranks: lineup_depth_per_team × 12 teams.
# QB=2 (starter+bye backup), WR=5 (3 starters+FLEX+depth), RB=4, TE=1.
# Matches the sim-derived positional scarcity used in train.py.
REPLACEMENT_RANK = {"QB": 24, "WR": 60, "RB": 48, "TE": 12}


# ── Name normalization ────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    """Lowercase, strip punctuation and common suffixes, collapse whitespace."""
    n = name.lower()
    n = re.sub(r"[.,'\-]", "", n)  # include comma so "Chase, Ja'Marr" → "chase jamarr"
    n = re.sub(r"\s+", " ", n).strip()
    n = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", n)  # strip generational suffixes
    aliases = {
        "hollywood brown": "marquise brown",
        "kenny gainwell": "kenneth gainwell",
    }
    return aliases.get(n, n)


# ── Load projections ──────────────────────────────────────────────────────────

def load_projections() -> tuple[list[dict], dict[str, dict]]:
    path = MODELS_DIR / "projections.json"
    if not path.exists():
        raise RuntimeError(
            "projections.json not found.\n"
            "Restore or regenerate server/models/projections.json before starting the server."
        )
    with open(path) as f:
        players = json.load(f)

    # A projection field can be present but NaN/null — e.g. draft_round/draft_pick/is_rookie
    # for a veteran, or age/prior_pts for a backfilled gap row (written as JSON null by the
    # backfill step). float(None), int(NaN) raise and bool(NaN) is True, which would crash
    # /players and /rank or mislabel veterans as rookies. Such a value is semantically
    # "missing", so drop the key and let .get(key, default) supply the intended default.
    def _missing(v) -> bool:
        return v is None or (isinstance(v, float) and math.isnan(v))
    for p in players:
        for k in [k for k, v in p.items() if _missing(v)]:
            del p[k]

    index: dict[str, dict] = {}

    def put_index(key: str, player: dict) -> None:
        current = index.get(key)
        if current is None or int(player.get("overall_rank", 9999)) < int(current.get("overall_rank", 9999)):
            index[key] = player

    # Build index with normalized names as keys for O(1) lookup. Keep the
    # better-ranked row when a name collides, independent of JSON row order.
    for p in players:
        name = p.get("player_display_name", "")
        if name:
            key = normalize(name)
            put_index(key, p)
            # Also index by last name for sites that show "Last, First"
            parts = key.split()
            if len(parts) > 1:
                last_first = f"{parts[-1]} {' '.join(parts[:-1])}"
                put_index(last_first, p)

    return players, index


try:
    ALL_PLAYERS, PLAYER_INDEX = load_projections()
    proj_season = ALL_PLAYERS[0].get("proj_season", "?") if ALL_PLAYERS else "?"
    print(f"Loaded {len(ALL_PLAYERS)} player projections (projecting {proj_season} season).")
except RuntimeError as e:
    print(f"WARNING: {e}")
    ALL_PLAYERS = []
    PLAYER_INDEX = {}
    proj_season = "?"


# ── ADP (best ball consensus) ─────────────────────────────────────────────────

def load_adp() -> dict[str, float]:
    path = MODELS_DIR / "adp.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    print(f"ADP loaded: {len(data)} players.")
    return data


ADP_INDEX: dict[str, float] = load_adp()


def load_dk_adp() -> tuple[dict[str, float], dict[str, int]]:
    path = MODELS_DIR / "dk_adp.json"
    if not path.exists():
        print("DK ADP not found - DK EV policy will use fallback ADP values.")
        return {}, {}
    with open(path) as f:
        blob = json.load(f)
    data = blob.get("adp", blob) if isinstance(blob, dict) else {}
    adp = {normalize(k): float(v) for k, v in data.items()}
    rank = {name: i + 1 for i, name in enumerate(sorted(adp, key=adp.get))}
    meta = blob.get("_meta", {}) if isinstance(blob, dict) else {}
    label = meta.get("source", "dk_adp.json")
    print(f"DK ADP loaded: {len(adp)} players ({label}).")
    return adp, rank


def load_underdog_adp() -> tuple[dict[str, float], dict[str, int]]:
    path = MODELS_DIR / "underdog_adp.json"
    if not path.exists():
        print("Underdog ADP not found - half-PPR policy will fall back to DK ADP values.")
        return {}, {}
    with open(path) as f:
        blob = json.load(f)
    data = blob.get("adp", blob) if isinstance(blob, dict) else {}
    adp = {normalize(k): float(v) for k, v in data.items()}
    rank = {name: i + 1 for i, name in enumerate(sorted(adp, key=adp.get))}
    meta = blob.get("_meta", {}) if isinstance(blob, dict) else {}
    label = meta.get("source", "underdog_adp.json")
    print(f"Underdog ADP loaded: {len(adp)} players ({label}).")
    return adp, rank


DK_ADP_INDEX, DK_ADP_RANK = load_dk_adp()
UNDERDOG_ADP_INDEX, UNDERDOG_ADP_RANK = load_underdog_adp()

# mtime of dk_adp.json at last load, so a running server can pick up the daily
# 6 AM refresh (training/refresh_dk_adp.py) without a restart. 0.0 = never loaded
# from a real file (e.g. missing), which forces a reload attempt if one appears.
_DK_ADP_PATH = MODELS_DIR / "dk_adp.json"
_UNDERDOG_ADP_PATH = MODELS_DIR / "underdog_adp.json"


def _dk_adp_mtime() -> float:
    try:
        return _DK_ADP_PATH.stat().st_mtime
    except OSError:
        return 0.0


_DK_ADP_MTIME: float = _dk_adp_mtime()
_UNDERDOG_ADP_MTIME: float = 0.0


def _underdog_adp_mtime() -> float:
    try:
        return _UNDERDOG_ADP_PATH.stat().st_mtime
    except OSError:
        return 0.0


_UNDERDOG_ADP_MTIME = _underdog_adp_mtime()


def maybe_reload_dk_adp() -> None:
    """Reload DK ADP if dk_adp.json changed on disk since we last read it.

    Cheap (one stat per call); only re-parses when the file actually changed.
    Lets the daily refresh reach a long-running server without a restart.
    """
    global DK_ADP_INDEX, DK_ADP_RANK, _DK_ADP_MTIME
    mtime = _dk_adp_mtime()
    if mtime and mtime != _DK_ADP_MTIME:
        DK_ADP_INDEX, DK_ADP_RANK = load_dk_adp()
        _DK_ADP_MTIME = mtime
        print(f"DK ADP hot-reloaded (mtime {mtime:.0f}).")


def maybe_reload_underdog_adp() -> None:
    global UNDERDOG_ADP_INDEX, UNDERDOG_ADP_RANK, _UNDERDOG_ADP_MTIME
    mtime = _underdog_adp_mtime()
    if mtime and mtime != _UNDERDOG_ADP_MTIME:
        UNDERDOG_ADP_INDEX, UNDERDOG_ADP_RANK = load_underdog_adp()
        _UNDERDOG_ADP_MTIME = mtime
        print(f"Underdog ADP hot-reloaded (mtime {mtime:.0f}).")


def load_dk_ev_policy_artifact(
    label: str,
    model_filename: str,
    cols_filename: str,
    meta_filename: str,
) -> dict:
    model_path = MODELS_DIR / model_filename
    cols_path = MODELS_DIR / cols_filename
    meta_path = MODELS_DIR / meta_filename
    if not model_path.exists() or not cols_path.exists():
        print(f"No {label} DK EV policy model found - using VOR/boom ranking when selected.")
        return {
            "label": label,
            "model": None,
            "feature_cols": [],
            "meta": {},
            "model_path": model_path,
            "cols_path": cols_path,
            "meta_path": meta_path,
        }
    model = joblib.load(model_path)
    with open(cols_path) as f:
        cols = json.load(f)
    meta: dict = {}
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        print(f"{label} DK EV policy metadata could not be read - assuming legacy candidate depth.")
    print(f"{label} DK EV policy loaded ({len(cols)} features).")
    return {
        "label": label,
        "model": model,
        "feature_cols": cols,
        "meta": meta,
        "model_path": model_path,
        "cols_path": cols_path,
        "meta_path": meta_path,
    }


DK_EV_POLICY_ARTIFACTS = {
    "full": load_dk_ev_policy_artifact(
        "full-PPR",
        "model_dk_ev_policy.joblib",
        "dk_ev_policy_feature_cols.json",
        "dk_ev_policy_meta.json",
    ),
    "half": load_dk_ev_policy_artifact(
        "half-PPR",
        "model_dk_ev_policy_halfppr.joblib",
        "dk_ev_policy_halfppr_feature_cols.json",
        "dk_ev_policy_halfppr_meta.json",
    ),
}

DK_EV_POLICY = DK_EV_POLICY_ARTIFACTS["full"]["model"]
DK_EV_FEATURE_COLS = DK_EV_POLICY_ARTIFACTS["full"]["feature_cols"]
DK_EV_POLICY_META = DK_EV_POLICY_ARTIFACTS["full"]["meta"]


DK_POS = ("QB", "RB", "WR", "TE")
DK_MIN_POS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
DK_COMFORT_POS = {"QB": 2, "RB": 5, "WR": 7, "TE": 2}
DK_HARD_CAP = {"QB": 5, "TE": 5}
DK_REALISM_CAP = {"QB": 4, "TE": 4}
DK_SOFT_CAP = {"QB": 3, "RB": 7, "WR": 8, "TE": 3}
DK_BASE_CANDIDATE_POOL = 12
DK_CANDIDATE_DEPTH_SCHEDULE = (
    (1, 12),
    (3, 14),
    (7, 18),
    (11, 24),
    (15, 32),
)
DK_TRAINED_CANDIDATE_POOL = int(DK_EV_POLICY_META.get("max_candidate_rank", 8) or 8)


def _dk_policy_trained_candidate_pool(artifact: dict | None = None) -> int:
    artifact = artifact or DK_EV_POLICY_ARTIFACTS["full"]
    meta = artifact.get("meta", {}) if isinstance(artifact, dict) else {}
    return int(meta.get("max_candidate_rank", 8) or 8)


def _normalize_scoring(scoring: str | None) -> str:
    val = str(scoring or "full").strip().lower()
    if val in {"", "full", "full_ppr", "full-ppr", "ppr", "1.0", "1ppr"}:
        return "full"
    if val in {"half", "half_ppr", "half-ppr", "0.5ppr", "0.5"}:
        return "half"
    raise ValueError("scoring must be 'full' or 'half'")


def dk_policy_artifact_for_scoring(scoring: str | None) -> dict:
    key = _normalize_scoring(scoring)
    return DK_EV_POLICY_ARTIFACTS.get(key) or DK_EV_POLICY_ARTIFACTS["full"]

# Defense-vs-position matchup tables for the playoff-matchup feature. Self-contained
# loader mirroring outcome_model._load_matchups so train and serve stay in lockstep.
_DATA_PROC = Path(__file__).parent.parent / "data" / "processed"
DK_PLAYOFF_WEEKS = (15, 16, 17)


def _load_dk_matchups() -> tuple[dict, dict]:
    import json as _json
    ratings: dict = {}
    opp: dict[int, dict[str, str]] = {}
    try:
        ratings = _json.loads((_DATA_PROC / "defense_vs_position_2026.json").read_text()).get("ratings", {})
    except Exception:
        ratings = {}
    try:
        sched = _json.loads((_DATA_PROC / "schedule_2026.json").read_text())
        for wk, pairs in sched.items():
            m: dict[str, str] = {}
            for pr in pairs:
                if len(pr) == 2:
                    a, b = str(pr[0]).upper(), str(pr[1]).upper()
                    m[a] = b
                    m[b] = a
            opp[int(wk)] = m
    except Exception:
        opp = {}
    return ratings, opp


DK_MATCHUP_RATINGS, DK_MATCHUP_OPP = _load_dk_matchups()


def _dk_playoff_matchup(team: str, pos: str) -> float:
    """Avg defense-vs-position rating in playoff weeks 15/16/17 (~1.0 neutral, >1 soft).
    Mirrors outcome_model.playoff_matchup_rating."""
    team = str(team or "").upper()
    vals = []
    for w in DK_PLAYOFF_WEEKS:
        o = DK_MATCHUP_OPP.get(w, {}).get(team)
        if o is None:
            continue
        r = DK_MATCHUP_RATINGS.get(o, {}).get(pos)
        if r:
            vals.append(float(r))
    return float(sum(vals) / len(vals)) if vals else 1.0


def _dk_candidate_pool_depth(
    current_pick: int,
    total_teams: int,
    base: int = DK_BASE_CANDIDATE_POOL,
) -> int:
    round_num = ((max(1, int(current_pick)) - 1) // max(1, int(total_teams))) + 1
    base_depth = max(1, int(base))
    depth = base_depth
    for start_round, scheduled_depth in DK_CANDIDATE_DEPTH_SCHEDULE:
        if round_num >= start_round:
            scaled_depth = (
                scheduled_depth * base_depth + DK_BASE_CANDIDATE_POOL - 1
            ) // DK_BASE_CANDIDATE_POOL
            depth = max(depth, scaled_depth)
    return depth


def _dk_effective_candidate_pool_depth(
    current_pick: int,
    total_teams: int,
    artifact: dict | None = None,
) -> int:
    return min(
        _dk_candidate_pool_depth(current_pick, total_teams),
        max(1, _dk_policy_trained_candidate_pool(artifact)),
    )


def _dk_adp(p: dict) -> float:
    raw = DK_ADP_INDEX.get(normalize(p.get("player_display_name", "")), 0.0)
    return float(raw) if raw > 0 else 9999.0


def _active_adp_index() -> dict[str, float]:
    override = _REQUEST_ADP_INDEX.get()
    if override:
        return override
    if current_platform() == "underdog" and UNDERDOG_ADP_INDEX:
        return UNDERDOG_ADP_INDEX
    return DK_ADP_INDEX


def _active_adp_rank() -> dict[str, int]:
    override = _REQUEST_ADP_RANK.get()
    if override:
        return override
    if current_platform() == "underdog" and UNDERDOG_ADP_RANK:
        return UNDERDOG_ADP_RANK
    return DK_ADP_RANK


def _market_adp(p: dict) -> float:
    raw = _active_adp_index().get(normalize(p.get("player_display_name", "")), 0.0)
    return float(raw) if raw > 0 else 9999.0


def _base_platform_adp_index() -> dict[str, float]:
    if current_platform() == "underdog" and UNDERDOG_ADP_INDEX:
        return UNDERDOG_ADP_INDEX
    return DK_ADP_INDEX


def _request_page_adp(req) -> dict[str, float]:
    raw = req.page_adp or {}
    out: dict[str, float] = {}
    base = _base_platform_adp_index()
    if not isinstance(raw, dict):
        return out
    for name, val in raw.items():
        try:
            adp = float(val)
        except (TypeError, ValueError):
            continue
        if not (0.1 <= adp <= 400.0):
            continue
        key = normalize(str(name))
        if not key:
            continue
        fallback = float(base.get(key, 0.0) or 0.0)
        if fallback > 0:
            # Protect against row-rank/slot numbers being misread as ADP.
            # Real ADP can move, but not from WR1 range to pick 1.0 between scans.
            if fallback <= 24:
                max_abs_delta = 4.0
            elif fallback <= 120:
                max_abs_delta = 18.0
            else:
                max_abs_delta = 36.0
            max_ratio = 2.25
            ratio = max(adp, fallback) / max(0.1, min(adp, fallback))
            if abs(adp - fallback) > max_abs_delta or ratio > max_ratio:
                continue
        out[key] = adp
    return out


def _install_request_adp(req):
    """Prefer live page ADP for scraped players, fallback to platform board."""
    page = _request_page_adp(req)
    base = _base_platform_adp_index()
    if page:
        merged = dict(base)
        merged.update(page)
        rank = {name: i + 1 for i, name in enumerate(sorted(merged, key=merged.get))}
        source = f"page+{current_platform()}_adp"
        return (
            _REQUEST_ADP_INDEX.set(merged),
            _REQUEST_ADP_RANK.set(rank),
            _REQUEST_ADP_SOURCE.set(source),
            len(page),
        )
    source = "underdog_adp.json" if current_platform() == "underdog" else "dk_adp.json"
    return (
        _REQUEST_ADP_INDEX.set(None),
        _REQUEST_ADP_RANK.set(None),
        _REQUEST_ADP_SOURCE.set(source),
        0,
    )


def _reset_request_adp(tokens) -> None:
    index_token, rank_token, source_token, _ = tokens
    _REQUEST_ADP_INDEX.reset(index_token)
    _REQUEST_ADP_RANK.reset(rank_token)
    _REQUEST_ADP_SOURCE.reset(source_token)


def _dk_market_tier(adp: float) -> int:
    if adp <= 36:
        return 0
    if adp <= 96:
        return 1
    if adp <= 168:
        return 2
    return 3


def _dk_same_pos_replacement_gap(
    candidate: dict,
    available: list[dict],
    candidate_name: str | None = None,
) -> float:
    cand_pos = candidate.get("position", "")
    cand_adp = _market_adp(candidate)
    cand_name = candidate_name or normalize(candidate.get("player_display_name", ""))
    same = sorted(
        (
            (_market_adp(p), normalize(p.get("player_display_name", "")))
            for p in available
            if p.get("position") == cand_pos
        ),
        key=lambda x: (x[0], x[1]),
    )
    for i, (_, name) in enumerate(same):
        if name == cand_name:
            if i + 1 >= len(same):
                return 9999.0
            return float(same[i + 1][0] - cand_adp)
    return 9999.0


def _dk_under_cap(counts: dict[str, int], player: dict, caps: dict[str, int]) -> bool:
    pos = player.get("position", "")
    cap = caps.get(pos)
    return cap is None or counts.get(pos, 0) < cap


def _dk_live_candidate_ranks(
    available: list[dict],
    counts: dict[str, int],
    picks_left: int,
) -> dict[str, int]:
    """Live candidate set for DK EV scoring.

    Hard caps always apply, and final picks are forced toward unfilled minimums.
    Soft construction caps are intentionally not hard-filtered here: the policy
    already sees roster counts/needs as features, and live drafting can still
    want a 9th WR or 8th RB when the board value is right.
    """
    legal = [p for p in available if _dk_under_cap(counts, p, DK_HARD_CAP)]

    unmet = {
        pos: max(0, DK_MIN_POS[pos] - counts.get(pos, 0))
        for pos in DK_POS
    }
    if picks_left <= sum(unmet.values()) and sum(unmet.values()) > 0:
        needed = {pos for pos, need in unmet.items() if need > 0}
        legal = [p for p in legal if p.get("position", "") in needed]

    return {
        normalize(p.get("player_display_name", "")): i + 1
        for i, p in enumerate(sorted(legal, key=_market_adp))
    }


def _snake_slot(global_pick_zero: int, total_teams: int) -> int:
    rnd, slot = divmod(global_pick_zero, total_teams)
    return slot + 1 if rnd % 2 == 0 else total_teams - slot


def _next_pick_window(current_pick: int, total_teams: int, my_pick_position: int, total_rounds: int) -> list[int]:
    total_picks = total_teams * total_rounds
    window: list[int] = []
    for pick in range(current_pick + 1, total_picks + 1):
        if _snake_slot(pick - 1, total_teams) == my_pick_position:
            break
        window.append(pick)
    return window


def _team_counts_from_request(req) -> dict[int, dict[str, int]]:
    counts = {
        slot: {pos: 0 for pos in DK_POS}
        for slot in range(1, req.total_teams + 1)
    }
    if req.team_rosters:
        for slot in range(1, req.total_teams + 1):
            picks = req.team_rosters.get(slot, req.team_rosters.get(str(slot), []))
            for p in picks:
                if not isinstance(p, dict):
                    continue
                pos = str(p.get("position", "")).upper()
                if pos in DK_POS:
                    counts[slot][pos] += 1
        return counts

    for i, pick in enumerate(req.drafted_players):
        slot = _snake_slot(i, req.total_teams)
        pos = pick.position.upper()
        if pos in DK_POS:
            counts[slot][pos] += 1
    return counts


def _dk_opponent_window_features(req) -> dict:
    window = _next_pick_window(req.current_pick, req.total_teams, req.my_pick_position, req.total_rounds)
    slots = sorted({_snake_slot(p - 1, req.total_teams) for p in window})
    counts_by_slot = _team_counts_from_request(req)

    min_need = {pos: 0 for pos in DK_POS}
    comfort_need = {pos: 0 for pos in DK_POS}
    for slot in slots:
        if slot == req.my_pick_position:
            continue
        counts = counts_by_slot.get(slot, {})
        for pos in DK_POS:
            if counts.get(pos, 0) < DK_MIN_POS[pos]:
                min_need[pos] += 1
            if counts.get(pos, 0) < DK_COMFORT_POS[pos]:
                comfort_need[pos] += 1

    recent6 = {pos: 0 for pos in DK_POS}
    recent12 = {pos: 0 for pos in DK_POS}
    for pick in req.drafted_players[-6:]:
        pos = pick.position.upper()
        if pos in recent6:
            recent6[pos] += 1
    for pick in req.drafted_players[-12:]:
        pos = pick.position.upper()
        if pos in recent12:
            recent12[pos] += 1

    return {
        "picks_until_next_pick": len(window),
        "teams_until_next_pick": len(slots),
        "opp_min_need_QB_before_next": min_need["QB"],
        "opp_min_need_RB_before_next": min_need["RB"],
        "opp_min_need_WR_before_next": min_need["WR"],
        "opp_min_need_TE_before_next": min_need["TE"],
        "opp_comfort_need_QB_before_next": comfort_need["QB"],
        "opp_comfort_need_RB_before_next": comfort_need["RB"],
        "opp_comfort_need_WR_before_next": comfort_need["WR"],
        "opp_comfort_need_TE_before_next": comfort_need["TE"],
        "last6_qb_taken": recent6["QB"],
        "last6_rb_taken": recent6["RB"],
        "last6_wr_taken": recent6["WR"],
        "last6_te_taken": recent6["TE"],
        "last12_qb_taken": recent12["QB"],
        "last12_rb_taken": recent12["RB"],
        "last12_wr_taken": recent12["WR"],
        "last12_te_taken": recent12["TE"],
    }


def picks_until_next_turn(current_pick: int, total_teams: int, my_pick_position: int) -> int:
    """
    Compute how many picks happen before the user's next turn (including current pick if it's an opponent's).
    current_pick is 1-indexed (the NEXT pick to be made on the board).
    """
    if total_teams <= 1:
        return 0
    round_idx    = (current_pick - 1) // total_teams   # 0-indexed
    pos_in_round = (current_pick - 1) % total_teams    # 0-indexed
    my_pos = (my_pick_position - 1) if round_idx % 2 == 0 else (total_teams - my_pick_position)

    if my_pos == pos_in_round:
        return 0  # it's my pick right now
    if my_pos > pos_in_round:
        return my_pos - pos_in_round  # still coming this round
    # My pick already passed — find it in the next round
    next_round  = round_idx + 1
    next_my_pos = (my_pick_position - 1) if next_round % 2 == 0 else (total_teams - my_pick_position)
    return (total_teams - pos_in_round - 1) + next_my_pos + 1



# ── Platform config ───────────────────────────────────────────────────────────

SCORING_RULES = {
    "draftkings": {
        "name": "DraftKings Best Ball",
        "reception": 1.0, "pass_bonus_300": 3.0,
        "rush_bonus_100": 3.0, "rec_bonus_100": 3.0,
        "int": -1.0, "fumble_lost": -1.0,
    },
    "underdog": {
        "name": "Underdog Best Ball",
        "reception": 0.5, "pass_bonus_300": 0.0,
        "rush_bonus_100": 0.0, "rec_bonus_100": 0.0,
        "int": -1.0, "fumble_lost": -1.0,
    },
}

_platform_store: dict = {"platform": "draftkings"}
_REQUEST_ADP_INDEX: ContextVar[dict[str, float] | None] = ContextVar("request_adp_index", default=None)
_REQUEST_ADP_RANK: ContextVar[dict[str, int] | None] = ContextVar("request_adp_rank", default=None)
_REQUEST_ADP_SOURCE: ContextVar[str] = ContextVar("request_adp_source", default="")


def current_platform() -> str:
    return _platform_store["platform"]


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="DraftManager", version="1.0")

# The server binds to 127.0.0.1, but a wildcard CORS policy would still let any
# site you happen to have open in the same browser read from it. The content
# script calls /rank from the draft page's own origin, so allow exactly the
# origins in extension/manifest.json plus the extension itself. Keep this list
# in sync with the manifest's host_permissions.
CORS_ORIGIN_REGEX = (
    r"^(chrome-extension://[a-p]{32}"
    r"|https?://(localhost|127\.0\.0\.1)(:\d+)?"
    r"|https?://([a-z0-9-]+\.)*(draftkings|underdogfantasy|playunderdog)\.com)$"
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────

class DraftedPlayer(BaseModel):
    name: str
    position: str
    round: int = Field(ge=1)
    by_user: bool = False


class RankRequest(BaseModel):
    available_players: list[str] = Field(default_factory=list)
    drafted_players: list[DraftedPlayer] = Field(default_factory=list)   # all drafted (by anyone)
    current_pick: int = Field(default=1, ge=1)
    total_teams: int = Field(default=12, ge=2, le=20)
    total_rounds: int = Field(default=20, ge=1, le=30)
    my_pick_position: int = Field(default=1, ge=1)
    scoring: str = "full"
    use_dk_ev_policy: bool = True
    # Optional: actual tracked team rosters from the extension.
    # When provided, used directly for demand modeling instead of inferring from pick order.
    # Format: {"1": [{"name": "Josh Allen", "position": "QB", "round": 1}], ...}
    team_rosters: Optional[dict] = None
    # Optional live ADP scraped from the draft room itself. Keys may be display
    # names or already-normalized names; values are current platform ADP/rank.
    page_adp: Optional[dict[str, float]] = None

    @model_validator(mode="after")
    def validate_draft_bounds(self):
        if self.my_pick_position > self.total_teams:
            raise ValueError("my_pick_position cannot exceed total_teams")
        max_next_pick = self.total_teams * self.total_rounds + 1
        if self.current_pick > max_next_pick:
            raise ValueError("current_pick cannot exceed total_teams * total_rounds + 1")
        self.scoring = _normalize_scoring(self.scoring)
        return self


class PlayerRec(BaseModel):
    name: str
    position: str
    team: str
    proj_points: float
    proj_boom_rate: float
    vor: float
    age: float
    carry_share: float = 0.0
    value_tier: str
    is_rookie: bool = False
    draft_round: int = 0
    draft_pick: int = 0
    bye_week: int = 0
    bye_alert: str = ""
    adp: float = 0.0
    going_soon: bool = False
    is_stack: bool = False
    stack_qb: str = ""    # QB name this player stacks with
    overall_rank: int = 0  # model's pre-draft overall rank
    prior_pts: float = 0.0       # prior season actuals (for trajectory display)
    prior_boom_rate: float = 0.0
    dk_ev_score: float = 0.0
    dk_ev_rank: int = 0
    dk_target_round: str = ""
    ranking_source: str = "legacy"
    dk_ev_margin_to_top: float = 0.0   # EV gap to the #1 candidate (0 for the leader)
    dk_ev_margin_to_next: float = 0.0  # EV gap to the next-ranked candidate


class RankResponse(BaseModel):
    recommendations: list[PlayerRec]
    stack_alerts: list[str]
    scarcity_warnings: list[str]
    total_available: int
    proj_season: str
    scoring: str = "full"
    dk_policy_label: str = ""
    dk_policy_model: str = ""
    dk_policy_max_candidate_rank: int = 0
    adp_source: str = ""
    page_adp_count: int = 0


# ── Matching ──────────────────────────────────────────────────────────────────

def match_player(name: str) -> Optional[dict]:
    if not name.strip():
        return None

    n = normalize(name)

    # 1. Exact normalized match
    if n in PLAYER_INDEX:
        return PLAYER_INDEX[n]

    # 2. "Last, First" → "First Last" swap (some sites format this way)
    if "," in name:
        parts = name.split(",", 1)
        swapped = normalize(f"{parts[1].strip()} {parts[0].strip()}")
        if swapped in PLAYER_INDEX:
            return PLAYER_INDEX[swapped]

    # 3. Substring match on full normalized name (handles suffixes like Jr./III)
    candidates = [
        p for key, p in PLAYER_INDEX.items()
        if n in key or key in n
    ]
    if len(candidates) == 1:
        return candidates[0]

    # 4. First-initial match: "d henry" → any "d* henry" in index (handles "D. Henry")
    parts = n.split()
    if len(parts) == 2 and len(parts[0]) == 1:
        initial, last = parts[0], parts[1]
        if len(last) >= 4:
            matches = [p for key, p in PLAYER_INDEX.items()
                       if key.startswith(initial) and key.endswith(f" {last}")]
            if len(matches) == 1:
                return matches[0]

    # 5. Last-name-only match (only if unambiguous)
    parts = n.split()
    if parts:
        last = parts[-1]
        if len(last) >= 4:  # skip short/common last names
            matches = [p for key, p in PLAYER_INDEX.items() if key.endswith(f" {last}") or key == last]
            if len(matches) == 1:
                return matches[0]

    return None


# ── Ranking helpers ───────────────────────────────────────────────────────────

def compute_dynamic_vor(available: list[dict]) -> dict[str, float]:
    by_pos: dict[str, list[float]] = {}
    for p in available:
        by_pos.setdefault(p["position"], []).append(float(p.get("proj_points", 0)))

    replacement: dict[str, float] = {}
    for pos, pts_list in by_pos.items():
        pts_list.sort(reverse=True)
        rank = REPLACEMENT_RANK.get(pos, 12)
        idx = min(rank - 1, len(pts_list) - 1)
        replacement[pos] = pts_list[idx]
    return replacement


def value_tier(vor: float, pos: str) -> str:
    thresholds = {
        "QB": (30, 10, -5),
        "WR": (40, 15, -5),
        "RB": (35, 12, -5),
        "TE": (30, 10, -5),
    }
    steal_t, value_t, reach_t = thresholds.get(pos, (30, 10, -5))
    if vor >= steal_t:  return "steal"
    if vor >= value_t:  return "value"
    if vor >= reach_t:  return "fair"
    return "reach"


def infer_opponent_demand(
    drafted_players: list,
    total_teams: int,
    my_pick_position: int,
) -> dict[str, int]:
    """
    Returns how many opponent teams currently lack each position (below best-ball minimums).
    Uses snake-draft math to infer which team made each pick.
    """
    MINIMUMS = {"QB": 2, "TE": 1, "RB": 3, "WR": 4}
    team_rosters: dict[int, dict] = {i: defaultdict(int) for i in range(1, total_teams + 1)}

    for i, pick in enumerate(drafted_players):
        overall_pick  = i + 1
        round_idx     = (overall_pick - 1) // total_teams   # 0-indexed
        pos_in_round  = (overall_pick - 1) % total_teams    # 0-indexed
        team_slot = (pos_in_round + 1) if round_idx % 2 == 0 else (total_teams - pos_in_round)
        if team_slot != my_pick_position:
            team_rosters[team_slot][pick.position] += 1

    demand: dict[str, int] = {}
    for pos, minimum in MINIMUMS.items():
        demand[pos] = sum(
            1 for slot, roster in team_rosters.items()
            if slot != my_pick_position and roster.get(pos, 0) < minimum
        )
    return demand


def demand_from_actual_rosters(
    team_rosters: dict,
    my_pick_position: int,
    total_teams: int = 12,
) -> dict[str, int]:
    """
    Compute opponent demand from actual tracked rosters (extension-provided).
    Teams not yet in team_rosters (no picks yet) count as having 0 of each position
    (maximum demand) — this correctly handles early rounds where most teams lack picks.
    """
    MINIMUMS = {"QB": 2, "TE": 1, "RB": 3, "WR": 4}
    demand: dict[str, int] = {}
    for pos, minimum in MINIMUMS.items():
        count = 0
        for slot in range(1, total_teams + 1):
            if slot == my_pick_position:
                continue
            # Look up this team's picks (keys may be int or str in the dict)
            picks = team_rosters.get(slot, team_rosters.get(str(slot), []))
            pos_count = sum(
                1 for p in picks
                if isinstance(p, dict) and str(p.get("position", "")).upper() == pos
            )
            if pos_count < minimum:
                count += 1
        demand[pos] = count
    return demand


def _safe_round(value, default: int) -> int:
    try:
        return max(1, int(value or default))
    except (TypeError, ValueError):
        return default


def my_roster_entries_from_request(req: RankRequest) -> list[dict]:
    if req.team_rosters:
        picks = req.team_rosters.get(req.my_pick_position, req.team_rosters.get(str(req.my_pick_position), None))
        if picks is not None:
            return [p for p in picks if isinstance(p, dict)]
    return [
        {"name": d.name, "position": d.position, "round": d.round}
        for d in req.drafted_players
        if d.by_user
    ]


def my_pick_data_from_request(req: RankRequest) -> list[dict]:
    out = []
    for pick in my_roster_entries_from_request(req):
        player = match_player(str(pick.get("name", "")))
        if player:
            out.append(player)
    return out


def my_drafted_players_from_request(req: RankRequest) -> list[DraftedPlayer]:
    return [
        DraftedPlayer(
            name=str(p.get("name", "")),
            position=str(p.get("position", "")),
            round=_safe_round(p.get("round"), i + 1),
            by_user=True,
        )
        for i, p in enumerate(my_roster_entries_from_request(req))
        if p.get("name")
    ]


def drafted_name_set_from_request(req: RankRequest) -> set[str]:
    names = {normalize(d.name) for d in req.drafted_players if d.name}
    if req.team_rosters:
        for picks in req.team_rosters.values():
            if not isinstance(picks, list):
                continue
            for p in picks:
                if isinstance(p, dict) and p.get("name"):
                    names.add(normalize(str(p["name"])))
    return names


def apply_going_soon(
    recs: list,
    current_pick: int,
    total_teams: int,
    my_pick_pos: int,
    demand: dict[str, int] | None = None,
) -> list:
    """
    Flag players whose ADP suggests they'll be taken before the user's next pick.
    Demand-adjusts the threshold: if many opponents need a position, flag it sooner.
    Caps at 5 "going soon" flags total so it stays actionable (not noise).
    """
    if not ADP_INDEX or current_pick <= 0:
        return recs
    gap = picks_until_next_turn(current_pick, total_teams, my_pick_pos)
    opponents = max(total_teams - 1, 1)

    # Score urgency: lower adp_gap = more urgent
    urgency_scores: list[tuple[float, int]] = []  # (adp_gap, index)
    updated = list(recs)
    for i, r in enumerate(updated):
        if r.adp > 0 and r.adp > current_pick:
            pos_demand    = (demand or {}).get(r.position, 0)
            demand_factor = pos_demand / opponents
            threshold     = current_pick + gap * (1.2 + 1.0 * demand_factor)
            if r.adp <= threshold:
                adp_gap = r.adp - current_pick
                urgency_scores.append((adp_gap, i))

    # Keep only the 5 most urgent picks (smallest gap to draft window)
    urgency_scores.sort(key=lambda x: x[0])
    flagged_idxs = {i for _, i in urgency_scores[:5]}

    return [
        r.model_copy(update={"going_soon": True}) if i in flagged_idxs else r
        for i, r in enumerate(updated)
    ]


def apply_bye_penalties(recs: list, my_picks: list[dict]) -> list:
    """
    Penalize players whose bye week creates a dead spot.
    - Two QBs on the same bye = guaranteed 0-QB week → big penalty
    - Two TEs on the same bye = same problem → moderate penalty
    Moves affected players down in the ranking and sets bye_alert.
    """
    my_qbs = [p for p in my_picks if p.get("position") == "QB"]
    my_tes = [p for p in my_picks if p.get("position") == "TE"]
    my_qb_byes = {int(p.get("bye_week", 0)) for p in my_qbs if p.get("bye_week")}
    my_te_byes = {int(p.get("bye_week", 0)) for p in my_tes if p.get("bye_week")}

    # No existing QBs/TEs → nothing to penalize
    if not my_qb_byes and not my_te_byes:
        return recs

    scored: list[tuple[float, PlayerRec]] = []
    for i, r in enumerate(recs):
        penalty   = 0.0
        bye_alert = ""
        bw        = r.bye_week

        if r.position == "QB" and bw and bw in my_qb_byes:
            penalty   = 40.0   # lose a QB week — severe
            bye_alert = f"Bye overlap Wk{bw} with your other QB"
        elif r.position == "TE" and bw and bw in my_te_byes:
            penalty   = 25.0   # lose a TE week — significant
            bye_alert = f"Bye overlap Wk{bw} with your other TE"

        # Use rank position as score base so the sort is stable
        score = -(i + penalty)
        if bye_alert:
            r = r.model_copy(update={"bye_alert": bye_alert})
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored]


def detect_stacks(my_players: list[DraftedPlayer], available: list[dict]) -> list[str]:
    alerts = []
    my_qbs  = [p for p in my_players if p.position.upper() == "QB" and p.by_user]
    my_data = [match_player(p.name) for p in my_players if p.by_user]
    my_data = [d for d in my_data if d]

    for qb in my_qbs:
        qb_data = match_player(qb.name)
        if not qb_data:
            continue
        qb_team = qb_data.get("recent_team", "")
        if not qb_team:
            continue

        # Count pass catchers already drafted by the user from this QB's team
        already_stacked = sum(
            1 for p in my_data
            if p.get("recent_team") == qb_team and p.get("position") in ("WR", "TE")
        )

        teammates = sorted(
            [p for p in available if p.get("recent_team") == qb_team and p.get("position") in ("WR", "TE")],
            key=lambda x: float(x.get("proj_points", 0)),
            reverse=True,
        )

        if not teammates:
            continue

        best = teammates[0]
        depth_label = "full stack" if already_stacked >= 1 else "mini stack"
        alerts.append(
            f"Stack: {best['player_display_name']} ({best['position']}, {best.get('recent_team','')}) + "
            f"{qb.name} = {depth_label} ({already_stacked} WR/TE already stacked)"
        )
    return alerts


def detect_scarcity(available: list[dict], total_teams: int) -> list[str]:
    # Count only players above replacement level (VOR > -30 pts, i.e., not pure streamers)
    repl_pts = compute_dynamic_vor(available)
    by_pos: dict[str, int] = {}
    for p in available:
        pos  = p.get("position", "")
        proj = float(p.get("proj_points", 0))
        repl = repl_pts.get(pos, 0)
        if proj >= repl - 30:  # include players within 30 pts of replacement
            by_pos[pos] = by_pos.get(pos, 0) + 1

    warnings_out = []
    # Danger thresholds: how many startable players at each pos should remain?
    danger = {
        "TE": total_teams,
        "QB": total_teams,
        "RB": int(total_teams * 1.5),
        "WR": total_teams * 2,
    }
    for pos, threshold in danger.items():
        count = by_pos.get(pos, 0)
        if count == 0:
            warnings_out.append(f"No startable {pos}s left!")
        elif count <= threshold // 2:
            warnings_out.append(f"CRITICAL — only {count} {pos}s worth drafting.")
        elif count <= threshold:
            warnings_out.append(f"Warning — {count} {pos}s left. Scarcity approaching.")
    return warnings_out


def _effective_adp(p: dict) -> float:
    """Raw ADP when available; 0 means no market ADP match."""
    raw = float(ADP_INDEX.get(normalize(p.get("player_display_name", "")), 0))
    if raw > 0:
        return round(raw, 1)
    return 0.0


def _display_adp(p: dict) -> float:
    if current_platform() in {"draftkings", "underdog"}:
        raw = _market_adp(p)
        return round(raw, 1) if raw < 9999.0 else 0.0
    return _effective_adp(p)


def _dk_policy_feature_row(
    req: RankRequest,
    rec: PlayerRec,
    player: dict,
    available: list[dict],
    my_pick_data: list[dict],
) -> dict:
    counts = {pos: 0 for pos in DK_POS}
    for p in my_pick_data:
        pos = p.get("position", "")
        if pos in counts:
            counts[pos] += 1

    cand_pos = rec.position
    cand_team = rec.team
    cand_name = normalize(rec.name)
    cand_adp = _market_adp(player)
    cand_bye = int(player.get("bye_week", 0) or 0)
    cand_tier = _dk_market_tier(cand_adp)

    qbs = {p.get("recent_team", "") for p in my_pick_data if p.get("position") == "QB"}
    pass_catcher_teams = {
        p.get("recent_team", "")
        for p in my_pick_data
        if p.get("position") in ("WR", "TE")
    }
    same_team_roster = [
        p for p in my_pick_data
        if cand_team and p.get("recent_team") == cand_team
    ]
    # Handcuff signal: same-team same-position players already on my roster (mirrors
    # train_policy.state_candidate_features["cand_handcuff_depth"]).
    same_team_same_pos = [
        p for p in my_pick_data
        if cand_team and p.get("recent_team") == cand_team and p.get("position") == cand_pos
    ]
    same_team_qbs = [
        p for p in my_pick_data
        if cand_team and p.get("position") == "QB" and p.get("recent_team") == cand_team
    ]
    same_team_catchers = [
        p for p in my_pick_data
        if cand_team and p.get("position") in ("WR", "TE") and p.get("recent_team") == cand_team
    ]
    bye_overlap = sum(
        1 for p in my_pick_data
        if cand_bye and int(p.get("bye_week", 0) or 0) == cand_bye
    )
    pos_bye_overlap = sum(
        1 for p in my_pick_data
        if cand_bye
        and p.get("position") == cand_pos
        and int(p.get("bye_week", 0) or 0) == cand_bye
    )

    by_pos: dict[str, list[float]] = {pos: [] for pos in DK_POS}
    tier_left = 0
    tier_pos_left = 0
    for p in available:
        pos = p.get("position", "")
        if pos not in by_pos:
            continue
        adp = _market_adp(p)
        by_pos[pos].append(adp)
        tier = _dk_market_tier(adp)
        if tier == cand_tier:
            tier_left += 1
            if pos == cand_pos:
                tier_pos_left += 1

    left_by_pos = {pos: len(vals) for pos, vals in by_pos.items()}
    best_by_pos = {pos: min(vals) if vals else 9999.0 for pos, vals in by_pos.items()}

    next_best_gap = _dk_same_pos_replacement_gap(player, available, cand_name)

    needs = {
        pos: max(0, DK_MIN_POS[pos] - counts.get(pos, 0))
        for pos in DK_POS
    }
    to_comfort = {
        pos: max(0, DK_COMFORT_POS[pos] - counts.get(pos, 0))
        for pos in DK_POS
    }
    opp = _dk_opponent_window_features(req)
    cand_need_key = f"opp_min_need_{cand_pos}_before_next"
    cand_comfort_key = f"opp_comfort_need_{cand_pos}_before_next"
    pick_window_adp_end = float(req.current_pick + opp["picks_until_next_pick"])
    overall_before_next = 0
    same_before_next = 0
    for p in available:
        adp = _market_adp(p)
        if adp <= pick_window_adp_end:
            overall_before_next += 1
            if p.get("position") == cand_pos:
                same_before_next += 1

    stack_depth_after = 0
    if cand_pos in ("WR", "TE"):
        stack_depth_after = len(same_team_qbs)
    elif cand_pos == "QB":
        stack_depth_after = len(same_team_catchers)

    rnd = (req.current_pick - 1) // max(req.total_teams, 1) + 1
    slot = (req.current_pick - 1) % max(req.total_teams, 1) + 1
    return {
        "pick_no": req.current_pick,
        "round": rnd,
        "slot_in_round": slot,
        "picks_made_by_team": len(my_pick_data),
        "picks_left": max(0, req.total_rounds - len(my_pick_data) - 1),
        "n_QB": counts["QB"],
        "n_RB": counts["RB"],
        "n_WR": counts["WR"],
        "n_TE": counts["TE"],
        "need_QB": needs["QB"],
        "need_RB": needs["RB"],
        "need_WR": needs["WR"],
        "need_TE": needs["TE"],
        "cand_adp": cand_adp,
        "cand_adp_minus_pick": cand_adp - req.current_pick,
        "cand_board_rank": _active_adp_rank().get(cand_name, 9999),
        "cand_has_real_adp": int(cand_adp < 9999.0),
        "cand_bye_week": cand_bye,
        "cand_market_tier": cand_tier,
        "cand_pos_QB": int(cand_pos == "QB"),
        "cand_pos_RB": int(cand_pos == "RB"),
        "cand_pos_WR": int(cand_pos == "WR"),
        "cand_pos_TE": int(cand_pos == "TE"),
        "cand_stack": int(cand_pos in ("WR", "TE") and cand_team in qbs),
        "cand_bringback": int(cand_pos == "QB" and cand_team in pass_catcher_teams),
        "cand_same_team_roster_count": len(same_team_roster),
        "cand_handcuff_depth": len(same_team_same_pos),
        "cand_playoff_matchup": _dk_playoff_matchup(cand_team, cand_pos),
        "cand_stack_depth_after_pick": stack_depth_after,
        "cand_bye_overlap_count": bye_overlap,
        "qb_bye_overlap": int(cand_pos == "QB" and pos_bye_overlap > 0),
        "te_bye_overlap": int(cand_pos == "TE" and pos_bye_overlap > 0),
        "rb_to_comfort": to_comfort["RB"],
        "wr_to_comfort": to_comfort["WR"],
        "qb_to_comfort": to_comfort["QB"],
        "te_to_comfort": to_comfort["TE"],
        "roster_wr_rb_ratio": counts["WR"] / max(counts["RB"], 1),
        "board_qb_left": left_by_pos["QB"],
        "board_rb_left": left_by_pos["RB"],
        "board_wr_left": left_by_pos["WR"],
        "board_te_left": left_by_pos["TE"],
        "best_qb_adp_left": best_by_pos["QB"],
        "best_rb_adp_left": best_by_pos["RB"],
        "best_wr_adp_left": best_by_pos["WR"],
        "best_te_adp_left": best_by_pos["TE"],
        **opp,
        "cand_pos_min_need_before_next": opp.get(cand_need_key, 0),
        "cand_pos_comfort_need_before_next": opp.get(cand_comfort_key, 0),
        "cand_pos_last6_taken": opp.get(f"last6_{cand_pos.lower()}_taken", 0),
        "cand_pos_last12_taken": opp.get(f"last12_{cand_pos.lower()}_taken", 0),
        "cand_pos_next_best_adp_gap": next_best_gap,
        "pick_window_adp_end": pick_window_adp_end,
        "cand_adp_minus_pick_window": cand_adp - pick_window_adp_end,
        "overall_players_adp_before_next": overall_before_next,
        "cand_pos_players_adp_before_next": same_before_next,
        "cand_tier_players_left": tier_left,
        "cand_tier_pos_players_left": tier_pos_left,
    }


def rank_with_dk_ev_policy(
    req: RankRequest,
    recs: list[PlayerRec],
    available: list[dict],
    my_pick_data: list[dict],
) -> list[PlayerRec] | None:
    artifact = dk_policy_artifact_for_scoring(req.scoring)
    model = artifact.get("model")
    feature_cols = artifact.get("feature_cols", [])
    if model is None or not feature_cols or not _active_adp_index():
        return None

    by_name = {
        normalize(p.get("player_display_name", "")): p
        for p in available
    }
    own_counts = {pos: 0 for pos in DK_POS}
    for p in my_pick_data:
        pos = p.get("position", "")
        if pos in own_counts:
            own_counts[pos] += 1
    picks_left = max(0, req.total_rounds - len(my_pick_data))
    live_rank_by_name = _dk_live_candidate_ranks(available, own_counts, picks_left)
    rows = []
    hard_illegal: list[bool] = []
    outside_candidate_pool: list[bool] = []
    live_ranks: list[int] = []
    candidate_pool_depth = _dk_effective_candidate_pool_depth(
        req.current_pick,
        req.total_teams,
        artifact,
    )
    for rec in recs:
        player = by_name.get(normalize(rec.name))
        if not player:
            rows.append({})
            hard_illegal.append(True)
            outside_candidate_pool.append(True)
            live_ranks.append(9999)
            continue
        rows.append(_dk_policy_feature_row(req, rec, player, available, my_pick_data))
        hard_cap = DK_HARD_CAP.get(rec.position)
        count = sum(1 for p in my_pick_data if p.get("position") == rec.position)
        hard_illegal.append(bool(hard_cap is not None and count + 1 > hard_cap))
        live_rank = live_rank_by_name.get(normalize(rec.name), 9999)
        live_ranks.append(live_rank)
        outside_candidate_pool.append(live_rank > candidate_pool_depth)

    X = pd.DataFrame(rows)
    for col in feature_cols:
        if col not in X:
            X[col] = 0
    scores = np.asarray(model.predict(X[feature_cols].fillna(0)), dtype=float)
    for i, illegal in enumerate(hard_illegal):
        if illegal:
            scores[i] = -999.0
        elif outside_candidate_pool[i]:
            # Keep very deep names visible, but do not let the model extrapolate
            # past the smaller of the round-aware pool and the artifact's
            # trained candidate depth; below that, fall back to live market order.
            scores[i] = -0.01 - (live_ranks[i] * 0.000001)

    order = np.argsort(-scores, kind="stable")
    ranked = []
    for rank, idx in enumerate(order, start=1):
        rec = recs[int(idx)].model_copy(update={
            "dk_ev_score": round(float(scores[int(idx)]), 8),
            "dk_ev_rank": rank,
            "dk_target_round": f"R{((req.current_pick - 1) // max(req.total_teams, 1)) + 1}",
            "ranking_source": "dk_ev_policy",
        })
        ranked.append(rec)
    return ranked


def _annotate_dk_order(recs: list[PlayerRec]) -> list[PlayerRec]:
    if not recs:
        return recs

    top_score = recs[0].dk_ev_score
    annotated: list[PlayerRec] = []
    for i, r in enumerate(recs):
        m_next = (
            round(r.dk_ev_score - recs[i + 1].dk_ev_score, 8)
            if i + 1 < len(recs) else 0.0
        )
        annotated.append(r.model_copy(update={
            "dk_ev_rank": i + 1,
            "dk_ev_margin_to_top": round(top_score - r.dk_ev_score, 8),
            "dk_ev_margin_to_next": m_next,
        }))
    return annotated


def apply_dk_ev_tiebreaker(
    req: RankRequest,
    recs: list[PlayerRec],
    my_pick_data: list[dict],
) -> list[PlayerRec]:
    """Annotate EV margins for transparency; DO NOT rerank.

    A paired backtest (training/paired_tiebreaker.py, 2026-06-17) measured the
    indifference-band reranker against the bare model in the same seats: it is neutral at
    the tightest band (1e-6: +0.08%, not significant) and significantly HARMFUL as the
    band widens (-1.5% at 5e-6, -2.2% at 1e-5, -3.9% at 2e-5, -8.8% at the original
    4.5e-5 band). The model already trains on survival, roster need, stack, and bye
    features, so a cruder hand-weighted rerank can only match it or fight it. So we keep
    the bare-model order and expose only the EV margins (dk_ev_margin_to_top/_to_next).
    """
    if not recs or recs[0].ranking_source != "dk_ev_policy":
        return recs
    return _annotate_dk_order(recs)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "players_loaded": len(ALL_PLAYERS),
        "proj_season": proj_season,
        "platform": current_platform(),
    }


@app.get("/platform")
def get_platform():
    """Return current scoring platform and its rules."""
    plat = current_platform()
    return {"platform": plat, "rules": SCORING_RULES.get(plat, {})}


@app.post("/platform")
def set_platform(body: dict):
    """Switch scoring platform. Body: {"platform": "draftkings"|"underdog"}"""
    plat = body.get("platform", "").lower()
    if plat not in SCORING_RULES:
        from fastapi import HTTPException
        raise HTTPException(
            400,
            f"Unsupported platform: {plat}",
        )
    _platform_store["platform"] = plat
    print(f"Platform switched to: {plat}")
    return {"platform": plat, "rules": SCORING_RULES[plat]}


@app.get("/adp")
def get_adp():
    """Return ADP data for all players that have ADP. Useful for debugging."""
    return sorted(
        [{"name": k, "adp": v} for k, v in ADP_INDEX.items()],
        key=lambda x: x["adp"],
    )


@app.get("/debug/history")
def debug_history(count: int = 0, names: str = ""):
    """
    Diagnose pick history detection. Pass ?count=N and/or ?names=Josh+Allen,CeeDee+Lamb
    to see what teams would be assigned via snake draft math.
    Usage: /debug/history?count=12&names=Josh+Allen,Jahmyr+Gibbs
    """
    teams = 12
    result = []
    name_list = [n.strip() for n in names.split(",") if n.strip()]
    for i, name in enumerate(name_list):
        overall_pick = i + 1
        round_idx = (overall_pick - 1) // teams
        pos_in_round = (overall_pick - 1) % teams
        slot = pos_in_round + 1 if round_idx % 2 == 0 else teams - pos_in_round
        matched = match_player(name)
        result.append({
            "pick": overall_pick,
            "name": name,
            "matched": matched.get("player_display_name") if matched else None,
            "team_slot": slot,
            "round": round_idx + 1,
        })
    return {"total_picks_detected": count, "picks": result}


@app.get("/debug/scoring")
def debug_scoring(name: str = "", season: int = 0, top: int = 20):
    """
    Verify DK scoring breakdown for any player or show top scoring weeks.
    Usage:
      /debug/scoring?name=Josh+Allen
      /debug/scoring?name=Josh+Allen&season=2025
      /debug/scoring?top=10
    """
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent / "training"))
    try:
        import pandas as _pd
        import numpy as _np
        from scoring import PLATFORMS, DEFAULT_PLATFORM

        raw = _Path(__file__).parent.parent / "data" / "raw"
        df = _pd.read_csv(raw / "player_stats_weekly.csv", low_memory=False)
        df = df[df["season_type"] == "REG"].copy()
        if season:
            df = df[df["season"] == season]

        s = PLATFORMS[DEFAULT_PLATFORM]

        def _col(c, d=0.0):
            return df[c].fillna(d) if c in df.columns else _pd.Series(d, index=df.index)

        py = _col("passing_yards"); ry = _col("rushing_yards"); cy = _col("receiving_yards")
        fum = (_col("rushing_fumbles_lost") + _col("receiving_fumbles_lost")
               + _col("sack_fumbles_lost") + _col("fumbles_lost")).clip(0, 4)

        df["_total"] = (
            py * s["pass_yd_per"] + _col("passing_tds") * s["pass_td"] + _col("interceptions") * s["int"]
            + ry * s["rush_yd_per"] + _col("rushing_tds") * s["rush_td"]
            + cy * s["rec_yd_per"] + _col("receiving_tds") * s["rec_td"] + _col("receptions") * s["reception"]
            + fum * s["fumble_lost"]
            + _np.where(py >= 300, s["pass_bonus_300"], 0)
            + _np.where(ry >= 100, s["rush_bonus_100"], 0)
            + _np.where(cy >= 100, s["rec_bonus_100"],  0)
        )
        df["_pass_bonus"] = _np.where(py >= 300, s["pass_bonus_300"], 0)
        df["_rush_bonus"] = _np.where(ry >= 100, s["rush_bonus_100"], 0)
        df["_rec_bonus"]  = _np.where(cy >= 100, s["rec_bonus_100"],  0)
        df["_fum_pen"]    = fum * s["fumble_lost"]

        if name:
            df = df[df["player_display_name"].str.contains(name, case=False, na=False)]

        rows = []
        for _, r in df.nlargest(top, "_total").iterrows():
            rows.append({
                "name":        r.get("player_display_name", ""),
                "position":    r.get("position", ""),
                "season":      int(r.get("season", 0)),
                "week":        int(r.get("week", 0)),
                "total_dk":    round(float(r["_total"]), 2),
                "pass_yds":    int(py.loc[r.name]),
                "rush_yds":    int(ry.loc[r.name]),
                "rec_yds":     int(cy.loc[r.name]),
                "pass_bonus":  float(r["_pass_bonus"]),
                "rush_bonus":  float(r["_rush_bonus"]),
                "rec_bonus":   float(r["_rec_bonus"]),
                "fum_penalty": float(r["_fum_pen"]),
            })
        return {"platform": DEFAULT_PLATFORM, "rules": s, "weeks": rows}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/debug/match")
def debug_match(name: str):
    """Diagnose why a player name isn't matching. Usage: /debug/match?name=Patrick+Mahomes"""
    n = normalize(name)
    exact = n in PLAYER_INDEX
    result = match_player(name)
    close = [k for k in PLAYER_INDEX if n in k or k in n][:5]
    return {
        "input": name,
        "normalized": n,
        "exact_match": exact,
        "matched_to": result.get("player_display_name") if result else None,
        "close_keys": close,
    }


@app.get("/players")
def get_players():
    """All player names + metadata — used by extension for page scanning."""
    return [
        {
            "name":           p.get("player_display_name", ""),
            "position":       p.get("position", ""),
            "team":           p.get("recent_team", ""),
            "proj_points":    round(float(p.get("proj_points", 0)), 1),
            "proj_boom_rate": round(float(p.get("proj_boom_rate", 0)), 3),
            "carry_share":    round(float(p.get("carry_share", 0)), 3),
            "bye_week":       int(p.get("bye_week", 0)),
            "age":            round(float(p.get("age", 0)), 1),
            "is_rookie":      bool(p.get("is_rookie", False)),
            "draft_round":    int(p.get("draft_round", 0)),
            "draft_pick":     int(p.get("draft_pick", 0)),
            "overall_rank":   p.get("overall_rank", 999),
            "adp":            _effective_adp(p),
            "prior_pts":      round(float(p.get("prior_pts", 0)), 1),
            "prior_boom_rate":round(float(p.get("prior_boom_rate", 0)), 3),
        }
        for p in ALL_PLAYERS
        if p.get("player_display_name")
    ]


@app.get("/board", response_class=None)
def get_board():
    from fastapi.responses import HTMLResponse
    players = sorted(ALL_PLAYERS, key=lambda p: p.get("overall_rank", 9999))
    rows = []
    for p in players:
        name     = p.get("player_display_name", "")
        pos      = p.get("position", "")
        team     = p.get("recent_team", "")
        rank     = p.get("overall_rank", "")
        proj     = round(float(p.get("proj_points", 0)))
        boom     = round(float(p.get("proj_boom_rate", 0)) * 100)
        adp      = _effective_adp(p)
        adp_str  = f"{adp:.0f}" if adp else "—"
        diff     = round(adp - rank) if adp and rank else 0
        diff_str = (f'<span style="color:#4caf50">▲{diff}</span>' if diff >= 15
                    else f'<span style="color:#f44336">▼{abs(diff)}</span>' if diff <= -15
                    else f'<span style="color:#666">{diff:+d}</span>')
        rookie   = ' <span style="color:#ff9900;font-size:10px">R</span>' if p.get("is_rookie") else ""
        bye      = int(p.get("bye_week", 0))
        bye_str  = f"Wk{bye}" if bye else "—"
        cs       = round(float(p.get("carry_share", 0)) * 100)
        cs_str   = f"{cs}%" if pos == "RB" and cs else "—"
        rows.append(
            f'<tr data-pos="{pos.lower()}">'
            f'<td class="num">{rank}</td>'
            f'<td class="num adp-col">{adp_str}</td>'
            f'<td class="num">{diff_str}</td>'
            f'<td><span class="pos pos-{pos.lower()}">{pos}</span></td>'
            f'<td class="name">{name}{rookie}</td>'
            f'<td>{team}</td>'
            f'<td class="num">{proj}</td>'
            f'<td class="num">{boom}%</td>'
            f'<td class="num">{cs_str}</td>'
            f'<td class="num">{bye_str}</td>'
            f'</tr>'
        )
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DraftManager — 2026 Board</title>
<style>
  body {{ background:#0d0d1a; color:#e0e0e0; font-family:monospace; font-size:13px; margin:0; padding:16px; }}
  h1 {{ color:#00d4aa; margin:0 0 12px; font-size:18px; }}
  .subtitle {{ color:#555; font-size:11px; margin-bottom:14px; }}
  .filters {{ margin-bottom:12px; display:flex; gap:8px; align-items:center; }}
  .filters button {{ background:#1a1a2e; border:1px solid #333; color:#aaa; padding:5px 14px;
    border-radius:4px; cursor:pointer; font-size:12px; font-family:monospace; }}
  .filters button.active {{ background:#00d4aa; color:#000; border-color:#00d4aa; font-weight:700; }}
  .filters input {{ background:#1a1a2e; border:1px solid #333; color:#e0e0e0; padding:5px 10px;
    border-radius:4px; font-size:12px; font-family:monospace; width:180px; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ background:#111; color:#555; font-size:10px; letter-spacing:0.8px; text-transform:uppercase;
    padding:6px 8px; text-align:left; position:sticky; top:0; z-index:10; border-bottom:1px solid #1a1a2e; }}
  td {{ padding:5px 8px; border-bottom:1px solid #111; }}
  tr:hover td {{ background:#111; }}
  tr.hidden {{ display:none; }}
  .num {{ text-align:right; color:#aaa; }}
  .name {{ color:#fff; }}
  .adp-col {{ color:#888; }}
  .pos {{ display:inline-block; padding:1px 5px; border-radius:3px; font-size:10px; font-weight:700; }}
  .pos-qb {{ background:#1a3a5c; color:#5ba3ff; }}
  .pos-wr {{ background:#1a4a2a; color:#4caf50; }}
  .pos-rb {{ background:#3a1a1a; color:#f44336; }}
  .pos-te {{ background:#3a2a1a; color:#ff9800; }}
</style>
</head>
<body>
<h1>DraftManager — 2026 Pre-Draft Board</h1>
<div class="subtitle">{len(players)} players · ranked by model (proj pts + VOR) · ADP from FantasyPros</div>
<div class="filters">
  <button class="active" onclick="filter('all',this)">ALL</button>
  <button onclick="filter('qb',this)">QB</button>
  <button onclick="filter('wr',this)">WR</button>
  <button onclick="filter('rb',this)">RB</button>
  <button onclick="filter('te',this)">TE</button>
  <input id="search" placeholder="Search name..." oninput="search(this.value)">
</div>
<table>
<thead><tr>
  <th>Rank</th><th>ADP</th><th>vs ADP</th><th>Pos</th><th>Name</th><th>Team</th>
  <th>Proj Pts</th><th>Boom</th><th>CarryShare</th><th>Bye</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<script>
let curPos = 'all', curSearch = '';
function filter(pos, btn) {{
  curPos = pos;
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  apply();
}}
function search(q) {{ curSearch = q.toLowerCase(); apply(); }}
function apply() {{
  document.querySelectorAll('tbody tr').forEach(r => {{
    const pos = r.dataset.pos;
    const name = r.querySelector('.name').textContent.toLowerCase();
    const posOk = curPos === 'all' || pos === curPos;
    const nameOk = !curSearch || name.includes(curSearch);
    r.classList.toggle('hidden', !(posOk && nameOk));
  }});
}}
</script>
</body>
</html>"""
    return HTMLResponse(html)


@app.post("/rank", response_model=RankResponse)
def rank_players(req: RankRequest):
    if not ALL_PLAYERS:
        raise HTTPException(503, "Projections not loaded — run the training pipeline first.")

    # Pick up a fresh ADP board (daily refresh) without needing a server restart.
    maybe_reload_dk_adp()
    maybe_reload_underdog_adp()
    if current_platform() == "underdog" and req.scoring != "half":
        req = req.model_copy(update={"scoring": "half"})
    adp_tokens = _install_request_adp(req)
    page_adp_count = int(adp_tokens[3])

    # Remove players already drafted (by anyone) from the available pool
    drafted_names = drafted_name_set_from_request(req)

    available: list[dict] = []
    seen_players: set = set()
    for name in req.available_players:
        if normalize(name) in drafted_names:
            continue
        player = match_player(name)
        if not player:
            continue
        # Dedupe by player identity: two incoming name variants (e.g. "Brock
        # Bowers" and "B. Bowers") can match the same player; without this they'd
        # produce duplicate recommendations.
        ident = player.get("player_id") or normalize(player.get("player_display_name", ""))
        if ident in seen_players:
            continue
        seen_players.add(ident)
        available.append(player)

    if not available:
        # Graceful: still return empty recs rather than 400
        resp = RankResponse(
            recommendations=[],
            stack_alerts=[],
            scarcity_warnings=["No players matched. Try clicking Rescan."],
            total_available=0,
            proj_season=str(proj_season),
            scoring=req.scoring,
            adp_source=_REQUEST_ADP_SOURCE.get(),
            page_adp_count=page_adp_count,
        )
        _reset_request_adp(adp_tokens)
        return resp

    replacement = compute_dynamic_vor(available)

    recs = []
    for p in available:
        pos   = p.get("position", "?")
        proj  = float(p.get("proj_points", 0))
        repl  = replacement.get(pos, 0)
        dyn_vor = proj - repl

        overall_rank = int(p.get("overall_rank", 999))
        recs.append(PlayerRec(
            name=p.get("player_display_name", "Unknown"),
            position=pos,
            team=p.get("recent_team", "?"),
            proj_points=round(proj, 1),
            proj_boom_rate=round(float(p.get("proj_boom_rate", 0)), 3),
            vor=round(dyn_vor, 1),
            age=round(float(p.get("age", 0)), 1),
            carry_share=round(float(p.get("carry_share", 0)), 3),
            value_tier=value_tier(dyn_vor, pos),
            is_rookie=bool(p.get("is_rookie", False)),
            draft_round=int(p.get("draft_round", 0)),
            draft_pick=int(p.get("draft_pick", 0)),
            bye_week=int(p.get("bye_week", 0)),
            adp=_display_adp(p),
            overall_rank=overall_rank,
            prior_pts=round(float(p.get("prior_pts", 0)), 1),
            prior_boom_rate=round(float(p.get("prior_boom_rate", 0)), 3),
        ))

    my_pick_data = my_pick_data_from_request(req)
    used_dk_policy = False
    selected_artifact = dk_policy_artifact_for_scoring(req.scoring)
    if req.use_dk_ev_policy and current_platform() in {"draftkings", "underdog"} and recs:
        ranked = rank_with_dk_ev_policy(req, recs, available, my_pick_data)
        if ranked is not None:
            recs = apply_dk_ev_tiebreaker(req, ranked, my_pick_data)
            used_dk_policy = True

    if not used_dk_policy and recs:
        max_boom = max(r.proj_boom_rate for r in recs) or 1.0
        max_vor  = max(abs(r.vor) for r in recs) or 1.0
        recs.sort(
            key=lambda r: 0.65 * (r.vor / max_vor) + 0.35 * (r.proj_boom_rate / max_boom),
            reverse=True,
        )
        recs = [r.model_copy(update={"ranking_source": "legacy_vor"}) for r in recs]

    my_picks = my_drafted_players_from_request(req)
    # Use real rosters from extension when available — far more accurate than inferring
    if req.team_rosters:
        opponent_demand = demand_from_actual_rosters(req.team_rosters, req.my_pick_position, req.total_teams)
    else:
        opponent_demand = infer_opponent_demand(req.drafted_players, req.total_teams, req.my_pick_position)
    recs = apply_bye_penalties(recs, my_pick_data)
    recs = apply_going_soon(recs, req.current_pick, req.total_teams, req.my_pick_position, opponent_demand)

    # Mark pass catchers who share a team with the user's QB(s) as stack targets
    my_qb_by_team = {
        p.get("recent_team", ""): p.get("player_display_name", "QB")
        for p in my_pick_data
        if p.get("position") == "QB" and p.get("recent_team")
    }
    if my_qb_by_team:
        recs = [
            r.model_copy(update={"is_stack": True, "stack_qb": my_qb_by_team[r.team]})
            if r.position in ("WR", "TE") and r.team in my_qb_by_team
            else r
            for r in recs
        ]

    stack_alerts  = detect_stacks(my_picks, available)
    scarcity_warn = detect_scarcity(available, req.total_teams)

    # Add opponent position-rush warnings (many teams still need a position)
    opponents = req.total_teams - 1
    if req.current_pick <= req.total_teams:
        opponent_demand = {}
    for pos, hungry in sorted(opponent_demand.items(), key=lambda x: -x[1]):
        if hungry >= max(3, opponents // 2):
            scarcity_warn.insert(0, f"Position rush: {hungry}/{opponents} opponents still need {pos} — expect a run soon")

    resp = RankResponse(
        recommendations=recs[:60],
        stack_alerts=stack_alerts,
        scarcity_warnings=scarcity_warn,
        total_available=len(available),
        proj_season=str(proj_season),
        scoring=req.scoring,
        dk_policy_label=str(selected_artifact.get("label", "")) if used_dk_policy else "",
        dk_policy_model=selected_artifact.get("model_path", Path()).name if used_dk_policy else "",
        dk_policy_max_candidate_rank=(
            _dk_policy_trained_candidate_pool(selected_artifact) if used_dk_policy else 0
        ),
        adp_source=_REQUEST_ADP_SOURCE.get(),
        page_adp_count=page_adp_count,
    )
    _reset_request_adp(adp_tokens)
    return resp


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nDraftManager server -> http://localhost:8765")
    print("Keep this terminal open during your draft.\n")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
