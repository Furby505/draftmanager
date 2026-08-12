"""
Step 3 — opponent field (ADP-with-noise draft rooms).

Generates complete 12-team best-ball draft rooms whose rosters feed directly
into season_sim / bracket. Opponents draft from the DK ADP board using
probabilistic ADP-weighted picks, obey the 20-man roster + position-cap rules,
avoid degenerate builds, and steer toward unfilled needs late.

Pick model
----------
ADP-with-noise (the main model): each available, legal player gets a draft key
    key = ADP - need_bonus + Normal(0, sigma)
and the team takes the smallest key (best perceived value still on the board).
Gaussian noise on ADP reproduces realistic reaches/falls; `need_bonus` nudges
toward positions the roster still needs (and a hard override guarantees a legal
roster in the final picks). `strategy="pure_adp"` disables noise/needs and just
takes best ADP — a deterministic baseline for testing.

Legality
--------
- Hard caps: <=5 QB, <=5 TE (DK rule); roster size 20.
- Soft caps (QB3/RB7/WR8/TE3): excluded unless that empties the pool or a need
  forces it -> prevents 4-QB / 1-RB type builds.
- Validity guarantee: if a team's remaining picks == its unfilled minimum slots
  (1QB/2RB/3WR/1TE), the pool is restricted to needed positions.

The DK EV lane uses `load_market_board()`, which is built directly from the
DraftKings ADP export and does not iterate over the old projection board.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import best_ball as bb
from outcome_model import normalize_name

ADP_JSON     = Path(__file__).resolve().parent.parent / "server" / "models" / "adp.json"
PROJECTIONS  = Path(__file__).resolve().parent.parent / "server" / "models" / "projections.json"
BBM_V_SAMPLE = Path(__file__).resolve().parent.parent / "data" / "raw" / "bbm_v_sample.csv"
WEEKLY_CSV   = Path(__file__).resolve().parent.parent / "data" / "raw" / "player_stats_weekly.csv"

POS          = ("QB", "RB", "WR", "TE")
POS_CODE     = {p: i for i, p in enumerate(POS)}
HARD_CAP     = np.array([5, 20, 20, 5])     # QB/TE per DK; RB/WR bounded by roster
REALISM_CAP  = np.array([4, 20, 20, 4])     # opponent rooms should not take QB5/TE5
SOFT_CAP     = np.array([3, 7, 8, 3])       # discourage degenerate builds
MIN_POS      = np.array([1, 2, 3, 1])       # validity minimums
COMFORT      = np.array([2, 4, 5, 2])       # "want a backup" targets (late nudge)
DEFAULT_SIGMA = 12.0                         # ADP noise, in pick-slot units
NEED_STRONG   = 20.0                         # pull toward unfilled minimums
NEED_MILD     = 6.0                          # late pull toward comfort backups
MARKET_PROJ_SCALE = 0.85                     # 2026-06-18 score-environment calibration


# ── Asymmetric field bias ────────────────────────────────────────────────────
# Real DK drafters don't deviate from ADP symmetrically — they deviate in
# CONSISTENT directions: they reach for hyped rookies and last-year's producers,
# and they let aging vets fall. Symmetric Gaussian noise (DEFAULT_SIGMA) averages
# those out, so the simulated field has no systematic soft spots for the policy
# to exploit. This adds a per-player, one-directional shift (in pick-slot units,
# NEGATIVE = the field reaches earlier) layered on top of the existing noise.
#
# The board's outcome value is unchanged (it still comes from market_proj_points
# of the player's own ADP); the field merely spends draft capital on hype, which
# mechanically lets the complementary "boring but solid" players fall to a patient
# drafter. That availability skew is the exploitable structure — fully consistent
# with this lane's design that true value is drawn from ADP, not a projection.
#
# Signal is roster FACTS only (is_rookie / age / prior-year points), never the
# retired projection model's outputs. Default OFF; opt-in for training A/Bs.

@dataclass
class FieldBiasParams:
    rookie_reach: float = 5.0      # pick-slots the field reaches for any rookie
    recency_max: float = 5.0       # max reach for last-year's top producer
    recency_ref_pts: float = 280.0 # prior-season points that earns the full reach
    age_fade_start: float = 30.0   # vets at/after this age start to fall
    age_fade_per_year: float = 1.0 # extra slots they fall per year past start-1
    age_fade_cap: float = 5.0      # max fall from age


DEFAULT_FIELD_BIAS = FieldBiasParams()


def load_field_bias_meta(projections: Path = PROJECTIONS) -> dict[str, dict]:
    """{normalized_name: {is_rookie, age, prior_pts}} from projections.json.

    These are public roster facts used only to shape opponent draft BEHAVIOR;
    no projected-value column is read (kept clear of the retired model lane).
    """
    out: dict[str, dict] = {}
    data = json.loads(Path(projections).read_text())
    for r in data:
        nm = normalize_name(r.get("player_display_name", ""))
        if not nm:
            continue
        out[nm] = {
            "is_rookie": bool(r.get("is_rookie", False)),
            "age": float(r.get("age") or 0.0),
            "prior_pts": float(r.get("prior_pts") or 0.0),
        }
    return out


def _player_field_bias(meta: dict, params: FieldBiasParams) -> float:
    """Per-player draft-key shift (negative = reached earlier than ADP)."""
    shift = 0.0
    if meta.get("is_rookie"):
        shift -= params.rookie_reach
    prior = meta.get("prior_pts", 0.0) or 0.0
    if prior > 0.0 and params.recency_ref_pts > 0.0:
        shift -= params.recency_max * min(1.0, prior / params.recency_ref_pts)
    age = meta.get("age", 0.0) or 0.0
    if age >= params.age_fade_start:
        shift += min(params.age_fade_cap,
                     params.age_fade_per_year * (age - (params.age_fade_start - 1.0)))
    return float(shift)


def compute_field_bias(names: np.ndarray, meta_by_name: dict[str, dict],
                       params: FieldBiasParams = DEFAULT_FIELD_BIAS) -> np.ndarray:
    """Per-board-row field-bias array; 0 where no metadata matches the name."""
    bias = np.zeros(len(names), dtype=float)
    for i, nm in enumerate(names):
        meta = meta_by_name.get(normalize_name(str(nm)))
        if meta:
            bias[i] = _player_field_bias(meta, params)
    return bias


# ── Board ────────────────────────────────────────────────────────────────────
@dataclass
class Board:
    player_id: np.ndarray   # [N] str
    name: np.ndarray        # [N] str (display)
    team: np.ndarray        # [N] str
    pos_code: np.ndarray    # [N] int (index into POS)
    adp: np.ndarray         # [N] float, ascending
    has_real_adp: np.ndarray  # [N] bool
    proj_points: np.ndarray | None = None  # market-implied points, not old model
    market_tier: np.ndarray | None = None
    bye_week: np.ndarray | None = None
    field_bias: np.ndarray | None = None   # [N] per-player draft-key shift (pick-slots)

    def __len__(self):
        return len(self.adp)

    def player_dict(self, i: int) -> dict:
        d = {"player_id": str(self.player_id[i]), "name": str(self.name[i]),
             "position": POS[self.pos_code[i]], "team": str(self.team[i]),
             "adp": float(self.adp[i])}
        if self.proj_points is not None:
            d["proj_points"] = float(self.proj_points[i])
        if self.market_tier is not None:
            d["market_tier"] = int(self.market_tier[i])
        if self.bye_week is not None:
            d["bye_week"] = int(self.bye_week[i])
        return d


def market_proj_points(pos: str, adp: float) -> float:
    """Market-implied season level from DK ADP only.

    This is not a player projection model. It gives the outcome simulator a
    rough scoring level by position/market slot so EV training can run without
    trusting the old projection board. Player-specific weekly shape still
    comes from history when available; otherwise the ADP tier selects a comparable
    position pool.
    """
    # Approximate active-season PPG ranges by market tier. The absolute scale is
    # less important than monotonic position/tier separation because labels are
    # comparative within DK draft contexts.
    hi = {"QB": 22.0, "RB": 18.0, "WR": 18.0, "TE": 15.5}
    lo = {"QB": 9.0, "RB": 4.0, "WR": 4.0, "TE": 3.0}
    x = float(np.clip(1.0 - (float(adp) - 1.0) / 240.0, 0.0, 1.0))
    ppg = lo.get(pos, 4.0) + (hi.get(pos, 15.0) - lo.get(pos, 4.0)) * (x ** 0.85)
    return float(ppg * 17.0 * MARKET_PROJ_SCALE)


def market_tier_from_adp(adp: float) -> int:
    if adp <= 36:
        return 0
    if adp <= 96:
        return 1
    if adp <= 168:
        return 2
    return 3


def load_market_board(source: Path | None = None,
                      field_bias: bool | FieldBiasParams = False) -> Board:
    """Build the DK EV board directly from real DK ADP rows.

    No `projections.json`, no old model rank fallback. If the DK source lacks
    position/team columns, the player is skipped because DK EV training needs
    both for legality and correlation.

    field_bias: False -> no asymmetric opponent bias (legacy). True or a
    FieldBiasParams -> attach a per-player draft-key shift so opponent rooms
    reach for hyped rookies / recent producers and let vets fall (see
    `compute_field_bias`). Only joins roster facts from projections.json.
    """
    from fetch_dk_adp import _read_rows, _resolve_default_source

    source = Path(source) if source else _resolve_default_source()
    if source is None or not source.exists():
        raise FileNotFoundError(
            "No DraftKings ADP source found. Expected data/raw/dk_adp.csv "
            "or draftkings_best_ball_adp_latest.csv."
        )

    rows = []
    seen = set()
    for r in _read_rows(source):
        pos = str(r.get("pos") or "").upper()
        team = str(r.get("team") or "").upper()
        adp = float(r["adp"])
        if pos not in POS_CODE or not team:
            continue
        key = (r["nm"], pos, team)
        if key in seen:
            continue
        seen.add(key)
        display = " ".join(part.capitalize() for part in r["nm"].split())
        rows.append({
            "player_id": f"DKADP-{int(round(adp * 10)):04d}-{r['nm'].replace(' ', '-').upper()}",
            "name": display,
            "team": team,
            "pos": pos,
            "adp": adp,
            "bye_week": int(r.get("bye_week") or 0),
            "proj_points": market_proj_points(pos, adp),
            "market_tier": market_tier_from_adp(adp),
        })

    if len(rows) < bb.TEAMS_PER_POD * bb.DRAFT_ROUNDS:
        raise RuntimeError(f"DK ADP board too small for a full draft: {len(rows)} rows")

    rows.sort(key=lambda r: r["adp"])
    name_arr = np.array([r["name"] for r in rows], dtype=object)
    bias_arr = None
    if field_bias is not False:
        params = field_bias if isinstance(field_bias, FieldBiasParams) else DEFAULT_FIELD_BIAS
        bias_arr = compute_field_bias(name_arr, load_field_bias_meta(), params)
    return Board(
        player_id=np.array([r["player_id"] for r in rows], dtype=object),
        name=name_arr,
        team=np.array([r["team"] for r in rows], dtype=object),
        pos_code=np.array([POS_CODE[r["pos"]] for r in rows], dtype=int),
        adp=np.array([r["adp"] for r in rows], dtype=float),
        has_real_adp=np.ones(len(rows), dtype=bool),
        proj_points=np.array([r["proj_points"] for r in rows], dtype=float),
        market_tier=np.array([r["market_tier"] for r in rows], dtype=int),
        bye_week=np.array([r["bye_week"] for r in rows], dtype=int),
        field_bias=bias_arr,
    )


def _team_map_from_weekly_2024(weekly_csv: Path = WEEKLY_CSV) -> dict[str, str]:
    """Best-effort 2024 name -> NFL team map for historical BBM V boards."""
    import pandas as pd

    if not Path(weekly_csv).exists():
        return {}
    cols = ["season", "player_display_name", "recent_team"]
    w = pd.read_csv(weekly_csv, usecols=lambda c: c in cols, low_memory=False)
    if "season" in w.columns:
        w = w[w["season"] == 2024]
    out: dict[str, str] = {}
    for _, r in w.dropna(subset=["player_display_name"]).iterrows():
        nm = normalize_name(r.get("player_display_name", ""))
        team = str(r.get("recent_team") or "").upper()
        if nm and team:
            out[nm] = team
    return out


def load_bbm_v_board(source: Path | None = None,
                     weekly_csv: Path = WEEKLY_CSV) -> Board:
    """Build a historical Underdog BBM V board from the real draft sample.

    This is for offline half-PPR validation prototypes only. It uses BBM V's
    `projection_adp` median by player as a historical market board; it is not a
    current Underdog ADP feed and should not be used for live product decisions.
    """
    import pandas as pd

    source = Path(source) if source else BBM_V_SAMPLE
    if not source.exists():
        raise FileNotFoundError(f"BBM V sample not found: {source}")

    usecols = ["player_id", "player_name", "position_name", "projection_adp"]
    df = pd.read_csv(source, usecols=usecols, low_memory=False)
    df = df[df["position_name"].isin(POS_CODE)].copy()
    df["projection_adp"] = pd.to_numeric(df["projection_adp"], errors="coerce")
    df = df.dropna(subset=["player_id", "player_name", "position_name", "projection_adp"])
    players = (
        df.groupby(["player_id", "player_name", "position_name"], as_index=False)
          ["projection_adp"].median()
          .sort_values("projection_adp", kind="stable")
    )
    team_map = _team_map_from_weekly_2024(weekly_csv)
    names = players["player_name"].astype(str).to_numpy(object)
    pos = players["position_name"].astype(str).to_numpy(object)
    adp = players["projection_adp"].astype(float).to_numpy()
    teams = np.array([team_map.get(normalize_name(n), "") for n in names], dtype=object)

    return Board(
        player_id=np.array([f"BBMV-{p}" for p in players["player_id"].astype(str)], dtype=object),
        name=names,
        team=teams,
        pos_code=np.array([POS_CODE[p] for p in pos], dtype=int),
        adp=adp,
        has_real_adp=np.ones(len(players), dtype=bool),
        proj_points=np.array([market_proj_points(p, a) for p, a in zip(pos, adp)], dtype=float),
        market_tier=np.array([market_tier_from_adp(a) for a in adp], dtype=int),
        bye_week=np.zeros(len(players), dtype=int),
    )


def load_adp_board(adp_json: Path | None = None,
                   projections: Path = PROJECTIONS) -> Board:
    """Build the draft board from projections, ADP where known else overall_rank.

    ADP source priority: explicit `adp_json` > canonical DK ADP (dk_adp.json,
    via fetch_dk_adp) > FantasyPros adp.json. The board prints its provenance so
    it's always clear whether real DK ADP or a stand-in is in use.
    """
    if adp_json is not None:
        adp_map = json.loads(Path(adp_json).read_text())
        if isinstance(adp_map, dict) and "adp" in adp_map:
            adp_map = adp_map["adp"]
        adp_map = {normalize_name(k): float(v) for k, v in adp_map.items()}
    else:
        from fetch_dk_adp import load_dk_adp
        adp_map, meta = load_dk_adp()
        if not meta.get("is_real_dk_adp", False):
            print("[board] ADP source is a STAND-IN, not real DK ADP "
                  f"({meta.get('source', '?')}). Edge numbers are provisional.")
    proj = json.loads(Path(projections).read_text())

    pid, name, team, pcode, adp, real = [], [], [], [], [], []
    for r in proj:
        p = r.get("position")
        if p not in POS_CODE:
            continue
        nm = normalize_name(r.get("player_display_name", ""))
        a = adp_map.get(nm)
        pid.append(str(r.get("player_id") or ""))
        name.append(r.get("player_display_name", ""))
        team.append(r.get("recent_team") or "")
        pcode.append(POS_CODE[p])
        if a is not None:
            adp.append(float(a)); real.append(True)
        else:
            # Synthetic fallback: place after the real-ADP pool by overall_rank.
            adp.append(float(r.get("overall_rank", 9999))); real.append(False)

    adp = np.array(adp); order = np.argsort(adp, kind="stable")
    take = lambda arr: np.array(arr, dtype=object)[order]
    return Board(take(pid), take(name), take(team),
                 np.array(pcode)[order], adp[order], np.array(real)[order])


# ── Draft ────────────────────────────────────────────────────────────────────
def _sample_team_caps(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw a per-team build archetype so the field spans real constructions
    (2- vs 3-QB, Zero-RB vs hero-RB, WR-heavy, etc.) instead of one uniform build.
    Returns (soft_cap[4], comfort[4]) for QB/RB/WR/TE."""
    qb = rng.choice([2, 3], p=[0.62, 0.38])
    te = rng.choice([2, 3], p=[0.58, 0.42])
    rb = rng.choice([5, 6, 7, 8], p=[0.12, 0.33, 0.35, 0.20])   # low rb = Zero-RB lean
    wr = rng.choice([7, 8, 9], p=[0.28, 0.42, 0.30])
    soft = np.array([qb, rb, wr, te])
    comfort = np.array([min(2, qb), max(2, rb - 3), max(3, wr - 3), min(2, te)])
    return soft, comfort


def _need_bonus(counts: np.ndarray, comfort: np.ndarray,
                rnd: int, n_rounds: int) -> np.ndarray:
    """Per-position pull (subtracted from ADP key): unmet minimums always, and
    comfort backups in the back half of the draft."""
    b = np.zeros(4)
    late = rnd >= n_rounds // 2
    for k in range(4):
        if counts[k] < MIN_POS[k]:
            b[k] += NEED_STRONG
        elif late and counts[k] < comfort[k]:
            b[k] += NEED_MILD
    return b


def _choose(board: Board, avail: np.ndarray, counts: np.ndarray,
            soft_cap: np.ndarray, comfort: np.ndarray,
            picks_left: int, rnd: int, n_rounds: int,
            sigma: float, strategy: str, rng: np.random.Generator) -> int:
    """Return the board index this team drafts."""
    # Hard-legal: available and under hard position cap.
    under_hard = counts[board.pos_code] < HARD_CAP[board.pos_code]
    legal = avail & under_hard
    under_realism = counts[board.pos_code] < REALISM_CAP[board.pos_code]
    if (legal & under_realism).any():
        legal = legal & under_realism

    # Validity guarantee: if every remaining pick is needed to hit minimums,
    # restrict to needed positions.
    unmet = np.maximum(0, MIN_POS - counts)
    if picks_left <= int(unmet.sum()) and unmet.sum() > 0:
        needed = np.isin(board.pos_code, np.where(unmet > 0)[0])
        legal = legal & needed
    else:
        # Soft caps: drop over-soft-cap positions unless that empties the pool.
        under_soft = counts[board.pos_code] < soft_cap[board.pos_code]
        if (legal & under_soft).any():
            legal = legal & under_soft

    idx = np.where(legal)[0]
    if idx.size == 0:                      # last-resort: anything legal under caps
        idx = np.where(avail & under_hard)[0]

    if strategy == "pure_adp":
        return int(idx[np.argmin(board.adp[idx])])

    bonus = _need_bonus(counts, comfort, rnd, n_rounds)
    keys = board.adp[idx] - bonus[board.pos_code[idx]] \
        + rng.normal(0.0, sigma, size=idx.size)
    # Asymmetric field bias: reach for hyped types, let vets fall (opt-in; the
    # board only carries this array when load_market_board(field_bias=...) is set).
    if board.field_bias is not None:
        keys = keys + board.field_bias[idx]
    return int(idx[np.argmin(keys)])


def simulate_draft(board: Board, rng: np.random.Generator,
                   n_teams: int = bb.TEAMS_PER_POD, n_rounds: int = bb.DRAFT_ROUNDS,
                   sigma: float = DEFAULT_SIGMA,
                   strategy: str = "adp_noise") -> list[list[dict]]:
    """Run one snake draft -> list of `n_teams` rosters (each a list of player dicts)."""
    avail = np.ones(len(board), dtype=bool)
    counts = np.zeros((n_teams, 4), dtype=int)
    rosters: list[list[dict]] = [[] for _ in range(n_teams)]
    # Each team gets a build archetype. pure_adp uses fixed caps so it stays a
    # deterministic baseline (no RNG consumed).
    if strategy == "pure_adp":
        team_caps = [(SOFT_CAP, COMFORT)] * n_teams
    else:
        team_caps = [_sample_team_caps(rng) for _ in range(n_teams)]

    for gp in range(n_teams * n_rounds):
        rnd, slot = divmod(gp, n_teams)
        team = slot if rnd % 2 == 0 else n_teams - 1 - slot   # snake
        picks_left = n_rounds - len(rosters[team])
        soft_cap, comfort = team_caps[team]
        i = _choose(board, avail, counts[team], soft_cap, comfort,
                    picks_left, rnd, n_rounds, sigma, strategy, rng)
        avail[i] = False
        counts[team][board.pos_code[i]] += 1
        rosters[team].append(board.player_dict(i))
    return rosters


def draft_field(n_rooms: int, rng: np.random.Generator,
                board: Board | None = None, sigma: float = DEFAULT_SIGMA,
                strategy: str = "adp_noise",
                n_rounds: int = bb.DRAFT_ROUNDS) -> list[list[dict]]:
    """Draft `n_rooms` rooms and return the flat population of opponent rosters."""
    board = board or load_market_board()
    field: list[list[dict]] = []
    for _ in range(n_rooms):
        field.extend(simulate_draft(board, rng, n_rounds=n_rounds,
                                    sigma=sigma, strategy=strategy))
    return field


# ── Bridge to the bracket: rosters -> Field ─────────────────────────────────
def build_field(rosters: list[list[dict]], model=None, n_seasons: int = 200,
                rng: np.random.Generator | None = None,
                contest: bb.ContestConfig = bb.DEFAULT_CONTEST,
                availability: bool | None = None):
    """
    Simulate every opponent roster and pool their per-round scores into a
    bracket.Field. This is the opponent score distribution the scorer needs.
    `availability` overrides the model's iron-man toggle (for A/B audits).
    """
    from outcome_model import load_default_model
    from season_sim import simulate_roster
    from bracket import Field

    rng = rng or np.random.default_rng()
    model = model or load_default_model()
    chunks = [simulate_roster(r, n_seasons=n_seasons, rng=rng, model=model,
                              contest=contest, availability=availability).round_scores
              for r in rosters]
    return Field(np.concatenate(chunks, axis=0))
