"""
Step 1 — Correlated player-week outcome model (simplest working version).

Given a roster and a simulated NFL Week, returns correlated DK fantasy scores
for every player on the roster. Correlation is the whole point: a QB and his
pass catchers must boom together, because that stack correlation is the core
best-ball edge. Independent per-player draws would destroy it.

How it works
------------
1. Marginals: each player gets their *own* empirical distribution of real DK
   weekly scores (a quantile grid). Sampling a uniform quantile reproduces that
   player's real boom/bust shape — exactly, regardless of correlation strength.

2. Correlation: one latent "passing-game" factor z_t per NFL team per week.
   Each player's latent draw is  g_i = L_i * z_team + sqrt(1 - L_i^2) * e_i,
   which is standard-normal (marginals untouched) but shares z_team with
   teammates. Position loadings L encode the real correlation hierarchy:
   QB/WR tie tightly, TE nearly so, RB barely (game script suppresses RB work
   in pass-heavy games). Different teams share nothing → independent.
   The quantile q_i = Phi(g_i) indexes the player's empirical grid.

Deliberate v1 simplifications (documented, easy to replace later):
- Weeks are independent across the season (no week-to-week autocorrelation).
- Only same-team correlation is modeled, not same-*game* (opponent) shootout
  correlation. Hook left in `team_idx` to extend to game-level factors.
- Injury / DNP weeks are not sampled (we draw from games actually played, i.e.
  "when healthy"). Missed-game modeling is a later layer.
- Rookies / team-changers fall back to a (position, tier) comparable pool; no
  attempt at exact player mapping yet.

Replace any single piece (marginals, loadings, factor structure) without
touching the rest — the roster->scores interface is stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scoring import DEFAULT_PLATFORM, recalc_weekly_df

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
WEEKLY_CSV     = DATA_DIR / "raw" / "player_stats_weekly.csv"
PROJECTIONS    = Path(__file__).resolve().parent.parent / "server" / "models" / "projections.json"

MIN_SEASON     = 2022     # recency window for empirical distributions
MIN_OWN_WEEKS  = 10       # below this, fall back to a comparable pool
GRID_K         = 256      # quantile-grid resolution per player
N_TIERS        = 4        # projection tiers for the fallback pool (0 = elite)
N_GAMES        = 17       # weeks sampled per season -> anchor target = proj/17
SCORE_POSITIONS = ("QB", "RB", "WR", "TE")
# Clamp on the projection anchor scale, so a wild history/projection mismatch
# can't produce an absurd distribution (e.g. tiny historical sample).
ANCHOR_CLIP    = (0.2, 4.0)
# Tail-shrinkage prior strength (pseudo-weeks). reliability w = n / (n + PRIOR):
# the convex prize-EV objective chases fat right tails, and small-sample tails
# are mostly noise -> blend a player's SHAPE toward its archetype pool by w.
SHRINK_PRIOR   = 30.0

# ── Availability / injury layer ─────────────────────────────────────────────
FULL_SEASON     = 17            # games a fully-available player plays
# Position priors: per-GAME probability a player misses (fallback when no/low
# history). RBs most fragile, QBs least — these set baseline games missed/season.
POS_MISS_RATE   = {"QB": 0.09, "RB": 0.16, "WR": 0.11, "TE": 0.11}
# Fraction of a player's missed games that come from a clustered MAJOR injury
# (vs independent one-week minor absences).
POS_MAJOR_FRAC  = {"QB": 0.55, "RB": 0.60, "WR": 0.55, "TE": 0.55}
SEASON_END_FRAC = 0.45         # a major injury is season-ending with this prob
MAJOR_DUR_RANGE = (2, 6)       # else lasts this many weeks (inclusive)
MAJOR_DUR_MEAN  = 6.0          # ~mean weeks lost per major injury (calibration)
K_AV            = 2.0          # seasons prior strength for historical miss rate
MISS_RATE_CLIP  = (0.0, 0.5)   # don't let any single player exceed 50% missed
P_MAJOR_CLIP    = (0.0, 0.6)
# Only infer injury durability from players who were actually STARTERS — a
# backup playing 5 games "missed" 12 to ROLE, not injury. A historical rate is
# used only if the player had a >=13-game season; it averages availability over
# their starter-level (>=6 game) seasons. Everyone else uses the position prior.
STARTER_SEASON_GAMES = 13
STARTED_MIN_GAMES    = 6

# ── Handcuff / workload-inheritance layer ────────────────────────────────────
# When a higher-workload same-team, same-position player is OUT a given week, the
# next-up teammate inherits a fraction of the vacated role (volume). Applied as a
# MULTIPLICATIVE scale on the backup's already-correlated score, so boom/bust shape
# (and CV) are preserved. This is what makes RB handcuffs (and, weaker, QB/TE)
# carry real best-ball EV: the backup spikes exactly when the starter is hurt, and
# best-ball auto-starts whoever produced. The fraction is the share of the
# (starter - backup) healthy-ppg gap the inheritor captures.
#
# EMPIRICALLY CALIBRATED (training/calibrate_handcuff.py, REG 2015-2025 DK scoring):
# frac = (backup_ppg_when_starter_OUT - backup_ppg_when_starter_in) / (starter_ppg - backup_in).
#   RB 0.66 (n=508), QB 0.74 (574), WR 0.11 (320), TE 0.39 (577).
# WR is low because a vacated WR1's targets scatter across many receivers, so the
# nominal WR2 barely inherits -> WR "handcuffs" are mostly a myth, correctly cheap.
HANDCUFF_TRANSFER = {"RB": 0.66, "QB": 0.74, "WR": 0.11, "TE": 0.39}

# ── Playoff-matchup layer ────────────────────────────────────────────────────
# Per-team, per-position defense-vs-position multipliers (~1.0), projected for 2026
# by build_defense_ratings.py. The sim modulates each player's WEEKLY output by the
# rating of the defense he faces that week, then NORMALIZES per player to mean 1.0 so
# the season total stays anchored to the market projection -- only the DISTRIBUTION
# across weeks shifts. This matters because the bracket's R2/R3/R4 are SINGLE weeks
# (15/16/17): a soft playoff matchup raises ceiling in exactly the weeks that decide
# advancement, which the season-long projection/ADP does not price.
DEFENSE_RATINGS = (Path(__file__).resolve().parent.parent / "data" / "processed"
                   / "defense_vs_position_2026.json")

# ── Structural correlation model (game -> team -> opportunity -> points) ─────
# Calibrated to historical DK fantasy correlations (correlation_targets.py):
#   QB1-WR1 ~0.39, QB1-TE1 ~0.31, same-team WR-WR ~0.04, bring-back ~0.03-0.04.
# A single shared team factor CAN'T produce QB-WR>>WR-WR; the trick is that the
# QB rides team passing VOLUME while pass-catchers ALSO compete for target share
# (zero-sum), which cancels the WR-WR team boost but not QB-WR.
SCHEDULE_2026 = (Path(__file__).resolve().parent.parent / "data" / "processed"
                 / "schedule_2026.json")
A_GAME   = 0.33   # shared GAME environment; tuned to avoid overvaluing bring-backs
A_QB     = 0.92   # QB loads on its team passing-volume factor (-> QB-WR ~0.39)
A_TEAM   = {"WR": 0.45, "TE": 0.40, "RB": 0.12, "QB": 0.0}   # catcher team-volume load
A_COMP   = {"WR": 0.40, "TE": 0.30, "RB": 0.0,  "QB": 0.0}   # target-competition load
CATCHER_POS = ("WR", "TE")   # who shares the zero-sum target-competition factor

# Target pairwise latent correlation between a QB and his WR (the anchor pair).
DEFAULT_TEAM_CORR = 0.35
# Per-position loading on the shared passing-game factor (relative to QB/WR=1).
# RB is nearly decoupled: pass-heavy game scripts suppress RB usage.
POS_LOADING = {"QB": 1.0, "WR": 1.0, "TE": 0.9, "RB": 0.2}


# ── Small utilities ─────────────────────────────────────────────────────────
def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Vectorized standard-normal CDF (Abramowitz-Stegun 7.1.26 erf, ~1e-7)."""
    z = np.asarray(x, dtype=float) / np.sqrt(2.0)
    sign = np.sign(z)
    az = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * az)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
               - 0.284496736) * t + 0.254829592) * t * np.exp(-az * az)
    erf = sign * y
    return 0.5 * (1.0 + erf)


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation/suffixes for fuzzy roster->history match."""
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    for ch in (".", ",", "'", "`", "-"):
        s = s.replace(ch, " " if ch == "-" else "")
    parts = [p for p in s.split() if p not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return " ".join(parts)


def _quantile_grid(scores: np.ndarray, k: int = GRID_K) -> np.ndarray:
    """Length-k quantile grid (q = 0..1) summarizing an empirical distribution."""
    return np.quantile(scores, np.linspace(0.0, 1.0, k))


# ── Roster context (precomputed once per roster) ────────────────────────────
@dataclass
class RosterContext:
    player_ids: list[str]
    names: list[str]
    positions: list[str]
    teams: list[str]
    team_idx: np.ndarray     # [n] index into the per-week team-factor vector
    loadings: np.ndarray     # [n] L_i on the shared team factor
    qgrids: np.ndarray       # [n, GRID_K] empirical quantile grids
    sources: list[str]       # "own" | "pool:POS:tier" | "miss"
    n_teams: int
    q_minor: np.ndarray      # [n] per-week minor-absence probability
    p_major: np.ndarray      # [n] per-season major-injury probability
    bye_weeks: np.ndarray | None = None   # [n] NFL bye week (1-based; 0/None = unknown)
    anchor: np.ndarray | None = None      # [n] healthy per-week level (qgrid mean) ~ workload
    handcuff_groups: list[np.ndarray] | None = None  # same-team same-pos index sets, anchor-desc


# ── The model ───────────────────────────────────────────────────────────────
class CorrelatedOutcomeModel:
    """Builds empirical distributions once; samples correlated weeks cheaply."""

    def __init__(self, team_corr: float = DEFAULT_TEAM_CORR,
                 pos_loading: dict | None = None,
                 anchor_to_projection: bool = True,
                 tail_shrink: bool = True,
                 availability_model: bool = True,
                 correlation_model: bool = True,
                 handcuff_model: bool = True,
                 matchup_model: bool = True,
                 platform: str = DEFAULT_PLATFORM):
        self.platform = str(platform)
        self.base_a = float(np.sqrt(np.clip(team_corr, 0.0, 1.0)))
        self.pos_loading = dict(pos_loading or POS_LOADING)
        # Anchor each player's empirical SHAPE to their 2026 projected LEVEL,
        # so role/team/QB changes (which the projection reflects) reprice the
        # historical distribution and stale "fake value" disappears. Disable
        # for a pure-historical baseline.
        self.anchor = bool(anchor_to_projection)
        # Shrink unreliable (small-sample) SHAPE toward the archetype pool so the
        # convex objective can't chase one-spike-week fake ceilings. A/B flag.
        self.tail_shrink = bool(tail_shrink)
        # Model per-week availability (injuries/DNP) instead of 17-game iron men.
        # A/B flag; per-season application lives in season_sim.
        self.availability_model = bool(availability_model)
        # Redistribute an injured starter's workload to his same-team same-position
        # backup (handcuff inheritance). A/B flag; needs availability_model on to fire.
        self.handcuff_model = bool(handcuff_model)
        self._handcuff_transfer = dict(HANDCUFF_TRANSFER)
        # Modulate weekly output by the week's opponent defense-vs-position rating
        # (mean-1 normalized per player; leverages the single-week playoff rounds).
        self.matchup_model = bool(matchup_model)
        self._def_ratings, self._opp_by_week = self._load_matchups()
        self._miss_rate: dict[str, float] = {}       # player_id -> historical miss rate
        self._miss_seasons: dict[str, int] = {}      # player_id -> qualifying seasons
        # Structural correlation: game -> team volume -> target competition.
        # A/B flag; when False, fall back to the single-team-factor copula.
        self.correlation_model = bool(correlation_model)
        self._sched: dict[int, dict[str, int]] = self._load_schedule()
        self._own: dict[str, np.ndarray] = {}        # player_id -> qgrid (shrunk)
        self._raw_own: dict[str, np.ndarray] = {}    # player_id -> qgrid (raw, for audit)
        self._own_name: dict[str, str] = {}          # norm name -> player_id
        self._n_weeks: dict[str, int] = {}           # player_id -> sample size
        self._display: dict[str, str] = {}           # player_id -> display name
        self._pos_of: dict[str, str] = {}            # player_id -> position
        self._pool: dict[tuple[str, int], np.ndarray] = {}  # (pos,tier) -> qgrid
        self._pool_shape: dict[tuple[str, int], np.ndarray] = {}  # mean-1 shape
        self._pos_pool_shape: dict[str, np.ndarray] = {}    # pos -> mean-1 shape
        self._proj_tier: dict[str, int] = {}         # player_id -> tier
        self._proj_tier_name: dict[str, int] = {}    # norm name -> tier
        self._proj_ppg: dict[str, float] = {}        # player_id -> projected ppg
        self._proj_ppg_name: dict[str, float] = {}   # norm name -> projected ppg
        self._built = False

    # -- build -------------------------------------------------------------
    def build(self, weekly_csv: Path = WEEKLY_CSV,
              projections: Path | None = PROJECTIONS,
              platform: str | None = None) -> "CorrelatedOutcomeModel":
        if platform is not None:
            self.platform = str(platform)
        df = pd.read_csv(weekly_csv, low_memory=False)
        if "season_type" in df.columns:
            df = df[df["season_type"].fillna("REG") == "REG"]
        df = df[df["season"] >= MIN_SEASON]
        df = df[df["position"].isin(SCORE_POSITIONS)].copy()
        df = recalc_weekly_df(df, platform=self.platform)

        # 1. Comparable pools + their mean-1 SHAPES (shrink targets). Tier each
        #    player-SEASON by within-position PPG rank, then pool that tier.
        season_ppg = (df.groupby(["player_id", "season", "position"])
                        ["fantasy_points_ppr"].mean().reset_index())
        season_ppg["tier"] = (
            season_ppg.groupby(["season", "position"])["fantasy_points_ppr"]
            .transform(lambda s: pd.qcut(s.rank(method="first"),
                                         q=min(N_TIERS, max(1, s.nunique())),
                                         labels=False, duplicates="drop"))
        )
        # qcut labels ascending (0 = worst); flip so tier 0 = elite.
        season_ppg["tier"] = (N_TIERS - 1) - season_ppg["tier"].fillna(0).astype(int)
        key = df.merge(season_ppg[["player_id", "season", "tier"]],
                       on=["player_id", "season"], how="left")
        for (pos, tier), g in key.groupby(["position", "tier"]):
            grid = _quantile_grid(g["fantasy_points_ppr"].to_numpy(dtype=float))
            self._pool[(pos, int(tier))] = grid
            if grid.mean() > 0:
                self._pool_shape[(pos, int(tier))] = grid / grid.mean()
        for pos, g in df.groupby("position"):
            grid = _quantile_grid(g["fantasy_points_ppr"].to_numpy(dtype=float))
            if grid.mean() > 0:
                self._pos_pool_shape[pos] = grid / grid.mean()

        # 2. Optional projection tiers / ppg. DK EV training passes market
        #    anchors and tiers on each roster player, so it builds with
        #    projections=None and does not read the old projection board.
        if projections is not None:
            self._build_projection_tiers(projections)

        # 3. Raw per-player grids + metadata + historical availability.
        raw: dict[str, np.ndarray] = {}
        for pid, g in df.groupby("player_id"):
            scores = g["fantasy_points_ppr"].to_numpy(dtype=float)
            # Historical injury durability — ONLY for players who were starters
            # (a >=13-game season exists), else games_played reflects depth-chart
            # ROLE, not injury. Average availability over starter-level seasons.
            gp_by_season = g.groupby("season").size()
            if gp_by_season.max() >= STARTER_SEASON_GAMES:
                started = gp_by_season[gp_by_season >= STARTED_MIN_GAMES]
                if len(started) >= 1:
                    a_hist = float(started.mean()) / FULL_SEASON   # avail when starting
                    self._miss_rate[pid] = float(np.clip(1.0 - a_hist, *MISS_RATE_CLIP))
                    self._miss_seasons[pid] = int(len(started))
            if len(scores) >= MIN_OWN_WEEKS:
                raw[pid] = _quantile_grid(scores)
                self._n_weeks[pid] = int(len(scores))
                self._pos_of[pid] = g["position"].iloc[-1]
                disp = g["player_display_name"].iloc[-1]
                self._display[pid] = disp
                nm = normalize_name(disp)
                if nm:
                    self._own_name.setdefault(nm, pid)

        # 4. Shrink unreliable shapes toward the archetype pool (or keep raw).
        for pid, grid in raw.items():
            self._raw_own[pid] = grid
            self._own[pid] = self._shrink_grid(pid, grid) if self.tail_shrink else grid

        self._built = True
        return self

    def _shrink_target(self, pid: str) -> np.ndarray | None:
        """Mean-1 archetype shape to shrink toward: same position + projected
        tier when known, else the position-wide pool shape."""
        pos = self._pos_of.get(pid)
        if pos is None:
            return None
        tier = self._proj_tier.get(pid)
        if tier is not None and (pos, tier) in self._pool_shape:
            return self._pool_shape[(pos, tier)]
        return self._pos_pool_shape.get(pos)

    def _shrink_grid(self, pid: str, grid: np.ndarray) -> np.ndarray:
        """Blend the player's mean-1 shape with the archetype shape by
        reliability w = n/(n+PRIOR), then restore the historical level. Anomalous
        (small-sample) tails collapse toward the archetype; stud ceilings that
        already match their archetype are barely touched."""
        gm = float(grid.mean())
        target = self._shrink_target(pid)
        if gm <= 0.5 or target is None:
            return grid
        n = self._n_weeks.get(pid, 0)
        w = n / (n + SHRINK_PRIOR)
        blended_shape = w * (grid / gm) + (1.0 - w) * target   # both mean ~1, sorted
        return blended_shape * gm                              # back to historical level

    def _build_projection_tiers(self, projections: Path) -> None:
        """Map each projected player to a fallback tier (0 = elite at position)."""
        import json
        try:
            recs = json.loads(Path(projections).read_text())
        except Exception:
            return
        proj = pd.DataFrame(recs)
        if proj.empty or "proj_points" not in proj:
            return
        proj["tier"] = (
            proj.groupby("position")["proj_points"]
            .transform(lambda s: (N_TIERS - 1)
                       - pd.qcut(s.rank(method="first"),
                                 q=min(N_TIERS, max(1, s.nunique())),
                                 labels=False, duplicates="drop").fillna(0).astype(int))
        )
        for _, r in proj.iterrows():
            t = int(r["tier"]) if pd.notna(r["tier"]) else N_TIERS // 2
            ppg = (float(r["proj_points"]) / N_GAMES
                   if pd.notna(r.get("proj_points")) else None)
            pid = str(r["player_id"]) if pd.notna(r.get("player_id")) else None
            nm = normalize_name(r.get("player_display_name", ""))
            if pid:
                self._proj_tier[pid] = t
                if ppg and ppg > 0:
                    self._proj_ppg[pid] = ppg
            if nm:
                self._proj_tier_name.setdefault(nm, t)
                if ppg and ppg > 0:
                    self._proj_ppg_name.setdefault(nm, ppg)

    def _anchor(self, grid: np.ndarray, pid: str, nm: str,
                src: str, active_frac: float = 1.0) -> tuple[np.ndarray, str]:
        """Rescale a historical SHAPE so its mean matches the 2026 projected
        LEVEL. Multiplicative scaling preserves boom/bust shape (and CV).

        proj_points is trained on actual season totals, so it already includes
        expected missed games. Anchor the per-ACTIVE-WEEK mean to the healthy
        level proj_ppg / active_frac; the availability layer then removes ~
        (1 - active_frac) of weeks, returning the expected season to proj_points
        rather than double-discounting it. With the availability model off,
        active_frac is 1.0 and the anchor is proj_ppg (full healthy season)."""
        if not self.anchor:
            return grid, src
        ppg = self._proj_ppg.get(pid) or self._proj_ppg_name.get(nm)
        gm = float(grid.mean())
        if ppg and gm > 0.5:
            healthy_ppg = ppg / max(active_frac, 0.5)
            scale = float(np.clip(healthy_ppg / gm, *ANCHOR_CLIP))
            return grid * scale, src + "+anc"
        return grid, src

    def _anchor_with_player(self, grid: np.ndarray, player: dict,
                            src: str, active_frac: float = 1.0) -> tuple[np.ndarray, str]:
        """Anchor to a roster-supplied market level when present.

        DK EV training passes `proj_points` from DK ADP market position, not the
        old projection board. If absent, fall back to the projection lookup used
        by older callers/tests.
        """
        pid = str(player.get("player_id") or "")
        name = player.get("name") or player.get("player_display_name") or ""
        nm = normalize_name(name)
        ppg = None
        try:
            pts = float(player.get("proj_points"))
            if pts > 0:
                ppg = pts / N_GAMES
        except (TypeError, ValueError):
            ppg = None

        if not self.anchor:
            return grid, src
        gm = float(grid.mean())
        if ppg and gm > 0.5:
            healthy_ppg = ppg / max(active_frac, 0.5)
            scale = float(np.clip(healthy_ppg / gm, *ANCHOR_CLIP))
            return grid * scale, src + "+market"
        return self._anchor(grid, pid, nm, src, active_frac)

    # -- availability params -----------------------------------------------
    def _blended_miss_rate(self, player: dict) -> float:
        """Expected fraction of the season a player misses: historical starter-level
        miss rate (when available) blended with the position prior by season count."""
        pos = player.get("position", "")
        prior = POS_MISS_RATE.get(pos, 0.11)
        pid = str(player.get("player_id") or "")
        nm = normalize_name(player.get("name") or player.get("player_display_name") or "")
        hist = self._miss_rate.get(pid)
        if hist is None and nm in self._own_name:
            hist = self._miss_rate.get(self._own_name[nm])
        if hist is not None:
            ns = self._miss_seasons.get(pid) or \
                self._miss_seasons.get(self._own_name.get(nm, ""), 1)
            w = ns / (ns + K_AV)
            m = w * hist + (1.0 - w) * prior
        else:
            m = prior
        return float(np.clip(m, *MISS_RATE_CLIP))

    def _expected_active_fraction(self, player: dict) -> float:
        """E[fraction of season active]. 1.0 when the availability model is off
        (iron-man mode), else 1 - blended miss rate. Used to gross the projection
        anchor up to a healthy-week level before the availability layer removes
        weeks, so expected availability-adjusted points return to proj_points
        instead of being discounted twice."""
        if not self.availability_model:
            return 1.0
        return float(np.clip(1.0 - self._blended_miss_rate(player), 0.5, 1.0))

    def _availability_params(self, player: dict) -> tuple[float, float]:
        """(q_minor, p_major) for a player. Historical miss rate blended with the
        position prior, split into independent per-week absences and a clustered
        major-injury hazard. Calibrated so expected games missed = miss_rate*17."""
        pos = player.get("position", "")
        m = self._blended_miss_rate(player)
        phi = POS_MAJOR_FRAC.get(pos, 0.55)
        q_minor = (1.0 - phi) * m                       # E[minor missed]=q_minor*17
        p_major = float(np.clip(phi * m * FULL_SEASON / MAJOR_DUR_MEAN, *P_MAJOR_CLIP))
        return q_minor, p_major

    def sample_availability(self, ctx: RosterContext, n_seasons: int,
                            n_weeks: int, rng: np.random.Generator) -> np.ndarray:
        """Active mask [n_players, n_seasons, n_weeks]: True = plays that week.
        Minor absences are independent per week; a major injury knocks the player
        out for a consecutive block (season-ending or 2-6 weeks) that can wipe
        the Wk15-17 playoffs."""
        n = len(ctx.player_ids)
        minor_active = rng.random((n, n_seasons, n_weeks)) >= ctx.q_minor[:, None, None]

        has_major = rng.random((n, n_seasons)) < ctx.p_major[:, None]
        start = rng.integers(1, n_weeks + 1, size=(n, n_seasons))          # 1..W
        ending = rng.random((n, n_seasons)) < SEASON_END_FRAC
        dur = rng.integers(MAJOR_DUR_RANGE[0], MAJOR_DUR_RANGE[1] + 1,
                           size=(n, n_seasons))
        end = np.where(ending, n_weeks + 1, start + dur)                   # exclusive
        wk = np.arange(1, n_weeks + 1)[None, None, :]
        out = (has_major[..., None] & (wk >= start[..., None]) & (wk < end[..., None]))
        return minor_active & ~out

    # -- resolve a single roster player to a quantile grid -----------------
    def _resolve_grid(self, player: dict) -> tuple[np.ndarray, str]:
        pid = str(player.get("player_id") or "")
        name = player.get("name") or player.get("player_display_name") or ""
        pos = player.get("position")
        nm = normalize_name(name)
        active_frac = self._expected_active_fraction(player)

        if pid and pid in self._own:
            return self._anchor_with_player(self._own[pid], player, "own", active_frac)
        if nm and nm in self._own_name:
            return self._anchor_with_player(self._own[self._own_name[nm]], player, "own", active_frac)

        # Fallback: comparable (position, tier) pool, also anchored to the
        # player's projected level when known (sharpens rookies/team-changers).
        if pos in SCORE_POSITIONS:
            tier = None
            try:
                tier = int(player.get("market_tier"))
            except (TypeError, ValueError):
                tier = None
            tier = self._proj_tier.get(pid) if tier is None else tier
            if tier is None:
                tier = self._proj_tier_name.get(nm, N_TIERS // 2)
            tier = int(np.clip(tier, 0, N_TIERS - 1))
            grid = self._pool.get((pos, tier))
            if grid is not None:
                return self._anchor_with_player(grid, player, f"pool:{pos}:{tier}", active_frac)
            # any tier for that position
            for t in range(N_TIERS):
                if (pos, t) in self._pool:
                    return self._anchor_with_player(self._pool[(pos, t)], player,
                                                    f"pool:{pos}:{t}", active_frac)
        # Last resort: flat low-scoring grid so a miss can't silently boom.
        return np.zeros(GRID_K), "miss"

    # -- prepare a roster --------------------------------------------------
    def prepare_roster(self, roster: list[dict]) -> RosterContext:
        if not self._built:
            raise RuntimeError("call build() before prepare_roster()")
        n = len(roster)
        ids, names, positions, teams, sources = [], [], [], [], []
        loadings = np.zeros(n)
        qgrids = np.zeros((n, GRID_K))
        q_minor = np.zeros(n)
        p_major = np.zeros(n)
        bye_weeks = np.zeros(n, dtype=int)

        # Map teams (None/'' -> unique pseudo-team so player is independent).
        team_key, team_idx = {}, np.zeros(n, dtype=int)
        next_team = 0
        for i, p in enumerate(roster):
            grid, src = self._resolve_grid(p)
            qgrids[i] = grid
            sources.append(src)
            ids.append(str(p.get("player_id") or ""))
            names.append(p.get("name") or p.get("player_display_name") or "")
            pos = p.get("position", "")
            positions.append(pos)
            loadings[i] = self.base_a * self.pos_loading.get(pos, 0.0)
            q_minor[i], p_major[i] = self._availability_params(p)
            try:
                bye_weeks[i] = int(p.get("bye_week") or 0)
            except (TypeError, ValueError):
                bye_weeks[i] = 0

            tm = p.get("team") or p.get("recent_team") or ""
            if not tm:
                tm = f"__solo_{i}"
            teams.append(p.get("team") or p.get("recent_team") or "")
            if tm not in team_key:
                team_key[tm] = next_team
                next_team += 1
            team_idx[i] = team_key[tm]

        # Handcuff groups: same team + same handcuff-eligible position with >=2
        # rostered players, ordered by healthy workload (qgrid mean) descending so
        # role 0 = the presumed starter. anchor[i] ~ the player's healthy ppg level.
        anchor = qgrids.mean(axis=1)
        grouped: dict[tuple[int, str], list[int]] = {}
        for i in range(n):
            if positions[i] in self._handcuff_transfer and teams[i]:
                grouped.setdefault((int(team_idx[i]), positions[i]), []).append(i)
        handcuff_groups = [
            np.array(sorted(members, key=lambda j: -anchor[j]), dtype=int)
            for members in grouped.values() if len(members) >= 2
        ]

        return RosterContext(ids, names, positions, teams, team_idx,
                             loadings, qgrids, sources, next_team, q_minor, p_major,
                             bye_weeks=bye_weeks, anchor=anchor,
                             handcuff_groups=handcuff_groups or None)

    # -- playoff-matchup multipliers ---------------------------------------
    @staticmethod
    def _load_matchups() -> tuple[dict, dict]:
        """(def_ratings, opponent_by_week). def_ratings[TEAM][POS] -> ~1.0 multiplier;
        opponent_by_week[nfl_week][TEAM] -> opponent TEAM (from the 2026 schedule)."""
        import json
        ratings: dict = {}
        try:
            ratings = json.loads(DEFENSE_RATINGS.read_text()).get("ratings", {})
        except Exception:
            ratings = {}
        opp: dict[int, dict[str, str]] = {}
        try:
            sched = json.loads(SCHEDULE_2026.read_text())
            for wk, pairs in sched.items():
                m: dict[str, str] = {}
                for pair in pairs:
                    if len(pair) == 2:
                        a, b = str(pair[0]).upper(), str(pair[1]).upper()
                        m[a] = b
                        m[b] = a
                opp[int(wk)] = m
        except Exception:
            opp = {}
        return ratings, opp

    def _matchup_multipliers(self, ctx: RosterContext, n_weeks: int) -> np.ndarray:
        """[n, n_weeks] per-week opponent matchup multiplier, normalized to mean 1.0
        per player so the season total is unchanged (only the across-week distribution
        shifts). Neutral 1.0 on bye/unknown weeks and when data is missing."""
        n = len(ctx.player_ids)
        M = np.ones((n, n_weeks))
        if not self._def_ratings or not self._opp_by_week:
            return M
        for i in range(n):
            team = str(ctx.teams[i]).upper()
            pos = ctx.positions[i]
            for w in range(n_weeks):
                opp = self._opp_by_week.get(w + 1, {}).get(team)
                if opp is None:
                    continue
                r = self._def_ratings.get(opp, {}).get(pos)
                if r:
                    M[i, w] = float(r)
            mu = M[i].mean()
            if mu > 1e-6:
                M[i] /= mu          # keep season total anchored; only redistribute
        return M

    # -- handcuff workload transfer ----------------------------------------
    def apply_workload_transfer(self, scores: np.ndarray, active: np.ndarray,
                                ctx: RosterContext) -> np.ndarray:
        """Scale up a same-team same-position backup on weeks his higher-workload
        teammate is inactive: the backup inherits a fraction of the vacated role.

        scores/active: [n, S, W]. Multiplicative (shape-preserving). The inactive
        starter is still zeroed afterwards by `scores * active` in the caller."""
        groups = getattr(ctx, "handcuff_groups", None)
        if not groups or ctx.anchor is None:
            return scores
        for idx in groups:
            a = ctx.anchor[idx]                                  # [k] desc, healthy ppg
            if a[0] <= 0.5:
                continue
            act = active[idx]                                    # [k, S, W]
            # Role a member occupies that week = number of higher-workload teammates
            # who ARE active (they take the slots above him). 0 => he's the top active.
            role = np.clip(np.cumsum(act, axis=0) - act, 0, len(idx) - 1)
            role_anchor = a[role]                                # [k, S, W]
            frac = np.array([self._handcuff_transfer.get(ctx.positions[i], 0.0)
                             for i in idx])[:, None, None]
            own = a[:, None, None]
            eff = own + frac * np.clip(role_anchor - own, 0.0, None)
            scale = np.where(own > 1e-6, eff / own, 1.0)
            scale = np.where(act, scale, 1.0)                    # only boost players on the field
            scores[idx] = scores[idx] * scale
        return scores

    # -- sample ------------------------------------------------------------
    def sample_weeks(self, ctx: RosterContext, n_weeks: int,
                     rng: np.random.Generator) -> np.ndarray:
        """Correlated DK scores, shape [n_players, n_weeks]."""
        n = len(ctx.player_ids)
        z = rng.standard_normal((ctx.n_teams, n_weeks))        # team factors
        z_p = z[ctx.team_idx, :]                               # [n, n_weeks]
        e = rng.standard_normal((n, n_weeks))                  # idiosyncratic
        L = ctx.loadings[:, None]
        g = L * z_p + np.sqrt(np.clip(1.0 - L * L, 0.0, 1.0)) * e
        q = _norm_cdf(g)                                       # uniforms in (0,1)
        idx = np.clip((q * GRID_K).astype(int), 0, GRID_K - 1)
        rows = np.arange(n)[:, None]
        return ctx.qgrids[rows, idx]

    def sample_week(self, ctx: RosterContext,
                    rng: np.random.Generator) -> np.ndarray:
        """Correlated DK scores for one week, shape [n_players]."""
        return self.sample_weeks(ctx, 1, rng)[:, 0]

    # -- structural correlation (game -> team -> opportunity -> points) -----
    @staticmethod
    def _load_schedule() -> dict[int, dict[str, int]]:
        """{nfl_week: {team_abbr: game_id}}. Two teams in a game share a game_id
        (-> shared game environment -> bring-back). Missing file -> no games."""
        import json
        try:
            grid = json.loads(SCHEDULE_2026.read_text())
        except Exception:
            return {}
        out: dict[int, dict[str, int]] = {}
        for wk, pairs in grid.items():
            m = {}
            for gid, pair in enumerate(pairs):
                for team in pair:
                    m[str(team).upper()] = gid
            out[int(wk)] = m
        return out

    def _week_game_ids(self, week: int, team_names: list[str]) -> tuple[np.ndarray, int]:
        """Per-team game id for an NFL week. Teams not playing (bye/unknown) get
        their own solo game (independent environment)."""
        sched = self._sched.get(week, {})
        ids = np.empty(len(team_names), dtype=int)
        solo = max(sched.values(), default=-1) + 1
        for t, name in enumerate(team_names):
            gid = sched.get(str(name).upper())
            if gid is None:
                gid = solo; solo += 1
            ids[t] = gid
        return ids, int(solo)

    def sample_scores(self, ctx: RosterContext, n_seasons: int, n_weeks: int,
                      rng: np.random.Generator, matchup: bool | None = None) -> np.ndarray:
        """Correlated DK scores [n_players, n_seasons, n_weeks].

        Structural path (correlation_model=True): for each NFL week, a shared
        GAME environment lifts both teams in a matchup (bring-back); each team's
        passing-VOLUME factor lifts its QB and pass-catchers; same-team catchers
        also draw a ZERO-SUM target-competition shock (cancels the WR-WR boost,
        not QB-WR). Marginals are untouched (latent -> quantile -> empirical grid).

        matchup: None -> use self.matchup_model; True/False -> A/B the per-week
        opponent defense-vs-position modulation (mean-1 per player, season-neutral).
        """
        use_mu = self.matchup_model if matchup is None else matchup
        n = len(ctx.player_ids)
        if not self.correlation_model:
            flat = self.sample_weeks(ctx, n_seasons * n_weeks, rng)
            out = flat.reshape(n, n_seasons, n_weeks)
            if use_mu:
                out = out * self._matchup_multipliers(ctx, n_weeks)[:, None, :]
            return out

        # Team structure (reuse ctx.team_idx grouping; name per team for schedule).
        n_teams = ctx.n_teams
        team_name = [""] * n_teams
        for i in range(n):
            team_name[ctx.team_idx[i]] = ctx.teams[i]
        team_idx = ctx.team_idx
        pos = np.array(ctx.positions)

        # Per-player structural loadings.
        team_load = np.where(pos == "QB", A_QB,
                             np.array([A_TEAM.get(p, 0.0) for p in pos]))
        comp_load = np.where(pos == "QB", 0.0,
                             np.array([A_COMP.get(p, 0.0) for p in pos]))
        idio = np.sqrt(np.clip(1.0 - team_load ** 2 - comp_load ** 2, 0.0, 1.0))

        # Same-team catcher sets (for the zero-sum competition factor).
        catcher_sets: list[np.ndarray] = []
        for t in range(n_teams):
            members = np.array([i for i in range(n)
                                if team_idx[i] == t and pos[i] in CATCHER_POS])
            if len(members) >= 2:
                catcher_sets.append(members)

        g = np.empty((n, n_seasons, n_weeks))
        for w in range(n_weeks):
            game_ids, n_games = self._week_game_ids(w + 1, team_name)  # NFL week = w+1
            E = rng.standard_normal((n_games, n_seasons))             # game environment
            wt = rng.standard_normal((n_teams, n_seasons))            # team-specific
            V = A_GAME * E[game_ids] + np.sqrt(1 - A_GAME ** 2) * wt  # [n_teams, S]

            c = rng.standard_normal((n, n_seasons))                  # raw share shocks
            Scomp = np.zeros((n, n_seasons))
            for members in catcher_sets:                             # zero-sum per team
                k = len(members)
                dev = c[members] - c[members].mean(axis=0, keepdims=True)
                Scomp[members] = dev / np.sqrt(1.0 - 1.0 / k)        # Corr=-1/(k-1), Var=1

            e = rng.standard_normal((n, n_seasons))
            g[:, :, w] = (team_load[:, None] * V[team_idx]
                          + comp_load[:, None] * Scomp + idio[:, None] * e)

        q = _norm_cdf(g)
        idx = np.clip((q * GRID_K).astype(int), 0, GRID_K - 1)
        rows = np.arange(n)[:, None, None]
        out = ctx.qgrids[rows, idx]
        if use_mu:
            out = out * self._matchup_multipliers(ctx, n_weeks)[:, None, :]
        return out


# ── Matchup helpers (shared by the policy feature builders) ──────────────────
PLAYOFF_WEEKS = (15, 16, 17)   # the single-week bracket rounds R2/R3/R4


def load_matchup_tables() -> tuple[dict, dict]:
    """(def_ratings, opponent_by_week) from the 2026 ratings + schedule artifacts."""
    return CorrelatedOutcomeModel._load_matchups()


def playoff_matchup_rating(team: str, pos: str, ratings: dict, opp: dict,
                           weeks: tuple[int, ...] = PLAYOFF_WEEKS) -> float:
    """Average defense-vs-position rating a player faces in `weeks` (default the
    playoff weeks). ~1.0 neutral; >1 = soft playoff slate (higher ceiling when it
    decides advancement); <1 = tough. 1.0 when unknown."""
    team = str(team).upper()
    vals = []
    for w in weeks:
        o = opp.get(w, {}).get(team)
        if o is None:
            continue
        r = ratings.get(o, {}).get(pos)
        if r:
            vals.append(float(r))
    return float(sum(vals) / len(vals)) if vals else 1.0


# ── Convenience API ─────────────────────────────────────────────────────────
_DEFAULT_MODELS: dict[str, CorrelatedOutcomeModel] = {}


def load_default_model(platform: str = DEFAULT_PLATFORM) -> CorrelatedOutcomeModel:
    """Build (and cache) the default model from repo data."""
    key = str(platform)
    if key not in _DEFAULT_MODELS:
        _DEFAULT_MODELS[key] = CorrelatedOutcomeModel(platform=key).build()
    return _DEFAULT_MODELS[key]


_MARKET_MODELS: dict[str, CorrelatedOutcomeModel] = {}


def load_market_model(platform: str = DEFAULT_PLATFORM) -> CorrelatedOutcomeModel:
    """Build/cache the market outcome model without reading projections.json."""
    key = str(platform)
    if key not in _MARKET_MODELS:
        _MARKET_MODELS[key] = CorrelatedOutcomeModel(platform=key).build(projections=None)
    return _MARKET_MODELS[key]


def simulate_roster_week(roster: list[dict],
                         rng: np.random.Generator | None = None,
                         model: CorrelatedOutcomeModel | None = None) -> dict:
    """
    The step-1 deliverable: given a roster and a simulated week, return
    correlated DK fantasy scores for every player.

    roster: list of {name|player_id, position, team} dicts.
    Returns: {player label -> DK score}.
    """
    rng = rng or np.random.default_rng()
    model = model or load_default_model()
    ctx = model.prepare_roster(roster)
    scores = model.sample_week(ctx, rng)
    return {ctx.names[i] or ctx.player_ids[i]: float(scores[i])
            for i in range(len(scores))}


if __name__ == "__main__":
    m = load_default_model()
    demo = [
        {"name": "Josh Allen", "position": "QB", "team": "BUF"},
        {"name": "Khalil Shakir", "position": "WR", "team": "BUF"},
        {"name": "Bijan Robinson", "position": "RB", "team": "ATL"},
    ]
    print("own/pool/miss sources:", m.prepare_roster(demo).sources)
    print("one simulated week:", simulate_roster_week(demo, np.random.default_rng(0)))
