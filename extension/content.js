// DraftManager — universal draft assistant
// Scans any draft page for known player names, shows AI recommendations.

const SERVER = "http://localhost:8765";
const SCAN_MS_FAST   = 1200;   // when it's your turn or a few picks away
const SCAN_MS_NORMAL = 3000;   // within the next ~8 picks
const SCAN_MS_SLOW   = 7000;   // far from your turn
const SCAN_MS_DK     = 5000;   // DraftKings is API/event-driven; keep polling very light


// ── State ─────────────────────────────────────────────────────────────────────

const state = {
  settings: { teams: 12, pickPos: 1, rounds: 20, scoring: "full" },  // DK Best Ball = 20-round/20-player roster
  playerIndex: {},       // normalize(name) → player object (includes abbrev aliases)
  canonicalNorms: new Set(),  // only full-name norms (used for available set)
  allNames: [],          // sorted player names for scanning
  available: new Set(),  // normalized names currently on the board
  queue: [],             // normalized canonical player names in draft preference order
  autodraftArmed: false,
  autodraftBusy: false,
  dkQueued: new Set(),   // player keys we've already starred this draft (stars TOGGLE — never re-click)
  lastAutodraftPick: null,
  autodraftCountdown: null,    // { pickNo, target, fireAt, timerId } while the 15s grace runs
  autodraftCancelledPick: null,// overall pick the user manually cancelled (don't re-arm it)
  autodraftBackstopKey: null,  // player key we proactively starred on DK as a backstop
  autodraftVerify: null,       // { pickNo, key, name, ... , tries } after a Draft-click, pending board confirmation
  myTeam: [],            // { name, position, round }
  manualMyRoster: [],    // legacy my-team override (migrated into manualAdds on load)
  manualAdds: [],        // user-assigned picks for ANY team: { key, name, position, slot }
  manualRemovals: new Set(), // draftKeys the user pulled back onto the board (sticky vs auto feed)
  allDrafted: [],        // { name, position, round, by_user }
  lastRecs: null,
  connected: false,
  scanTimer: null,
  mutationBuffer: null,
  activePosFilter: "all",
  pageCurrentPick: null,   // next pick on the board when detected from the site
  teamRosters: {},         // {teamSlot: [{name, position, round}]}
  viewingTeamSlot: null,   // which team is shown in the team viewer (null = user's team)
  scanInFlight: false,
  pendingScan: false,
  customHistorySelector: null,  // user-set CSS selector override for pick history
  customRosterSelector: null,   // user-set CSS selector override for DK roster panels
  customRosterSlot: null,       // optional team slot for a single visible DK roster panel
  containerPlayerCounts: new Map(), // container key → last known player count (behavioral detection)
  historyNewestFirst: null,    // null=auto-detect, true/false=forced by user
  _orderedNewestFirst: null,   // what ordering was used in last sync (for flip detection)
  scanCount: 0,                // total scans since connect (used for delayed warning)
  _domSnapshot: new Set(),     // DK: player norms visible in DOM on last scan
  _pickPending: false,         // DK: SelectionRecorded fired but pick not yet identified
  _dkPickEventSeq: 0,
  _lastDkPickDebug: null,
  _lastRosterSyncDebug: null,
  // DraftKings authoritative JSON pick feed (see syncFromDKApi). When active, the
  // DOM/WS pick recorders are suppressed because this feed is ground truth.
  dkDraftables: {},        // draftableId -> { name, position, team, bye, playerId }
  dkByPlayerId: {},        // playerId    -> same record (fallback lookup)
  dkApiActive: false,      // true once the DK JSON feed is driving allDrafted
  dkUserKey: null,         // our entrant userKey (self) as seen in the feed
  _pendingDraftBoard: null,
  dkLastStatusBoard: null,
  feedPollTimer: null,     // periodic draftStatus re-fetch (keeps the feed live)
};

function currentSitePlatform() {
  const host = location.hostname.toLowerCase();
  if (host.includes("underdog") || host.includes("playunderdog")) return "underdog";
  if (host.includes("draftkings")) return "draftkings";
  return "draftkings";
}

function applyPlatformDefaults(platform) {
  if (platform === "underdog") {
    state.settings.teams = 12;
    state.settings.rounds = 18;
    state.settings.scoring = "half";
  } else if (platform === "draftkings" && state.settings.rounds === 18 && state.settings.scoring === "half") {
    state.settings.rounds = 20;
    state.settings.scoring = "full";
  }
}

// Round is derived from total picks — never track it manually.
function currentRound() {
  return Math.floor((getNextBoardPick() - 1) / state.settings.teams) + 1;
}

function getTrackedNextPick() {
  let maxPick = 0;
  state.allDrafted.forEach((pick, idx) => {
    const explicitPick = Number(pick.overall_pick);
    const fallbackPick = idx + 1;
    const pickNum = Number.isFinite(explicitPick) && explicitPick >= 1 ? explicitPick : fallbackPick;
    maxPick = Math.max(maxPick, pickNum);
  });
  return maxPick + 1;
}

function getNextBoardPick() {
  const fromHistory = getTrackedNextPick();
  const fromPage    = state.pageCurrentPick || 0;

  // Ground truth: how many players have actually left the board. Until at least
  // ONE player is drafted, the draft is at pick 1 — never let a misread page
  // number (a jersey #, our own draft-slot label, "Pick 8 of 12", etc.) jump the
  // pick ahead and falsely trigger "your pick" on a full board.
  const draftedCount = (state.canonicalNorms.size && state.available.size)
    ? Math.max(0, state.canonicalNorms.size - state.available.size)
    : state.allDrafted.filter((d) => !d._phNorm).length;
  if (state.allDrafted.length === 0 && draftedCount <= 0) return 1;

  // Once we are tracking picks, the tracked next pick (max overall pick + 1, which
  // reads DK's "Pick #N" labels) is the reliable source. The page-text number is
  // demoted because DK frequently renders our own draft SLOT as "Pick N", which is
  // NOT the current overall pick — trusting it re-introduced the false "your pick".
  if (fromHistory > 1) return fromHistory;

  // No usable tracking yet, but the board shows the draft has started: fall back to
  // the page number as a last-resort bootstrap (history sync will take over).
  return fromPage >= 1 ? fromPage : fromHistory;
}

function pickNumberForNewDetectedPick() {
  const fromHistory = getTrackedNextPick();
  const fromPage = Number(state.pageCurrentPick) || 0;
  const maxPicks = state.settings.teams * (state.settings.rounds || 20);

  // DraftKings page text often contains draft-slot labels and other "Pick N"
  // strings. For live detection, count actual detected picks from 1 and only
  // jump when a parser gives an explicit overall pick number.
  if (location.hostname.toLowerCase().includes("draftkings")) return fromHistory;

  // On refresh, DK may only expose partial history. The page's current pick keeps
  // snake team assignment aligned when the extension sees the next pick event.
  if (fromPage >= fromHistory && fromPage <= maxPicks) {
    if (fromHistory <= 1 || fromPage <= fromHistory + state.settings.teams) return fromPage;
  }
  return fromHistory;
}

// ── Normalization ─────────────────────────────────────────────────────────────

function normalize(name) {
  return name.toLowerCase()
    .replace(/[.,'\-]/g, "")   // include comma so "Chase, Ja'Marr" → "chase jamarr"
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\s+(jr|sr|ii|iii|iv)$/i, "");  // strip generational suffixes
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

// ── Settings persistence ──────────────────────────────────────────────────────

let _settingsLoadPromise = null;

function saveAutodraftState() {
  savePerDraftCfg();  // armed flag is per-draft, not global
}

function loadSettings() {
  if (_settingsLoadPromise) return _settingsLoadPromise;
  _settingsLoadPromise = new Promise((resolve) => {
    try {
      chrome.storage.local.get(["dmSettings", "dmHistorySelector", "dmRosterSelector", "dmRosterSlot", "dmHistoryNewestFirst", "dmPlatform", "dmAutodraftArmed"], (r) => {
      if (r.dmSettings) {
        // Never apply a global pickPos (legacy stored value) — slot is per-draft.
        const { pickPos: _ignoredSlot, ...globalSettings } = r.dmSettings;
        Object.assign(state.settings, globalSettings);
      }
      // Migrate the legacy 15-round default: the extension only supports DraftKings,
      // whose Best Ball contest is a 20-round/20-player roster. A stored 15 is stale.
      if (state.settings.rounds === 15) {
        state.settings.rounds = 20;
        saveSettings();
      }
      const sitePlatform = currentSitePlatform();
      applyPlatformDefaults(sitePlatform);
      if (r.dmHistorySelector) {
        state.customHistorySelector = r.dmHistorySelector;
        const inp = document.getElementById("dm-history-selector");
        if (inp) inp.value = r.dmHistorySelector;
      }
      if (r.dmRosterSelector) state.customRosterSelector = r.dmRosterSelector;
      if (r.dmRosterSlot) {
        const rs = Number(r.dmRosterSlot);
        if (Number.isFinite(rs) && rs >= 1) state.customRosterSlot = rs;
      }
      if (r.dmHistoryNewestFirst !== undefined) state.historyNewestFirst = r.dmHistoryNewestFirst;
      const platformToRestore = sitePlatform === "underdog" ? "underdog" : (r.dmPlatform || "draftkings");
      if (platformToRestore === "draftkings" || platformToRestore === "underdog") {
        // Restore saved platform to server on reconnect
        fetch(`${SERVER}/platform`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ platform: platformToRestore }),
        }).catch(() => {});
        const platSel = document.getElementById("dm-platform");
        if (platSel) platSel.value = platformToRestore;
        try { chrome.storage.local.set({ dmPlatform: platformToRestore }); } catch {}
      }
      // Armed state is per-draft (loadPerDraftCfg), not global — do not inherit
      // another draft's armed flag here.
        updateSettingsUI();
        resolve();
      });
    } catch {
      resolve();
    }
  });
  return _settingsLoadPromise;
}

function saveSettings() {
  // Global defaults are teams/rounds only. pickPos is deliberately EXCLUDED so a
  // brand-new draft can't inherit the last draft's slot — it lives only in the
  // per-draft config (savePerDraftCfg) keyed by draft id.
  const { pickPos, ...globalSettings } = state.settings;
  try { chrome.storage.local.set({ dmSettings: globalSettings }); } catch {}
  savePerDraftCfg();
}

function dmQueueKey() {
  return `dmQueue:${draftId()}`;  // per-draft so concurrent drafts don't share a queue
}

function loadQueue() {
  try {
    const qKey = dmQueueKey();
    chrome.storage.local.get([qKey], (r) => {
      const rawQueue = Array.isArray(r[qKey]) ? r[qKey] : [];
      state.queue = rawQueue
        .map((name) => draftKeyForName(name))
        .filter((key) => state.canonicalNorms.has(key));
      pruneQueue();
      renderQueue();
    });
  } catch {}
}

function saveQueue() {
  try {
    chrome.storage.local.set({
      [dmQueueKey()]: state.queue.map((key) => state.playerIndex[key]?.name || key),
    });
  } catch {}
}

// Stable per-draft identity. Used so that running several drafts at once keeps
// each draft's picks, pick position, and armed state isolated instead of sharing
// one global slot.
//
// Prefer the draft id embedded in the path (DraftKings: /draft/<type>/<id>, e.g.
// /draft/snake/191306202 or /draft/tournament/<hex>). This is deliberately
// immune to transient query params like ?tournamentSuccess=true that DK strips
// shortly after load — including them would re-key the draft mid-session and
// orphan its tracked picks.
function draftId() {
  const host = location.hostname.replace(/^www\./, "");
  const m = location.pathname.match(/\/draft\/[^/]+\/([A-Za-z0-9_-]+)/);
  if (m) return `${host}:${m[1]}`;
  const ud = location.href.match(/(?:draft|contest|entry|room|slate|tournament)[=/_-]([A-Za-z0-9_-]{6,})/i);
  if (ud) return `${host}:${ud[1]}`;
  // Fallback for other sites: pathname only (no search/hash, which can change).
  const path = location.pathname
    .replace(/[^a-zA-Z0-9_-]+/g, "_")
    .slice(0, 160) || "draft";
  return `${host}:${path}`;
}

function currentDraftFingerprint() {
  return {
    host: location.hostname.replace(/^www\./, ""),
    path: location.pathname,
    href: location.href.split("#")[0],
    draftId: draftId(),
  };
}

// NOTE: deliberately does NOT include pickPos/teams/rounds — keying on those
// orphaned a draft's saved picks the moment you set or auto-corrected your slot.
function draftStateKey() {
  return `dmDraftState:${draftId()}`;
}

function draftCfgKey() {
  return `dmDraftCfg:${draftId()}`;
}

// Per-draft config: your pick slot + whether autodraft is armed for THIS draft.
// Stored per draft URL so a second draft tab can't overwrite this draft's slot
// (the bug where a fresh draft thought it was your pick on the very first board
// pick) and so arming one draft doesn't arm all of them.
function savePerDraftCfg() {
  try {
    chrome.storage.local.set({
      [draftCfgKey()]: {
        pickPos: state.settings.pickPos,
        autodraftArmed: !!state.autodraftArmed,
        savedAt: Date.now(),
      },
    });
  } catch {}
}

function loadPerDraftCfg() {
  return new Promise((resolve) => {
    try {
      chrome.storage.local.get([draftCfgKey()], (r) => {
        const cfg = r[draftCfgKey()];
        if (cfg) {
          const pp = Number(cfg.pickPos);
          if (Number.isFinite(pp) && pp >= 1) state.settings.pickPos = pp;
          state.autodraftArmed = cfg.autodraftArmed === true;
        } else {
          // No saved config for this draft yet: never inherit another draft's
          // armed state. Start disarmed until explicitly armed here.
          state.autodraftArmed = false;
        }
        resolve(!!cfg);
      });
    } catch { resolve(false); }
  });
}

function normalizeDraftEntry(entry, idx) {
  if (!entry || !entry.name) return null;
  const explicitPick = Number(entry.overall_pick);
  const overallPick = Number.isFinite(explicitPick) && explicitPick >= 1 ? explicitPick : idx + 1;
  const round = Number(entry.round) || Math.floor((overallPick - 1) / state.settings.teams) + 1;
  const slot = Number(entry.team_slot) || pickToTeamSlot(overallPick, state.settings.teams);
  const out = {
    name: String(entry.name),
    position: String(entry.position || ""),
    round,
    by_user: !!entry.by_user,
    team_slot: slot,
    overall_pick: overallPick,
  };
  if (entry._draftKey) out._draftKey = String(entry._draftKey);
  if (entry._phNorm) out._phNorm = String(entry._phNorm);
  return out;
}

function normalizeManualRosterEntry(entry, idx) {
  const rawName = typeof entry === "string" ? entry : entry?.name;
  if (!rawName || typeof rawName !== "string") return null;
  const player = matchPlayer(rawName);
  const name = player?.name || rawName.trim();
  const position = player?.position || entry?.position || "";
  return {
    name,
    position,
    round: Number(entry?.round) || idx + 1,
    _draftKey: draftKeyForName(name),
  };
}

// ── Unified sticky manual override layer (any team) ──────────────────────────
// Source of truth for "what's drafted" = auto-detected picks (DK JSON feed / DOM)
// MINUS state.manualRemovals PLUS state.manualAdds. The override is re-applied
// after every auto rebuild so a feed refresh can never wipe a hand correction.
//   manualAdds:     [{ key, name, position, slot }]  — user-assigned to a team
//   manualRemovals: Set<draftKey>                    — user pulled back to board

function manualRosterKeys() {
  // Keys the user manually assigned (any team) — treated as drafted/off-board.
  return new Set((state.manualAdds || []).map((p) => p.key || draftKeyForName(p.name)).filter(Boolean));
}

function hasManualRosterOverride() {
  return (state.manualAdds && state.manualAdds.length > 0) ||
         (state.manualRemovals && state.manualRemovals.size > 0);
}

// Re-derive state.allDrafted from the current auto layer + manual overrides.
// Idempotent: strips its own prior _manualAdd entries first, so calling it after
// each auto rebuild yields the same effective board regardless of feed churn.
function applyManualOverrides() {
  if (!Array.isArray(state.manualAdds)) state.manualAdds = [];
  if (!(state.manualRemovals instanceof Set)) state.manualRemovals = new Set(state.manualRemovals || []);

  // 1. Drop prior manual-add rows (re-added fresh below) and any auto row the
  //    user explicitly removed.
  state.allDrafted = state.allDrafted.filter((d) => {
    if (d._manualAdd) return false;
    const key = d._draftKey || draftKeyForName(d.name);
    return !state.manualRemovals.has(key);
  });

  // 2. Splice in manual adds, each to its target team's next free roster slot.
  const perSlotCount = {};
  state.allDrafted.forEach((d) => {
    if (d._phNorm) return;
    const s = Number(d.team_slot) || pickToTeamSlot(d.overall_pick || 1, state.settings.teams);
    perSlotCount[s] = (perSlotCount[s] || 0) + 1;
  });
  const teams = state.settings.teams;
  state.manualAdds.forEach((add) => {
    const key = add.key || draftKeyForName(add.name);
    if (!key) return;
    // Skip if an auto row already covers this player (avoid duplicates).
    if (state.allDrafted.some((d) => (d._draftKey || draftKeyForName(d.name)) === key)) return;
    const slot = Number(add.slot) || state.settings.pickPos;
    const ordinal = (perSlotCount[slot] || 0) + 1;
    perSlotCount[slot] = ordinal;
    const overall = teamSlotPickNumber(slot, ordinal, teams);
    const round = Math.floor((overall - 1) / teams) + 1;
    state.allDrafted.push({
      name: add.name,
      position: add.position || "",
      round,
      by_user: slot === state.settings.pickPos,
      team_slot: slot,
      overall_pick: overall,
      _draftKey: key,
      _manualAdd: true,
    });
  });

  state.allDrafted.sort((a, b) => (Number(a.overall_pick) || 9999) - (Number(b.overall_pick) || 9999));
}

// Legacy alias — old call sites expect the my-team override to be (re)applied.
function applyManualMyRosterOverride() { applyManualOverrides(); }

// Single choke point: apply overrides, then rebuild rosters + available board.
// Every path that rebuilds allDrafted from an auto source funnels through here.
function reconcileDraftState() {
  applyManualOverrides();
  rebuildTeamStateFromDrafted();
  refreshAvailableFromDrafted();
}

function manualRosterText() {
  return state.myTeam.map((p) => p.name).join("\n");
}

function parseManualRosterInput(raw) {
  const entries = Array.isArray(raw)
    ? raw
    : String(raw || "").split(/[,;\n]/).map((s) => s.trim()).filter(Boolean);
  return entries
    .map((entry, idx) => normalizeManualRosterEntry(entry, idx))
    .filter(Boolean);
}

// ── Manual edit API (drives the unified override layer) ──────────────────────

// Assign a player to a team by hand. Removes them from the board and, if the
// user had previously pulled them back, clears that removal. Sticky vs the feed.
function manualAddPlayer(name, slot, source = "manual edit") {
  const entry = normalizeManualRosterEntry(name, (state.manualAdds || []).length);
  if (!entry) return false;
  const key = entry._draftKey || draftKeyForName(entry.name);
  if (!key) return false;
  const targetSlot = Number(slot) || state.settings.pickPos;

  // Un-remove if it was on the manual-removal list.
  if (state.manualRemovals instanceof Set) state.manualRemovals.delete(key);

  // De-dupe across manual adds; re-point slot if it moved teams.
  state.manualAdds = (state.manualAdds || []).filter((a) => (a.key || draftKeyForName(a.name)) !== key);
  state.manualAdds.push({ key, name: entry.name, position: entry.position || "", slot: targetSlot });

  reconcileDraftState();
  pruneQueue();
  saveDraftState();
  state._lastRenderSig = null;
  renderTeamRosters();
  const resolved = !!state.playerIndex[key];
  setStatus(`added ${entry.name} → T${targetSlot}${resolved ? "" : " (unmatched)"}`);
  console.log(`[DM] Manual add (${source}): ${entry.name} (${entry.position || "?"}) → T${targetSlot}${resolved ? "" : " — not in player DB, check spelling"}`);
  scanAndRank(true);
  return true;
}

// Pull a player back onto the board. If it was a hand-added pick, just forget the
// add; if it was auto-detected, record a sticky removal so the feed won't re-add.
function manualRemoveByKey(key, source = "manual edit") {
  if (!key) return false;
  if (!Array.isArray(state.manualAdds)) state.manualAdds = [];
  if (!(state.manualRemovals instanceof Set)) state.manualRemovals = new Set(state.manualRemovals || []);

  const wasAdd = state.manualAdds.some((a) => (a.key || draftKeyForName(a.name)) === key);
  if (wasAdd) {
    state.manualAdds = state.manualAdds.filter((a) => (a.key || draftKeyForName(a.name)) !== key);
  } else {
    state.manualRemovals.add(key);
  }

  reconcileDraftState();
  pruneQueue();
  saveDraftState();
  state._lastRenderSig = null;
  renderTeamRosters();
  setStatus("removed pick — back on board");
  console.log(`[DM] Manual remove (${source}): ${key}${wasAdd ? " (forgot manual add)" : " (sticky removal)"}`);
  scanAndRank(true);
  return true;
}

// Back-compat: set MY whole roster from a name list. Implemented as manual adds
// to my slot (replacing any prior manual adds for my slot).
function applyManualRosterInput(raw, source = "manual roster editor") {
  const roster = parseManualRosterInput(raw);
  const unresolved = roster.filter((p) => !state.playerIndex[draftKeyForName(p.name)]);
  const mySlot = state.settings.pickPos;
  // Drop existing manual adds for my slot, then re-add the new list.
  state.manualAdds = (state.manualAdds || []).filter((a) => (Number(a.slot) || mySlot) !== mySlot);
  roster.forEach((p) => {
    const key = draftKeyForName(p.name);
    if (!key) return;
    if (state.manualRemovals instanceof Set) state.manualRemovals.delete(key);
    state.manualAdds.push({ key, name: p.name, position: p.position || "", slot: mySlot });
  });
  reconcileDraftState();
  pruneQueue();
  saveDraftState();
  state._lastRenderSig = null;
  renderTeamRosters();
  setStatus(roster.length ? `manual team set: ${roster.length} players` : "manual team cleared");
  console.log(`[DM] Manual my-roster set from ${source} (${roster.length} players):`, roster.map((p) => `${p.name} (${p.position || "?"})`));
  if (unresolved.length > 0) {
    console.warn("[DM] These names did not exactly resolve to the player DB; fix spelling if they still show available:", unresolved.map((p) => p.name));
  }
  scanAndRank(true);
  return { roster, unresolved };
}

function clearManualRosterOverride() {
  state.manualMyRoster = [];
  state.manualAdds = [];
  state.manualRemovals = new Set();
  recomputeUserPicks();
  state._lastRenderSig = null;
  renderTeamRosters();
  setStatus("manual overrides cleared");
  console.log("[DM] All manual overrides cleared.");
  scanAndRank(true);
}

function rebuildTeamStateFromDrafted() {
  state.myTeam = [];
  state.teamRosters = {};
  state.allDrafted.forEach((entry) => {
    if (entry._phNorm) return;
    const slot = Number(entry.team_slot) || pickToTeamSlot(entry.overall_pick || 1, state.settings.teams);
    if (!state.teamRosters[slot]) state.teamRosters[slot] = [];
    state.teamRosters[slot].push({ name: entry.name, position: entry.position, round: entry.round });
    if (entry.by_user) {
      state.myTeam.push({ name: entry.name, position: entry.position, round: entry.round });
    }
  });
  // NOTE: manual overrides are already baked into state.allDrafted by
  // applyManualOverrides() (see reconcileDraftState) — do NOT re-derive here.
}

function refreshAvailableFromDrafted() {
  if (!state.canonicalNorms || state.canonicalNorms.size === 0) return;
  state.available = new Set([...state.canonicalNorms].filter((norm) => !draftedKeys().has(norm)));
}

// Re-derive each tracked pick's team_slot (pure snake math from its overall pick,
// independent of OUR slot) and by_user (= that slot is our slot) for the current
// pickPos/teams, then rebuild rosters/my-team. Call this whenever the pick slot
// or team count changes so a corrected slot doesn't leave stale by_user/team_slot
// in saved state (which would send the server wrong team_rosters).
function recomputeUserPicks() {
  const { pickPos, teams } = state.settings;
  state.allDrafted.forEach((d) => {
    if (d._phNorm) return;  // unknown placeholder — no roster identity
    const slot = pickToTeamSlot(d.overall_pick || 1, teams);
    d.team_slot = slot;
    d.by_user = (slot === pickPos);
  });
  reconcileDraftState();
  saveDraftState();
}

function addMissingPickPlaceholders(nextPick, reason = "catch-up") {
  const teams = state.settings.teams || 12;
  const maxPicks = teams * (state.settings.rounds || 20);
  const targetNextPick = Math.min(Math.max(Number(nextPick) || 1, 1), maxPicks + 1);
  let currentNextPick = getTrackedNextPick();
  let added = 0;

  while (currentNextPick < targetNextPick) {
    const phNorm = `__ph_${currentNextPick}`;
    if (!draftedKeys().has(phNorm)) {
      const slot = pickToTeamSlot(currentNextPick, teams);
      const round = Math.floor((currentNextPick - 1) / teams) + 1;
      state.allDrafted.push({
        name: `Unknown Pick ${currentNextPick}`,
        position: "?",
        round,
        by_user: false,
        team_slot: slot,
        overall_pick: currentNextPick,
        _phNorm: phNorm,
        _createdAt: Date.now(),
      });
      added++;
    }
    currentNextPick++;
  }

  if (added > 0) {
    state.pageCurrentPick = targetNextPick;
    state._lastDkPickDebug = {
      source: reason,
      added_placeholders: added,
      next_pick: targetNextPick,
      at: Date.now(),
    };
    console.warn(`[DM] Added ${added} placeholder pick(s) to catch up to pick ${targetNextPick} (${reason}).`);
    recomputeUserPicks();
    renderTeamRosters();
  }
  return added;
}

function recordUnknownDKPick(reason = "unresolved DK pick event") {
  if (state.dkApiActive && state.allDrafted.length > 0) return false;  // non-empty DK JSON feed is authoritative
  const nextPick = getTrackedNextPick();
  return addMissingPickPlaceholders(nextPick + 1, reason) > 0;
}

function saveDraftState() {
  try {
    const maxPicks = state.settings.teams * (state.settings.rounds || 20);
    chrome.storage.local.set({
      [draftStateKey()]: {
        savedAt: Date.now(),
        draftFingerprint: currentDraftFingerprint(),
        settings: { ...state.settings },
        pageCurrentPick: state.pageCurrentPick,
        historyNewestFirst: state.historyNewestFirst,
        orderedNewestFirst: state._orderedNewestFirst,
        manualAdds: (state.manualAdds || []).map(({ key, name, position, slot }) => ({ key, name, position, slot })),
        manualRemovals: [...(state.manualRemovals instanceof Set ? state.manualRemovals : [])],
        allDrafted: state.allDrafted
          .map((entry, idx) => normalizeDraftEntry(entry, idx))
          .filter(Boolean)
          .filter((entry) => entry.overall_pick <= maxPicks),
      },
    });
  } catch {}
}

function clearDraftState() {
  try { chrome.storage.local.remove([draftStateKey()]); } catch {}
}

function loadDraftState() {
  return new Promise((resolve) => {
    try {
      chrome.storage.local.get([draftStateKey()], (r) => {
        const saved = r[draftStateKey()];
        const maxAgeMs = 12 * 60 * 60 * 1000;
        const savedSettings = saved?.settings || {};
        // pickPos is intentionally NOT part of this check (it's per-draft config
        // now, and changing your slot must not discard the draft's tracked picks).
        const sameDraftShape =
          savedSettings.teams === state.settings.teams &&
          savedSettings.rounds === state.settings.rounds;
        const fp = saved?.draftFingerprint || {};
        const curFp = currentDraftFingerprint();
        const isUnderdog = currentSitePlatform() === "underdog";
        const sameDraftIdentity = isUnderdog
          ? (fp.draftId === curFp.draftId && fp.path === curFp.path && fp.href === curFp.href)
          : (!fp.draftId || (fp.draftId === curFp.draftId && fp.path === curFp.path && fp.href === curFp.href));
        if (!saved || !sameDraftShape || !sameDraftIdentity || Date.now() - Number(saved.savedAt || 0) > maxAgeMs) {
          if (saved && !sameDraftIdentity) clearDraftState();
          resolve(false);
          return;
        }

        const maxPicks = state.settings.teams * (state.settings.rounds || 20);
        state.allDrafted = (Array.isArray(saved.allDrafted) ? saved.allDrafted : [])
          .map((entry, idx) => normalizeDraftEntry(entry, idx))
          .filter(Boolean)
          .filter((entry) => entry.overall_pick <= maxPicks);
        state.pageCurrentPick = Number(saved.pageCurrentPick) || state.pageCurrentPick;

        // Restore the unified override layer.
        state.manualAdds = (Array.isArray(saved.manualAdds) ? saved.manualAdds : [])
          .map((a, idx) => {
            const e = normalizeManualRosterEntry(a, idx);
            if (!e) return null;
            return { key: e._draftKey || draftKeyForName(e.name), name: e.name, position: e.position || "", slot: Number(a.slot) || state.settings.pickPos };
          })
          .filter(Boolean);
        state.manualRemovals = new Set(Array.isArray(saved.manualRemovals) ? saved.manualRemovals : []);
        // Legacy migration: old saves stored a my-team-only manualMyRoster.
        if (state.manualAdds.length === 0 && Array.isArray(saved.manualMyRoster) && saved.manualMyRoster.length) {
          saved.manualMyRoster.forEach((entry, idx) => {
            const e = normalizeManualRosterEntry(entry, idx);
            if (e) state.manualAdds.push({ key: e._draftKey || draftKeyForName(e.name), name: e.name, position: e.position || "", slot: state.settings.pickPos });
          });
        }
        state.manualMyRoster = [];

        if (saved.historyNewestFirst !== undefined) state.historyNewestFirst = saved.historyNewestFirst;
        if (saved.orderedNewestFirst !== undefined) state._orderedNewestFirst = saved.orderedNewestFirst;

        // Drop any baked manual-add rows (saved unflagged) so reconcile re-adds
        // them flagged as _manualAdd — keeps the override layer recognizable.
        const addKeys = new Set(state.manualAdds.map((a) => a.key));
        if (addKeys.size > 0) {
          state.allDrafted = state.allDrafted.filter((d) => !addKeys.has(d._draftKey || draftKeyForName(d.name)));
        }
        reconcileDraftState();
        if (state.allDrafted.length > 0) {
          setStatus(`restored ${state.allDrafted.length} tracked picks`);
          renderTeamRosters();
        }
        resolve(state.allDrafted.length > 0);
      });
    } catch {
      resolve(false);
    }
  });
}

// ── Panel HTML ────────────────────────────────────────────────────────────────

function createPanel() {
  if (document.getElementById("dm-panel")) return;

  const panel = document.createElement("div");
  panel.id = "dm-panel";
  panel.innerHTML = `
    <div id="dm-header">
      <span id="dm-title-wrap">
        <span id="dm-title">DraftManager</span>
        <span id="dm-season-badge"></span>
      </span>
      <span id="dm-status">connecting...</span>
      <div id="dm-controls">
        <button id="dm-settings-btn" title="Settings">⚙</button>
        <button id="dm-collapse-btn" title="Collapse">−</button>
      </div>
    </div>

    <!-- Settings panel (hidden by default) -->
    <div id="dm-settings-panel" style="display:none">
      <div class="dm-setting-row">
        <label>Teams</label>
        <input id="dm-teams-input" type="number" min="8" max="16" value="12" />
      </div>
      <div class="dm-setting-row">
        <label>My pick #</label>
        <input id="dm-pick-input" type="number" min="1" max="16" value="1" />
      </div>
      <div class="dm-setting-row">
        <label>Rounds</label>
        <input id="dm-rounds-input" type="number" min="10" max="25" value="20" />
      </div>
      <div class="dm-setting-row">
        <label>Override pick#</label>
        <input id="dm-pick-override" type="number" min="1" max="300" placeholder="auto" />
      </div>
      <div class="dm-setting-row" style="flex-direction:column;align-items:flex-start;gap:2px;">
        <label>Pick history selector <span style="color:#555;font-weight:400">(leave blank = auto)</span></label>
        <input id="dm-history-selector" type="text" placeholder="e.g. [data-testid='pick-list']" style="width:100%;font-size:10px;" />
      </div>
      <div class="dm-setting-row">
        <label>History order</label>
        <select id="dm-history-order" style="background:#1a1a2e;border:1px solid #333;color:#e0e0e0;padding:3px 6px;border-radius:3px;font-size:11px;">
          <option value="auto">Auto-detect</option>
          <option value="newest">Newest first (reversed)</option>
          <option value="oldest">Oldest first (normal)</option>
        </select>
      </div>
      <div class="dm-setting-row">
        <label>Platform</label>
        <select id="dm-platform" style="background:#1a1a2e;border:1px solid #333;color:#e0e0e0;padding:3px 6px;border-radius:3px;font-size:11px;">
          <option value="draftkings">DraftKings</option>
          <option value="underdog">Underdog</option>
        </select>
      </div>
      <div class="dm-setting-row">
        <label>Scoring</label>
        <select id="dm-scoring" style="background:#1a1a2e;border:1px solid #333;color:#e0e0e0;padding:3px 6px;border-radius:3px;font-size:11px;">
          <option value="full">Full PPR</option>
          <option value="half">Half PPR</option>
        </select>
      </div>
      <button id="dm-settings-save">Save & Rescan</button>
    </div>

    <div id="dm-tabs">
      <button class="dm-tab dm-tab-active" data-tab="picks">PICKS</button>
      <button class="dm-tab" data-tab="teams">TEAMS</button>
    </div>

    <div id="dm-body">
      <!-- PICKS tab content -->
      <div id="dm-tab-picks">
        <div id="dm-search-wrap">
          <input id="dm-search" type="text" placeholder="Filter by name or team (KC...)" autocomplete="off" />
          <button id="dm-search-clear" title="Clear search (Esc)" style="display:none">×</button>
        </div>
        <div id="dm-pos-filters">
          <button class="dm-pos-btn dm-pos-btn-active" data-pos="all">ALL</button>
          <button class="dm-pos-btn" data-pos="qb">QB</button>
          <button class="dm-pos-btn" data-pos="wr">WR</button>
          <button class="dm-pos-btn" data-pos="rb">RB</button>
          <button class="dm-pos-btn" data-pos="te">TE</button>
        </div>
        <div id="dm-roster-summary"></div>
        <div id="dm-queue-section"></div>
        <div id="dm-recs-section"></div>
        <div id="dm-myteam-section"></div>
        <div id="dm-alerts-section"></div>
        <div id="dm-scarcity-section"></div>
      </div>

      <!-- TEAMS tab content -->
      <div id="dm-tab-teams" style="display:none">
        <div id="dm-teams-section"></div>
      </div>
    </div>

    <div id="dm-footer">
      <button id="dm-scan-btn">↺ Rescan</button>
      <button id="dm-undraft-btn" title="Undo last pick">↩ Undo</button>
      <button id="dm-copy-btn" title="Copy team to clipboard">⎘</button>
    </div>

    <div id="dm-resize-handle" title="Drag to resize"></div>
  `;
  document.body.appendChild(panel);

  makeDraggable(panel, document.getElementById("dm-header"));
  makeResizable(panel, document.getElementById("dm-resize-handle"));
  loadPanelPosition(panel);
  loadPanelSize(panel);
  bindEvents();
  loadSettings();
}

function switchTab(tab) {
  document.getElementById("dm-tab-picks").style.display  = tab === "picks" ? "" : "none";
  document.getElementById("dm-tab-teams").style.display  = tab === "teams" ? "" : "none";
  document.querySelectorAll(".dm-tab").forEach((btn) => {
    btn.classList.toggle("dm-tab-active", btn.dataset.tab === tab);
  });
  if (tab === "teams") {
    renderTeamRosters();
  }
}

function bindEvents() {
  document.getElementById("dm-collapse-btn").onclick = toggleCollapse;

  document.querySelectorAll(".dm-tab").forEach((btn) => {
    btn.onclick = () => switchTab(btn.dataset.tab);
  });

  document.getElementById("dm-settings-btn").onclick = () => {
    const sp = document.getElementById("dm-settings-panel");
    sp.style.display = sp.style.display === "none" ? "" : "none";
  };

  document.getElementById("dm-scoring")?.addEventListener("change", (e) => {
    const roundsInput = document.getElementById("dm-rounds-input");
    if (!roundsInput) return;
    const currentRounds = parseInt(roundsInput.value, 10) || 0;
    if (e.target.value === "half" && (!currentRounds || currentRounds === 20)) {
      roundsInput.value = "18";
    } else if (e.target.value === "full" && currentRounds === 18) {
      roundsInput.value = "20";
    }
  });

  document.getElementById("dm-settings-save").onclick = () => {
    const prevPickPos = state.settings.pickPos;
    const prevTeams   = state.settings.teams;
    state.settings.teams   = parseInt(document.getElementById("dm-teams-input").value) || 12;
    state.settings.pickPos = parseInt(document.getElementById("dm-pick-input").value)  || 1;
    state.settings.rounds  = parseInt(document.getElementById("dm-rounds-input").value) || 20;
    state.settings.scoring = document.getElementById("dm-scoring")?.value === "half" ? "half" : "full";
    const overridePick = parseInt(document.getElementById("dm-pick-override")?.value);
    if (overridePick >= 1) {
      state.pageCurrentPick = overridePick;
      addMissingPickPlaceholders(overridePick, "manual override");
    }
    // History selector override
    const selInput = document.getElementById("dm-history-selector")?.value.trim();
    state.customHistorySelector = selInput || null;
    _cachedHistoryEl = null;  // force re-detect
    try { chrome.storage.local.set({ dmHistorySelector: selInput || "" }); } catch {}
    // History order
    const orderVal = document.getElementById("dm-history-order")?.value;
    state.historyNewestFirst = orderVal === "newest" ? true : orderVal === "oldest" ? false : null;
    try { chrome.storage.local.set({ dmHistoryNewestFirst: state.historyNewestFirst }); } catch {}
    // Platform
    const platVal = document.getElementById("dm-platform")?.value;
    if (platVal) {
      applyPlatformDefaults(platVal);
      updateSettingsUI();
      fetch(`${SERVER}/platform`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ platform: platVal }),
      }).catch(() => {});
      try { chrome.storage.local.set({ dmPlatform: platVal }); } catch {}
    }
    saveSettings();
    _pickPosDetected = false;  // allow re-detection after settings change
    // The draft state key is URL-stable now, so changing your slot/teams keeps
    // the tracked picks — but their by_user/team_slot must be re-derived for the
    // new slot, or the server gets stale team_rosters.
    if (state.settings.pickPos !== prevPickPos || state.settings.teams !== prevTeams) {
      recomputeUserPicks();
    } else {
      saveDraftState();
    }
    document.getElementById("dm-settings-panel").style.display = "none";
    scanAndRank();
  };

  document.getElementById("dm-scan-btn").onclick = async (e) => {
    if (e.altKey) {
      // Alt+Rescan: show pick history debug info in console + status
      const histEl = findPickHistory();
      if (histEl) {
        const rawPicks     = readPickHistory(histEl);
        const orderedPicks = orderPickHistory(rawPicks);
        const tag  = histEl.tagName.toLowerCase();
        const id   = histEl.id ? `#${histEl.id}` : "";
        const cls  = [...histEl.classList].slice(0, 2).join(".");
        const hint = `${tag}${id}${cls ? "." + cls : ""}`;
        setStatus(`✓ History: ${hint} — ${orderedPicks.length} picks`);
        console.log("[DraftManager] History element:", histEl);
        console.log("[DraftManager] Picks (ordered):", orderedPicks);
        console.log("[DraftManager] Tip: call window.dm.highlight() to see all player containers");
        console.log("[DraftManager] Tip: call window.dm.setHistorySelector('YOUR_CSS_SEL') to override");
      } else {
        setStatus("No history detected — open console for help");
        console.log("[DraftManager] No pick history container found.");
        console.log("  → Call window.dm.highlight() to see colored outlines on all player containers");
        console.log("  → Right-click the pick history panel → Inspect → find its selector");
        console.log("  → Call window.dm.setHistorySelector('your-css-selector')");
        console.log("  → Or paste the selector in ⚙ Settings > Pick history selector");
      }
      return;
    }
    await scanAndRank(true);
    startAdaptiveScan();
  };

  document.getElementById("dm-copy-btn").onclick = () => {
    if (state.myTeam.length === 0) return;
    const lines = state.myTeam.map((p) => {
      const data = state.playerIndex[normalize(p.name)];
      const pts  = data?.proj_points ? ` (${data.proj_points.toFixed(0)} pts)` : "";
      return `Rd${p.round} ${p.position}: ${p.name}${pts}`;
    });
    const header = `My Team — ${state.myTeam.length} picks · Round ${Math.max(1, currentRound() - 1)}`;
    const text   = [header, ...lines].join("\n");
    navigator.clipboard?.writeText(text).then(() => {
      const btn = document.getElementById("dm-copy-btn");
      if (btn) { btn.textContent = "✓"; setTimeout(() => { btn.textContent = "⎘"; }, 2000); }
    });
  };

  document.getElementById("dm-undraft-btn").onclick = () => {
    // Undo the most recent pick (user or opponent)
    if (state.allDrafted.length === 0) return;
    const last = state.allDrafted.pop();
    if (last.by_user) state.myTeam = state.myTeam.filter((p) => p.name !== last.name);
    if (last.team_slot && state.teamRosters[last.team_slot]) {
      state.teamRosters[last.team_slot] = state.teamRosters[last.team_slot].filter(
        (p) => p.name !== last.name
      );
    }
    state.available.add(normalize(last.name));
    saveDraftState();
    setStatus(`Undid: ${last.name}`);
    renderTeamRosters();
    scanAndRank();
  };

  document.getElementById("dm-search").oninput = (e) => {
    const q = normalize(e.target.value);
    const clearBtn = document.getElementById("dm-search-clear");
    if (clearBtn) clearBtn.style.display = q ? "" : "none";
    filterRows(q, state.activePosFilter || "all");
  };

  document.getElementById("dm-search-clear").onclick = () => {
    const box = document.getElementById("dm-search");
    if (box) { box.value = ""; box.focus(); }
    document.getElementById("dm-search-clear").style.display = "none";
    filterRows("", state.activePosFilter || "all");
  };

  document.getElementById("dm-pos-filters").addEventListener("click", (e) => {
    const btn = e.target.closest(".dm-pos-btn");
    if (!btn) return;
    document.querySelectorAll(".dm-pos-btn").forEach((b) => b.classList.remove("dm-pos-btn-active"));
    btn.classList.add("dm-pos-btn-active");
    state.activePosFilter = btn.dataset.pos;
    const searchVal = normalize(document.getElementById("dm-search").value);
    filterRows(searchVal, state.activePosFilter);
  });

  // Text selection → instant filter (player name or team abbreviation)
  document.addEventListener("mouseup", () => {
    const sel = window.getSelection()?.toString()?.trim();
    if (!sel || sel.length < 2 || sel.length > 45) return;
    const norm = normalize(sel);
    const isPlayer = !!state.playerIndex[norm];
    const isTeam   = /^[a-z]{2,4}$/.test(norm) &&
                     Object.values(state.playerIndex).some((p) => (p.team || "").toLowerCase() === norm);
    if (!isPlayer && !isTeam) return;
    const searchBox = document.getElementById("dm-search");
    if (searchBox && !document.getElementById("dm-panel").contains(document.activeElement)) {
      searchBox.value = isTeam ? norm.toUpperCase() : sel;
      filterRows(norm, state.activePosFilter || "all");
    }
  });

  // Keyboard shortcuts (only when panel is not collapsed)
  document.addEventListener("keydown", (e) => {
    const panel = document.getElementById("dm-panel");
    if (!panel || panel.classList.contains("dm-collapsed")) return;
    // "/" → focus search
    if (e.key === "/" && !isTextEntryElement(document.activeElement)) {
      e.preventDefault();
      document.getElementById("dm-search")?.focus();
    }
    // 1/2/3/4/0 → position filter shortcuts (when not typing in search)
    if (!isTextEntryElement(document.activeElement)) {
      const posMap = { "1": "qb", "2": "wr", "3": "rb", "4": "te", "0": "all" };
      if (posMap[e.key]) {
        const targetPos = posMap[e.key];
        document.querySelectorAll(".dm-pos-btn").forEach((b) => {
          b.classList.toggle("dm-pos-btn-active", b.dataset.pos === targetPos);
        });
        state.activePosFilter = targetPos;
        filterRows(normalize(document.getElementById("dm-search")?.value || ""), targetPos);
      }
    }
    // Escape → clear search
    if (e.key === "Escape") {
      const searchBox = document.getElementById("dm-search");
      if (searchBox && searchBox.value) {
        searchBox.value = "";
        const clearBtn = document.getElementById("dm-search-clear");
        if (clearBtn) clearBtn.style.display = "none";
        filterRows("", state.activePosFilter);
      }
    }
    // Shift+R → force rescan (without stealing plain "r" from the page)
    if (e.key === "R" && e.shiftKey && !isTextEntryElement(document.activeElement)) {
      scanAndRank(true);
    }
  });
}

function isTextEntryElement(el) {
  const tag = el?.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || el?.isContentEditable;
}

function updateSettingsUI() {
  const ti = document.getElementById("dm-teams-input");
  const pi = document.getElementById("dm-pick-input");
  const ri = document.getElementById("dm-rounds-input");
  if (ti) ti.value = state.settings.teams;
  if (pi) pi.value = state.settings.pickPos;
  if (ri) ri.value = state.settings.rounds;
  const scoringSel = document.getElementById("dm-scoring");
  if (scoringSel) scoringSel.value = state.settings.scoring === "half" ? "half" : "full";
  // Sync history order dropdown
  const orderSel = document.getElementById("dm-history-order");
  if (orderSel) {
    orderSel.value = state.historyNewestFirst === true  ? "newest"
                   : state.historyNewestFirst === false ? "oldest"
                   :                                      "auto";
  }
  // Sync platform from server
  fetch(`${SERVER}/platform`).then(r => r.json()).then(d => {
    const platSel = document.getElementById("dm-platform");
    if (platSel && d.platform) platSel.value = d.platform;
  }).catch(() => {});
}

function toggleCollapse() {
  const panel = document.getElementById("dm-panel");
  const btn   = document.getElementById("dm-collapse-btn");
  panel.classList.toggle("dm-collapsed");
  btn.textContent = panel.classList.contains("dm-collapsed") ? "+" : "−";
}

// ── Draggable ─────────────────────────────────────────────────────────────────

function ensurePanelVisible() {
  let panel = document.getElementById("dm-panel");
  if (!panel) {
    createPanel();
    panel = document.getElementById("dm-panel");
  }
  if (!panel) return;

  panel.style.display = "";
  const rect = panel.getBoundingClientRect();
  const offscreen = rect.right < 40
    || rect.left > window.innerWidth - 40
    || rect.bottom < 30
    || rect.top > window.innerHeight - 30;
  if (offscreen) resetPanelLayout(panel);
}

function savePanelPosition(el) {
  try {
    chrome.storage.local.set({ dmPanelPos: { left: el.style.left, top: el.style.top } });
  } catch {}
}

// Keep at least a corner of the panel on-screen so it can never get lost.
function clampPanelToViewport(el, left, top) {
  const w = el.getBoundingClientRect().width || 340;
  const margin = 80;  // px of panel that must stay reachable horizontally
  const maxLeft = Math.max(0, window.innerWidth  - margin);
  const maxTop  = Math.max(0, window.innerHeight - 30);  // keep header row visible
  return {
    left: Math.min(Math.max(left, -(w - margin)), maxLeft),
    top:  Math.min(Math.max(top, 0), maxTop),
  };
}

function loadPanelPosition(el) {
  try {
    chrome.storage.local.get(["dmPanelPos"], (r) => {
      if (r.dmPanelPos?.left) {
        const left = parseFloat(r.dmPanelPos.left);
        const top  = parseFloat(r.dmPanelPos.top);
        if (!isNaN(left) && !isNaN(top)) {
          const c = clampPanelToViewport(el, left, top);
          el.style.right = "auto";
          el.style.left  = c.left + "px";
          el.style.top   = c.top + "px";
        }
      }
    });
  } catch {}
}

function savePanelSize(el) {
  try {
    chrome.storage.local.set({ dmPanelSize: { width: el.style.width, height: el.style.height } });
  } catch {}
}

function loadPanelSize(el) {
  try {
    chrome.storage.local.get(["dmPanelSize"], (r) => {
      if (r.dmPanelSize?.width)  el.style.width = r.dmPanelSize.width;
      if (r.dmPanelSize?.height) {
        el.style.height = r.dmPanelSize.height;
        el.style.maxHeight = "none";  // honor the user's explicit height
      }
    });
  } catch {}
}

// Double-click the header to snap back to the default top-right size/position.
function resetPanelLayout(el) {
  el.style.left = "auto";
  el.style.top = "20px";
  el.style.right = "20px";
  el.style.width = "";
  el.style.height = "";
  el.style.maxHeight = "";
  try { chrome.storage.local.remove(["dmPanelPos", "dmPanelSize"]); } catch {}
}

function makeDraggable(el, handle) {
  let ox, oy, startL, startT;
  handle.addEventListener("dblclick", (e) => {
    if (e.target.tagName === "BUTTON" || e.target.tagName === "INPUT") return;
    resetPanelLayout(el);
  });
  handle.addEventListener("mousedown", (e) => {
    if (e.target.tagName === "BUTTON" || e.target.tagName === "INPUT") return;
    ox = e.clientX; oy = e.clientY;
    const r = el.getBoundingClientRect();
    startL = r.left; startT = r.top;
    el.style.right = "auto";
    const onMove = (e) => {
      const c = clampPanelToViewport(el, startL + e.clientX - ox, startT + e.clientY - oy);
      el.style.left = c.left + "px";
      el.style.top  = c.top + "px";
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      savePanelPosition(el);  // persist final position
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    e.preventDefault();
  });
}

function makeResizable(el, handle) {
  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    e.stopPropagation();  // don't trigger a drag
    const startX = e.clientX, startY = e.clientY;
    const r = el.getBoundingClientRect();
    const startW = r.width, startH = r.height;
    el.style.maxHeight = "none";  // allow free vertical resize past the 85vh cap
    const onMove = (ev) => {
      const maxW = window.innerWidth  - r.left - 6;
      const maxH = window.innerHeight - r.top  - 6;
      const w = Math.max(260, Math.min(startW + ev.clientX - startX, maxW));
      const h = Math.max(140, Math.min(startH + ev.clientY - startY, maxH));
      el.style.width  = w + "px";
      el.style.height = h + "px";
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      savePanelSize(el);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
}

// ── Server communication ──────────────────────────────────────────────────────

// All fetches go through the background service worker to bypass page CSP.
function bgFetch(url, options = {}) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(
      { type: "fetch", url, method: options.method || "GET", body: options.body || null },
      (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else if (response?.ok) {
          resolve(response.data);
        } else {
          reject(new Error(response?.error || "fetch failed"));
        }
      }
    );
  });
}

async function fetchPlayerIndex() {
  try {
    const players = await bgFetch(`${SERVER}/players`);

    state.playerIndex = {};
    state.canonicalNorms = new Set();
    state.allNames = [];
    for (const p of players) {
      if (!p.name) continue;
      const key = normalize(p.name);
      state.playerIndex[key] = p;
      state.canonicalNorms.add(key);
      state.allNames.push(p.name);
    }

    // Also index abbreviated names: "d henry" → Derrick Henry (handles sites that show "D. Henry")
    // Add AFTER the main loop so full names take priority
    const abbrevMap = {};
    for (const [fullNorm, player] of Object.entries(state.playerIndex)) {
      const parts = fullNorm.split(" ");
      if (parts.length >= 2 && parts[0].length > 1) {
        const abbrev = parts[0][0] + " " + parts.slice(1).join(" ");
        // Only add if unambiguous (no two players share the same first-initial + last name)
        if (!abbrevMap[abbrev]) abbrevMap[abbrev] = player;
        else abbrevMap[abbrev] = null;  // collision — mark ambiguous
      }
    }
    for (const [abbrev, player] of Object.entries(abbrevMap)) {
      if (player && !state.playerIndex[abbrev]) {
        state.playerIndex[abbrev] = player;
      }
    }

    // Add "Last First" reversed aliases for "Last, First" display format (common on draft sites).
    // E.g., "Chase, Ja'Marr" normalizes to "chase jamarr" → alias resolves to Ja'Marr Chase.
    const reversedMap = {};
    for (const key of [...state.canonicalNorms]) {
      const parts = key.split(" ");
      if (parts.length >= 2) {
        const lastFirst = parts[parts.length - 1] + " " + parts.slice(0, -1).join(" ");
        if (!reversedMap[lastFirst]) reversedMap[lastFirst] = state.playerIndex[key];
        else reversedMap[lastFirst] = null;  // collision
      }
    }
    for (const [alias, player] of Object.entries(reversedMap)) {
      if (player && !state.playerIndex[alias]) {
        state.playerIndex[alias] = player;
      }
    }

    // Sort by longest name first so multi-word names match before fragments
    state.allNames.sort((a, b) => b.length - a.length);

    setStatus(`${players.length} players · round ${currentRound()}`);
    state.connected = true;
    loadQueue();

    // Show projected season
    try {
      const info = await bgFetch(`${SERVER}/`);
      const badge = document.getElementById("dm-season-badge");
      if (badge && info.proj_season) badge.textContent = `'${String(info.proj_season).slice(2)}`;
    } catch {}

    return true;
  } catch {
    setStatus("server offline");
    state.connected = false;
    return false;
  }
}

function picksUntilMyTurn() {
  const { teams, pickPos } = state.settings;
  if (teams <= 1) return 0;
  const nextBoardPick = getNextBoardPick();                    // next pick on the board
  const roundIdx   = Math.floor((nextBoardPick - 1) / teams);  // 0-indexed round
  const posInRound = (nextBoardPick - 1) % teams;              // 0-indexed pos in round
  // My slot in this round (0-indexed), accounting for snake reversal
  const mySlot = roundIdx % 2 === 0 ? (pickPos - 1) : (teams - pickPos);
  if (mySlot === posInRound) return 0;    // it's my pick now
  if (mySlot > posInRound)  return mySlot - posInRound;  // still coming this round
  // My slot already passed — find it in the next round
  const nextRound    = roundIdx + 1;
  const nextMySlot   = nextRound % 2 === 0 ? (pickPos - 1) : (teams - pickPos);
  return (teams - posInRound) + nextMySlot;
}

async function getRankings() {
  const availableList = [...state.available].map((norm) => {
    const p = state.playerIndex[norm];
    return p ? p.name : norm;
  });

  // current_pick is the next pick number on the board (1-indexed total picks made + 1)
  const currentPick = getNextBoardPick();

  // Send actual tracked team rosters so the server can model opponent demand accurately
  const teamRostersForServer = {};
  Object.entries(state.teamRosters).forEach(([slot, picks]) => {
    teamRostersForServer[slot] = picks.map(({ name, position, round }) => ({ name, position, round }));
  });
  const pageAdp = scrapeLivePageAdp();

  const body = JSON.stringify({
    available_players: availableList,
    drafted_players: effectiveDraftedPlayersForServer(),
    current_pick: currentPick,
    total_teams: state.settings.teams,
    total_rounds: state.settings.rounds,
    my_pick_position: state.settings.pickPos,
    scoring: state.settings.scoring === "half" ? "half" : "full",
    team_rosters: Object.keys(teamRostersForServer).length > 0 ? teamRostersForServer : null,
    page_adp: Object.keys(pageAdp).length > 0 ? pageAdp : null,
  });

  try {
    return await bgFetch(`${SERVER}/rank`, { method: "POST", body });
  } catch (err) {
    console.error("DraftManager /rank failed:", err);
    return null;
  }
}

// ── Page scanning ─────────────────────────────────────────────────────────────

function isVisibleElement(el) {
  if (!el || !(el instanceof Element)) return false;
  const style = window.getComputedStyle(el);
  if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") return false;
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}

function collectTextChunks(root = document.body) {
  const panel = document.getElementById("dm-panel");
  const chunks = [];

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent || panel?.contains(parent)) return NodeFilter.FILTER_REJECT;
      const tag = parent.tagName?.toLowerCase();
      if (["script", "style", "noscript"].includes(tag)) return NodeFilter.FILTER_REJECT;
      if (!isVisibleElement(parent)) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  let node;
  while ((node = walker.nextNode())) {
    const t = node.textContent?.trim();
    if (t && t.length > 2) chunks.push(t);
  }

  root.querySelectorAll?.("[alt],[aria-label],[data-player-name],[title]").forEach((el) => {
    if (panel?.contains(el) || !isVisibleElement(el)) return;
    const vals = [
      el.getAttribute("alt") || "",
      el.getAttribute("aria-label") || "",
      el.getAttribute("data-player-name") || "",
      el.getAttribute("title") || "",
    ];
    vals.forEach((v) => { if (v.length > 2) chunks.push(v); });
  });

  return chunks;
}

function matchKnownPlayers(chunks) {
  const fullText = normalize(chunks.join(" "));
  const found = new Set();
  for (const norm of Object.keys(state.playerIndex)) {
    if (fullText.includes(norm)) found.add(norm);
  }
  return found;
}

function scanRoot(root = document.body) {
  if (Object.keys(state.playerIndex).length === 0) return new Set();
  return matchKnownPlayers(collectTextChunks(root));
}

function parseAdpFromText(text) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) return null;

  const labeled = [
    /\bADP\b\D{0,16}(\d{1,3}\.\d{1,2})\b/i,
    /\bavg(?:\.|erage)?\s*(?:pick|draft\s*position)?\D{0,16}(\d{1,3}\.\d{1,2})\b/i,
    /\bdraft\s*(?:position|pos)\D{0,16}(\d{1,3}\.\d{1,2})\b/i,
  ];
  for (const re of labeled) {
    const m = clean.match(re);
    if (!m) continue;
    const val = Number(m[1]);
    if (Number.isFinite(val) && val >= 0.1 && val <= 400) return val;
  }

  // Fallback for compact draft-board rows where columns are just:
  // Name | POS | Team | ADP. Prefer decimal numbers because ranks/bye weeks
  // are usually integers.
  const decimals = [...clean.matchAll(/\b(\d{1,3}\.\d{1,2})\b/g)]
    .map((m) => Number(m[1]))
    .filter((v) => Number.isFinite(v) && v >= 0.1 && v <= 400);
  if (decimals.length) return decimals[0];

  return null;
}

function scrapeLivePageAdp() {
  if (currentSitePlatform() !== "underdog") return {};
  const rows = visibleNonPanelElements(
    "tr,li,[role='row'],button,[role='button'],[data-testid*='player' i],[class*='player' i],[class*='draftable' i]"
  );
  const found = {};
  const bestScore = {};
  const canonical = [...state.canonicalNorms]
    .map((key) => state.playerIndex[key])
    .filter(Boolean);

  rows.forEach((el) => {
    const text = el.textContent || "";
    if (text.length < 4 || text.length > 700) return;
    const matches = canonical.filter((p) => textHasExactPlayerName(text, p.name));
    if (matches.length !== 1) return;
    const adp = parseAdpFromText(text);
    if (adp == null) return;

    const p = matches[0];
    const key = normalize(p.name);
    const rect = el.getBoundingClientRect();
    let score = 1000 - Math.min(text.length, 1000);
    if (/\bADP\b/i.test(text)) score += 500;
    if (/draftable|player|available/i.test(rosterHintForElement(el))) score += 100;
    score += Math.min(100, Math.round(rect.width + rect.height) / 10);
    if (!(key in found) || score > bestScore[key]) {
      found[key] = adp;
      bestScore[key] = score;
    }
  });

  return found;
}

function findDKDraftableBoardRoot() {
  if (!location.hostname.toLowerCase().includes("draftkings")) return document.body;
  const panel = document.getElementById("dm-panel");
  const selectors = [
    ".LiveDraft_draftable-players",
    "[class*='draftable-players' i]",
    "[class*='DraftablePlayers']",
    "[data-testid*='draftable' i]",
    "[data-testid*='available' i]",
    "[aria-label*='available' i]",
    "[class*='available-player' i]",
    "[class*='player-list' i]",
  ];

  const candidates = [];
  selectors.forEach((sel) => {
    try {
      document.querySelectorAll(sel).forEach((el) => {
        if (!isVisibleElement(el) || panel?.contains(el)) return;
        const rect = el.getBoundingClientRect();
        if (rect.width < 140 || rect.height < 80) return;
        const hint = rosterHintForElement(el);
        if (/roster|pick.?order|history|queue|my.?team/i.test(hint)) return;
        const players = scanRoot(el);
        if (players.size < 3) return;
        let score = players.size * 10;
        if (/draftable|available/i.test(hint)) score += 80;
        if (/player/i.test(hint)) score += 20;
        candidates.push({ el, score, players: players.size, area: rect.width * rect.height });
      });
    } catch {}
  });

  candidates.sort((a, b) => b.score - a.score || b.players - a.players || b.area - a.area);
  return candidates[0]?.el || document.body;
}

function scanDKDraftableBoard() {
  const root = findDKDraftableBoardRoot();
  return scanRoot(root || document.body);
}

function isDKDraftableContext(el) {
  if (!el) return false;
  let node = el;
  const hints = [];
  for (let depth = 0; node && depth < 8; depth++, node = node.parentElement) {
    hints.push(rosterHintForElement(node));
  }
  const hint = hints.join(" ");
  if (/roster|pick.?order|history|queue|recent|last.?pick|my.?team/i.test(hint)) return false;
  return /draftable|available|player.?list|draft.?board|live.?draft/i.test(hint);
}

function isDKPlayerStillDraftable(player) {
  if (!player) return false;
  const key = draftKeyForName(player.name);
  if (scanDKDraftableBoard().has(key)) return true;

  for (const el of visibleNonPanelElements("button,[role='button'],li,div,tr")) {
    if (!textHasExactPlayerName(el.textContent || "", player.name)) continue;
    if (isDKDraftableContext(el)) return true;
  }
  return false;
}

function matchPlayer(name) {
  if (!name || typeof name !== "string") return null;

  const n = normalize(name);
  if (state.playerIndex[n]) return state.playerIndex[n];

  // Some draft rooms render "Last, First" or otherwise preserve the comma in
  // text before normalization. Try the human-order equivalent explicitly.
  if (name.includes(",")) {
    const [last, first] = name.split(",", 2);
    const swapped = normalize(`${first || ""} ${last || ""}`);
    if (state.playerIndex[swapped]) return state.playerIndex[swapped];
  }

  const parts = n.split(" ").filter(Boolean);
  if (parts.length === 2 && parts[0].length === 1) {
    const [initial, last] = parts;
    if (last.length >= 4) {
      const matches = Object.entries(state.playerIndex)
        .filter(([key]) => key.startsWith(initial) && key.endsWith(` ${last}`))
        .map(([, player]) => player);
      const unique = [...new Map(matches.map((p) => [normalize(p.name), p])).values()];
      if (unique.length === 1) return unique[0];
    }
  }

  if (parts.length > 0) {
    const last = parts[parts.length - 1];
    if (last.length >= 4) {
      const matches = Object.entries(state.playerIndex)
        .filter(([key]) => key.endsWith(` ${last}`) || key === last)
        .map(([, player]) => player);
      const unique = [...new Map(matches.map((p) => [normalize(p.name), p])).values()];
      if (unique.length === 1) return unique[0];
    }
  }

  const substringMatches = Object.entries(state.playerIndex)
    .filter(([key]) => n.length >= 5 && (n.includes(key) || key.includes(n)))
    .map(([, player]) => player);
  const unique = [...new Map(substringMatches.map((p) => [normalize(p.name), p])).values()];
  return unique.length === 1 ? unique[0] : null;
}

function draftKeyForName(name) {
  const player = matchPlayer(name);
  return player ? normalize(player.name) : normalize(name);
}

function draftedKeys() {
  // state.allDrafted is already the effective board (auto − removals + adds), so
  // every entry here is genuinely off the board.
  const keys = new Set();
  state.allDrafted.forEach((d) => {
    if (d._phNorm) keys.add(d._phNorm);
    if (d._draftKey) keys.add(d._draftKey);
    keys.add(draftKeyForName(d.name));
    keys.add(normalize(d.name));
  });
  return keys;
}

function effectiveDraftedPlayersForServer() {
  // Overrides are baked into allDrafted; send placeholders too (server treats the
  // _phNorm name as an unknown opponent pick, which is correct for pick counting).
  return state.allDrafted;
}

function pruneQueue() {
  const drafted = draftedKeys();
  const before = state.queue.length;
  const hasAvailableBoard = state.available.size > 0;
  state.queue = state.queue.filter((key) =>
    state.canonicalNorms.has(key) && !drafted.has(key) && (!hasAvailableBoard || state.available.has(key))
  );
  if (state.queue.length !== before) saveQueue();
}

function queuedKeys() {
  return new Set(state.queue);
}

function addToQueue(name) {
  const key = draftKeyForName(name);
  if (!key || !state.canonicalNorms.has(key) || !state.available.has(key)) return;
  if (!state.queue.includes(key)) {
    state.queue.push(key);
    saveQueue();
    state._lastRenderSig = null;
    renderAll(state.lastRecs || { recommendations: [], stack_alerts: [], scarcity_warnings: [] });
  }
}

function removeFromQueue(key) {
  const before = state.queue.length;
  state.queue = state.queue.filter((k) => k !== key);
  if (state.queue.length !== before) {
    saveQueue();
    state._lastRenderSig = null;
    renderAll(state.lastRecs || { recommendations: [], stack_alerts: [], scarcity_warnings: [] });
  }
}

function moveQueueItem(key, dir) {
  const idx = state.queue.indexOf(key);
  const nextIdx = idx + dir;
  if (idx < 0 || nextIdx < 0 || nextIdx >= state.queue.length) return;
  [state.queue[idx], state.queue[nextIdx]] = [state.queue[nextIdx], state.queue[idx]];
  saveQueue();
  state._lastRenderSig = null;
  renderAll(state.lastRecs || { recommendations: [], stack_alerts: [], scarcity_warnings: [] });
}

function clearQueue() {
  if (state.queue.length === 0) return;
  state.queue = [];
  saveQueue();
  state._lastRenderSig = null;
  renderAll(state.lastRecs || { recommendations: [], stack_alerts: [], scarcity_warnings: [] });
}

function toggleAutodraft() {
  if (currentSitePlatform() !== "draftkings") {
    state.autodraftArmed = false;
    state.lastAutodraftPick = null;
    state.autodraftVerify = null;
    saveAutodraftState();
    state._lastRenderSig = null;
    renderAll(state.lastRecs || { recommendations: [], stack_alerts: [], scarcity_warnings: [] });
    setStatus("autodraft not verified for this platform");
    return;
  }
  state.autodraftArmed = !state.autodraftArmed;
  state.lastAutodraftPick = null;
  state.autodraftVerify = null;
  saveAutodraftState();
  state._lastRenderSig = null;
  renderAll(state.lastRecs || { recommendations: [], stack_alerts: [], scarcity_warnings: [] });
  setStatus(state.autodraftArmed ? "top EV autodraft armed" : "autodraft off");
}

function getNextQueuedPlayer() {
  pruneQueue();
  for (const key of state.queue) {
    if (state.available.has(key)) return state.playerIndex[key];
  }
  return null;
}

function getAutodraftTarget(recs = []) {
  // 1. Queue-first: a player the user queued in OUR panel is an explicit override.
  const queued = getNextQueuedPlayer();
  if (queued) {
    return {
      name: queued.name,
      position: queued.position,
      team: queued.team || "",
      source: "QUEUE",
      key: draftKeyForName(queued.name),
    };
  }
  // 2. Fallback: the top EV model recommendation (only the real policy, not legacy VOR).
  const top = recs.find((p) => state.available.has(draftKeyForName(p.name)));
  if (!top) return null;
  if (top.ranking_source !== "dk_ev_policy") return null;
  return {
    name: top.name,
    position: top.position,
    team: top.team || "",
    source: "TOP EV",
    key: draftKeyForName(top.name),
  };
}

function visibleNonPanelElements(selector) {
  const panel = document.getElementById("dm-panel");
  return [...document.querySelectorAll(selector)].filter((el) => isVisibleElement(el) && !panel?.contains(el));
}

function textHasExactPlayerName(text, name) {
  const n = normalize(name);
  if (!n) return false;
  const haystack = normalize(text || "");
  if (haystack.split(" ").length >= n.split(" ").length && haystack.includes(n)) return true;
  // DK's draft board abbreviates first names: "Baker Mayfield" renders as
  // "B. Mayfield" -> normalized "b mayfield". Also accept the first-initial form.
  const parts = n.split(" ");
  if (parts.length >= 2 && parts[0].length > 1) {
    const abbrev = parts[0][0] + " " + parts.slice(1).join(" ");
    if (haystack.includes(abbrev)) return true;
  }
  return false;
}

function findClickableDraftKingsPlayer(name) {
  const candidates = [];
  for (const el of visibleNonPanelElements("button,[role='button'],li,div,tr")) {
    const text = el.textContent || "";
    if (!textHasExactPlayerName(text, name)) continue;
    const rect = el.getBoundingClientRect();
    if (rect.width < 20 || rect.height < 12) continue;
    const clickable = el.closest("button,[role='button'],li,tr,[data-testid],div") || el;
    if (!clickable || !isVisibleElement(clickable)) continue;
    if (candidates.some((c) => c === clickable || c.contains(clickable))) continue;
    candidates.push(clickable);
  }
  candidates.sort((a, b) => {
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    return (ar.width * ar.height) - (br.width * br.height);
  });
  return candidates[0] || null;
}

function findDraftKingsSearchInput() {
  const inputs = visibleNonPanelElements("input,textarea").filter((el) => {
    const type = String(el.getAttribute("type") || "text").toLowerCase();
    if (!["", "text", "search"].includes(type)) return false;
    const label = [
      el.getAttribute("placeholder") || "",
      el.getAttribute("aria-label") || "",
      el.getAttribute("name") || "",
      el.id || "",
    ].join(" ").toLowerCase();
    return /search|player|find/.test(label);
  });
  inputs.sort((a, b) => {
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    return ar.top - br.top || ar.left - br.left;
  });
  return inputs[0] || null;
}

const DK_SEARCH_WAIT_MS = 2500;
let dkSearchRestore = null;

function setDraftKingsSearchValue(input, value) {
  if (!input) return;
  input.focus();
  input.select?.();

  const proto = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const setter =
    Object.getOwnPropertyDescriptor(proto, "value")?.set ||
    Object.getOwnPropertyDescriptor(input.constructor.prototype, "value")?.set;
  if (setter) setter.call(input, value);
  else input.value = value;

  try {
    input.dispatchEvent(new InputEvent("beforeinput", {
      bubbles: true,
      cancelable: true,
      inputType: value ? "insertText" : "deleteContentBackward",
      data: value,
    }));
  } catch (_) {}
  input.dispatchEvent(new InputEvent("input", {
    bubbles: true,
    inputType: value ? "insertText" : "deleteContentBackward",
    data: value,
  }));
  input.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true, key: value ? value.slice(-1) : "Backspace" }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

async function waitForDraftKingsPlayer(name, timeoutMs = DK_SEARCH_WAIT_MS) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() <= deadline) {
    const playerEl = findDraftKingsPlayerActionRow(name) || findClickableDraftKingsPlayer(name);
    if (playerEl) return playerEl;
    await new Promise((resolve) => setTimeout(resolve, 125));
  }
  return null;
}

function restoreDraftKingsSearch() {
  const restore = dkSearchRestore;
  dkSearchRestore = null;
  const input = restore?.input;
  if (!input || !document.contains(input)) return;
  setDraftKingsSearchValue(input, restore.value || "");
  input.blur?.();
}

function scheduleDraftKingsSearchRestore(delayMs = 700) {
  if (!dkSearchRestore) return;
  setTimeout(restoreDraftKingsSearch, delayMs);
}

async function revealDraftKingsPlayer(name) {
  const visible = findDraftKingsPlayerActionRow(name) || findClickableDraftKingsPlayer(name);
  if (visible) return visible;

  const input = findDraftKingsSearchInput();
  if (!input) return null;

  const previousValue = input.value || "";
  dkSearchRestore = { input, value: previousValue };
  setDraftKingsSearchValue(input, name);
  return await waitForDraftKingsPlayer(name);
}

function findDraftKingsConfirmButton(name) {
  const actionRe = /\b(draft|select|pick|confirm)\b/i;
  const containers = visibleNonPanelElements("div,section,aside,dialog,[role='dialog'],[data-testid],form").filter((el) => {
    const rect = el.getBoundingClientRect();
    if (rect.width < 80 || rect.height < 30) return false;
    if (rect.width > window.innerWidth * 0.95 && rect.height > window.innerHeight * 0.95) return false;
    return textHasExactPlayerName(el.textContent || "", name);
  });
  containers.sort((a, b) => {
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    return (ar.width * ar.height) - (br.width * br.height);
  });

  for (const container of containers) {
    const btn = [...container.querySelectorAll("button,[role='button']")].find((candidate) => {
      if (!isVisibleElement(candidate)) return false;
      const text = (candidate.textContent || candidate.getAttribute("aria-label") || "").trim();
      const disabled = candidate.disabled || candidate.getAttribute("aria-disabled") === "true";
      return !disabled && actionRe.test(text);
    });
    if (btn) return btn;
  }
  return null;
}

function findDraftKingsPlayerActionRow(name, startEl = null) {
  const exactRows = visibleNonPanelElements('[role="row"], tr, [class*="BaseTable__row" i], [class*="dk-grid-row" i]')
    .filter((el) => textHasExactPlayerName(el.textContent || "", name) && hasDraftKingsRowAction(el));
  exactRows.sort((a, b) => {
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    return (ar.width * ar.height) - (br.width * br.height);
  });
  if (exactRows[0]) return exactRows[0];

  let row = startEl || findClickableDraftKingsPlayer(name);
  for (let i = 0; i < 10 && row; i++) {
    const text = row.textContent || "";
    const hasName = textHasExactPlayerName(text, name);
    const hasAction = hasDraftKingsRowAction(row);
    const role = (row.getAttribute?.("role") || "").toLowerCase();
    const cls = typeof row.className === "string" ? row.className : "";
    if (hasName && hasAction && (role === "row" || /basetable__row|draftable|player/i.test(cls))) {
      return row;
    }
    row = row.parentElement;
  }
  return null;
}

async function clickDraftKingsRowThenConfirm(name, playerEl) {
  if (!playerEl) return null;

  const before = findDraftKingsConfirmButton(name);
  if (before) {
    before.click();
    return before;
  }

  const target = findDraftKingsPlayerActionRow(name, playerEl);
  if (!target || !isVisibleElement(target)) return null;
  target.scrollIntoView?.({ block: "center", inline: "nearest" });
  target.click();
  await new Promise((resolve) => setTimeout(resolve, 500));

  const rowBtn = findDraftKingsRowDraftButton(name, target);
  if (rowBtn) {
    rowBtn.click();
    return rowBtn;
  }

  const confirm = findDraftKingsConfirmButton(name);
  if (!confirm) return null;
  confirm.click();
  return confirm;
}

// Confirmed from live DK inspection (June 2026): the per-row queue control is
// <button class="QueueIcon_queue-button…"> wrapping a 20px SVG. The class is
// CSS-module-hashed, so match a stable substring case-insensitively.
const DK_QUEUE_SELECTOR =
  'button[class*="queue-button" i], [class*="QueueIcon" i], button[class*="queue" i], [class*="queue-button" i]';

function hasDraftKingsRowAction(row) {
  if (!row?.querySelector) return false;
  if (row.querySelector('[class*="DraftButton" i], [class*="draft-button" i]')) return true;
  if (row.querySelector(DK_QUEUE_SELECTOR)) return true;
  return !!queueStarWithin(row);
}

// Find the per-row "queue"/star control for a player on DraftKings. DK drafts
// from the queue, so starring the top-EV player is the reliable way to auto-pick.
function findDraftKingsQueueStar(name, startEl = null) {
  const row = findDraftKingsPlayerActionRow(name, startEl);
  if (!row || !textHasExactPlayerName(row.textContent || "", name)) return null;

  // Deterministic DK class match within this exact row (preferred).
  const dk = [...row.querySelectorAll(DK_QUEUE_SELECTOR)].find(isVisibleElement);
  if (dk) return dk.closest("button,[role='button']") || dk;

  // Heuristic fallback, still scoped to the exact player row.
  const star = queueStarWithin(row);
  if (star) return star;
  return null;
}

// The per-row "Draft" button. Confirmed DK structure (June 2026):
//   div.DraftButton_draft-button--container > button.DKButtonV2_dk-button-v2 > span "Draft"
// NOTE the "DraftButton_*" class is on the CONTAINER DIV, not the <button> (the
// button is "DKButtonV2_*"). So we must NOT require the button itself to carry the
// draft-button class. It's disabled (aria-disabled / "Auto") until it's our pick and
// the player is draftable; when enabled, clicking it drafts immediately (auto-submit).
function findDraftKingsRowDraftButton(name, startEl = null) {
  const enabledClickable = (b) =>
    b && isVisibleElement(b) && !b.disabled && b.getAttribute("aria-disabled") !== "true";

  const row = findDraftKingsPlayerActionRow(name, startEl);
  if (!row || !textHasExactPlayerName(row.textContent || "", name)) return null;

  // 1. Preferred: the draft-button container (DIV), then its inner <button>.
  for (const container of row.querySelectorAll('[class*="DraftButton" i], [class*="draft-button" i]')) {
    const btn = container.matches("button") ? container : container.querySelector("button");
    if (enabledClickable(btn)) return btn;
  }

  // 2. Fallback: any enabled button in the row whose label is exactly "Draft"
  //    (DK's button class is hashed/generic, so match on the visible text).
  const byText = [...row.querySelectorAll("button")].find((b) => {
    if (!enabledClickable(b)) return false;
    const label = (b.textContent || b.getAttribute("aria-label") || "").trim().toLowerCase();
    return label === "draft" || /^draft\b/.test(label);
  });
  if (byText) return byText;
  return null;
}

function queueStarWithin(row) {
  if (!row?.querySelectorAll) return null;
  const QUEUE_RE = /(queue|add to queue|favou?rite|\bstar\b|watch ?list)/i;
  const controls = [...row.querySelectorAll("button,[role='button'],[data-testid],svg,span,i")].filter((el) => {
    if (!isVisibleElement(el)) return false;
    const r = el.getBoundingClientRect();
    if (r.width < 8 || r.height < 8 || r.width > 64 || r.height > 64) return false;  // small icon only
    const cls = typeof el.className === "string" ? el.className : (el.className?.baseVal || "");
    const label = [
      el.getAttribute?.("aria-label") || "",
      el.getAttribute?.("title") || "",
      el.getAttribute?.("data-testid") || "",
      cls,
    ].join(" ").toLowerCase();
    return QUEUE_RE.test(label);
  });
  if (!controls.length) return null;
  controls.sort((a, b) => a.getBoundingClientRect().left - b.getBoundingClientRect().left);  // star is on the left
  const el = controls[0];
  return el.closest("button,[role='button'],[data-testid]") || el;
}

// Human-priority grace period: once it's our pick we wait this long (into the 30s DK
// clock) before auto-firing, so a manual pick always wins. User-chosen fixed 15s
// (leaves a comfortable margin over the 5s scan poll, well clear of the 30s clock).
const AUTODRAFT_GRACE_MS = 15000;

function clearAutodraftCountdown() {
  if (state.autodraftCountdown?.timerId) clearInterval(state.autodraftCountdown.timerId);
  state.autodraftCountdown = null;
  renderAutodraftCountdown();
}

// Undo a backstop star we added on DK, so a leftover queue entry can never auto-draft a
// LATER pick. Safe no-op if the star is already gone (e.g. the player got drafted).
function undoAutodraftBackstop() {
  const key = state.autodraftBackstopKey;
  state.autodraftBackstopKey = null;
  if (!key) return;
  const p = state.playerIndex[key];
  if (!p) return;
  const star = findDraftKingsQueueStar(p.name);
  const label = (star?.getAttribute("aria-label") || "").toLowerCase();
  if (star && /remove|in queue|queued/.test(label)) star.click();  // toggle back off
  state.dkQueued.delete(key);
}

// Proactive backstop (model-fallback case only): star the model pick on DK so its native
// auto-draft covers us if the 15s click ever fails. Best-effort; failure is non-fatal.
function armAutodraftBackstop(target) {
  if (!target?.key || state.dkQueued.has(target.key)) return;
  Promise.resolve(revealDraftKingsPlayer(target.name)).then((playerEl) => {
    try {
      const star = findDraftKingsQueueStar(target.name, playerEl);
      if (!star) return;
      const label = (star.getAttribute("aria-label") || "").toLowerCase();
      if (/remove|in queue|queued/.test(label)) return;  // already on DK's queue
      star.click();
      state.dkQueued.add(target.key);
      state.autodraftBackstopKey = target.key;
    } finally {
      restoreDraftKingsSearch();
    }
  }).catch(() => { restoreDraftKingsSearch(); });
}

// After a Draft-button click we don't trust it blindly: confirm on a later scan that
// the player actually left the board. Retry the click up to this many times (each scan
// ~3-5s, and we fire at 15s of the 30s clock → time for a couple retries) before
// handing off to DK's native queue auto-pick as the last-resort safety net.
const AUTODRAFT_MAX_TRIES = 3;
const AUTODRAFT_VERIFY_STALE_MS = 45000;

// Ensure DK will auto-pick our player when the clock expires (star it in DK's queue).
async function queueAutodraftFallback(target) {
  if (!target?.name || !target.key) return false;
  try {
    const playerEl = await revealDraftKingsPlayer(target.name);
    const star = findDraftKingsQueueStar(target.name, playerEl);
    if (!star) {
      restoreDraftKingsSearch();
      return false;
    }
    const label = (star.getAttribute("aria-label") || "").toLowerCase();
    if (!/remove|in queue|queued/.test(label) && !state.dkQueued.has(target.key)) {
      star.click();
      state.dkQueued.add(target.key);
    }
    scheduleDraftKingsSearchRestore();
    return true;
  } catch (_) {
    restoreDraftKingsSearch();
    return false;
  }
}

// Called every scan. Manages the 15s grace countdown + post-click verification.
async function maybeAutodraft(recs = []) {
  if (!location.hostname.toLowerCase().includes("draftkings")) return;

  const onClock = state.autodraftArmed && picksUntilMyTurn() === 0;
  const pickNo = getNextBoardPick();

  // Verify a pending Draft-button click actually committed. DK removes a drafted player
  // from the board immediately, so "target gone" (or our turn passing) means it landed.
  const v = state.autodraftVerify;
  if (v) {
    const turnPassed = picksUntilMyTurn() !== 0 || pickNo !== v.pickNo;
    const targetGone = !state.available.has(v.key);
    const staleVerify = v.startedAt && Date.now() - v.startedAt > AUTODRAFT_VERIFY_STALE_MS;
    if (targetGone || turnPassed) {
      if (targetGone) setStatus(`autodraft confirmed: ${v.name}`);
      state.lastAutodraftPick = v.pickNo;
      state.autodraftVerify = null;
      undoAutodraftBackstop();
      clearAutodraftCountdown();
      return;
    }
    if (staleVerify) {
      triggerDraftStatusRefetch();
      state.autodraftVerify = null;
      state.autodraftCancelledPick = v.pickNo;
      undoAutodraftBackstop();
      clearAutodraftCountdown();
      setStatus(`autodraft stopped: couldn't verify ${v.name}`);
      return;
    }
    // Still our pick and the player is still on the board → the click didn't take.
    if (!state.autodraftBusy) {
      if (v.tries < AUTODRAFT_MAX_TRIES) {
        setStatus(`autodraft retry ${v.tries + 1}/${AUTODRAFT_MAX_TRIES}: ${v.name}`);
        executeAutodraft({ name: v.name, position: v.position, team: v.team, source: v.source, key: v.key }, v.pickNo);
      } else {
        // Out of click retries: fall back to DK's native auto-pick of our player.
        if (!v.queueFallbackArmed) {
          const queued = await queueAutodraftFallback(v);
          if (queued) {
            setStatus(`autodraft: clicking failed - queued ${v.name} for DK auto-pick`);
            state.autodraftVerify = { ...v, queueFallbackArmed: true };
          } else {
            triggerDraftStatusRefetch();
            state.autodraftVerify = null;
            state.autodraftCancelledPick = v.pickNo;
            setStatus(`autodraft stopped: couldn't queue ${v.name}`);
          }
        }
      }
    }
    return;
  }

  // Not our turn (or disarmed): tear down any running countdown and undo a backstop.
  if (!onClock) {
    if (state.autodraftCountdown) { undoAutodraftBackstop(); clearAutodraftCountdown(); }
    return;
  }
  // Already drafted, or the user cancelled auto for this exact pick.
  if (state.lastAutodraftPick === pickNo || state.autodraftCancelledPick === pickNo) {
    if (state.autodraftCountdown) clearAutodraftCountdown();
    return;
  }

  const target = getAutodraftTarget(recs);
  if (!target || !target.key || !state.available.has(target.key)) return;  // retry next scan

  // Start the grace countdown once per pick; otherwise just refresh the shown target.
  if (!state.autodraftCountdown || state.autodraftCountdown.pickNo !== pickNo) {
    if (state.autodraftCountdown) clearAutodraftCountdown();
    state.autodraftCountdown = { pickNo, target, fireAt: Date.now() + AUTODRAFT_GRACE_MS, timerId: null };
    if (target.source === "TOP EV") armAutodraftBackstop(target);  // backstop only when queue empty
    state.autodraftCountdown.timerId = setInterval(tickAutodraft, 1000);
    renderAutodraftCountdown();
  } else {
    state.autodraftCountdown.target = target;
  }
}

// 1Hz tick: updates the countdown UI, bails if the pick resolved, fires at 0.
function tickAutodraft() {
  const cd = state.autodraftCountdown;
  if (!cd) return;
  const resolved = !state.autodraftArmed
    || picksUntilMyTurn() !== 0
    || state.lastAutodraftPick === cd.pickNo
    || state.autodraftCancelledPick === cd.pickNo;
  if (resolved) {
    // If the pick was filled by something other than our own draft, undo the backstop.
    if (state.lastAutodraftPick !== cd.pickNo) undoAutodraftBackstop();
    clearAutodraftCountdown();
    return;
  }
  renderAutodraftCountdown();
  if (Date.now() >= cd.fireAt) {
    const target = getAutodraftTarget(state.lastRecs?.recommendations || []) || cd.target;
    const pickNo = cd.pickNo;
    clearAutodraftCountdown();
    executeAutodraft(target, pickNo);
  }
}

function cancelAutodraftForCurrentPick() {
  const cd = state.autodraftCountdown;
  if (cd) state.autodraftCancelledPick = cd.pickNo;
  state.autodraftVerify = null;  // stop any pending click-verification/retries
  undoAutodraftBackstop();
  clearAutodraftCountdown();
  setStatus("autodraft cancelled for this pick");
}

// The actual commit: reveal the player, click the row's Draft button (instant), or fall
// back to starring (DK auto-drafts the queue at expiry). Only marks the pick handled
// once it has actually clicked something — never silently skips a pick.
async function executeAutodraft(target, pickNo) {
  if (state.autodraftBusy) return;
  if (!target || !target.key || !state.available.has(target.key)) return;
  if (state.lastAutodraftPick === pickNo) return;

  state.autodraftBusy = true;
  setStatus(`autodraft: ${target.name}`);
  let committedVia = null;  // "button"/"confirm"/"queue" (all verified on later scans)

  try {
    const playerEl = await revealDraftKingsPlayer(target.name);
    if (!playerEl) {
      console.warn("[DM Autodraft] Target not visible/selectable:", target.name);
      setStatus(`autodraft blocked: ${target.name} not found`);
      return;
    }

    // Preferred path: the row's real Draft button (enabled only on our pick for a
    // draftable player). Clicking it drafts immediately — a true auto-submit. We do NOT
    // mark the pick done here; maybeAutodraft confirms the player left the board (and
    // retries the click if it didn't), so a silently-failed click can't waste the pick.
    const draftBtn = findDraftKingsRowDraftButton(target.name, playerEl);
    if (draftBtn) {
      draftBtn.click();
      committedVia = "button";
      setStatus(`autodraft clicked: ${target.name}`);
      return;
    }

    // Some DK builds expose the enabled Draft action only after opening/selecting
    // the row. Try that two-step flow before falling back to the queue star.
    const confirmBtn = await clickDraftKingsRowThenConfirm(target.name, playerEl);
    if (confirmBtn) {
      committedVia = "confirm";
      setStatus(`autodraft confirmed click: ${target.name}`);
      return;
    }

    // Otherwise queue via the row star (DK auto-drafts the queue when time expires).
    const star = findDraftKingsQueueStar(target.name, playerEl);
    if (star) {
      const label = (star.getAttribute("aria-label") || "").toLowerCase();
      const alreadyQueued = /remove|in queue|queued/.test(label) || state.dkQueued.has(target.key);
      if (!alreadyQueued) {
        star.click();
        state.dkQueued.add(target.key);
      }
      committedVia = "queue";
      setStatus(`queued (DK auto-picks at expiry): ${target.name}`);
      return;
    }

    console.warn("[DM Autodraft] No draft button or queue star found for:", target.name);
    setStatus(`autodraft: control not found`);
  } catch (err) {
    console.error("[DM Autodraft] Failed:", err);
    setStatus("autodraft failed");
  } finally {
    if (committedVia) {
      triggerDraftStatusRefetch();
      scheduleDraftKingsSearchRestore();
      // Pending confirmation: keep retry/verification alive until the board
      // confirms the target left or our turn passes.
      const prev = state.autodraftVerify;
      const tries = (prev && prev.pickNo === pickNo) ? prev.tries + 1 : 1;
      const startedAt = (prev && prev.pickNo === pickNo && prev.startedAt) ? prev.startedAt : Date.now();
      state.autodraftVerify = {
        pickNo, key: target.key, name: target.name,
        position: target.position, team: target.team, source: target.source, tries,
        queueFallbackArmed: committedVia === "queue",
        startedAt,
      };
      if (committedVia === "queue") state.autodraftBackstopKey = null;
    } else {
      restoreDraftKingsSearch();
    }
    setTimeout(() => { state.autodraftBusy = false; }, 1500);
  }
}

// Live countdown banner at the top of the panel: shows who will be drafted + a Cancel.
function renderAutodraftCountdown() {
  const panel = document.getElementById("dm-panel");
  if (!panel) return;
  let el = document.getElementById("dm-autodraft-countdown");
  const cd = state.autodraftCountdown;
  if (!cd) { el?.remove(); return; }
  if (!el) {
    el = document.createElement("div");
    el.id = "dm-autodraft-countdown";
    el.className = "dm-autodraft-countdown";
    panel.insertBefore(el, panel.firstChild);
  }
  const secs = Math.max(0, Math.ceil((cd.fireAt - Date.now()) / 1000));
  const t = cd.target || {};
  const srcTag = t.source === "QUEUE" ? "QUEUE" : "TOP EV";
  el.innerHTML =
    `<span class="dm-adc-label">⏱ Auto-draft <b>${srcTag}</b></span>`
    + `<span class="dm-adc-name">${t.name || "?"}</span>`
    + `<span class="dm-adc-timer">in ${secs}s</span>`
    + `<button id="dm-adc-cancel" class="dm-adc-cancel" type="button">Cancel</button>`;
  el.querySelector("#dm-adc-cancel")?.addEventListener("click", cancelAutodraftForCurrentPick);
}

function getPageText() {
  const panel = document.getElementById("dm-panel");
  const bodyText = document.body?.innerText || "";
  const panelText = panel?.innerText || "";
  return panelText ? bodyText.replace(panelText, " ") : bodyText;
}

function detectCurrentPickFromPage() {
  const text = getPageText();
  const teams  = state.settings.teams || 12;
  const rounds = state.settings.rounds || 20;
  const total  = teams * rounds;

  const valid = (n) => n >= 1 && n <= total;

  // "Round 2, Pick 3" / "Round 2 · Pick 3" / "Round 2 | Pick 3"
  let m = text.match(/round\s+(\d+)\s*[|·:,\-]?\s*pick\s+#?\s*(\d+)/i);
  if (m) {
    const r = parseInt(m[1], 10), p = parseInt(m[2], 10);
    if (r >= 1 && p >= 1 && p <= teams) return (r - 1) * teams + p;
  }

  // "Pick 24 of 180" / "Pick #24 of 180" / "Overall pick 24"
  m = text.match(/(?:overall\s+)?pick\s+#?\s*(\d{1,3})\s+of\s+\d+/i) ||
      text.match(/overall\s+pick[:\s]+#?\s*(\d{1,3})/i);
  if (m) {
    const n = parseInt(m[1], 10);
    if (valid(n)) return n;
  }

  // "On the clock ... pick N"
  m = text.match(/on\s+the\s+clock[\s\S]{0,60}?pick\s+#?\s*(\d{1,3})/i);
  if (m) {
    const n = parseInt(m[1], 10);
    if (valid(n)) return n;
  }

  // "Pick 1.12" (Underdog style: round.pick-in-round)
  m = text.match(/\bpick\s+(\d+)\.(\d+)\b/i);
  if (m) {
    const r = parseInt(m[1], 10), p = parseInt(m[2], 10);
    if (r >= 1 && p >= 1 && p <= teams) return (r - 1) * teams + p;
  }

  // Standalone "Pick 24" / "Pick #24" — only if the number is clearly a pick number
  // (avoid matching "Pick 2026" or other numbers)
  m = text.match(/\bpick\s+#\s*(\d{1,3})\b/i);
  if (m) {
    const n = parseInt(m[1], 10);
    if (valid(n)) return n;
  }

  // "Making pick N" / "Selecting pick N" / "Pick N is on the clock"
  m = text.match(/(?:making|selecting|current)\s+pick\s+(?:#\s*)?(\d{1,3})/i) ||
      text.match(/pick\s+(?:#\s*)?(\d{1,3})\s+(?:is\s+)?on\s+the\s+clock/i);
  if (m) {
    const n = parseInt(m[1], 10);
    if (valid(n)) return n;
  }

  // "#N of M" or "N/M picks" (pick counters)
  m = text.match(/#\s*(\d{1,3})\s+of\s+\d{2,3}\b/) ||
      text.match(/(\d{1,3})\s*\/\s*(\d{2,3})\s+picks?/i);
  if (m) {
    const n = parseInt(m[1], 10);
    if (valid(n)) return n;
  }

  // Underdog: "Rd N, Pk M" or "R N P M"
  m = text.match(/rd?\s+(\d{1,2})\s*,?\s*pk?\s+(\d{1,2})\b/i);
  if (m) {
    const r = parseInt(m[1], 10), p = parseInt(m[2], 10);
    if (r >= 1 && p >= 1 && p <= teams) return (r - 1) * teams + p;
  }

  return null;
}

// ── Team-slot math (snake draft) ──────────────────────────────────────────────

// Returns 1-indexed team slot for the Nth overall pick in a snake draft.
function pickToTeamSlot(overallPick, totalTeams) {
  const roundIdx    = Math.floor((overallPick - 1) / totalTeams);
  const posInRound  = (overallPick - 1) % totalTeams;
  return roundIdx % 2 === 0 ? posInRound + 1 : totalTeams - posInRound;
}

function teamSlotPickNumber(teamSlot, rosterPickOrdinal, totalTeams) {
  const roundIdx = Math.max(0, rosterPickOrdinal - 1);
  const posInRound = roundIdx % 2 === 0 ? teamSlot - 1 : totalTeams - teamSlot;
  return roundIdx * totalTeams + posInRound + 1;
}

function extractRosterSlotFromText(text, maxTeams) {
  const clean = String(text || "");
  const patterns = [
    /\bteam\s*#?\s*(\d{1,2})\b/i,
    /\bdraft\s*(?:slot|position)\s*#?\s*(\d{1,2})\b/i,
    /\bpick\s*(?:slot|position)\s*#?\s*(\d{1,2})\b/i,
    /\bslot\s*#?\s*(\d{1,2})\b/i,
  ];
  for (const re of patterns) {
    const m = clean.match(re);
    if (!m) continue;
    const n = parseInt(m[1], 10);
    if (n >= 1 && n <= maxTeams) return n;
  }
  return null;
}

function rosterHintForElement(el) {
  if (!el) return "";
  return [
    el.id || "",
    typeof el.className === "string" ? el.className : "",
    el.getAttribute?.("data-testid") || "",
    el.getAttribute?.("data-qa") || "",
    el.getAttribute?.("aria-label") || "",
    el.getAttribute?.("role") || "",
  ].join(" ");
}

function findDKRosterPanels() {
  if (!location.hostname.toLowerCase().includes("draftkings")) return [];
  if (Object.keys(state.playerIndex).length === 0) return [];

  const panel = document.getElementById("dm-panel");
  const teams = state.settings.teams || 12;
  const candidates = [];

  if (state.customRosterSelector) {
    try {
      const panels = [...document.querySelectorAll(state.customRosterSelector)]
        .filter((el) => isVisibleElement(el) && !panel?.contains(el))
        .map((el, idx) => buildRosterPanelCandidate(el, state.customRosterSlot || (idx + 1)))
        .filter(Boolean);
      if (panels.length > 0) return panels;
    } catch (e) {
      console.warn("[DM] custom roster selector error:", e);
    }
  }

  const selector = [
    "[class*='roster' i]", "[data-testid*='roster' i]", "[aria-label*='roster' i]",
    "[class*='lineup' i]", "[data-testid*='lineup' i]", "[aria-label*='lineup' i]",
    "[class*='drafted' i]", "[data-testid*='drafted' i]",
    "[class*='entry' i]", "[data-testid*='entry' i]",
    "[class*='team' i]", "[data-testid*='team' i]", "[aria-label*='team' i]",
  ].join(",");

  for (const el of document.querySelectorAll(selector)) {
    if (!isVisibleElement(el) || panel?.contains(el)) continue;
    const rect = el.getBoundingClientRect();
    if (rect.width < 80 || rect.height < 25) continue;
    if (el.children.length > 120) continue;

    const hint = rosterHintForElement(el);
    const strongHint = /roster|lineup|drafted|entry|team.?roster|user.?roster|team.?card|draft.?team/i.test(hint);
    if (!strongHint) continue;

    const cand = buildRosterPanelCandidate(el);
    if (!cand) continue;

    const text = (el.textContent || "").replace(/\s+/g, " ").trim();
    let score = cand.players.length * 8;
    if (/roster|lineup/i.test(hint)) score += 35;
    if (/entry|drafted|team/i.test(hint)) score += 12;
    if (/\b(QB|RB|WR|TE)\b/.test(text)) score += 8;
    if (extractRosterSlotFromText(`${hint} ${text}`, teams)) score += 20;
    if (rect.height > window.innerHeight * 0.85 && cand.players.length > 8) score -= 30;

    candidates.push({ ...cand, score });
  }

  candidates.sort((a, b) => b.score - a.score);
  const selected = [];
  for (const cand of candidates) {
    if (selected.some((s) => s.el === cand.el || s.el.contains(cand.el) || cand.el.contains(s.el))) continue;
    selected.push(cand);
    if (selected.length >= teams) break;
  }
  selected.sort((a, b) => a.top - b.top || a.left - b.left);

  const explicitCount = selected.filter((p) => p.slot).length;
  if (selected.length < 2) return [];
  if (explicitCount < selected.length && selected.length !== teams) return [];

  const usedSlots = new Set();
  return selected.map((panelInfo, idx) => {
    let slot = panelInfo.slot;
    if (!slot) slot = idx + 1;
    if (usedSlots.has(slot) || slot < 1 || slot > teams) return null;
    usedSlots.add(slot);
    return { ...panelInfo, slot };
  }).filter(Boolean);
}

function buildRosterPanelCandidate(el, fallbackSlot = null) {
  const teams = state.settings.teams || 12;
  const rect = el.getBoundingClientRect();
  const hint = rosterHintForElement(el);
  const text = (el.textContent || "").replace(/\s+/g, " ").trim();
  if (text.length > 5000) return null;

  const players = [...scanRoot(el)]
    .filter((norm) => state.canonicalNorms.has(norm))
    .map((norm) => state.playerIndex[norm])
    .filter(Boolean);
  const uniq = [];
  const seen = new Set();
  for (const p of players) {
    const key = draftKeyForName(p.name);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    uniq.push(p);
  }
  if (uniq.length < 1 || uniq.length > state.settings.rounds) return null;

  return {
    el,
    players: uniq,
    slot: extractRosterSlotFromText(`${hint} ${text}`, teams) || fallbackSlot,
    score: 0,
    top: rect.top,
    left: rect.left,
    hint,
  };
}

function findDKRosterCandidates(limit = 20) {
  if (!location.hostname.toLowerCase().includes("draftkings")) return [];
  if (Object.keys(state.playerIndex).length === 0) return [];
  const panel = document.getElementById("dm-panel");
  const candidates = [];
  for (const el of document.querySelectorAll("section,aside,article,nav,main,div,ul,ol")) {
    if (!isVisibleElement(el) || panel?.contains(el)) continue;
    const rect = el.getBoundingClientRect();
    if (rect.width < 80 || rect.height < 24 || el.children.length > 180) continue;
    const cand = buildRosterPanelCandidate(el);
    if (!cand) continue;
    if (cand.players.length > 28) continue;

    let score = cand.players.length * 6;
    if (/roster|lineup|team|entry|drafted|pick/i.test(cand.hint)) score += 25;
    if (/\b(QB|RB|WR|TE)\b/.test(el.textContent || "")) score += 10;
    if (cand.slot) score += 20;
    if (rect.height > window.innerHeight * 0.8) score -= 20;
    candidates.push({ ...cand, score });
  }
  candidates.sort((a, b) => b.score - a.score);
  const selected = [];
  for (const cand of candidates) {
    if (selected.some((s) => s.el === cand.el || s.el.contains(cand.el) || cand.el.contains(s.el))) continue;
    selected.push(cand);
    if (selected.length >= limit) break;
  }
  return selected;
}

function reconcileFromDKRosters() {
  if (state.dkApiActive && state.allDrafted.length > 0) return 0;  // non-empty DK JSON feed is authoritative
  const panels = findDKRosterPanels();
  if (panels.length < 2) return 0;

  const teams = state.settings.teams || 12;
  const rosterBySlot = {};
  let changed = 0;
  let totalRostered = 0;
  const rosteredKeys = new Set();

  panels.forEach((panelInfo) => {
    const slot = panelInfo.slot;
    rosterBySlot[slot] = [];
    panelInfo.players.forEach((player, idx) => {
      const key = draftKeyForName(player.name);
      if (!key || rosteredKeys.has(key)) return;
      rosteredKeys.add(key);
      totalRostered++;

      const ordinal = idx + 1;
      const overallPick = teamSlotPickNumber(slot, ordinal, teams);
      const round = Math.floor((overallPick - 1) / teams) + 1;
      const byUser = slot === state.settings.pickPos;
      const entry = {
        name: player.name,
        position: player.position,
        round,
        by_user: byUser,
        team_slot: slot,
        overall_pick: overallPick,
        _draftKey: key,
      };

      rosterBySlot[slot].push({ name: player.name, position: player.position, round });
      state.available.delete(key);

      const existingIdx = state.allDrafted.findIndex((d) => draftKeyForName(d.name) === key || d._draftKey === key);
      if (existingIdx >= 0) {
        const existing = state.allDrafted[existingIdx];
        if (existing.team_slot !== slot || existing.by_user !== byUser || existing.position !== player.position) {
          state.allDrafted[existingIdx] = { ...existing, ...entry, overall_pick: existing.overall_pick || overallPick };
          changed++;
        }
        return;
      }

      const phIdx = state.allDrafted.findIndex((d) => d._phNorm && Number(d.overall_pick) === overallPick);
      if (phIdx >= 0) state.allDrafted[phIdx] = entry;
      else state.allDrafted.push(entry);
      changed++;
    });
  });

  if (totalRostered === 0) return 0;

  state.allDrafted = state.allDrafted
    .filter((entry, idx, arr) => {
      if (!entry._phNorm) return true;
      return !arr.some((other, otherIdx) => otherIdx !== idx && !other._phNorm && Number(other.overall_pick) === Number(entry.overall_pick));
    })
    .sort((a, b) => (Number(a.overall_pick) || 9999) - (Number(b.overall_pick) || 9999));

  if (Object.keys(rosterBySlot).length >= 2) {
    state.teamRosters = rosterBySlot;
    state.myTeam = rosterBySlot[state.settings.pickPos] || [];
    changed++;
  }

  const nextFromRosters = totalRostered + 1;
  if (nextFromRosters > getTrackedNextPick()) {
    addMissingPickPlaceholders(nextFromRosters, "DK roster reconciliation");
    changed++;
  } else if (!state.pageCurrentPick || nextFromRosters > state.pageCurrentPick) {
    state.pageCurrentPick = nextFromRosters;
  }

  if (changed > 0) {
    state._lastRosterSyncDebug = {
      panels: panels.length,
      players: totalRostered,
      next_pick: nextFromRosters,
      slots: Object.keys(rosterBySlot).map(Number).sort((a, b) => a - b),
      at: Date.now(),
    };
    reconcileDraftState();
    saveDraftState();
    renderTeamRosters();
  }
  return changed;
}

// ── Pick-history scanning ─────────────────────────────────────────────────────
// Strategy: scan the PICK HISTORY (log of completed picks) rather than the
// available board. History only grows — no need to scroll anything.
// available = full player DB − history = always correct.

// Returns site-specific CSS selectors to try first, based on current hostname.
function getSiteSpecificSelectors() {
  const host = location.hostname.toLowerCase();

  // DraftKings — confirmed class names from live inspection (June 2026)
  if (host.includes("draftkings")) {
    return [
      // Confirmed live class names (CSS modules, NOT hashed on DK)
      ".PickOrder_pick-order",
      ".LiveDraft_pick-order-container",
      // data-testid patterns (stable across deploys)
      "[data-testid='draft-log-container']",
      "[data-testid='pick-log']",
      "[data-testid='pick-list']",
      "[data-testid='draft-pick-list']",
      "[data-testid='picks-feed']",
      "[data-testid='draft-history']",
      "[data-testid='picks-list']",
      "[data-testid='draft-queue']",
      "[data-testid='pick-queue']",
      // ARIA roles (semantic, stable)
      "[aria-label='Draft log']",
      "[aria-label='Pick history']",
      "[aria-label='Draft picks']",
      "[aria-label='Picks made']",
      "[aria-label='Recent picks']",
      "[aria-label='Pick log']",
      // DK class patterns (CSS modules partially match)
      "[class*='PickLog']",
      "[class*='DraftLog']",
      "[class*='PickFeed']",
      "[class*='draftLog']",
      "[class*='pickLog']",
      "[class*='pickFeed']",
      "[class*='DraftBoard__picks']",
      "[class*='PickHistory']",
      "[class*='pickHistory']",
      "[class*='RecentPicks']",
      "[class*='recentPicks']",
      "[class*='DraftQueue']",
      "[class*='draftQueue']",
    ];
  }

  // Underdog Fantasy — playunderdog.com
  if (host.includes("underdog") || host.includes("playunderdog")) {
    return [
      // data-testid / data-qa patterns
      "[data-testid='draft-log']",
      "[data-testid='pick-log']",
      "[data-testid='draft-history']",
      "[data-testid='picks-made']",
      "[data-testid='picks-list']",
      "[data-testid='pick-history']",
      "[data-qa='draft-log']",
      "[data-qa='pick-history']",
      "[data-qa='picks-made']",
      // ARIA
      "[aria-label='Draft log']",
      "[aria-label='Picks made']",
      "[aria-label='Pick history']",
      "[aria-label='Recent picks']",
      // Class patterns
      "[class*='DraftLog']",
      "[class*='draftLog']",
      "[class*='PickLog']",
      "[class*='pickLog']",
      "[class*='draft-log']",
      "[class*='pick-log']",
      "[class*='PickHistory']",
      "[class*='pickHistory']",
      "[class*='PicksMade']",
      "[class*='picksMade']",
      "[class*='DraftPicks']",
      "[class*='draftPicks']",
    ];
  }

  // Sleeper — sleeper.com
  if (host.includes("sleeper")) {
    return [
      ".pick-list",
      ".draft-log",
      ".draft-history",
      "[data-testid='pick-list']",
      "[data-testid='draft-history']",
      "[class*='PickList']",
      "[class*='DraftHistory']",
      "[class*='pick-list']",
      "[class*='draft-history']",
      "[class*='DraftLog']",
      "[aria-label='Draft history']",
      "[aria-label='Pick list']",
    ];
  }

  // ESPN Fantasy
  if (host.includes("espn")) {
    return [
      ".pick-history",
      ".draft-history",
      "[class*='PickHistory']",
      "[class*='DraftHistory']",
      "[data-testid='pick-history']",
      "[data-testid='draft-history']",
    ];
  }

  // Yahoo Fantasy
  if (host.includes("yahoo")) {
    return [
      "#pick-list",
      ".pick-list",
      "[class*='pick-list']",
      "[class*='draft-history']",
      "[data-tst='picks-feed']",
    ];
  }

  return [];
}

// Cache the detected history element so we skip the full cascade on subsequent scans.
// Key: element identity via id/class fingerprint; value: the element reference.
let _cachedHistoryEl = null;
let _cachedHistoryKey = "";

function findPickHistory() {
  // DraftKings renders only the last pick, not a scrollable history.
  // We use addDKLastPick() (event-driven + per-scan polling) instead.
  if (location.hostname.includes("draftkings")) return null;

  const panel = document.getElementById("dm-panel");

  // Fast path: verify the previously-found element is still valid and growing
  if (_cachedHistoryEl && document.contains(_cachedHistoryEl) && isVisibleElement(_cachedHistoryEl)) {
    if (!panel?.contains(_cachedHistoryEl) && scanRoot(_cachedHistoryEl).size >= 1) {
      return _cachedHistoryEl;
    }
    // Element became empty or stale — fall through to re-detect
    _cachedHistoryEl = null;
    _cachedHistoryKey = "";
  }

  // Strategy 1: User override via settings panel or window.dm.setHistorySelector()
  if (state.customHistorySelector) {
    try {
      for (const el of document.querySelectorAll(state.customHistorySelector)) {
        if (isVisibleElement(el) && !panel?.contains(el)) {
          const found = scanRoot(el);
          if (found.size >= 1) return _cacheAndReturn(el);
        }
      }
    } catch (e) {
      console.warn("[DraftManager] custom selector error:", e);
    }
  }

  // Strategy 2: ARIA role=log — the semantic standard for live activity logs
  for (const el of document.querySelectorAll("[role='log']")) {
    if (isVisibleElement(el) && !panel?.contains(el)) {
      if (scanRoot(el).size >= 1) return _cacheAndReturn(el);
    }
  }

  // Strategy 3: Site-specific selectors (most reliable for known sites)
  for (const sel of getSiteSpecificSelectors()) {
    try {
      for (const el of document.querySelectorAll(sel)) {
        if (isVisibleElement(el) && !panel?.contains(el)) {
          if (scanRoot(el).size >= 1) return _cacheAndReturn(el);
        }
      }
    } catch {}
  }

  // Strategy 4: Attribute-based patterns (stable across CSS module obfuscation)
  const attrSelectors = [
    "[aria-label*='pick history' i]", "[aria-label*='draft history' i]",
    "[aria-label*='picks made' i]",   "[aria-label*='recent picks' i]",
    "[aria-label*='pick feed' i]",    "[aria-label*='draft feed' i]",
    "[aria-label*='draft log' i]",    "[aria-label*='pick log' i]",
    "[data-testid*='pick-history']",  "[data-testid*='draft-history']",
    "[data-testid*='picks-list']",    "[data-testid*='draft-feed']",
    "[data-testid*='pick-log']",      "[data-testid*='draft-log']",
    "[data-testid*='picks-made']",    "[data-testid*='pick-order']",
    "[data-testid*='pick-list']",     "[data-testid*='picks-feed']",
    "[data-qa*='pick']",              "[data-qa*='draft-log']",
  ];
  for (const sel of attrSelectors) {
    try {
      for (const el of document.querySelectorAll(sel)) {
        if (isVisibleElement(el) && !panel?.contains(el)) {
          if (scanRoot(el).size >= 1) return _cacheAndReturn(el);
        }
      }
    } catch {}
  }

  // Strategy 5: Class/id/data-testid keyword patterns
  const keywordRe = /pick.?hist|draft.?hist|pick.?log|draft.?log|pick.?feed|draft.?feed|picks.?made|pick.?order|recent.?pick|PickHist|DraftHist|PickLog|DraftLog|PickFeed|DraftFeed|pickHist|draftHist|pickLog|draftLog/i;
  for (const el of document.querySelectorAll("section,div,ul,ol,aside,nav,article,main")) {
    if (!isVisibleElement(el) || panel?.contains(el)) continue;
    const testId = el.getAttribute("data-testid") || el.getAttribute("data-qa") || "";
    const hint = `${el.className || ""} ${el.id || ""} ${testId}`;
    if (!keywordRe.test(hint)) continue;
    if (el.children.length < 1 || el.children.length > 400) continue;
    if (scanRoot(el).size >= 1) return _cacheAndReturn(el);
  }

  // Strategy 6: Behavioral detection — find the container that grows over time
  // and has pick-card-like structure (1 player per child element)
  const behaviorEl = findPickHistoryByBehavior(panel);
  return behaviorEl ? _cacheAndReturn(behaviorEl) : null;
}

// Cache-and-return helper: validates size before caching to reject the player board.
function _cacheAndReturn(el) {
  const maxPicks = (state.settings.teams || 12) * (state.settings.rounds || 20);
  if (scanRoot(el).size > maxPicks) {
    return null;  // more players than the whole draft — this is the player board
  }
  _cachedHistoryEl  = el;
  _cachedHistoryKey = el.id || el.className.slice(0, 60);
  return el;
}

// Behavioral detection: score containers by structural and growth signals.
// Pick history containers: grow one item at a time, each child = one pick card.
// Available board: shrinks as picks are made, has many players per "section".
function findPickHistoryByBehavior(panel) {
  const maxPicks = (state.settings.teams || 12) * (state.settings.rounds || 20);
  let bestEl = null;
  let bestScore = -Infinity;

  for (const el of document.querySelectorAll("section,div,ul,ol,aside,article,nav,main")) {
    if (!isVisibleElement(el) || panel?.contains(el)) continue;

    const childCount = el.children.length;
    if (childCount < 1 || childCount > 400) continue;

    const players = scanRoot(el);
    // Hard cap: pick history can never have more players than the entire draft
    if (players.size < 1 || players.size > maxPicks) continue;

    // Skip containers dominated by a parent already found
    if (bestEl && bestEl.contains(el)) continue;

    let score = 0;

    // Growth signal — stable key (NO childCount) so DOM mutations don't reset tracking.
    // DraftKings virtualizes its player board: childCount changes constantly, which was
    // resetting the key and firing growth at full strength on every scan.
    const elKey = (el.id || "") + "|" + el.className.slice(0, 60);
    const prevCount = state.containerPlayerCounts.get(elKey) || 0;
    const growth = players.size - prevCount;
    state.containerPlayerCounts.set(elKey, players.size);
    // Real picks: at most 1–2 between scans. Reward small growth; penalize rapid jumps
    // (rapid growth = player board loading more rows, not real picks being made).
    if (growth > 0 && growth <= 2) score += growth * 15;
    else if (growth > 2)           score -= (growth - 2) * 8;

    // Pick-card structure: check how many children contain exactly 1 player name
    const childSample = [...el.children].slice(0, 20);
    let singlePlayerChildren = 0;
    let noPlayerChildren = 0;
    for (const child of childSample) {
      const cp = scanRoot(child).size;
      if (cp === 1) singlePlayerChildren++;
      else if (cp === 0) noPlayerChildren++;
    }
    const sampleRatio = childSample.length / Math.max(childCount, 1);
    singlePlayerChildren = Math.round(singlePlayerChildren / sampleRatio);
    const pickCardRatio = singlePlayerChildren / Math.max(childCount, 1);
    score += pickCardRatio * 25;

    if (singlePlayerChildren === childCount) score += 20;

    score += Math.min(players.size, 30) * 1.5;

    // Steeper penalty for large containers — player board easily hits 50-150 players.
    // Old penalty was too gentle: (size-60)*0.5. New: (size-40)*2.
    if (players.size > 40) score -= (players.size - 40) * 2;

    if (/\bround\b|\bpick\b|\brd\s*\d/i.test(el.textContent || "")) score += 5;

    let ancestor = el.parentElement;
    for (let depth = 0; depth < 4 && ancestor; depth++, ancestor = ancestor.parentElement) {
      const hint = `${ancestor.className || ""} ${ancestor.id || ""}`;
      if (/pick.?hist|draft.?hist|pick.?log|draft.?log|pick.?feed|picks.?made/i.test(hint)) {
        score += 8;
        break;
      }
    }

    if (score > bestScore) {
      bestScore = score;
      bestEl = el;
    }
  }

  return bestScore >= 10 ? bestEl : null;
}

// Extract a known NFL position abbreviation from pick-card text.
// Used when a player isn't in our DB but we can still read their position.
function extractPositionFromRowText(rowText) {
  const m = rowText.match(/\b(QB|WR|RB|TE)\b/);
  return m ? m[1] : "?";
}

// Extract player picks from the history container in DOM order.
// Returns [{norm, name, position, pickNum}] — each item is one pick.
// DOM order may be newest-first (most draft sites); we detect and correct this.
function readPickHistory(container) {
  // Try progressively broader row selectors
  const rowSelectors = [
    // Most specific first (explicit pick card attributes)
    "[data-pick-number]", "[data-pick]", "[data-pick-id]",
    // DK/Underdog player/athlete id attributes (one per pick card)
    "[data-athlete-id]", "[data-player-id]", "[data-player-key]",
    // data-testid pick card patterns
    "[data-testid*='pick-item']", "[data-testid*='pick-card']",
    "[data-testid*='draft-pick-']", "[data-testid*='pick-entry']",
    // Semantic list rows
    "[role='listitem']", "[role='row']",
    // Common pick-card class patterns (non-hashed CSS)
    "[class*='pick-row']", "[class*='pickRow']",
    "[class*='pick-item']", "[class*='PickItem']",
    "[class*='draft-pick']", "[class*='DraftPick']",
    "[class*='pick-card']", "[class*='PickCard']",
    "[class*='pick-slot']", "[class*='PickSlot']",
    "[class*='roster-pick']", "[class*='RosterPick']",
    "[class*='draft-slot']", "[class*='DraftSlot']",
    // Standard HTML list/table items
    "li", "tr",
  ];

  let rows = [];
  for (const sel of rowSelectors) {
    try {
      const found = [...container.querySelectorAll(sel)];
      if (found.length >= 2) { rows = found; break; }
    } catch {}
  }

  // Fallback 1: direct children — but only if the container is plausibly pick-sized.
  // Skipping this for huge containers (player board) prevents mass-adding available players.
  const maxTotalPicks = (state.settings.teams || 12) * (state.settings.rounds || 20);
  if (rows.length < 2 && container.children.length <= maxTotalPicks + 5) {
    rows = [...container.children];
  }

  // Fallback 2: if children each have multiple sub-elements (round-grouped layout),
  // go one level deeper — grab grandchildren that each have exactly one player name.
  if (rows.length >= 2 && rows.length < 20) {
    const grandkids = rows.flatMap((r) => [...r.children]);
    if (grandkids.length > rows.length) {
      const withPlayer = grandkids.filter((g) => scanRoot(g).size === 1);
      if (withPlayer.length > rows.length) rows = grandkids;
    }
  }

  // Keep only the deepest non-overlapping elements
  if (rows.length > 5) {
    rows = rows.filter((r) => !rows.some((o) => o !== r && o.contains(r)));
  }

  // Extract picks and explicit pick numbers from each row
  const picks = [];
  const seenNorms = new Set();
  const teams  = state.settings.teams  || 12;
  const rounds = state.settings.rounds || 20;
  const maxPick = teams * rounds;

  for (const row of rows) {
    const rowText = row.textContent || "";
    let pickNum = null;

    // 1. "Round.Pick" / "Rd.Pk" format: "2.12" or "1.04" (most common on Underdog/Sleeper)
    const rdPkDot = rowText.match(/\b(\d{1,2})\.(\d{1,2})\b/);
    if (rdPkDot) {
      const r = parseInt(rdPkDot[1], 10), p = parseInt(rdPkDot[2], 10);
      if (r >= 1 && r <= rounds && p >= 1 && p <= teams) pickNum = (r - 1) * teams + p;
    }

    // 2. "Rd 2, Pk 12" / "Round 2 Pick 12" / "R2P12" style
    if (pickNum === null) {
      const rdPkText = rowText.match(/r(?:d|ound)?\s*\.?\s*(\d{1,2})\s*[,·|\s]\s*p(?:k|ick)?\s*#?\s*(\d{1,2})\b/i);
      if (rdPkText) {
        const r = parseInt(rdPkText[1], 10), p = parseInt(rdPkText[2], 10);
        if (r >= 1 && r <= rounds && p >= 1 && p <= teams) pickNum = (r - 1) * teams + p;
      }
    }

    // 3. Explicit overall pick: "Pick 24", "Pick #24", "#24", "Overall pick 24"
    if (pickNum === null) {
      const overallMatch = rowText.match(/(?:overall\s+)?pick\s*#?\s*(\d{1,3})\b/i) ||
                           rowText.match(/#\s*(\d{1,3})\b/);
      if (overallMatch) {
        const n = parseInt(overallMatch[1], 10);
        if (n >= 1 && n <= maxPick) pickNum = n;
      }
    }

    // 4. Number at start of row text: "24." or "24 PlayerName" (numbered list style)
    if (pickNum === null) {
      const startNum = rowText.trim().match(/^(\d{1,3})\s*[.\s]/);
      if (startNum) {
        const n = parseInt(startNum[1], 10);
        if (n >= 1 && n <= maxPick) pickNum = n;
      }
    }

    // Also try data attributes on the row element itself
    if (pickNum === null) {
      const attrNum = row.getAttribute("data-pick-number") ||
                      row.getAttribute("data-pick") ||
                      row.getAttribute("data-pick-id") ||
                      row.getAttribute("data-pick-index");
      if (attrNum) {
        const n = parseInt(attrNum, 10);
        if (n >= 1 && n <= maxPick) pickNum = n;
      }
    }

    // Also check aria-label / title / data-player-name on the row element itself —
    // DK/Underdog sometimes encode player names in these attributes
    const rowAttrText = [
      row.getAttribute("aria-label") || "",
      row.getAttribute("title") || "",
      row.getAttribute("data-player-name") || "",
      row.getAttribute("data-player") || "",
    ].join(" ");
    const normalizedAttr = rowAttrText.length > 2 ? normalize(rowAttrText) : "";

    let foundPlayer = false;

    // Priority 1: aria-label / title / data-player-name on the row element itself
    // DK/Underdog sometimes encode player names in these attributes
    if (!foundPlayer && normalizedAttr) {
      for (const [norm, player] of Object.entries(state.playerIndex)) {
        if (!seenNorms.has(norm) && normalizedAttr.includes(norm) && player) {
          seenNorms.add(norm);
          picks.push({ norm, name: player.name, position: player.position, pickNum });
          foundPlayer = true;
          break;
        }
      }
    }

    // Priority 2: scan specific "player name" sub-elements first to avoid false matches
    // from "drafted by" team names or other text in the pick card
    if (!foundPlayer) {
      const nameSel = '[class*="player-name"],[class*="playerName"],[class*="player_name"],' +
        '[class*="PlayerName"],[class*="athlete-name"],[class*="AthleteName"],' +
        'strong,b,[data-player-name]';
      const nameEl = row.querySelector(nameSel);
      if (nameEl) {
        for (const norm of scanRoot(nameEl)) {
          if (!seenNorms.has(norm)) {
            seenNorms.add(norm);
            const player = state.playerIndex[norm];
            if (player) {
              picks.push({ norm, name: player.name, position: player.position, pickNum });
              foundPlayer = true;
            }
            break;
          }
        }
      }
    }

    // Priority 3: full DOM scan of the row (catches any player name anywhere in the card)
    if (!foundPlayer) {
      for (const norm of scanRoot(row)) {
        if (!seenNorms.has(norm)) {
          seenNorms.add(norm);
          const player = state.playerIndex[norm];
          if (player) {
            picks.push({ norm, name: player.name, position: player.position, pickNum });
            foundPlayer = true;
          }
          break;
        }
      }
    }

    // If no known player found but this row has an explicit pick number, it's almost
    // certainly a real pick for a player not in our DB. Add a placeholder so subsequent
    // picks still get correct team slot assignments (prevent slot drift).
    if (!foundPlayer && pickNum !== null && pickNum >= 1 && pickNum <= maxPick) {
      const phNorm = `__ph_${pickNum}`;
      if (!seenNorms.has(phNorm)) {
        seenNorms.add(phNorm);
        const position = extractPositionFromRowText(rowText);
        picks.push({ norm: phNorm, name: "Unknown", position, pickNum, unknown: true });
      }
    }
  }

  return picks;
}

// Determine if pick history is newest-first or oldest-first.
// Returns the correctly ordered list (chronological: pick1 first).
// When ordering is ambiguous on first scan, defaults to newest-first (DK/Underdog behavior).
// If a later scan detects the initial assumption was wrong, resets picks and re-processes.
function orderPickHistory(picks) {
  if (picks.length === 0) return picks;

  // User has forced an order via settings (persisted)
  if (state.historyNewestFirst === true)  return picks.slice().reverse();
  if (state.historyNewestFirst === false) return picks;

  // Try to detect from explicit pick numbers embedded in the DOM rows (most reliable)
  const numbered = picks.filter((p) => p.pickNum !== null);
  if (numbered.length >= 2) {
    const firstNum = numbered[0].pickNum;
    const lastNum  = numbered[numbered.length - 1].pickNum;
    if (firstNum > lastNum) return picks.slice().reverse();  // newest-first
    if (firstNum < lastNum) return picks;                    // oldest-first
  }

  // Try to detect from existing allDrafted — but only when we have few picks committed
  // and haven't already locked in an ordering. Once we have 4+ picks, the ordering
  // is settled; flipping would just re-add the same players and create duplicates.
  const flipCount = state._orderingFlipCount || 0;
  if (state.allDrafted.length >= 3 && state.allDrafted.length < 4 && flipCount < 1) {
    const knownNorms = state.allDrafted.map((d) => normalize(d.name));
    const histNorms  = picks.map((p) => p.norm);
    const revNorms   = picks.slice().reverse().map((p) => p.norm);
    const fwdScore = knownNorms.slice(0, 5).filter((n, i) => histNorms[i] === n).length;
    const revScore = knownNorms.slice(0, 5).filter((n, i) => revNorms[i]  === n).length;

    const curNewestFirst = state._orderedNewestFirst;
    // Require strong evidence (≥3 gap) to correct — prevents flip-flopping on ambiguous data
    if (curNewestFirst === true && fwdScore >= revScore + 3) {
      console.log("[DraftManager] Ordering correction → oldest-first");
      state.allDrafted  = [];
      state.myTeam      = [];
      state.teamRosters = {};
      state._orderedNewestFirst = false;
      state._orderingFlipCount  = flipCount + 1;
      clearDraftState();
      return picks;
    }
    if (curNewestFirst === false && revScore >= fwdScore + 3) {
      console.log("[DraftManager] Ordering correction → newest-first");
      state.allDrafted  = [];
      state.myTeam      = [];
      state.teamRosters = {};
      state._orderedNewestFirst = true;
      state._orderingFlipCount  = flipCount + 1;
      clearDraftState();
      return picks.slice().reverse();
    }

    if (revScore > fwdScore) { state._orderedNewestFirst = true;  return picks.slice().reverse(); }
    if (fwdScore > revScore) { state._orderedNewestFirst = false; return picks; }
  }

  // Default: DraftKings and Underdog show newest pick at the top (newest-first)
  state._orderedNewestFirst = true;
  return picks.slice().reverse();
}

// Sync picks found in history into state.allDrafted and state.teamRosters.
// pickHistoryList = [{norm, name, position, pickNum}] in chronological order (pick1 first).
function syncPickHistory(pickHistoryList) {
  if (state.dkApiActive && state.allDrafted.length > 0) return;  // non-empty DK JSON feed is authoritative
  const teams    = state.settings.teams;
  const mySlot   = state.settings.pickPos;
  const maxPicks = teams * (state.settings.rounds || 20);
  let added      = 0;
  const hasExplicitPickNums = pickHistoryList.some((pick) => Number(pick.pickNum) >= 1);

  const knownNorms = draftedKeys();
  const prevLength = state.allDrafted.length;

  pickHistoryList.forEach((pick, idx) => {
    if (state.allDrafted.length >= maxPicks) return;  // never exceed entire draft
    const player    = pick.unknown ? null : (state.playerIndex[pick.norm] || matchPlayer(pick.name));
    const draftKey  = pick.unknown ? pick.norm : (player ? normalize(player.name) : normalize(pick.name));
    const lookupKey = draftKey;
    if (knownNorms.has(lookupKey)) return;

    // Prefer explicit pick number from DOM (accurate even for partial history views).
    // Fall back to position in the chronological list (correct when full history is in DOM).
    const overallPick = pick.pickNum || (idx + 1);
    const slot        = pickToTeamSlot(overallPick, teams);
    const byUser      = !pick.unknown && slot === mySlot;
    const round       = Math.floor((overallPick - 1) / teams) + 1;
    const name        = player?.name || pick.name;
    const position    = player?.position || pick.position;

    const entry = {
      name,
      position,
      round,
      by_user: byUser,
      team_slot: slot,
      overall_pick: overallPick,
      ...(pick.unknown ? { _phNorm: pick.norm } : { _draftKey: draftKey }),  // track canonical key
    };
    state.allDrafted.push(entry);
    knownNorms.add(lookupKey);
    knownNorms.add(pick.norm);

    // Unknown placeholders still consume a pick slot but don't go into rosters
    if (!pick.unknown) {
      if (byUser) state.myTeam.push({ name, position, round });
      if (!state.teamRosters[slot]) state.teamRosters[slot] = [];
      state.teamRosters[slot].push({ name, position, round });
    }
    added++;
  });

  // Runaway guard: if we just added way more picks than any single scan could legitimately
  // produce (>1 round worth of new picks, and allDrafted was empty before), the cached
  // container is probably the player board. Reset everything and re-detect.
  if (added > teams * 2 && prevLength === 0 && !state.pageCurrentPick && !hasExplicitPickNums) {
    console.warn(`[DM] Runaway sync: ${added} picks added cold — likely wrong container. Resetting.`);
    _cachedHistoryEl  = null;
    _cachedHistoryKey = "";
    state.allDrafted  = [];
    state.myTeam      = [];
    state.teamRosters = {};
    clearDraftState();
    return 0;
  }

  if (added > 0) {
    reconcileDraftState();
    saveDraftState();
  }
  return added;
}

function scanPage() {
  const isDK = location.hostname.toLowerCase().includes("draftkings");
  const currentPick = isDK ? getTrackedNextPick() : detectCurrentPickFromPage();
  // Derive available from CANONICAL norms minus everyone drafted
  // (exclude abbreviated aliases like "d henry" which would create duplicates)
  const draftedNorms = draftedKeys();
  state.available = new Set([...state.canonicalNorms].filter((n) => !draftedNorms.has(n)));
  pruneQueue();
  return { currentPick };
}

// ── Draft a player ────────────────────────────────────────────────────────────

function gradePickLabel(playerRec) {
  const tier = playerRec.value_tier;
  const urgency = playerRec.going_soon ? " (beat the run)" : "";
  if (tier === "steal") return `A+ steal${urgency}`;
  if (tier === "value") return `A value${urgency}`;
  if (tier === "fair")  return playerRec.going_soon ? `B+ smart (was going soon)` : "B fair value";
  if (tier === "reach") return playerRec.vor < -20 ? "D big reach" : "C slight reach";
  return "";
}

function draftPlayer(name, position, byUser, playerRec = null) {
  const player = matchPlayer(name);
  const canonName = player?.name || name;
  const canonPos = player?.position || position;
  const norm = draftKeyForName(name);

  // If already tracked (history scanner already added it), just update by_user if needed
  const existingIdx = state.allDrafted.findIndex((d) => draftedKeys().has(norm) && draftKeyForName(d.name) === norm);
  if (existingIdx !== -1) {
    if (byUser && !state.allDrafted[existingIdx].by_user) {
      // Re-mark as by_user (user confirmed a pick that history assigned to wrong team)
      state.allDrafted[existingIdx].by_user = true;
      if (!state.myTeam.some((p) => normalize(p.name) === norm)) {
        state.myTeam.push({ name: canonName, position: canonPos, round: state.allDrafted[existingIdx].round });
      }
    }
    state.available.delete(norm);
    saveDraftState();
    scanAndRank();
    return;
  }

  state.available.delete(norm);
  const overallPick = pickNumberForNewDetectedPick();
  const round = Math.floor((overallPick - 1) / state.settings.teams) + 1;

  // Auto-detect pick position: if user claims a pick and pick-history tracking
  // is active, the actual slot from snake math should match their setting.
  // If it doesn't, offer to auto-correct.
  if (byUser) {
    const inferredSlot = pickToTeamSlot(overallPick, state.settings.teams);
    if (inferredSlot !== state.settings.pickPos && state.allDrafted.length > 0) {
      console.log(`[DraftManager] Pick position hint: you picked at overall #${overallPick} → slot ${inferredSlot} (settings say ${state.settings.pickPos})`);
      // Auto-correct only if pick history is being tracked (not a manual-only session)
      if (state.allDrafted.some(d => !d.by_user)) {
        state.settings.pickPos = inferredSlot;
        saveSettings();
        updateSettingsUI();
        recomputeUserPicks();  // re-derive by_user/team_slot for the corrected slot
        setStatus(`Pick # auto-corrected to ${inferredSlot}`);
      }
    }
  }

  const slot = byUser ? state.settings.pickPos : pickToTeamSlot(overallPick, state.settings.teams);
  state.allDrafted.push({ name: canonName, position: canonPos, round, by_user: byUser, team_slot: slot, overall_pick: overallPick, _draftKey: norm });
  if (!state.teamRosters[slot]) state.teamRosters[slot] = [];
  state.teamRosters[slot].push({ name: canonName, position: canonPos, round });

  if (byUser) {
    state.myTeam.push({ name: canonName, position: canonPos, round });

    // Flash pick grade in status bar for 4 seconds
    if (playerRec) {
      const grade = gradePickLabel(playerRec);
      if (grade) {
        const prevStatus = document.getElementById("dm-status")?.textContent || "";
        setStatus(`${grade} — ${canonName}`);
        setTimeout(() => setStatus(prevStatus), 4000);
      }
    }
  }
  saveDraftState();
  scanAndRank();
}

// ── Render ────────────────────────────────────────────────────────────────────

function renderAll(data) {
  const recs = data.recommendations || [];
  const renderSig = JSON.stringify({
    recs: recs.map((p) => [
      p.name,
      p.position,
      p.vor,
      p.dk_ev_score,
      p.dk_ev_rank,
      p.ranking_source,
      p.going_soon,
      p.is_stack,
      p.bye_alert,
    ]),
    drafted: state.allDrafted.map((p) => [p.name, p.position, p.round, p.by_user, p.team_slot]),
    myTeam: state.myTeam.map((p) => [p.name, p.position, p.round]),
    queue: state.queue,
    autodraftArmed: state.autodraftArmed,
    alerts: [data.stack_alerts || [], data.scarcity_warnings || []],
    scoring: data.scoring || state.settings.scoring || "full",
    policy: data.dk_policy_model || "",
    tab: document.getElementById("dm-tab-teams")?.style.display === "none" ? "picks" : "teams",
  });
  if (state._lastRenderSig === renderSig) return;
  state._lastRenderSig = renderSig;

  const body = document.getElementById("dm-body");
  const prevScrollTop = body?.scrollTop ?? 0;
  const prevScrollHeight = body?.scrollHeight ?? 0;

  renderTrackingBanner();
  renderRosterSummary();
  renderQueue(recs);
  renderRecs(recs, data);
  clearPickTabExtras();
  renderTeamRosters();

  if (body) {
    const heightDelta = body.scrollHeight - prevScrollHeight;
    body.scrollTop = Math.max(0, prevScrollTop + Math.min(heightDelta, 0));
  }
}

function clearPickTabExtras() {
  const myTeamSec = document.getElementById("dm-myteam-section");
  const alertsSec = document.getElementById("dm-alerts-section");
  const scarcitySec = document.getElementById("dm-scarcity-section");
  if (myTeamSec && myTeamSec.innerHTML) myTeamSec.innerHTML = "";
  if (alertsSec && alertsSec.innerHTML) alertsSec.innerHTML = "";
  if (scarcitySec && scarcitySec.innerHTML) scarcitySec.innerHTML = "";
}

function renderQueue(recs = []) {
  const sec = document.getElementById("dm-queue-section");
  if (!sec) return;

  pruneQueue();
  const target = getAutodraftTarget(recs);
  const targetLabel = target?.source || "TOP EV";
  const autodraftSupported = currentSitePlatform() === "draftkings";

  let html = `<div class="dm-queue-head">
    <span>Autodraft</span>
    <span class="dm-queue-actions">
      <button id="dm-autodraft-toggle" class="${state.autodraftArmed && autodraftSupported ? "dm-autodraft-armed" : ""}" title="${autodraftSupported ? "Toggle DraftKings autodraft (queue-first, fires 15s into your clock)" : "Autodraft is not verified for this platform"}"${autodraftSupported ? "" : " disabled"}>${state.autodraftArmed && autodraftSupported ? "AUTODRAFT ARMED" : "AUTODRAFT OFF"}</button>
      <button id="dm-queue-clear" title="Clear queue"${state.queue.length ? "" : " disabled"}>Clear</button>
    </span>
  </div>`;

  if (target) {
    const name = target.name || target.player_display_name;
    const pos = target.position || target.pos || "";
    const team = target.team || target.recent_team || "";
    html += `<div class="dm-queue-target">
      <span class="dm-queue-source">${targetLabel}</span>
      <span class="dm-queue-target-name">${name}</span>
      <span class="dm-queue-target-meta">${[pos, team].filter(Boolean).join(" · ")}</span>
    </div>`;
  } else {
    html += `<div class="dm-queue-target dm-queue-target-empty">
      <span class="dm-queue-source">TOP EV</span>
      <span class="dm-queue-target-name">No live recommendation yet</span>
    </div>`;
  }

  if (state.queue.length === 0) {
    html += `<div class="dm-queue-empty">Queue empty → autodraft uses the top EV pick. Add players to override who gets drafted.</div>`;
  } else {
    html += `<div class="dm-queue-empty">Autodraft drafts the top available queued player; top EV is the fallback.</div>`;
    html += `<div class="dm-queue-list">`;
    state.queue.forEach((key, idx) => {
      const p = state.playerIndex[key];
      if (!p) return;
      html += `<div class="dm-queue-row" data-key="${key}">
        <span class="dm-queue-num">${idx + 1}</span>
        <span class="dm-queue-name">${p.name}</span>
        <span class="dm-queue-meta">${p.position} · ${p.team || ""}</span>
        <button class="dm-queue-up" title="Move up"${idx === 0 ? " disabled" : ""}>↑</button>
        <button class="dm-queue-down" title="Move down"${idx === state.queue.length - 1 ? " disabled" : ""}>↓</button>
        <button class="dm-queue-remove" title="Remove">×</button>
      </div>`;
    });
    html += `</div>`;
  }

  const changed = sec.innerHTML !== html;
  if (changed) sec.innerHTML = html;
  if (!changed) return;

  document.getElementById("dm-queue-clear")?.addEventListener("click", clearQueue);
  document.getElementById("dm-autodraft-toggle")?.addEventListener("click", toggleAutodraft);
  sec.querySelectorAll(".dm-queue-row").forEach((row) => {
    const key = row.dataset.key;
    row.querySelector(".dm-queue-up")?.addEventListener("click", () => moveQueueItem(key, -1));
    row.querySelector(".dm-queue-down")?.addEventListener("click", () => moveQueueItem(key, 1));
    row.querySelector(".dm-queue-remove")?.addEventListener("click", () => removeFromQueue(key));
  });
}

function renderTrackingBanner() {
  const picksSoFar  = state.allDrafted.length;
  const pagePickNum = state.pageCurrentPick;
  const expectedMin = pagePickNum ? pagePickNum - 1 : 0;

  // Condition 1: tracked significantly fewer picks than expected
  const bigDiscrepancy = picksSoFar > 0 && expectedMin > 0 &&
    (expectedMin - picksSoFar) > 5 && (expectedMin - picksSoFar) / expectedMin > 0.3;

  // Condition 2: page shows picks in progress but we've tracked none
  const pageShowsProgress = expectedMin > 3;

  // Condition 3: running 10+ scans, 0 picks, AND page suggests draft has started
  // (only warn when it looks like picks SHOULD be happening but aren't detected)
  const draftSeemStarted = (pagePickNum !== null && pagePickNum > 1) ||
    /\bon\s+the\s+clock\b|pick\s+#?\d|round\s+\d/i.test(getPageText().slice(0, 5000));
  const likelyMissed = picksSoFar === 0 && state.scanCount >= 10 && draftSeemStarted;

  if ((!bigDiscrepancy && !pageShowsProgress && !likelyMissed) || (picksSoFar > 0 && !bigDiscrepancy)) {
    document.getElementById("dm-tracking-banner")?.remove();
    return;
  }

  const detail = bigDiscrepancy
    ? `Tracking ${picksSoFar} of ~${expectedMin} picks.`
    : "Pick log not detected.";
  const html =
    `⚠ <strong>${detail}</strong> Alt+↺ Rescan to diagnose, ` +
    `or open ⚙ Settings → <em>Pick history selector</em>.`;

  let banner = document.getElementById("dm-tracking-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "dm-tracking-banner";
    banner.style.cssText = [
      "background:#2a1500;border-left:3px solid #ff9900;padding:5px 7px;",
      "margin-bottom:6px;border-radius:0 3px 3px 0;font-size:10px;color:#ffcc88;",
      "cursor:pointer;"
    ].join("");
    document.getElementById("dm-recs-section")?.before(banner);
  }
  if (banner.innerHTML !== html) banner.innerHTML = html;
  banner.title = "Click to open Settings";
  banner.onclick = () => {
    const sp = document.getElementById("dm-settings-panel");
    if (sp) sp.style.display = "";
    document.getElementById("dm-history-selector")?.focus();
  };
}

// Build the shared player-name datalist once (≈800 options is too heavy to
// rebuild every render). Reused by the inline team editor's add input.
function ensurePlayerDatalist() {
  let dl = document.getElementById("dm-player-list");
  if (dl && dl.childElementCount > 0) return dl;
  if (!dl) {
    dl = document.createElement("datalist");
    dl.id = "dm-player-list";
    document.body.appendChild(dl);
  }
  if (state.allNames && state.allNames.length) {
    dl.innerHTML = state.allNames.map((name) => `<option value="${escapeHtml(name)}"></option>`).join("");
  }
  return dl;
}

function renderTeamRosters() {
  const sec = document.getElementById("dm-teams-section");
  if (!sec) return;

  const teams  = state.settings.teams;
  const mySlot = state.settings.pickPos;

  // Default viewing slot to user's own team
  if (!state.viewingTeamSlot) state.viewingTeamSlot = mySlot || 1;

  const nextPick  = getNextBoardPick();
  const clockSlot = pickToTeamSlot(nextPick, teams);

  const viewSlot  = state.viewingTeamSlot;
  const picks     = state.teamRosters[viewSlot] || [];
  const isMe      = viewSlot === mySlot;
  const isOnClock = viewSlot === clockSlot;

  const posCount = { QB: 0, WR: 0, RB: 0, TE: 0 };
  picks.forEach((p) => { if (posCount[p.position] !== undefined) posCount[p.position]++; });
  const NEEDS = { QB: 2, WR: 4, RB: 3, TE: 1 };
  const needy = Object.entries(NEEDS)
    .filter(([pos, min]) => posCount[pos] < min)
    .map(([pos]) => pos);

  // ── header: ← label → clock ───────────────────────────────────────────────
  const prevSlot = viewSlot === 1 ? teams : viewSlot - 1;
  const nextSlot = viewSlot === teams ? 1 : viewSlot + 1;

  let clockBadge = "";
  if (isOnClock && isMe)  clockBadge = `<span style="color:#ffdd00;font-weight:900;font-size:9px;margin-left:6px;">⚡ YOUR PICK</span>`;
  else if (isOnClock)     clockBadge = `<span style="color:#ffdd00;font-size:9px;margin-left:6px;">⚡ on clock</span>`;
  else if (clockSlot === mySlot) clockBadge = `<span style="color:#ffdd00;font-size:9px;margin-left:6px;">⚡ YOUR TURN</span>`;

  const teamLabel = isMe ? `<span style="color:#00d4aa;font-weight:700;">T${viewSlot} — YOU</span>`
                         : `<span style="color:#ccc;font-weight:700;">Team ${viewSlot}</span>`;

  const manualKeys   = manualRosterKeys();
  const overrideCount = (state.manualAdds || []).length;
  const removedCount  = state.manualRemovals instanceof Set ? state.manualRemovals.size : 0;
  const overrideNote  = (overrideCount || removedCount)
    ? `<div style="display:flex;justify-content:space-between;align-items:center;font-size:9px;color:#777;padding:3px 2px 0;">
         <span>✎ manual: ${overrideCount} added${removedCount ? `, ${removedCount} removed` : ""}</span>
         <button id="dm-manual-clear-all" style="background:none;border:1px solid #333;color:#999;font-size:8px;border-radius:3px;padding:1px 5px;cursor:pointer;">Clear all</button>
       </div>`
    : "";

  sec.innerHTML = `
    <div style="border-top:1px solid #1a1a2e;padding:6px 0 2px;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
        <button id="dm-team-prev" style="background:none;border:none;color:#555;font-size:14px;cursor:pointer;padding:0 4px;line-height:1;">◀</button>
        <div style="flex:1;text-align:center;font-size:10px;">
          ${teamLabel}${clockBadge}
        </div>
        <button id="dm-team-next" style="background:none;border:none;color:#555;font-size:14px;cursor:pointer;padding:0 4px;line-height:1;">▶</button>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;font-size:9px;color:#555;margin-bottom:4px;padding:0 2px;">
        <span>${["QB","WR","RB","TE"].map(pos => `${pos}:<b style="color:${posCount[pos]>0?'#aaa':'#444'}">${posCount[pos]}</b>`).join("  ")}</span>
        ${needy.length ? `<span style="color:#ff9900;">needs ${needy.join(" ")}</span>` : ""}
      </div>
      <div id="dm-team-player-list" style="max-height:150px;overflow-y:auto;"></div>
      <div style="display:flex;gap:4px;align-items:center;padding:5px 2px 1px;">
        <input id="dm-team-add-name" list="dm-player-list" autocomplete="off" placeholder="Add player to T${viewSlot}${isMe ? " (you)" : ""}…"
          style="flex:1;min-width:0;background:#0d0d18;border:1px solid #2a2a3e;color:#ddd;font-size:10px;border-radius:3px;padding:3px 5px;" />
        <button id="dm-team-add-btn" style="background:#1a3a32;border:1px solid #00d4aa;color:#00d4aa;font-size:10px;border-radius:3px;padding:3px 8px;cursor:pointer;flex-shrink:0;">Add</button>
      </div>
      <div style="font-size:8px;color:#555;padding:0 2px 2px;">Use this when the auto pick-up misses or misassigns a pick. Edits stick.</div>
      ${overrideNote}
    </div>`;

  ensurePlayerDatalist();

  // ── navigation buttons ─────────────────────────────────────────────────────
  document.getElementById("dm-team-prev").onclick = () => {
    state.viewingTeamSlot = prevSlot;
    renderTeamRosters();
  };
  document.getElementById("dm-team-next").onclick = () => {
    state.viewingTeamSlot = nextSlot;
    renderTeamRosters();
  };

  // ── add player to the viewed team ──────────────────────────────────────────
  const addInput = document.getElementById("dm-team-add-name");
  const submitAdd = () => {
    const name = addInput?.value?.trim();
    if (!name) return;
    if (manualAddPlayer(name, viewSlot, "team editor")) {
      if (addInput) addInput.value = "";
    }
  };
  document.getElementById("dm-team-add-btn")?.addEventListener("click", submitAdd);
  addInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); submitAdd(); }
    e.stopPropagation();  // don't let panel keyboard shortcuts hijack typing
  });
  document.getElementById("dm-manual-clear-all")?.addEventListener("click", clearManualRosterOverride);

  // ── player list ────────────────────────────────────────────────────────────
  const list = document.getElementById("dm-team-player-list");
  if (picks.length === 0) {
    list.innerHTML = `<div style="font-size:9px;color:#333;font-style:italic;padding:2px 2px;">No picks tracked yet</div>`;
    return;
  }

  picks.forEach((p) => {
    const key = draftKeyForName(p.name);
    const isManual = manualKeys.has(key);
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:5px;padding:2px 2px;border-bottom:1px solid #111;";
    row.innerHTML = `
      <span class="dm-pos-badge dm-pos-${p.position}" style="font-size:9px;padding:1px 4px;border-radius:3px;flex-shrink:0;">${p.position || "?"}</span>
      <span style="font-size:10px;color:#ddd;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(p.name)}${isManual ? ` <span style="color:#00d4aa;font-size:8px;" title="hand-added">✎</span>` : ""}</span>
      <span style="font-size:9px;color:#444;flex-shrink:0;">Rd${p.round}</span>
      <button class="dm-team-remove" title="Remove ${escapeHtml(p.name)} — back on board" style="background:none;border:none;color:#a44;font-size:12px;line-height:1;cursor:pointer;flex-shrink:0;padding:0 2px;">×</button>`;
    row.querySelector(".dm-team-remove")?.addEventListener("click", (e) => {
      e.stopPropagation();
      manualRemoveByKey(key, "team editor");
    });
    list.appendChild(row);
  });
}

// Shared board-stat formatter: DK EV edge when available, else ADP. Defensive so
// a malformed field degrades to a dash instead of throwing during render.
function formatBoardStat(p) {
  const adp = p && p.adp > 0 ? p.adp.toFixed(1) : "—";
  const hasDkEv = p && p.ranking_source === "dk_ev_policy" && Number.isFinite(Number(p.dk_ev_score));
  if (!hasDkEv) return { value: adp, label: "ADP", cls: "", title: "" };
  const evScaled = Number(p.dk_ev_score) * 100000;
  return {
    value: `${evScaled >= 0 ? "+" : ""}${evScaled.toFixed(2)}`,  // EV (top, colored)
    label: `ADP ${adp}`,                                          // ADP shown beneath the EV
    cls: evScaled >= 0 ? "dm-ev-positive" : "dm-ev-negative",
    title: `DK EV edge: ${Number(p.dk_ev_score).toExponential(3)} · ADP ${adp}`,
  };
}

function renderRecs(recs, data = {}) {
  const sec = document.getElementById("dm-recs-section");
  if (!sec) return;
  sec.innerHTML = "";
  const titleEl = document.createElement("div");
  titleEl.className = "dm-section-title";
  const usingDkEv = recs.some((p) => p.ranking_source === "dk_ev_policy");
  const scoringLabel = (data.scoring || state.settings.scoring) === "half" ? "Half PPR" : "Full PPR";
  const policyLabel = scoringLabel === "Half PPR" ? "Half-PPR EV Board" : "DK EV Board";
  titleEl.textContent = usingDkEv ? `${policyLabel} - ${scoringLabel}` : "ADP Board";
  if (usingDkEv && data.dk_policy_model) {
    const adpInfo = data.adp_source
      ? ` · ADP: ${data.adp_source}${data.page_adp_count ? ` (${data.page_adp_count} page rows)` : ""}`
      : "";
    titleEl.title = `${data.dk_policy_label || "DK EV policy"}: ${data.dk_policy_model}${adpInfo}`;
  }
  sec.appendChild(titleEl);

  if (recs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "dm-empty";
    empty.innerHTML = "No players found on page.<br>Click ↺ Rescan after board loads.";
    sec.appendChild(empty);
    return;
  }

  const seenRecNorms = new Set();
  recs.forEach((p, i) => {
    try {
      const recNorm = normalize(p.name);
      if (seenRecNorms.has(recNorm)) return;  // never render the same player twice
      seenRecNorms.add(recNorm);
      const key = draftKeyForName(p.name);
      const isQueued = state.queue.includes(key);
      const row = document.createElement("div");
      row.className = "dm-player-row";
      row.dataset.name = normalize(p.name);
      row.dataset.pos  = (p.position || "").toLowerCase();
      row.dataset.team = (p.team || "").toLowerCase();
      const age = p.age ? `${p.age.toFixed(0)}yo` : "";
      const metaParts = [p.team || "", age].filter(Boolean).join(" · ");
      const stat = formatBoardStat(p);
      row.innerHTML = `
        <span class="dm-rank">${i + 1}</span>
        <span class="dm-pos-badge dm-pos-${p.position}">${p.position}</span>
        <div class="dm-player-info">
          <div class="dm-player-name">${p.name}</div>
          <div class="dm-player-meta">${metaParts}</div>
        </div>
        <div class="dm-player-stats">
          <div class="dm-proj-pts ${stat.cls}" title="${stat.title}">${stat.value}</div>
          <div class="dm-board-label">${stat.label}</div>
        </div>
        <button class="dm-queue-add" title="${isQueued ? "Already queued" : "Add to queue"}" ${isQueued ? "disabled" : ""}>${isQueued ? "✓" : "+"}</button>
      `;
      row.querySelector(".dm-queue-add")?.addEventListener("click", (e) => {
        e.stopPropagation();
        addToQueue(p.name);
      });
      sec.appendChild(row);
    } catch (err) {
      console.warn("DraftManager: skipped a board row that failed to render", p?.name, err);
    }
  });

  // Re-apply any active search/position filter after re-render
  const searchVal = normalize(document.getElementById("dm-search")?.value || "");
  if (searchVal || state.activePosFilter !== "all") {
    filterRows(searchVal, state.activePosFilter);
  }
}

function renderMyTeam() {
  const sec = document.getElementById("dm-myteam-section");
  if (!sec || state.myTeam.length === 0) {
    if (sec) sec.innerHTML = "";
    return;
  }

  // Compute total projected points from playerIndex
  let totalProj = 0;
  state.myTeam.forEach((p) => {
    const data = state.playerIndex[normalize(p.name)];
    if (data?.proj_points) totalProj += data.proj_points;
  });
  const projLabel = totalProj > 0 ? ` · ${totalProj.toFixed(0)} proj pts` : "";

  const lastPickRound = state.myTeam.length > 0 ? state.myTeam[state.myTeam.length - 1].round : 1;
  sec.innerHTML = `<div class="dm-section-title">My Team (${state.myTeam.length} picks · Rd ${lastPickRound}${projLabel})</div>`;
  state.myTeam.forEach((p) => {
    const chip = document.createElement("span");
    chip.className = `dm-team-chip dm-pos-${p.position}`;
    const data = state.playerIndex[normalize(p.name)];
    const pts    = data?.proj_points ? ` ${data.proj_points.toFixed(0)}` : "";
    const byeWk  = (p.position === "QB" || p.position === "TE") && data?.bye_week
                   ? `(W${data.bye_week})` : "";
    const lastName = p.name.split(" ").slice(-1)[0];
    chip.textContent = `${lastName}${byeWk}${pts}`;
    chip.title = `${p.name} · ${p.position} · Rd ${p.round}${data?.proj_points ? ` · ${data.proj_points.toFixed(0)} proj pts` : ""}${data?.bye_week ? ` · Bye Wk${data.bye_week}` : ""}`;
    sec.appendChild(chip);
  });
}

function detectPositionRun() {
  // Look at the last 6 opponent picks and detect if 3+ are the same position
  const recentOpponent = state.allDrafted.filter((d) => !d.by_user).slice(-6);
  if (recentOpponent.length < 3) return [];

  const runs = [];
  const WINDOW = 5;
  const last = recentOpponent.slice(-WINDOW);
  const posCounts = {};
  last.forEach((d) => { posCounts[d.position] = (posCounts[d.position] || 0) + 1; });
  for (const [pos, count] of Object.entries(posCounts)) {
    if (count >= 3) {
      runs.push(`${pos} run: ${count} of last ${last.length} picks — consider targeting ${pos}`);
    }
  }
  return runs;
}

function renderAlerts(stacks, scarcity) {
  const alertSec    = document.getElementById("dm-alerts-section");
  const scarcitySec = document.getElementById("dm-scarcity-section");
  if (!alertSec || !scarcitySec) return;

  alertSec.innerHTML = "";
  const posRuns = detectPositionRun();
  const allAlerts = [...posRuns, ...stacks];
  if (allAlerts.length > 0) {
    alertSec.innerHTML = `<div class="dm-section-title">Alerts</div>`;
    allAlerts.forEach((s) => {
      const d = document.createElement("div");
      d.className = "dm-alert";
      d.textContent = s;
      alertSec.appendChild(d);
    });
  }

  scarcitySec.innerHTML = "";
  if (scarcity.length > 0) {
    scarcitySec.innerHTML = `<div class="dm-section-title">Scarcity</div>`;
    scarcity.forEach((s) => {
      const d = document.createElement("div");
      d.className = "dm-scarcity";
      d.textContent = s;
      scarcitySec.appendChild(d);
    });
  }
}

function filterRows(query, pos = "all") {
  const filtering = query || pos !== "all";
  // Hide supplementary headers when any filter is active
  document.querySelectorAll(".dm-tier-divider").forEach((d) => {
    d.style.display = filtering ? "none" : "";
  });
  const bbp = document.getElementById("dm-best-by-pos");
  if (bbp) bbp.style.display = filtering ? "none" : "";

  // Detect team abbreviation search: 2–4 lowercase letters matching a known team
  const isTeamSearch = !!(query && /^[a-z]{2,4}$/.test(query) &&
    Object.values(state.playerIndex).some((p) => (p.team || "").toLowerCase() === query));

  document.querySelectorAll(".dm-player-row:not(.dm-extra-result)").forEach((row) => {
    const nameMatch = !query || (isTeamSearch ? row.dataset.team === query : row.dataset.name?.includes(query));
    const posMatch  = pos === "all" || row.dataset.pos === pos;
    row.style.display = (nameMatch && posMatch) ? "" : "none";
  });

  // Remove previous extra-search results
  document.querySelectorAll(".dm-extra-result").forEach((r) => r.remove());
  document.getElementById("dm-extra-title")?.remove();

  if (!query || !state.playerIndex) return;

  // Find rendered player names to avoid duplicates
  const rendered = new Set();
  document.querySelectorAll(".dm-player-row:not(.dm-extra-result)").forEach((r) => {
    if (r.dataset.name) rendered.add(r.dataset.name);
  });

  const sec = document.getElementById("dm-recs-section");
  if (!sec) return;

  // playerIndex holds alias keys (abbrev "b bowers", reversed "bowers brock")
  // that point to the SAME player object, so Object.values() yields a player
  // once per alias. Dedupe by normalized name — against both already-rendered
  // rows AND earlier extras — so a player can't appear 2–3× (the "3 Brock
  // Bowers" bug).
  const seen = new Set(rendered);
  const extras = Object.values(state.playerIndex).filter((p) => {
    const norm = normalize(p.name);
    if (seen.has(norm)) return false;
    const posOk = pos === "all" || p.position.toLowerCase() === pos;
    const ok = isTeamSearch
      ? posOk && (p.team || "").toLowerCase() === query
      : posOk && norm.includes(query);
    if (ok) seen.add(norm);
    return ok;
  }).sort((a, b) => a.overall_rank - b.overall_rank).slice(0, 20);

  if (extras.length === 0) return;

  const titleEl = document.createElement("div");
  titleEl.className = "dm-section-title";
  titleEl.id = "dm-extra-title";
  titleEl.textContent = isTeamSearch ? `All ${query.toUpperCase()} Players` : "More Matches";
  sec.appendChild(titleEl);

  extras.forEach((p) => {
    const key = draftKeyForName(p.name);
    const isQueued = state.queue.includes(key);
    const age = p.age ? `${p.age.toFixed(0)}yo` : "";
    const metaParts = [p.team || "", age].filter(Boolean).join(" · ");
    const adp = p.adp > 0 ? p.adp.toFixed(1) : "—";
    const row = document.createElement("div");
    row.className = "dm-player-row dm-extra-result";
    row.dataset.name = normalize(p.name);
    row.dataset.pos  = p.position.toLowerCase();
    row.dataset.team = (p.team || "").toLowerCase();
    row.innerHTML = `
      <span class="dm-rank">${p.overall_rank}</span>
      <span class="dm-pos-badge dm-pos-${p.position}">${p.position}</span>
      <div class="dm-player-info">
        <div class="dm-player-name">${p.name}</div>
        <div class="dm-player-meta">${metaParts}</div>
      </div>
      <div class="dm-player-stats">
        <div class="dm-proj-pts">${adp}</div>
        <div class="dm-board-label">ADP</div>
      </div>
      <button class="dm-queue-add" title="${isQueued ? "Already queued" : "Add to queue"}" ${isQueued ? "disabled" : ""}>${isQueued ? "✓" : "+"}</button>
    `;
    row.querySelector(".dm-queue-add")?.addEventListener("click", (e) => {
      e.stopPropagation();
      addToQueue(p.name);
    });
    sec.appendChild(row);
  });
}

function renderRosterSummary() {
  const sec = document.getElementById("dm-roster-summary");
  if (!sec) return;
  if (state.myTeam.length === 0) {
    if (sec.innerHTML) sec.innerHTML = "";
    return;
  }

  const counts    = { QB: 0, RB: 0, WR: 0, TE: 0 };
  const byeByPos  = { QB: [], TE: [] };   // bye weeks for each pos we care about

  state.myTeam.forEach((p) => {
    if (counts[p.position] !== undefined) counts[p.position]++;
    if (byeByPos[p.position]) {
      const data = state.playerIndex[normalize(p.name)];
      if (data?.bye_week) byeByPos[p.position].push(data.bye_week);
    }
  });

  const r = currentRound();
  const warnings = [];

  // Round-aware best-ball coaching, with specific bye weeks when known
  if (counts.QB === 0 && r > 6)  warnings.push("0 QBs — draft one soon");
  if (counts.QB === 1 && r > 9) {
    const bw = byeByPos.QB[0] ? ` (Wk${byeByPos.QB[0]})` : "";
    warnings.push(`1 QB${bw} — need a backup, avoid same bye`);
  }
  if (counts.TE === 0 && r > 5)  warnings.push("0 TEs — getting late");
  if (counts.TE === 1 && r > 10) {
    const bw = byeByPos.TE[0] ? ` (Wk${byeByPos.TE[0]})` : "";
    warnings.push(`1 TE${bw} — need a backup, avoid same bye`);
  }
  if (counts.WR < 4 && r > 7)   warnings.push(`${counts.WR} WRs — best ball needs 6+`);
  if (counts.RB < 2 && r > 5)   warnings.push(`${counts.RB} RBs — very thin`);

  const needed = {
    QB: counts.QB < 2 ? (r > 9 ? "URGENT: backup needed" : "need backup") : "",
    TE: counts.TE < 2 ? (r > 10 ? "URGENT: backup needed" : "need backup") : "",
    WR: counts.WR < 6 ? `need ${6 - counts.WR} more` : "",
    RB: counts.RB < 4 ? `need ${4 - counts.RB} more` : "",
  };
  // Build chip tooltip with bye weeks when relevant
  const chipTip = (pos, n) => {
    const base = needed[pos] || "on track";
    const uniqueByes = [...new Set(byeByPos[pos] || [])];
    const overlap = uniqueByes.length < (byeByPos[pos] || []).length;  // duplicate → overlap
    const byeLabel = uniqueByes.length
      ? ` · bye${uniqueByes.length > 1 ? "s" : ""} Wk${uniqueByes.join(",")}${overlap ? " ⚠ OVERLAP" : ""}`
      : "";
    return base + byeLabel;
  };
  const chips = Object.entries(counts).map(([pos, n]) => {
    const isUrgent = needed[pos]?.startsWith("URGENT");
    const warn = needed[pos] ? (isUrgent ? " ⚠⚠" : " ⚠") : "";
    return `<span class="dm-summary-chip dm-pos-${pos}" title="${chipTip(pos, n)}">${pos}: ${n}${warn}</span>`;
  }).join("");

  // Pick countdown line — suppress until we have at least one pick tracked or
  // the user has explicitly set their pick position (pickPos != 1 default)
  const gap = picksUntilMyTurn();
  const hasTracking = state.allDrafted.length > 0 || state.pageCurrentPick !== null || state.settings.pickPos !== 1;
  const pickLine = !hasTracking
    ? `<div class="dm-pick-countdown" style="color:#444;">Set pick # in ⚙ Settings</div>`
    : gap === 0
      ? `<div class="dm-pick-countdown dm-pick-now">⚡ YOUR PICK NOW</div>`
      : gap <= 3
        ? `<div class="dm-pick-countdown dm-pick-soon">${gap} picks away — get ready</div>`
        : `<div class="dm-pick-countdown">${gap} picks away · Rd ${r}</div>`;

  let html = `<div id="dm-roster-chips">${chips}</div>${pickLine}`;
  if (warnings.length > 0) {
    html += `<div class="dm-roster-warn">${warnings[0]}</div>`;
  }
  if (sec.innerHTML !== html) sec.innerHTML = html;
}

function setStatus(text) {
  const el = document.getElementById("dm-status");
  if (el) el.textContent = text;
}

// ── Auto-detect user's pick position ─────────────────────────────────────────
// When the site shows "YOU'RE ON THE CLOCK" and we can read the current pick
// number, we can infer the user's pick slot (1-12) in the snake draft.
// Only runs once per draft session (stops after first confident detection).

let _pickPosDetected = false;

function tryDetectMyPickPos() {
  if (_pickPosDetected) return;
  if (state.dkApiActive) return;  // DK feed aligns our slot authoritatively
  if (state.allDrafted.length === 0) return;  // need at least one pick to know round length

  const text = getPageText().toLowerCase();
  // Phrases used by different sites to indicate it's the current user's turn
  const myTurnPhrases = [
    "you're on the clock",
    "your on the clock",
    "your turn",
    "you are on the clock",
    "make your pick",
    "your pick",
    "it's your turn",
    "its your turn",
    "select a player",
    "you're picking",
    "time to pick",
    "make a selection",
  ];
  const isMyTurn = myTurnPhrases.some((phrase) => text.includes(phrase));
  if (!isMyTurn) return;

  // Use getNextBoardPick() which clamps unreliable pageCurrentPick values
  // (raw pageCurrentPick can contain jersey numbers or rankings from the page)
  const overallPick = getNextBoardPick();
  if (!overallPick || overallPick < 1) return;

  const inferredSlot = pickToTeamSlot(overallPick, state.settings.teams);
  if (inferredSlot === state.settings.pickPos) {
    _pickPosDetected = true;  // already correct, mark as done
    return;
  }

  // Only auto-correct if we haven't committed to the old position heavily
  // (less than 2 picks tracked for "my team" means we can safely re-assign)
  if (state.myTeam.length <= 1) {
    console.log(`[DraftManager] Auto-detected pick position: slot ${inferredSlot} (was ${state.settings.pickPos})`);
    state.settings.pickPos = inferredSlot;
    saveSettings();
    updateSettingsUI();
    setStatus(`Pick position auto-detected: slot ${inferredSlot}`);
    // Reset team tracking so everything re-assigns from scratch with correct slot
    state.allDrafted = [];
    state.myTeam = [];
    state.teamRosters = {};
    state._orderedNewestFirst = null;
    clearDraftState();
    _pickPosDetected = true;
    return true;  // signal: state was reset, re-scan needed
  }
  return false;
}

// ── Main scan / rank flow ─────────────────────────────────────────────────────

async function scanAndRank(force = false) {
  if (!state.connected) return;
  if (state.scanInFlight) { state.pendingScan = true; return; }
  state.scanInFlight = true;
  state.scanCount++;
  if (force) setStatus("scanning...");

  try {
    // 1. Detect pick history on the page and sync any new picks into state
    const historyEl = findPickHistory();
    if (historyEl) {
      // Skip the DOM history pipeline entirely when the DK JSON feed is authoritative
      // — orderPickHistory() can wipe/rebuild allDrafted and would fight the feed.
      if (!state.dkApiActive) {
        const rawPicks     = readPickHistory(historyEl);
        const orderedPicks = orderPickHistory(rawPicks);
        const added        = syncPickHistory(orderedPicks);
        if (added > 0) renderTeamRosters();
      }
    } else if (state.scanCount === 3) {
      // Log actionable debug info on the 3rd scan (page is probably fully loaded by then)
      const host = location.hostname.toLowerCase();
      console.log("[DraftManager] Pick history not detected after several scans.");
      if (host.includes("draftkings")) {
        console.log("  [DK] Pick detection is event-driven — picks are read from the 'last drafted' element");
        console.log("  [DK] when DraftKings fires a SelectionRecorded event.");
        console.log("  [DK] If picks aren't tracking: reload the draft page so ws_interceptor.js runs at startup.");
      } else if (host.includes("underdog") || host.includes("playunderdog")) {
        console.log("  [Underdog] Look for the picks feed at the top or side of the draft room.");
        console.log("  [Underdog] Try window.dm.highlight() to see what containers were found.");
      }
      console.log("  1. Run window.dm.highlight() → colored outlines show all containers");
      console.log("  2. Find the pick history panel (use DevTools to inspect highlighted elements)");
      console.log("  3. Run window.dm.setHistorySelector('your-css-selector')");
      console.log("  4. Or paste the selector in ⚙ Settings → 'Pick history selector'");
    }

    // DK fallback: only do DOM work after an actual pick event. Passive scans
    // during DK load are expensive and can create bad inferred picks.
    if (location.hostname.includes("draftkings") && !state.dkApiActive && state._pickPending) {
      const diffed = detectDKPickByDiff();
      if (diffed && recordDKPick(diffed.name, diffed.position)) {
        state._pickPending = false;
      } else {
        tryDKDomPick(); // last resort DOM selector fallback
      }
    }

    // 2. Update current pick number from page text
    const { currentPick } = scanPage();
    // Only trust detected pick numbers that are plausible relative to tracked history
    // — prevents jersey numbers / player rankings from being read as pick numbers.
    if (currentPick) {
      const fromHistory = getTrackedNextPick();
      const draftedCount = (state.canonicalNorms.size && state.available.size)
        ? Math.max(0, state.canonicalNorms.size - state.available.size)
        : state.allDrafted.filter((d) => !d._phNorm).length;
      if (draftedCount <= 0) {
        // Draft hasn't started (full board). DK may render our own slot as
        // "Pick 8" — only a detected pick of 1 is believable here; never store a
        // higher number that would later resurface as a false "your pick".
        if (currentPick === 1) state.pageCurrentPick = 1;
      } else if (fromHistory <= 1 || currentPick <= fromHistory + state.settings.teams) {
        state.pageCurrentPick = currentPick;
      }
    }

    // 3a. Try to auto-detect user's pick position from "on the clock" page text.
    // If it resets state, bail out — the queued rescan will have correct data.
    if (tryDetectMyPickPos()) return;

    // 3b. available = full DB − all drafted (no board scrolling ever)
    // (scanPage() already sets state.available)

    if (state.available.size === 0) {
      setStatus("no players found — click ↺ Rescan");
      renderRecs([]);
      return;
    }

    if (force) setStatus(`ranking ${state.available.size} avail...`);
    const data = await getRankings();
    if (!data) { setStatus("server error"); return; }

    state.lastRecs = data;
    renderAll(data);

    const gap       = picksUntilMyTurn();
    const pickLabel = gap === 0 ? " · YOUR PICK" : ` · ${gap} picks away`;
    const histNote = (historyEl || state.allDrafted.length > 0)
      ? ` · ${state.allDrafted.length}p tracked`
      : " · Alt+↺ to diagnose";
    setStatus(`${data.total_available} avail · rd ${currentRound()}${pickLabel}${histNote}`);
    await maybeAutodraft(data.recommendations || []);

    const panel = document.getElementById("dm-panel");
    if (panel) {
      if (gap === 0) {
        panel.classList.add("dm-your-turn");
        if (!document.title.startsWith("🏈 YOUR PICK")) {
          document._dmOrigTitle = document.title;
          document.title = "🏈 YOUR PICK — " + document.title;
        }
      } else {
        panel.classList.remove("dm-your-turn");
        if (document._dmOrigTitle) {
          document.title = document._dmOrigTitle;
          document._dmOrigTitle = null;
        }
      }
    }
  } finally {
    state.scanInFlight = false;
    if (state.pendingScan) {
      state.pendingScan = false;
      setTimeout(() => scanAndRank(), 0);
    }
  }
}

// ── DraftKings pick detection ─────────────────────────────────────────────────
// Primary:  diff scanRoot() snapshots — whoever disappears from the visible board
//           after a SelectionRecorded event was just drafted.
// Secondary: parse raw WS payload JSON for player name (no DOM needed).
// Tertiary:  DOM selectors for a "last pick" widget as last resort.
// All paths feed into recordDKPick() which handles dedup + snake math.

function parseWsPickedPlayer(raw) {
  if (!raw) return null;
  try {
    const jsonStr = raw.match(/\{[\s\S]*\}/)?.[0];
    if (!jsonStr) return null;
    const data = JSON.parse(jsonStr);

    // Recursively search any nesting depth for name + position fields
    function findIn(obj, depth) {
      if (!obj || typeof obj !== "object" || depth > 6) return null;
      const name = obj.playerName || obj.player_name || obj.displayName || obj.display_name ||
                   (obj.firstName && obj.lastName ? `${obj.firstName} ${obj.lastName}` : null) ||
                   (obj.first_name && obj.last_name ? `${obj.first_name} ${obj.last_name}` : null) ||
                   obj.name;
      const pos  = obj.position || obj.positionAbbreviation || obj.position_abbreviation ||
                   obj.posAbbr  || obj.pos;
      // Require a real two-word name so we don't match noise like "pick" or "data"
      if (name && typeof name === "string" && name.trim().includes(" ") && name.trim().length > 5) {
        return { name: name.trim(), position: (pos || "").toUpperCase() };
      }
      for (const val of Object.values(obj)) {
        if (val && typeof val === "object") {
          const found = findIn(val, depth + 1);
          if (found) return found;
        }
      }
      return null;
    }
    return findIn(data, 0);
  } catch { return null; }
}

function readDKLastPickFromDOM() {
  // Try every plausible selector for a "last drafted player" display on DK.
  const selectors = [
    '[class*="last-drafted-player"]', '[class*="LastDraftedPlayer"]',
    '[class*="lastDraftedPlayer"]',   '[class*="last-pick"]',
    '[class*="LastPick"]',            '[class*="recentPick"]',
    '[class*="recent-pick"]',         '[class*="currentPick"]',
    '[data-testid*="last-pick"]',     '[data-testid*="recent-pick"]',
    '[aria-label*="last pick"]',
  ];
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (!el) continue;
    const text = (el.textContent || "").replace(/\s+/g, " ").trim();
    // Expect something like "Josh Allen | QB BUF" or "Josh Allen QB"
    const posMatch = text.match(/\b(QB|WR|RB|TE)\b/);
    if (!posMatch) continue;
    const pickNum = parseOverallPickFromText(text);
    const beforePos = text.slice(0, text.search(/\b(QB|WR|RB|TE)\b/))
      .replace(/round\s+\d{1,2}\s*[|.,:\-\s]+pick\s+#?\s*\d{1,2}/ig, "")
      .replace(/\brd\s*\d{1,2}\s*[|.,:\-\s]+pk\s*#?\s*\d{1,2}/ig, "")
      .replace(/(?:overall\s+)?pick\s*#?\s*\d{1,3}\b/ig, "")
      .replace(/#\s*\d{1,3}\b/g, "")
      .replace(/[|:,]/g, "")
      .replace(/last\s*pick/i, "")
      .trim();
    // Name should be 2+ words and not obviously noise
    if (beforePos.split(" ").filter(Boolean).length >= 2) {
      return { name: beforePos, position: posMatch[1], pickNum };
    }
  }
  return null;
}

// DraftKings has NO dedicated "last pick" element — it renders the current pick
// and most-recent pick as PLAIN TEXT inside .LiveDraft_live-draft:
//   "On the clock: Pick 78"           → current overall pick (last done = N−1)
//   "Last Pick: Rhamondre Stevenson | RB NE"  → NAME | POS TEAM at pick N−1
// Reading both together lets us record the last pick with an EXACT overall
// number, which is far more reliable than parsing the WS payload. This is the
// primary DK DOM fallback when the JSON feed isn't resolving.
function readDKLiveDom() {
  const root = document.querySelector('[class*="LiveDraft_live-draft"]')
            || document.querySelector('[class*="SnakeDraft"]')
            || document.body;
  const text = ((root.innerText || root.textContent || "")).replace(/ /g, " ");
  const out = { onClockPick: null, lastPick: null };

  const maxPick = (state.settings.teams || 12) * (state.settings.rounds || 20);
  const clk = text.match(/on\s+the\s+clock:?\s*pick\s*#?\s*(\d{1,3})/i);
  if (clk) {
    const n = parseInt(clk[1], 10);
    if (n >= 1 && n <= maxPick) out.onClockPick = n;
  }

  // "Last Pick: First Last | POS TEAM" (TEAM optional). Non-greedy name capture
  // stops at the pipe; require a real 2+ word name so we don't grab noise.
  const lp = text.match(/last\s*pick:?\s*([A-Za-z][A-Za-z0-9 .'\-]+?)\s*\|\s*(QB|RB|WR|TE)\b\s*([A-Z]{2,3})?/i);
  if (lp) {
    const name = lp[1].replace(/\s+/g, " ").trim();
    if (name.split(" ").filter(Boolean).length >= 2) {
      out.lastPick = { name, position: lp[2].toUpperCase(), team: (lp[3] || "").toUpperCase() };
    }
  }
  return out;
}

// Record DK's "Last Pick" with its exact overall number (onClock − 1). recordDKPick
// dedupes by name, so calling this repeatedly (poll) is safe and idempotent.
function tryDKLiveDomPick() {
  const dom = readDKLiveDom();
  if (!dom.lastPick) return false;
  const overall = Number.isFinite(dom.onClockPick) && dom.onClockPick > 1
    ? dom.onClockPick - 1
    : null;
  return recordDKPick(dom.lastPick.name, dom.lastPick.position, overall);
}

function parseOverallPickFromText(text) {
  const teams = state.settings.teams || 12;
  const rounds = state.settings.rounds || 20;
  const maxPick = teams * rounds;
  const clean = String(text || "");

  let m = clean.match(/round\s+(\d{1,2})\s*[|.,:\-\s]+pick\s+#?\s*(\d{1,2})/i) ||
          clean.match(/\brd\s*(\d{1,2})\s*[|.,:\-\s]+pk\s*#?\s*(\d{1,2})/i);
  if (m) {
    const r = parseInt(m[1], 10);
    const p = parseInt(m[2], 10);
    if (r >= 1 && r <= rounds && p >= 1 && p <= teams) return (r - 1) * teams + p;
  }

  m = clean.match(/(?:overall\s+)?pick\s*#?\s*(\d{1,3})\b/i) ||
      clean.match(/#\s*(\d{1,3})\b/);
  if (m) {
    const n = parseInt(m[1], 10);
    if (n >= 1 && n <= maxPick) return n;
  }
  return null;
}

// Diff the current DOM snapshot against the previous one.
// Called after SelectionRecorded fires — whoever was visible before but isn't
// now (and is still in our available set) is the pick.
function detectDKPickByDiff(baseSnapshot = state._domSnapshot) {
  const current = scanDKDraftableBoard();

  const disappeared = [];
  for (const norm of baseSnapshot || []) {
    if (!current.has(norm) && state.available.has(norm)) {
      disappeared.push(norm);
    }
  }

  // Always update snapshot so next diff is against current reality
  state._domSnapshot = current;

  if (disappeared.length === 1) {
    const player = state.playerIndex[disappeared[0]];
    if (player) {
      if (isDKPlayerStillDraftable(player)) {
        state._lastDkPickDebug = {
          source: "rejected-board-diff",
          player: player.name,
          reason: "player still visible as draftable",
          at: Date.now(),
        };
        console.warn(`[DM] Rejected DK board diff for ${player.name}: still visible as draftable.`);
        return null;
      }
      return { name: player.name, position: player.position };
    }
  }

  if (disappeared.length > 1) {
    const names = disappeared
      .map((n) => state.playerIndex[n]?.name || n)
      .slice(0, 12);
    state._lastDkPickDebug = {
      source: "ambiguous-board-diff",
      disappeared: disappeared.length,
      names,
      at: Date.now(),
    };
    console.warn(`[DM] Ambiguous DK board diff (${disappeared.length} players disappeared). Skipping guess:`, names);
  }

  return null; // nothing conclusive
}

// Core function — records a pick regardless of how it was detected.
function recordDKPick(name, position, overallPickOverride = null) {
  if (state.dkApiActive && state.allDrafted.length > 0) return false;  // non-empty DK JSON feed is authoritative
  const norm = draftKeyForName(name);
  if (!norm || norm.length < 4) return false;

  // Dedup — ignore if we already have this player
  const knownNorms = draftedKeys();
  if (knownNorms.has(norm)) return false;

  // Resolve against our player DB for canonical name + position
  const player    = matchPlayer(name);
  const canonName = player?.name || name;
  const canonPos  = player?.position || position || "WR";

  const explicitPick = Number(overallPickOverride);
  const lastIdx = state.allDrafted.length - 1;
  const lastEntry = lastIdx >= 0 ? state.allDrafted[lastIdx] : null;
  const canReplaceFreshPlaceholder =
    !Number.isFinite(explicitPick) &&
    lastEntry?._phNorm &&
    Date.now() - Number(lastEntry._createdAt || 0) < 15000;

  if (Number.isFinite(explicitPick) && explicitPick >= getTrackedNextPick()) {
    addMissingPickPlaceholders(explicitPick, `pre-fill before detected pick ${explicitPick}`);
  }
  const overallPick = canReplaceFreshPlaceholder
    ? Number(lastEntry.overall_pick)
    : (Number.isFinite(explicitPick) && explicitPick >= getTrackedNextPick()
      ? explicitPick
      : pickNumberForNewDetectedPick());
  const slot    = pickToTeamSlot(overallPick, state.settings.teams);
  const byUser  = slot === state.settings.pickPos;
  const round   = Math.floor((overallPick - 1) / state.settings.teams) + 1;

  if (canReplaceFreshPlaceholder) {
    state.allDrafted[lastIdx] = { name: canonName, position: canonPos, round, by_user: byUser, team_slot: slot, overall_pick: overallPick, _draftKey: norm };
  } else {
    state.allDrafted.push({ name: canonName, position: canonPos, round, by_user: byUser, team_slot: slot, overall_pick: overallPick, _draftKey: norm });
  }

  // Re-apply manual overrides and rebuild rosters/available off the new pick.
  reconcileDraftState();
  saveDraftState();

  console.log(`[DM] Pick ${overallPick}: ${canonName} (${canonPos}) → T${slot}${byUser ? " ← YOU" : ""}`);
  renderTeamRosters();
  return true;
}

// Called every 500ms poll and after WS events as DOM fallback
function tryDKDomPick() {
  // Prefer the LiveDraft "Last Pick + On the clock" text (exact pick number).
  if (tryDKLiveDomPick()) return true;
  // Fall back to any generic last-pick widget (other sites / DK layout changes).
  const pick = readDKLastPickFromDOM();
  if (!pick) return false;
  return recordDKPick(pick.name, pick.position, pick.pickNum);
}

// ── DraftKings authoritative JSON pick feed ─────────────────────────────────
// DK serves the entire draft as JSON (intercepted in ws_interceptor.js):
//   draftables  → draftableId -> player (name/pos/team/bye)
//   draftStatus → ordered list of every pick { userKey, draftableId, overall }
// Joining them maps every pick to player + team + order with ZERO DOM scraping
// and ZERO snake-math inference of *who* picked (the userKey says it directly).
// When this feed is present it is authoritative and the DOM/WS recorders below
// are suppressed via state.dkApiActive.

function ingestDraftables(list) {
  if (!Array.isArray(list) || list.length === 0) return;
  for (const d of list) {
    if (!d || d.draftableId == null) continue;
    const rec = {
      name: d.name || "",
      position: d.position || "",
      team: d.team || "",
      bye: Number(d.bye) || 0,
      playerId: d.playerId,
    };
    state.dkDraftables[d.draftableId] = rec;
    if (d.playerId != null && !(d.playerId in state.dkByPlayerId)) {
      state.dkByPlayerId[d.playerId] = rec;
    }
  }
  // A board may have arrived before the dictionary did — resolve it now.
  if (state._pendingDraftBoard) syncFromDKApi(state._pendingDraftBoard);
  else if (state.dkLastStatusBoard) syncFromDKApi(state.dkLastStatusBoard);
}

function detectSelfUserKey(board) {
  // Once locked, keep it: the roster dropdown can later be switched to VIEW other
  // entrants' rosters, so re-reading it every sync would flip "my team".
  if (state.dkUserKey && board.some((p) => p.userKey === state.dkUserKey)) return state.dkUserKey;
  // First time: the dropdown defaults to OUR team; its container id is our userKey.
  const dd = document.querySelector("[class*='RosterTable_roster-dropdown'][id$='_container']");
  if (dd && dd.id) {
    const key = dd.id.replace(/_container$/, "");
    if (key && board.some((p) => p.userKey === key)) return key;
  }
  // Fall back to whichever entrant sits in our configured snake slot.
  const { teams, pickPos } = state.settings;
  const mine = board.find((p) => pickToTeamSlot(p.overall, teams) === pickPos);
  return mine ? mine.userKey : null;
}

function syncFromDKApi(board) {
  if (!Array.isArray(board)) return;
  if (board.length === 0) {
    state._pendingDraftBoard = null;
    state.dkApiActive = false;
    state.pageCurrentPick = 1;
    state._pickPending = false;
    state._domSnapshot = new Set();
    state.allDrafted = [];
    state.teamRosters = {};
    state.myTeam = [];
    state._lastDkApiBoardSig = "0|0|0";
    reconcileDraftState();  // empty auto board, but manual overrides still apply
    pruneQueue();
    saveDraftState();
    state._lastRosterSyncDebug = {
      source: "dk-api-empty-board", picks: 0, self: state.dkUserKey, at: Date.now(),
    };
    renderTeamRosters();
    scanAndRank();
    return;
  }
  state.dkLastStatusBoard = board;
  state._pendingDraftBoard = board;  // always remember the most recent feed

  // Need the player DB (for matchPlayer + available set) and the draftables
  // dictionary first. If either is missing, hold the board and ask for buffered
  // draftables; init re-requests after the DB finishes loading.
  if (!state.canonicalNorms || state.canonicalNorms.size === 0) return;
  if (Object.keys(state.dkDraftables).length === 0) {
    window.dispatchEvent(new CustomEvent("dm:request_draft_data"));
  }

  const board2 = state._pendingDraftBoard;
  state._pendingDraftBoard = null;
  const teams = state.settings.teams;

  const lastOverall = board2.reduce((m, p) => Math.max(m, Number(p.overall) || 0), 0);
  const boardSig = `${board2.length}|${lastOverall}|${Object.keys(state.dkDraftables).length}`;
  if (boardSig === state._lastDkApiBoardSig) return;
  state._lastDkApiBoardSig = boardSig;

  const self = detectSelfUserKey(board2);
  if (self) {
    state.dkUserKey = self;
    // Authoritatively align our pick slot to the feed so by_user is correct.
    const mine = board2.find((p) => p.userKey === self);
    if (mine) {
      const slot = pickToTeamSlot(mine.overall, teams);
      if (slot !== state.settings.pickPos) {
        state.settings.pickPos = slot;
        _pickPosDetected = true;
        saveSettings();
        updateSettingsUI();
      }
    }
  }

  const sorted = [...board2].sort((a, b) => (a.overall || 0) - (b.overall || 0));
  const drafted = [];
  let unresolved = 0;
  let resolved = 0;
  for (const p of sorted) {
    const dk = state.dkDraftables[p.draftableId] || state.dkByPlayerId[p.playerId] || {
      name: p.name || "",
      position: p.position || "",
      team: "",
    };
    const overall = Number(p.overall) || (drafted.length + 1);
    const slot = pickToTeamSlot(overall, teams);
    const round = p.round || Math.floor((overall - 1) / teams) + 1;
    if (!dk || !dk.name) {
      unresolved++;
      const existing = state.allDrafted.find((d) => Number(d.overall_pick) === overall && !d._phNorm);
      if (existing) drafted.push(existing);
      continue;
    }
    resolved++;
    const matched = matchPlayer(dk.name);
    const name = matched?.name || dk.name;
    const position = dk.position || matched?.position || "";
    drafted.push({
      name,
      position,
      team: dk.team,
      round,
      overall_pick: overall,
      team_slot: slot,
      by_user: self ? p.userKey === self : slot === state.settings.pickPos,
      _draftKey: draftKeyForName(name),
      _userKey: p.userKey,
    });
  }
  if (resolved === 0) {
    state.dkApiActive = false;
    state.allDrafted = state.allDrafted.filter((d) => !d._phNorm);
    reconcileDraftState();
    state.pageCurrentPick = Math.max(state.pageCurrentPick || 1, lastOverall + 1);
    state._lastRosterSyncDebug = {
      source: "dk-api-unresolved-ignored",
      picks: board2.length,
      resolved,
      unresolved,
      draftables: Object.keys(state.dkDraftables).length,
      self: state.dkUserKey,
      at: Date.now(),
    };
    triggerDraftStatusRefetch();
    saveDraftState();
    renderTeamRosters();
    return;
  }

  state.allDrafted = drafted;
  state.dkApiActive = resolved > 0;
  reconcileDraftState();  // bake in manual adds/removals on top of the feed
  state._lastRosterSyncDebug = {
    source: "dk-api", picks: drafted.length, resolved, unresolved, self: state.dkUserKey, at: Date.now(),
  };
  saveDraftState();
  scanAndRank();
}

// Ask the MAIN-world interceptor to re-pull DK's draftStatus JSON. The new pick
// may not be in draftStatus the instant the WS event fires, so poke a few times;
// the interceptor throttles the actual network hits (REFETCH_MIN_MS).
function triggerDraftStatusRefetch() {
  [0, 500, 1500].forEach((d) =>
    setTimeout(() => window.dispatchEvent(new CustomEvent("dm:refetch_draft_status")), d)
  );
}

// Safety net: even if WS pick-event detection breaks (DK changes the payload),
// poll draftStatus on a slow interval so the authoritative feed never drifts more
// than a few seconds out of date. The MAIN-world refetch no-ops until it has
// captured a draftStatus URL, so this also helps bootstrap the feed before
// state.dkApiActive flips on.
function startFeedRefreshPoll() {
  if (state.feedPollTimer) clearInterval(state.feedPollTimer);
  state.feedPollTimer = setInterval(() => {
    if (!state.connected || !location.hostname.toLowerCase().includes("draftkings")) return;
    // 1. Keep the authoritative JSON feed fresh.
    window.dispatchEvent(new CustomEvent("dm:refetch_draft_status"));
    // 2. Safety net: if the JSON feed isn't driving picks, read DK's plain-text
    //    "Last Pick" + "On the clock" so picks still track from the DOM alone.
    if (!state.dkApiActive) {
      if (tryDKLiveDomPick()) scanAndRank();
    }
  }, 3000);
}

// ── WebSocket / console event listeners (from ws_interceptor.js) ─────────────
function setupPickEventListeners() {
  window.addEventListener("dm:draftables", (e) => {
    ingestDraftables(e.detail?.draftables || []);
  });
  window.addEventListener("dm:draft_status", (e) => {
    if (e.detail?.url) state.dkStatusUrl = e.detail.url;
    syncFromDKApi(e.detail?.board || []);
  });

  window.addEventListener("dm:pick_recorded", (e) => {
    if (!state.connected) return;
    // DK's JSON feed is authoritative — but it only refreshes when draftStatus is
    // re-fetched, and DK pushes picks over the WS without always re-fetching. Poke
    // a re-fetch so the new pick lands in the feed, then refresh recs.
    triggerDraftStatusRefetch();
    if (state.dkApiActive) {
      const activeFeedNextPick = getTrackedNextPick();
      setTimeout(() => {
        if (getTrackedNextPick() > activeFeedNextPick) return;
        state.dkApiActive = false;
        state._pickPending = true;
        state._lastDkPickDebug = {
          source: "stale-dk-api-fallback",
          started_next_pick: activeFeedNextPick,
          status_url: state.dkStatusUrl || "?",
          at: Date.now(),
        };
        if (!tryDKDomPick()) recordUnknownDKPick("stale DK API after pick event");
        state._pickPending = false;
        scanAndRank();
      }, 2400);
      scanAndRank();
      return;
    }
    // Snapshot the board RIGHT NOW before React re-renders. DK can update at
    // different times, so this event is resolved by several delayed attempts.
    const eventSeq = ++state._dkPickEventSeq;
    const eventStartNextPick = getTrackedNextPick();
    const baseSnapshot = scanDKDraftableBoard();
    const raw = e.detail?.raw || "";
    state._domSnapshot = baseSnapshot;
    state._pickPending = true;
    state._lastDkPickDebug = {
      source: e.detail?.source || "unknown",
      event_seq: eventSeq,
      started_next_pick: eventStartNextPick,
      raw_prefix: raw ? raw.slice(0, 300) : "",
      at: Date.now(),
    };

    let resolved = false;
    const tryResolve = (attemptLabel) => {
      if (resolved) return true;
      if (getTrackedNextPick() > eventStartNextPick) {
        resolved = true;
        state._pickPending = false;
        return true;
      }

      // DK's "Last Pick + On the clock" text is the cleanest signal (exact pick
      // number, canonical name) — try it before the WS-payload guesswork.
      if (tryDKLiveDomPick()) {
        state._lastDkPickDebug = { source: "live-dom", attempt: attemptLabel, at: Date.now() };
        resolved = true;
        state._pickPending = false;
        scanAndRank();
        return true;
      }

      const ws = parseWsPickedPlayer(raw);
      if (ws && recordDKPick(ws.name, ws.position)) {
        if (raw) console.log("[DM] WS payload (SelectionRecorded) — paste this to dev to improve parser:", raw.slice(0, 1000));
        state._lastDkPickDebug = { source: "ws", attempt: attemptLabel, player: ws.name, at: Date.now() };
        resolved = true;
        state._pickPending = false;
        scanAndRank();
        return true;
      }

      const diffed = detectDKPickByDiff(baseSnapshot);
      if (diffed && recordDKPick(diffed.name, diffed.position)) {
        state._lastDkPickDebug = { source: "diff", attempt: attemptLabel, player: diffed.name, at: Date.now() };
        resolved = true;
        state._pickPending = false;
        scanAndRank();
        return true;
      }

      if (tryDKDomPick()) {
        state._lastDkPickDebug = { source: "last-pick-dom", attempt: attemptLabel, at: Date.now() };
        resolved = true;
        state._pickPending = false;
        scanAndRank();
        return true;
      }

      return false;
    };

    [150, 650, 1400].forEach((delay, idx, delays) => {
      setTimeout(() => {
        if (tryResolve(`${idx + 1}/${delays.length}@${delay}ms`)) return;
        if (idx !== delays.length - 1) return;

        if (getTrackedNextPick() <= eventStartNextPick) {
          recordUnknownDKPick("unresolved DK SelectionRecorded");
        }
        resolved = true;
        state._pickPending = false;
        scanAndRank();
      }, delay);
    });
  });

  window.addEventListener("dm:on_clock", () => {
    if (!state.connected) return;
    // "On the clock" means the prior pick just completed — pull it into the feed.
    triggerDraftStatusRefetch();
    scanAndRank();
  });

  window.addEventListener("dm:ws_message", (e) => {
    console.log("[DM WS]", e.detail);
  });

  // Relay commands from window.dm bridge (ws_interceptor.js MAIN world → content.js isolated world)
  window.addEventListener("dm:cmd", (e) => {
    const { cmd, arg } = e.detail || {};
    const api = window._dmInternal;
    if (api && typeof api[cmd] === "function") api[cmd](arg);
    else console.warn("[DM] Unknown command:", cmd);
  });
}

// ── MutationObserver (throttled) ──────────────────────────────────────────────

function setupMutationObserver() {
  // DraftKings is driven by intercepted draftStatus JSON and websocket events.
  // Its React app mutates constantly during load, so a whole-page observer causes
  // avoidable rescans and can make the draft room feel frozen.
  if (location.hostname.toLowerCase().includes("draftkings")) return;

  let debounceTimer = null;
  const observer = new MutationObserver(() => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      if (state.connected) scanAndRank();
    }, 800);  // 800ms: fast enough to catch picks, slow enough to batch burst mutations
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

// ── Message from toolbar click ────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "toggle_panel") {
    const panel = document.getElementById("dm-panel");
    if (!panel || panel.style.display === "none") {
      ensurePanelVisible();
      return;
    }
    panel.style.display = "none";
  }
});

// ── Adaptive scan timer ───────────────────────────────────────────────────────

function startAdaptiveScan() {
  if (state.scanTimer) clearTimeout(state.scanTimer);
  const isDK = location.hostname.includes("draftkings");
  const gap  = picksUntilMyTurn();
  const delay = isDK                ? SCAN_MS_DK
              : gap === 0 || gap <= 2 ? SCAN_MS_FAST
              : gap <= 8              ? SCAN_MS_NORMAL
              :                        SCAN_MS_SLOW;
  state.scanTimer = setTimeout(async () => {
    if (state.connected) await scanAndRank();
    startAdaptiveScan();  // reschedule after each scan
  }, delay);
}

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  createPanel();
  await loadSettings();
  // Override global defaults with THIS draft's saved slot + armed state, so
  // concurrent drafts in other tabs can't change our pick position.
  await loadPerDraftCfg();
  updateSettingsUI();  // reflect this draft's pick slot in the panel inputs

  setupPickEventListeners();
  startFeedRefreshPoll();  // keep DK's authoritative feed live (no-op until feed active)
  const ok = await fetchPlayerIndex();
  if (ok) {
    await loadDraftState();
    window.dispatchEvent(new CustomEvent("dm:request_draft_data"));
    await new Promise((resolve) => setTimeout(resolve, 25));
    await scanAndRank(true);
    setupMutationObserver();
    startAdaptiveScan();
    // Pull again in case DraftKings fetched JSON during the first scan.
    window.dispatchEvent(new CustomEvent("dm:request_draft_data"));
  } else {
    const sec = document.getElementById("dm-recs-section");
    if (sec) {
      sec.innerHTML = `
        <div class="dm-empty">
          <strong>Server not running.</strong><br><br>
          In your terminal:<br>
          <code>python server/server.py</code><br><br>
          Retrying automatically...
        </div>`;
    }
    // Retry connection every 5 seconds
    const retryInterval = setInterval(async () => {
      const ok2 = await fetchPlayerIndex();
      if (ok2) {
        clearInterval(retryInterval);
        await loadDraftState();
        window.dispatchEvent(new CustomEvent("dm:request_draft_data"));
        await new Promise((resolve) => setTimeout(resolve, 25));
        await scanAndRank(true);
        setupMutationObserver();
        startAdaptiveScan();
        window.dispatchEvent(new CustomEvent("dm:request_draft_data"));
      }
    }, 5000);
  }
}

// ── Developer debug API ───────────────────────────────────────────────────────
// Available in the browser console as window.dm
// Use this when auto-detection isn't working on a specific site.

window._dmInternal = {
  status() {
    const realPicks = state.allDrafted.filter(d => !d._phNorm);
    const phantoms  = state.allDrafted.filter(d => !!d._phNorm);
    console.log(`[DM] Picks tracked: ${realPicks.length} real + ${phantoms.length} unknown placeholders`);
    console.log(`[DM] DK JSON feed: ${state.dkApiActive ? "ACTIVE (authoritative)" : "inactive"} | `
      + `draftables=${Object.keys(state.dkDraftables).length} | self userKey=${state.dkUserKey || "?"} | `
      + `statusUrl=${state.dkStatusUrl ? "yes" : "no"}`);
    console.log(`[DM] Available: ${state.available.size} | My team: ${state.myTeam.length} picks`);
    if (hasManualRosterOverride()) {
      console.log("[DM] Manual overrides ACTIVE:",
        `${(state.manualAdds || []).length} added`,
        state.manualAdds.map(a => `${a.name}→T${a.slot}`),
        `| ${state.manualRemovals instanceof Set ? state.manualRemovals.size : 0} removed`,
        [...(state.manualRemovals instanceof Set ? state.manualRemovals : [])]);
    }
    console.log(`[DM] Next pick: tracked=${getTrackedNextPick()} board=${getNextBoardPick()} page=${state.pageCurrentPick || "none"}`);
    if (location.hostname.toLowerCase().includes("draftkings")) {
      const dom = readDKLiveDom();
      console.log(`[DM] DK DOM reader: onClock=Pick ${dom.onClockPick ?? "?"} | lastPick=`,
        dom.lastPick ? `${dom.lastPick.name} (${dom.lastPick.position} ${dom.lastPick.team})` : "(not found)");
    }
    if (state._lastDkPickDebug) console.log("[DM] Last DK pick debug:", state._lastDkPickDebug);
    if (state._lastRosterSyncDebug) console.log("[DM] Last roster sync:", state._lastRosterSyncDebug);
    // Show last 5 picks for quick sanity check
    const last5 = state.allDrafted.slice(-5);
    if (last5.length > 0) {
      console.log("[DM] Last 5 picks (most recent last):");
      last5.forEach(d => {
        const me = d.by_user ? "✓ YOU" : `T${d.team_slot}`;
        const pickNo = d.overall_pick ? `#${d.overall_pick}` : "#?";
        console.log(`  Pick ${pickNo}${d._phNorm ? " unknown" : ""} · ${me} · ${d.name} (${d.position}) Rd${d.round}`);
      });
    }
    console.log(`[DM] My team (${state.myTeam.length}):`, state.myTeam.map(p => `${p.name} (${p.position}) Rd${p.round}`));
    const rosters = Object.entries(state.teamRosters).map(([slot, picks]) =>
      `Team ${slot} (${picks.length}): ${picks.map(p => `${p.name}(${p.position})`).join(", ")}`);
    if (rosters.length > 0) rosters.forEach(r => console.log("[DM] " + r));
    else console.log("[DM] No team rosters tracked yet.");
    const cacheInfo = _cachedHistoryEl
      ? `✓ ${_cachedHistoryEl.tagName.toLowerCase()}${_cachedHistoryEl.id ? "#" + _cachedHistoryEl.id : ""}` +
        ` · ${_cachedHistoryEl.children.length} children`
      : "none (will re-detect next scan)";
    console.log(`[DM] History cache: ${cacheInfo}`);
    console.log(`[DM] Pick pos auto-detected: ${_pickPosDetected} | pickPos: ${state.settings.pickPos}`);
    console.log(`[DM] Scan count: ${state.scanCount} | scanInFlight: ${state.scanInFlight}`);
  },

  readPicks() {
    const el = findPickHistory();
    if (!el) {
      console.log("[DM] readPicks: no pick history element detected. Run window.dm.highlight().");
      return { found: false, raw: [], ordered: [] };
    }
    const raw = readPickHistory(el);
    const ordered = orderPickHistory(raw);
    const tag = el.tagName.toLowerCase();
    const id = el.id ? `#${el.id}` : "";
    const cls = typeof el.className === "string" && el.className
      ? `.${el.className.split(/\s+/).slice(0, 3).join(".")}`
      : "";
    console.log(`[DM] readPicks: ${raw.length} raw / ${ordered.length} ordered from ${tag}${id}${cls}`, el);
    ordered.slice(-12).forEach((p, i) => {
      const n = p.pickNum || "?";
      console.log(`  [${Math.max(0, ordered.length - 12) + i}] pick#${n} ${p.name} (${p.position})${p.unknown ? " [UNKNOWN]" : ""}`);
    });
    return { found: true, raw, ordered };
  },

  pageAdp() {
    const adp = scrapeLivePageAdp();
    const rows = Object.entries(adp)
      .sort((a, b) => a[1] - b[1])
      .map(([key, val]) => ({ player: state.playerIndex[key]?.name || key, adp: val }));
    console.table(rows.slice(0, 80));
    console.log(`[DM] page ADP rows scraped: ${rows.length}`);
    return rows;
  },

  smoke() {
    const recs = state.lastRecs?.recommendations || [];
    const out = {
      draftId: draftId(),
      platform: currentSitePlatform(),
      settings: { ...state.settings },
      connected: state.connected,
      nextPick: getNextBoardPick(),
      picksUntilMyTurn: picksUntilMyTurn(),
      trackedPicks: state.allDrafted.length,
      myTeam: state.myTeam.map((p) => `${p.name} (${p.position || "?"})`),
      available: state.available.size,
      adpSource: state.lastRecs?.adp_source || "",
      pageAdpCount: state.lastRecs?.page_adp_count || 0,
      policyModel: state.lastRecs?.dk_policy_model || "",
      topRecs: recs.slice(0, 8).map((r) => ({
        name: r.name,
        pos: r.position,
        adp: r.adp,
        ev: r.dk_ev_score,
        src: r.ranking_source,
      })),
    };
    console.log("[DM smoke]", out);
    console.table(out.topRecs);
    return out;
  },

  // Show what DK's plain-text DOM reader sees right now, and force-record the
  // last pick from it. Use this to diagnose the DOM fallback on DraftKings.
  livedom(record = false) {
    const dom = readDKLiveDom();
    console.log("[DM] DK live DOM:", dom);
    if (record) {
      const ok = tryDKLiveDomPick();
      console.log(`[DM] Force-record from DOM: ${ok ? "recorded" : "nothing new"}`);
      if (ok) scanAndRank(true);
    }
    return dom;
  },

  // Dry-run the autodraft DOM finders WITHOUT drafting. Pass a player name (defaults to
  // the current autodraft target). Reports whether the player row and an ENABLED Draft
  // button are locatable, and outlines the button so you can eyeball it. Costs nothing.
  async testDraftBtn(name) {
    if (!name) {
      const t = getAutodraftTarget(state.lastRecs?.recommendations || []);
      name = t?.name;
    }
    if (!name) { console.warn("[DM] testDraftBtn: no name and no autodraft target"); return null; }
    try {
    const playerEl = await revealDraftKingsPlayer(name);
    console.log(`[DM] testDraftBtn("${name}") → player row found: ${!!playerEl}`, playerEl || "");
    if (!playerEl) { console.warn("[DM] Player not found on board (try scrolling / check the name)"); return { name, playerFound: false }; }
    const row = findDraftKingsPlayerActionRow(name, playerEl);
    const rawButtons = row ? [...row.querySelectorAll("button")] : [];
    console.log("[DM] action row:", row || "");
    console.log("[DM] row buttons:", rawButtons.map((b) => ({
      text: (b.textContent || "").trim(),
      aria: b.getAttribute("aria-label") || "",
      disabled: !!b.disabled,
      ariaDisabled: b.getAttribute("aria-disabled") || "",
      cls: b.className || "",
    })));
    const btn = findDraftKingsRowDraftButton(name, row || playerEl);
    console.log(`[DM] → enabled Draft button found: ${!!btn}`, btn || "");
    if (btn) {
      btn.style.outline = "4px solid magenta";
      btn.style.outlineOffset = "2px";
      setTimeout(() => { btn.style.outline = ""; btn.style.outlineOffset = ""; }, 4000);
      console.log("[DM] ✅ Would click this button (outlined magenta for 4s). It's only enabled on your pick.");
    } else {
      const star = findDraftKingsQueueStar(name, row || playerEl);
      console.log(`[DM] → no enabled Draft button. Queue-star fallback found: ${!!star}`, star || "");
      console.log("[DM] If you are NOT on the clock this is expected (Draft button is disabled until your pick).");
    }
    return { name, playerFound: !!playerEl, draftButtonFound: !!btn };
    } finally {
      restoreDraftKingsSearch();
    }
  },

  // Show what containers have player names — use this to find the pick history element
  highlight() {
    const panel = document.getElementById("dm-panel");
    const containers = [];
    for (const el of document.querySelectorAll("section,div,ul,ol,aside,article")) {
      if (!isVisibleElement(el) || panel?.contains(el)) continue;
      if (el.children.length < 1 || el.children.length > 400) continue;
      const players = scanRoot(el);
      if (players.size < 1) continue;
      // Skip if dominated by a parent we already found
      if (containers.some(c => c.el.contains(el))) continue;
      // Remove dominated children
      for (let i = containers.length - 1; i >= 0; i--) {
        if (el.contains(containers[i].el)) containers.splice(i, 1);
      }
      containers.push({ el, count: players.size });
    }
    containers.sort((a, b) => b.count - a.count);
    // Mark the currently-cached history element distinctly
    if (_cachedHistoryEl && document.contains(_cachedHistoryEl)) {
      _cachedHistoryEl.style.outline = "4px solid lime";
      _cachedHistoryEl.style.outlineOffset = "4px";
      console.log(
        "%c[DM] ★ Currently using this as pick history (lime border):",
        "color:lime;font-weight:bold", _cachedHistoryEl,
        `\n  → window.dm.setHistorySelector('') to clear and re-detect`
      );
    }

    containers.forEach(({ el, count }, i) => {
      if (el === _cachedHistoryEl) return;  // already highlighted above
      const color = `hsl(${i * 47 % 360}, 80%, 55%)`;
      el.style.outline = `3px solid ${color}`;
      el.style.outlineOffset = "2px";
      const tag = el.tagName.toLowerCase();
      const id  = el.id ? `#${el.id}` : "";
      const cls = [...el.classList].slice(0, 2).join(".");

      // Build suggested CSS selector (most specific first)
      let suggested = "";
      if (el.id) suggested = `#${el.id}`;
      else if ([...el.attributes].find(a => a.name.startsWith("data-testid")))
        suggested = `[data-testid='${el.getAttribute("data-testid")}']`;
      else if ([...el.attributes].find(a => a.name === "aria-label"))
        suggested = `[aria-label='${el.getAttribute("aria-label")}']`;
      else if ([...el.attributes].find(a => a.name === "role"))
        suggested = `${tag}[role='${el.getAttribute("role")}']`;
      else if (el.className) suggested = `${tag}.${[...el.classList][0]}`;
      else suggested = tag;

      console.log(
        `%c[DM] ${i}: ${count} players — ${tag}${id}${cls ? "." + cls : ""}`,
        `color:${color}`, el,
        `\n  → window.dm.setHistorySelector('${suggested}')`
      );
    });
    setTimeout(() => {
      containers.forEach(({ el }) => {
        el.style.outline = "";
        el.style.outlineOffset = "";
      });
      if (_cachedHistoryEl) {
        _cachedHistoryEl.style.outline = "";
        _cachedHistoryEl.style.outlineOffset = "";
      }
    }, 6000);
    console.log("[DM] Outlines shown for 6 seconds. Use the index above to setHistorySelector.");
    return containers;
  },

  // Override the pick history CSS selector (persists across reloads)
  setHistorySelector(css) {
    state.customHistorySelector = css || null;
    _cachedHistoryEl  = null;
    _cachedHistoryKey = "";
    try { chrome.storage.local.set({ dmHistorySelector: css || "" }); } catch {}
    const inp = document.getElementById("dm-history-selector");
    if (inp) inp.value = css || "";
    console.log("[DM] History selector set to:", css || "(auto)");
    scanAndRank(true);
  },

  // Manually set which pick position (team slot) you are in the draft
  setPickPos(slot) {
    state.settings.pickPos = slot;
    saveSettings();
    updateSettingsUI();
    _pickPosDetected = true;
    // Reset tracked state so team assignments re-calculate with new slot
    state.allDrafted  = [];
    state.myTeam      = [];
    state.teamRosters = {};
    reconcileDraftState();  // re-applies manual overrides onto the cleared board
    state._orderedNewestFirst = null;
    clearDraftState();
    if (hasManualRosterOverride()) saveDraftState();
    console.log(`[DM] Pick position set to slot ${slot}. State reset. Rescanning...`);
    scanAndRank(true);
  },

  setCurrentPick(nextPick) {
    const n = parseInt(nextPick, 10);
    if (!Number.isFinite(n) || n < 1) {
      console.warn("[DM] setCurrentPick expects the next overall pick number, e.g. window.dm.setCurrentPick(17)");
      return;
    }
    const added = addMissingPickPlaceholders(n, "console setCurrentPick");
    state.pageCurrentPick = n;
    console.log(`[DM] Current pick set to ${n}; added ${added} placeholder pick(s).`);
    scanAndRank(true);
  },

  setMyRoster(players) {
    applyManualRosterInput(players, "console");
  },

  clearMyRoster() {
    clearManualRosterOverride();
  },

  correctPick(arg, maybeName) {
    const pickNo = typeof arg === "object" && arg ? Number(arg.pickNo || arg.pick || arg.overall_pick) : Number(arg);
    const rawName = typeof arg === "object" && arg ? arg.name : maybeName;
    if (!Number.isFinite(pickNo) || pickNo < 1 || !rawName) {
      console.warn("[DM] correctPick expects: window.dm.correctPick(6, 'Jaxon Smith-Njigba')");
      return;
    }
    const player = matchPlayer(String(rawName));
    if (!player) {
      console.warn("[DM] Could not match player:", rawName);
      return;
    }
    const idx = state.allDrafted.findIndex((d) => Number(d.overall_pick) === pickNo);
    const slot = pickToTeamSlot(pickNo, state.settings.teams);
    const round = Math.floor((pickNo - 1) / state.settings.teams) + 1;
    const entry = {
      name: player.name,
      position: player.position,
      round,
      by_user: slot === state.settings.pickPos,
      team_slot: slot,
      overall_pick: pickNo,
      _draftKey: draftKeyForName(player.name),
    };
    if (idx >= 0) state.allDrafted[idx] = entry;
    else state.allDrafted.push(entry);
    state.allDrafted.sort((a, b) => (Number(a.overall_pick) || 9999) - (Number(b.overall_pick) || 9999));
    reconcileDraftState();
    saveDraftState();
    state._lastRenderSig = null;
    console.log(`[DM] Corrected pick ${pickNo}: ${player.name} (${player.position}) -> T${slot}`);
    scanAndRank(true);
  },

  rosters() {
    const panels = findDKRosterPanels();
    console.log(`[DM] DK roster panels detected: ${panels.length}`);
    panels.forEach((panel, i) => {
      console.log(
        `[DM] roster ${i + 1}: slot=${panel.slot} players=${panel.players.length} score=${panel.score}`,
        panel.players.map((p) => `${p.name} (${p.position})`),
        panel.el
      );
    });
    if (panels.length === 0) {
      const candidates = findDKRosterCandidates(20);
      console.log(`[DM] Roster auto-detect found 0. Broad candidates with player names: ${candidates.length}`);
      candidates.forEach((cand, i) => {
        const color = `hsl(${i * 47 % 360}, 80%, 55%)`;
        cand.el.style.outline = `3px solid ${color}`;
        cand.el.style.outlineOffset = "2px";
        const tag = cand.el.tagName.toLowerCase();
        const id = cand.el.id ? `#${cand.el.id}` : "";
        const cls = [...cand.el.classList].slice(0, 2).join(".");
        let suggested = "";
        if (cand.el.id) suggested = `#${cand.el.id}`;
        else if (cand.el.getAttribute("data-testid")) suggested = `[data-testid='${cand.el.getAttribute("data-testid")}']`;
        else if (cand.el.getAttribute("aria-label")) suggested = `[aria-label='${cand.el.getAttribute("aria-label")}']`;
        else if (cand.el.classList.length) suggested = `${tag}.${[...cand.el.classList][0]}`;
        else suggested = tag;
        console.log(
          `%c[DM roster candidate ${i}] slot=${cand.slot || "?"} players=${cand.players.length} score=${cand.score} ${tag}${id}${cls ? "." + cls : ""}`,
          `color:${color}`,
          cand.players.map((p) => `${p.name} (${p.position})`),
          cand.el,
          `\n  If this is your roster, run: window.dm.setRosterSelector('${suggested}', YOUR_PICK_SLOT)`
        );
      });
      setTimeout(() => {
        candidates.forEach((cand) => {
          cand.el.style.outline = "";
          cand.el.style.outlineOffset = "";
        });
      }, 8000);
    }
    return panels;
  },

  setRosterSelector(arg) {
    const css = typeof arg === "object" && arg ? arg.css : arg;
    const slot = typeof arg === "object" && arg ? Number(arg.slot) : null;
    state.customRosterSelector = css || null;
    state.customRosterSlot = Number.isFinite(slot) && slot >= 1 ? slot : null;
    try {
      chrome.storage.local.set({
        dmRosterSelector: css || "",
        dmRosterSlot: state.customRosterSlot || "",
      });
    } catch {}
    console.log("[DM] Roster selector set to:", css || "(auto)", state.customRosterSlot ? `slot ${state.customRosterSlot}` : "");
    const panels = findDKRosterPanels();
    console.log(`[DM] Roster panels after selector: ${panels.length}`);
    panels.forEach((panel, i) => console.log(`[DM] roster ${i + 1}: slot=${panel.slot} players=${panel.players.length}`, panel.players.map((p) => p.name), panel.el));
    scanAndRank(true);
  },

  // Force pick history ordering: true=newest-first (reverse), false=oldest-first
  setOrder(newestFirst) {
    state.historyNewestFirst = newestFirst;
    try { chrome.storage.local.set({ dmHistoryNewestFirst: newestFirst }); } catch {}
    console.log("[DM] Order set to:", newestFirst === null ? "auto-detect" : newestFirst ? "newest-first (reversed)" : "oldest-first");
    // Reset allDrafted + ordering cache so corrected order takes effect on next scan
    state.allDrafted = [];
    state.myTeam = [];
    state.teamRosters = {};
    state._orderedNewestFirst = null;
    _cachedHistoryEl  = null;
    _cachedHistoryKey = "";
    clearDraftState();
    scanAndRank(true);
  },

  // Show what the current pick history detection finds
  debug() {
    const el = findPickHistory();
    if (!el) {
      console.log("[DM] No pick history element detected. Try window.dm.highlight()");
      return;
    }
    const rawPicks = readPickHistory(el);
    const ordered  = orderPickHistory(rawPicks);
    const tag = el.tagName.toLowerCase();
    const id  = el.id ? `#${el.id}` : "";
    const cls = [...el.classList].slice(0, 3).join(".");
    console.log(`[DM] History element: ${tag}${id}${cls ? "." + cls : ""}`, el);
    console.log(`[DM] Raw picks (${rawPicks.length}, DOM order):`);
    rawPicks.slice(0, 20).forEach((p, i) => {
      const pn = p.pickNum ? `pick#${p.pickNum}` : `idx${i}`;
      console.log(`  [${i}] ${pn} → ${p.name} (${p.position})${p.unknown ? " [UNKNOWN]" : ""}`);
    });
    console.log(`[DM] Ordered picks (${ordered.length}, chronological):`);
    ordered.slice(0, 20).forEach((p, i) => {
      const overallPick = p.pickNum || (i + 1);
      const slot = pickToTeamSlot(overallPick, state.settings.teams);
      const isMe = slot === state.settings.pickPos ? " ← YOU" : "";
      console.log(`  [${i}] pick#${overallPick} → Team${slot}${isMe} · ${p.name} (${p.position})`);
    });
    const isNewestFirst = rawPicks.length > 0 && ordered[0]?.name === rawPicks[rawPicks.length-1]?.name;
    console.log(`[DM] Ordering: ${isNewestFirst ? "NEWEST-FIRST (reversed)" : "OLDEST-FIRST (normal)"}`);
    const missing = ordered.filter(p => p.unknown).length;
    if (missing > 0) console.log(`[DM] Unknown players (not in DB): ${missing} — won't appear in 'available'`);
    console.log(`[DM] Tip: if wrong player appears, run window.dm.setHistorySelector('') to re-detect`);
  },

  // Reset all tracked picks and start fresh
  reset() {
    state.allDrafted  = [];
    state.myTeam      = [];
    state.manualMyRoster = [];
    state.manualAdds     = [];
    state.manualRemovals = new Set();
    state.teamRosters = {};
    state.viewingTeamSlot    = null;
    state._orderingFlipCount = 0;
    state.pageCurrentPick    = null;
    state._orderedNewestFirst = null;
    state.containerPlayerCounts.clear();
    _pickPosDetected = false;
    _cachedHistoryEl  = null;
    _cachedHistoryKey = "";
    state.scanCount = 0;
    clearDraftState();
    console.log("[DM] State reset. Rescanning...");
    scanAndRank(true);
  },
};

setTimeout(init, 1200);
