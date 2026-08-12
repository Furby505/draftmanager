/**
 * DraftManager — WebSocket + console interceptor.
 * Runs in the page's MAIN world at document_start, before any site JS executes.
 * Intercepts WebSocket messages and DraftKings' console.log events, then
 * dispatches CustomEvents that content.js listens to.
 *
 * Known DraftKings event names (from console observation):
 *   SelectionRecorded   — a pick was just made
 *   SelectionOnTheClock — next team's turn started
 */
(function () {
  const CAPTURE_LIMIT = 120;
  const CAPTURE_TEXT_LIMIT = 12000;
  const capture = {
    enabled: sessionStorage.getItem("dmCaptureDK") === "1",
    items: [],
  };

  function safeStringify(value) {
    if (typeof value === "string") return value;
    try { return JSON.stringify(value); } catch (_) { return String(value); }
  }

  function shouldCapture(url, text) {
    const u = String(url || "");
    const t = String(text || "");
    return /draftkings|dkng|pusher|centrifuge|socket|draft|selection|entrant|roster/i.test(u) ||
      /SelectionRecorded|SelectionOnTheClock|selectionRecorded|selection|draft|entrant|roster|player/i.test(t);
  }

  function relevantUrlText(url) {
    try {
      const parsed = new URL(String(url || ""), location.href);
      return `${parsed.hostname} ${parsed.pathname} ${parsed.search}`;
    } catch (_) {
      return String(url || "");
    }
  }

  function isDraftDataUrl(url) {
    const u = relevantUrlText(url).toLowerCase();
    return u.includes("/draftstatus") || u.includes("/draftables");
  }

  function shouldInspectCapturedUrl(url) {
    if (isDraftDataUrl(url)) return true;
    const u = relevantUrlText(url);
    return /dkng|pusher|centrifuge|socket|drafts?\/|draftgroups?|selection|entrant|roster|player/i.test(u);
  }

  function shouldReadFetchBody(url, response) {
    if (isDraftDataUrl(url)) return true;
    if (!capture.enabled || !shouldInspectCapturedUrl(url)) return false;

    const type = String(response?.headers?.get?.("content-type") || "").toLowerCase();
    if (type && !/json|text|javascript|x-www-form-urlencoded/.test(type)) return false;

    const len = Number(response?.headers?.get?.("content-length"));
    return !Number.isFinite(len) || len <= 300000;
  }

  function shouldReadXhrBody(xhr) {
    const url = xhr?.__dmUrl || "";
    if (isDraftDataUrl(url)) return true;
    if (!capture.enabled || !shouldInspectCapturedUrl(url)) return false;

    const responseType = String(xhr.responseType || "");
    if (responseType && responseType !== "text") return false;

    const type = String(xhr.getResponseHeader?.("content-type") || "").toLowerCase();
    if (type && !/json|text|javascript|x-www-form-urlencoded/.test(type)) return false;

    const len = Number(xhr.getResponseHeader?.("content-length"));
    return !Number.isFinite(len) || len <= 300000;
  }

  function storeCapture(type, url, text, extra = {}) {
    if (!capture.enabled) return;
    if (!shouldCapture(url, text)) return;
    const raw = String(text || "");
    capture.items.push({
      i: capture.items.length,
      t: new Date().toISOString(),
      type,
      url: String(url || ""),
      text: raw.slice(0, CAPTURE_TEXT_LIMIT),
      truncated: raw.length > CAPTURE_TEXT_LIMIT,
      ...extra,
    });
    if (capture.items.length > CAPTURE_LIMIT) capture.items.shift();
  }

  function dumpCapture(filter = "") {
    const needle = String(filter || "").toLowerCase();
    const items = capture.items.filter((item) => {
      if (!needle) return true;
      return `${item.type} ${item.url} ${item.text}`.toLowerCase().includes(needle);
    });
    console.log(`[DM capture] ${items.length}/${capture.items.length} payload(s)${needle ? ` matching "${filter}"` : ""}`);
    items.forEach((item, idx) => {
      console.log(`[DM capture ${idx}] ${item.t} ${item.type} ${item.url}${item.truncated ? " [truncated]" : ""}`, item.text);
    });
    return items;
  }

  // ── DraftKings draft-data interception (authoritative pick feed) ───────────
  // DK serves the whole draft as JSON, so content.js never has to scrape the DOM
  // or infer pick order:
  //   …/drafts/v1/{draftId}/entries/{entryId}/draftStatus → ordered all-team picks
  //   …/draftgroups/v1/draftgroups/{id}/draftables        → draftableId → player
  // Payloads are slimmed before crossing the world boundary and buffered so
  // content.js (which loads later) can pull whatever was captured at page load.
  const draftData = { status: null, draftables: null, statusUrl: null, statusInit: null };

  function rememberDraftStatusRequest(args, url) {
    if (!String(url || "").toLowerCase().includes("/draftstatus")) return;
    try {
      const req = args[0] instanceof Request ? args[0] : null;
      const init = args[1] || {};
      const headers = new Headers();
      const srcHeaders = init.headers || req?.headers;
      if (srcHeaders) new Headers(srcHeaders).forEach((value, key) => headers.set(key, value));
      draftData.statusInit = {
        method: init.method || req?.method || "GET",
        headers,
        credentials: init.credentials || req?.credentials || "include",
        cache: "no-store",
      };
    } catch (_) {
      draftData.statusInit = { credentials: "include", cache: "no-store" };
    }
  }

  function slimDraftables(json) {
    const list = json && (json.draftables || json.players || json.data?.draftables || json.data?.players);
    if (!Array.isArray(list)) return [];
    const out = [];
    for (const d of list) {
      if (!d) continue;
      const src = d.player || d.draftable || d;
      const draftableId = d.draftableId ?? src.draftableId ?? d.id;
      if (draftableId == null) continue;
      let bye = null;
      const attr = (d.playerAttributes || src.playerAttributes || []).find((a) => a && a.name === "ByeWeek");
      if (attr) bye = attr.value;
      out.push({
        draftableId,
        playerId: d.playerId ?? src.playerId,
        name: d.displayName || src.displayName || d.playerName || src.playerName ||
          `${d.firstName || src.firstName || ""} ${d.lastName || src.lastName || ""}`.trim(),
        position: d.position || src.position || d.positionAbbreviation || src.positionAbbreviation || "",
        team: d.teamAbbreviation || src.teamAbbreviation || d.team || src.team || "",
        bye: bye,
      });
    }
    return out;
  }

  function slimDraftStatus(json) {
    const board = json && json.draftBoard;
    if (!Array.isArray(board)) return null;
    return board
      .map((p) => ({
        userKey: p.userKey || p.entrantUserKey || p.entryUserKey,
        draftableId: p.draftableId ?? p.draftable?.draftableId ?? p.selection?.draftableId,
        playerId: p.playerId ?? p.draftable?.playerId ?? p.selection?.playerId,
        name: p.displayName || p.playerName || p.draftable?.displayName || p.selection?.displayName || "",
        position: p.position || p.draftable?.position || p.selection?.position || "",
        round: p.roundNumber ?? p.round,
        overall: p.overallSelectionNumber ?? p.overallSelection ?? p.overall ?? p.pickNumber,
      }))
      .filter((p) => Number(p.overall) >= 1 && (p.userKey || p.draftableId != null || p.playerId != null || p.name));
  }

  function emitDraftData(url, text) {
    const u = String(url || "");
    const lowerUrl = u.toLowerCase();
    if (!text || (!lowerUrl.includes("/draftstatus") && !lowerUrl.includes("/draftables"))) return;
    try {
      if (lowerUrl.includes("/draftstatus")) {
        draftData.statusUrl = u;  // remember for triggered re-fetch (keeps the feed live)
        const board = slimDraftStatus(JSON.parse(text));
        if (board) {
          draftData.status = board;
          window.dispatchEvent(new CustomEvent("dm:draft_status", { detail: { board, url: u } }));
        }
      } else {
        const draftables = slimDraftables(JSON.parse(text));
        if (draftables.length) {
          draftData.draftables = draftables;
          window.dispatchEvent(new CustomEvent("dm:draftables", { detail: { draftables, url: u } }));
        }
      }
    } catch (_) {}
  }

  // content.js asks for buffered payloads once its listeners are up (load race).
  window.addEventListener("dm:request_draft_data", () => {
    if (draftData.draftables)
      window.dispatchEvent(new CustomEvent("dm:draftables", { detail: { draftables: draftData.draftables } }));
    if (draftData.status)
      window.dispatchEvent(new CustomEvent("dm:draft_status", { detail: { board: draftData.status, url: draftData.statusUrl } }));
  });

  // ── Triggered draftStatus re-fetch (keeps the authoritative feed live) ──────
  // DK pushes picks over the WebSocket but does NOT reliably re-fetch the full
  // draftStatus JSON on every pick, so passive interception goes stale mid-draft
  // (missed picks / fallback fills gaps to the wrong team). content.js pokes
  // `dm:refetch_draft_status` on each pick + on a periodic safety poll; we re-pull
  // the last-seen draftStatus URL using the PAGE's own session (credentials), so
  // the response flows back through emitDraftData → syncFromDKApi unchanged.
  const REFETCH_MIN_MS = 700;  // throttle floor between actual network hits
  let refetchInFlight = false;
  let refetchPending = false;
  let refetchLastAt = 0;
  let refetchTimer = null;

  function doRefetch() {
    const url = draftData.statusUrl;
    if (!url || !OrigFetch) return;
    refetchInFlight = true;
    refetchLastAt = Date.now();
    OrigFetch.call(window, url, draftData.statusInit || { credentials: "include", cache: "no-store" })
      .then((r) => {
        if (r && r.ok) return r.text();
        storeCapture("refetch:error", url, `HTTP ${r ? r.status : "no response"}`);
        return null;
      })
      .then((text) => { if (text) emitDraftData(url, text); })
      .catch(() => {})
      .finally(() => {
        refetchInFlight = false;
        if (refetchPending) { refetchPending = false; refetchDraftStatus(); }
      });
  }

  function refetchDraftStatus() {
    if (!draftData.statusUrl) return;
    if (refetchInFlight) { refetchPending = true; return; }  // coalesce; run trailing edge
    const wait = REFETCH_MIN_MS - (Date.now() - refetchLastAt);
    if (wait > 0) {
      if (!refetchTimer) refetchTimer = setTimeout(() => { refetchTimer = null; doRefetch(); }, wait);
      return;
    }
    doRefetch();
  }

  window.addEventListener("dm:refetch_draft_status", refetchDraftStatus);

  // ── WebSocket interception ─────────────────────────────────────────────────
  const OrigWS = window.WebSocket;
  if (OrigWS) {
    function wrappedWS(...args) {
      const ws = new OrigWS(...args);
      const wsUrl = args[0] || "";
      ws.addEventListener("message", function (e) {
        try {
          const raw = typeof e.data === "string" ? e.data : "";
          if (!raw) return;
          storeCapture("ws:message", wsUrl, raw);

          if (raw.includes("SelectionRecorded") || raw.includes("selectionRecorded")) {
            window.dispatchEvent(
              new CustomEvent("dm:pick_recorded", { detail: { raw: raw.slice(0, 4000), source: "ws" } })
            );
          }
          if (raw.includes("SelectionOnTheClock") || raw.includes("selectionOnTheClock")) {
            window.dispatchEvent(
              new CustomEvent("dm:on_clock", { detail: { raw: raw.slice(0, 4000), source: "ws" } })
            );
          }
          // Broadcast ALL WS messages in debug mode so content.js can log them
          if (window.__dmDebugWS) {
            window.dispatchEvent(
              new CustomEvent("dm:ws_message", { detail: raw.slice(0, 2000) })
            );
          }
        } catch (_) {}
      });
      return ws;
    }
    wrappedWS.prototype = OrigWS.prototype;
    window.WebSocket = wrappedWS;
  }

  // ── console.log interception (fallback / DraftKings-specific) ─────────────
  // DraftKings logs "processDraftStatusUpdateEvent: Handling SelectionRecorded"
  // for every pick even when the WS message format changes. Catching this is
  // a reliable fallback that doesn't depend on the WS payload structure.
  const OrigFetch = window.fetch;
  if (OrigFetch) {
    window.fetch = async function (...args) {
      const url = typeof args[0] === "string" ? args[0] : args[0]?.url || "";
      rememberDraftStatusRequest(args, url);
      const response = await OrigFetch.apply(this, args);
      try {
        if (shouldReadFetchBody(url, response)) {
          const clone = response.clone();
          clone.text()
            .then((text) => { emitDraftData(url, text); storeCapture("fetch", url, text, { status: response.status }); })
            .catch(() => {});
        }
      } catch (_) {}
      return response;
    };
  }

  const OrigXHROpen = XMLHttpRequest.prototype.open;
  const OrigXHRSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this.__dmUrl = url;
    this.__dmMethod = method;
    return OrigXHROpen.call(this, method, url, ...rest);
  };
  XMLHttpRequest.prototype.send = function (...args) {
    this.addEventListener("loadend", () => {
      try {
        if (shouldReadXhrBody(this)) {
          const text = typeof this.responseText === "string" ? this.responseText : "";
          emitDraftData(this.__dmUrl || "", text);
          storeCapture("xhr", this.__dmUrl || "", text, { status: this.status, method: this.__dmMethod || "" });
        }
      } catch (_) {}
    });
    return OrigXHRSend.apply(this, args);
  };

  const origLog = console.log;
  console.log = function (...args) {
    origLog.apply(console, args);
    try {
      const msg = args.map((a) => (typeof a === "string" ? a : "")).join(" ");
      const hasSelectionEvent = msg.includes("SelectionRecorded") || msg.includes("SelectionOnTheClock");
      const rawMsg = hasSelectionEvent
        ? args.map(safeStringify).join(" ")
        : args.map((a) => (typeof a === "string" ? a : Object.prototype.toString.call(a))).join(" ");
      if (capture.enabled) storeCapture("console", "console.log", rawMsg);
      if (msg.includes("SelectionRecorded")) {
        window.dispatchEvent(
          new CustomEvent("dm:pick_recorded", { detail: { source: "console", raw: rawMsg.slice(0, 4000) } })
        );
      }
      if (msg.includes("SelectionOnTheClock")) {
        window.dispatchEvent(
          new CustomEvent("dm:on_clock", { detail: { source: "console" } })
        );
      }
    } catch (_) {}
  };
  // ── window.dm debug bridge ────────────────────────────────────────────────
  // content.js runs in an isolated world so window.dm set there is NOT visible
  // from the DevTools console. This bridge lives in the MAIN world and relays
  // commands to content.js via CustomEvents, which CAN cross the world boundary.
  window.dm = {
    debug()              { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "debug" } })); },
    highlight()          { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "highlight" } })); },
    reset()              { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "reset" } })); },
    status()             { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "status" } })); },
    readPicks()          { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "readPicks" } })); },
    setHistorySelector(css) { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "setHistorySelector", arg: css } })); },
    setOrder(v)          { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "setOrder", arg: v } })); },
    setPickPos(n)        { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "setPickPos", arg: n } })); },
    setCurrentPick(n)    { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "setCurrentPick", arg: n } })); },
    setMyRoster(players) { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "setMyRoster", arg: players } })); },
    clearMyRoster()      { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "clearMyRoster" } })); },
    correctPick(pickNo, name) { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "correctPick", arg: { pickNo, name } } })); },
    rosters()            { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "rosters" } })); },
    setRosterSelector(css, slot) { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "setRosterSelector", arg: { css, slot } } })); },
    testDraftBtn(name)   { window.dispatchEvent(new CustomEvent("dm:cmd", { detail: { cmd: "testDraftBtn", arg: name } })); },
    captureDK(on = true) {
      const wasEnabled = capture.enabled;
      capture.enabled = on !== false;
      sessionStorage.setItem("dmCaptureDK", capture.enabled ? "1" : "0");
      if (capture.enabled && !wasEnabled) capture.items = [];
      console.log(`[DM capture] ${capture.enabled ? `ON - ${capture.items.length} payload(s) stored` : "OFF"}`);
      return capture.enabled;
    },
    statusDK() {
      console.log(`[DM capture] enabled=${capture.enabled} stored=${capture.items.length} session=${sessionStorage.getItem("dmCaptureDK") || "0"}`);
      return { enabled: capture.enabled, stored: capture.items.length };
    },
    dumpDK(filter = "") { return dumpCapture(filter); },
    clearDK() { capture.items = []; console.log("[DM capture] cleared"); },
    refetch() {
      console.log(`[DM] draftStatus re-fetch requested${draftData.statusUrl ? "" : " (no draftStatus URL captured yet — load the draft page first)"}`);
      refetchDraftStatus();
    },
  };
})();
