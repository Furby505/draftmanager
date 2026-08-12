// DraftManager background service worker
// Routes fetch requests from content.js to bypass page Content-Security-Policy.
// Content scripts run in the page context and are blocked by site CSP;
// the service worker has no such restriction.

function isInjectablePage(url) {
  try {
    const protocol = new URL(url || "").protocol;
    return protocol === "http:" || protocol === "https:";
  } catch {
    return false;
  }
}

chrome.action.onClicked.addListener((tab) => {
  if (!tab?.id) return;
  chrome.tabs.sendMessage(tab.id, { type: "toggle_panel" }, async () => {
    const missingReceiver = !!chrome.runtime.lastError;
    if (!missingReceiver || !isInjectablePage(tab.url)) return;
    try {
      await chrome.scripting.insertCSS({
        target: { tabId: tab.id },
        files: ["panel.css"],
      });
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["ws_interceptor.js"],
        world: "MAIN",
      });
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content.js"],
      });
      chrome.tabs.sendMessage(tab.id, { type: "toggle_panel" }, () => {
        void chrome.runtime.lastError;
      });
    } catch {
      // Unsupported browser pages, closed tabs, or site restrictions can reject
      // injection. The content-script match patterns still handle normal loads.
    }
  });
});

// The proxy below exists so the content script can reach the local server
// without tripping page CSP. Restrict it to the hosts we actually talk to, so a
// compromised or hostile draft page can't turn it into a general-purpose
// credentialed fetch relay.
const FETCH_ALLOWED_HOSTS = new Set([
  "localhost",
  "127.0.0.1",
  "draftkings.com",
  "underdogfantasy.com",
  "playunderdog.com",
]);

function isAllowedFetchUrl(raw) {
  let url;
  try {
    url = new URL(String(raw));
  } catch {
    return false;
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return false;
  const host = url.hostname.toLowerCase();
  for (const allowed of FETCH_ALLOWED_HOSTS) {
    if (host === allowed || host.endsWith(`.${allowed}`)) return true;
  }
  return false;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type !== "fetch") return;

  let settled = false;
  const reply = (payload) => {
    if (settled) return;
    settled = true;
    try { sendResponse(payload); } catch {}
  };

  if (!isAllowedFetchUrl(msg.url)) {
    reply({ ok: false, error: "blocked: url not in allowlist" });
    return true;
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), Number(msg.timeoutMs) || 45000);
    const opts = { method: msg.method || "GET", signal: controller.signal };
    if (msg.body) {
      opts.body = msg.body;
      opts.headers = { "Content-Type": "application/json" };
    }

    fetch(msg.url, opts)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => reply({ ok: true, data }))
      .catch((err) => reply({ ok: false, error: err.name === "AbortError" ? "request timed out" : err.message }))
      .finally(() => clearTimeout(timeout));
  } catch (err) {
    reply({ ok: false, error: err.message || String(err) });
  }

  return true; // keep channel open for async sendResponse
});
