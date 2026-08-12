# DraftManager

DraftManager is a local best-ball draft assistant. A Chrome extension observes
the supported draft room, sends roster state to a FastAPI service bound to the
loopback interface, and renders ranked player recommendations from trained
tournament-EV policies.

## Architecture and privacy

```text
Supported draft page -> Chrome extension -> http://127.0.0.1:8765 -> ranking model
```

- Draft state is processed on the user's computer.
- The API binds to `127.0.0.1`; it is not exposed to the local network.
- Browser-to-server requests are limited to an explicit origin allowlist.
- The project contains no analytics, telemetry endpoint, user database, API key,
  or cloud credential.
- Training-only refresh scripts access the public data sources named in their
  source code. They are not called by the live server.

The extension necessarily reads draft-room state on supported sites. Review the
extension source and requested permissions before loading it.

## Run locally

Requirements: Python 3.10 or newer, Node.js for JavaScript syntax checks, and a
Chromium-based browser.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python server/server.py
```

In Chrome, open `chrome://extensions`, enable Developer mode, choose **Load
unpacked**, and select `extension/`.

The `/rank` endpoint accepts `scoring: "full"` or `scoring: "half"`. Full PPR
and half PPR use separately trained policy artifacts. If a policy is unavailable
or explicitly disabled, the server falls back to a deterministic VOR/boom
ranking.

## Model pipeline

The training pipeline drafts ADP-based opponent rooms, simulates correlated
weekly outcomes, applies tournament advancement and payout rules, generates
rollout EV labels, and fits the low-latency policy used by the server.

```powershell
# Small end-to-end smoke run
python training/run_dk_ev_training.py --skip-tests --states 2 --candidates 3 --rollouts 1 --eval-seasons 100 --field-rooms 1 --field-seasons 40

# Fit the serving policy from generated rollout labels
python training/fit_dk_ev_policy.py
```

See `training/DK_EV_PIPELINE.md` for the pipeline layout and artifact lineage.

## Verification

```powershell
python training/test_server_dk_ev_policy.py
python -m pytest training
node --check extension/content.js
node --check extension/background.js
python -m compileall -q server training
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/audit_public_repo.ps1
```

## Security and data notice

Only load the tracked model artifacts from a trusted checkout: Python model
serialization formats are not safe for untrusted files. Generated datasets,
local configuration, credentials, logs, and experimental model outputs are
excluded from version control. See `SECURITY.md` for the repository's security
boundaries and audit procedure.

DraftKings, Underdog Fantasy, and data-source names are trademarks of their
respective owners. This independent project is not endorsed by or affiliated
with those companies. Users are responsible for complying with applicable site
terms and local laws.
