# Security policy

DraftManager is designed as a local application. The API listens only on
`127.0.0.1`, and the browser extension sends live draft state only to that local
service. No API key or hosted account is required.

## Trust boundaries

- Install the extension only from a checkout you trust.
- Load `.joblib` model files only from a trusted checkout. Model serialization
  can execute code when a malicious artifact is loaded.
- Do not expose port `8765` through a proxy, tunnel, router, or permissive
  firewall rule.
- Training refresh scripts make outbound requests to the data sources declared
  in those scripts; the live ranking service does not run those refresh jobs.
- Never commit environment files, credentials, browser data, TLS private keys,
  generated datasets, or local AI-tool configuration.

## Repository audit

Run the public-safety audit before pushing:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/audit_public_repo.ps1
```

The same audit runs in GitHub Actions for every push and pull request. It checks
tracked content and reachable history for common credential formats,
secret-bearing filenames, embedded credentials in remote URLs, and public commit
email addresses.

If you discover a vulnerability, use GitHub's private vulnerability-reporting
feature when it is available. Do not post credentials or exploit details in a
public issue.
