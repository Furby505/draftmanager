[CmdletBinding()]
param(
    [switch]$SkipHistory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (& git rev-parse --show-toplevel 2>$null)
if (-not $repoRoot) {
    throw "Run this script from inside the DraftManager Git repository."
}

$failures = [System.Collections.Generic.List[string]]::new()

function Add-Failure {
    param([string]$Message)
    if (-not $failures.Contains($Message)) {
        $failures.Add($Message)
    }
}

Push-Location $repoRoot
try {
    $secretPatterns = [ordered]@{
        "private key"       = "-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
        "AWS access key"    = "\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"
        "GitHub token"      = "\bgh[pousr]_[A-Za-z0-9_]{20,}\b"
        "OpenAI token"      = "\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"
        "Google API key"    = "\bAIza[0-9A-Za-z_-]{30,}\b"
        "Slack token"       = "\bxox[baprs]-[A-Za-z0-9-]{10,}\b"
        "Stripe live key"   = "\b(?:sk|rk)_live_[A-Za-z0-9]{16,}\b"
        "JSON Web Token"    = "\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
        "credentialed URL"  = "[A-Za-z][A-Za-z0-9+.-]*://[^\s/:]+:[^\s/@]+@"
        "secret literal"    = "\b(?:api[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|password|passwd)\b\s*[:=]\s*[`"'][^`"'\r\n]{8,}[`"']"
        "personal email"    = "\b[A-Z0-9._%+-]+@(?:gmail|outlook|yahoo|hotmail)\.[A-Z]{2,}\b"
        "Windows user path" = "\b[A-Z]:\\Users\\[^\\\s`"']+"
        "Unix user path"    = "(?:^|[`"'\s])/(?:home|Users)/[^/\s`"']+"
    }

    $trackedFiles = @(& git ls-files)
    $binaryEncoding = [System.Text.Encoding]::GetEncoding(28591)
    foreach ($path in $trackedFiles) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            continue
        }
        $content = $binaryEncoding.GetString(
            [System.IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $path))
        )
        foreach ($entry in $secretPatterns.GetEnumerator()) {
            if ([regex]::IsMatch(
                    $content,
                    $entry.Value,
                    [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
                )) {
                Add-Failure "Tracked content matches $($entry.Key): $path"
            }
        }
    }

    $blockedPathPattern = "(?i)(^|/)(\.env(?:\..*)?|id_rsa.*|id_ed25519.*|[^/]+\.(?:key|pem|p12|pfx|jks|keystore|kdbx))$"
    foreach ($path in $trackedFiles) {
        if ($path -eq ".env.example") {
            continue
        }
        if ($path -match $blockedPathPattern) {
            Add-Failure "Secret-bearing filename is tracked: $path"
        }
    }

    foreach ($remoteName in @(& git remote)) {
        $remoteUrl = (& git remote get-url $remoteName)
        if ($remoteUrl -match "^[A-Za-z][A-Za-z0-9+.-]*://[^\s/:]+:[^\s/@]+@") {
            Add-Failure "Git remote '$remoteName' contains embedded credentials"
        }
    }

    if (-not $SkipHistory) {
        foreach ($email in @(& git log --all --format="%ae%n%ce" | Sort-Object -Unique)) {
            if ($email -and $email -notmatch "^[^@]+@users\.noreply\.github\.com$") {
                Add-Failure "Reachable commit uses a public author/committer email"
            }
        }

        foreach ($objectLine in @(& git rev-list --objects --all)) {
            $separator = $objectLine.IndexOf(" ")
            if ($separator -lt 0) {
                continue
            }
            $historicalPath = $objectLine.Substring($separator + 1)
            if ($historicalPath -ne ".env.example" -and $historicalPath -match $blockedPathPattern) {
                Add-Failure "Reachable history contains a secret-bearing filename: $historicalPath"
            }
        }

        $historyPattern = "-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----|(AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9_]{20,}|sk-(proj-)?[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}|(sk|rk)_live_[A-Za-z0-9]{16,}|(api[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|password|passwd)[[:space:]]*[:=][[:space:]]*[`"'][^`"']{8,}[`"']"
        foreach ($commit in @(& git rev-list --all)) {
            & git grep -I -q -E -- $historyPattern $commit
            if ($LASTEXITCODE -eq 0) {
                Add-Failure "Reachable commit contains a credential pattern: $($commit.Substring(0, 12))"
            } elseif ($LASTEXITCODE -ne 1) {
                throw "git grep failed while auditing commit $commit"
            }
        }
    }
} finally {
    Pop-Location
}

if ($failures.Count -gt 0) {
    Write-Error ("Public-safety audit failed:`n - " + ($failures -join "`n - "))
    exit 1
}

Write-Output "Public-safety audit passed."
