# Smoke test script for REST endpoints (PowerShell)
# Requirements: Windows PowerShell 5.1+ or PowerShell 7+, Invoke-WebRequest available
# Usage examples:
#   .\scripts\smoke\curl-smoke.ps1 -ApiUrl "http://localhost:8000"
#   $jwt = python .\scripts\smoke\jwt\gen_test_jwt.py --private-key .\private.pem --sub tester
#   .\scripts\smoke\curl-smoke.ps1 -ApiUrl "http://localhost:8000" -Jwt $jwt
#
# Validations:
#   - health_check: expect HTTP 200 or 503 JSON
#   - protected_check_no_token: expect HTTP 401 when AUTH is enabled
#   - protected_check_with_token: expect HTTP 200 when JWT is provided
#
# Notes:
# - Health endpoints are limiter-exempt per [python.function health()](fba_bench_api/main.py:226) and [python.function health_v1()](fba_bench_api/main.py:265).
# - Rate limiting is configured in [python.module main](fba_bench_api/main.py:114) via API_RATE_LIMIT.

[CmdletBinding()]
param(
  [string]$ApiUrl = "http://localhost:8000",
  [string]$Jwt = ""
)

function Write-Pass([string]$msg) { Write-Host "PASS: $msg" -ForegroundColor Green }
function Write-Fail([string]$msg) { Write-Host "FAIL: $msg" -ForegroundColor Red }

function Invoke-GetWithStatus {
  param(
    [Parameter(Mandatory=$true)][string]$Url,
    [hashtable]$Headers = @{}
  )
  try {
    $resp = Invoke-WebRequest -Uri $Url -Method Get -Headers $Headers -UseBasicParsing -ErrorAction Stop
    return @{ Code = [int]$resp.StatusCode; Body = $resp.Content }
  } catch {
    $ex = $_.Exception
    $httpResponse = $ex.Response
    if ($null -ne $httpResponse) {
      try { $code = [int]$httpResponse.StatusCode } catch { $code = -1 }
      $body = ""
      try {
        $stream = $httpResponse.GetResponseStream()
        if ($null -ne $stream) {
          $reader = New-Object System.IO.StreamReader($stream)
          $body = $reader.ReadToEnd()
          $reader.Close()
          $stream.Close()
        }
      } catch { }
      return @{ Code = $code; Body = $body }
    } else {
      return @{ Code = -1; Body = $ex.Message }
    }
  }
}

function Health-Check {
  $url = "$ApiUrl/api/v1/health"
  Write-Host "==> Health check: $url"
  $res = Invoke-GetWithStatus -Url $url
  if ($res.Body) { Write-Host $res.Body }
  if ($res.Code -eq 200 -or $res.Code -eq 503) {
    Write-Pass "Health returned $($res.Code)"
    return $true
  } else {
    Write-Fail "Health unexpected status: $($res.Code)"
    return $false
  }
}

function Protected-Check-NoToken {
  $url = "$ApiUrl/api/v1/settings"
  Write-Host "==> Protected endpoint without token (expect 401): $url"
  $res = Invoke-GetWithStatus -Url $url
  if ($res.Code -eq 401) {
    Write-Pass "Unauthorized as expected (401)"
    return $true
  } else {
    Write-Fail "Expected 401 without token, got $($res.Code)"
    return $false
  }
}

function Protected-Check-WithToken {
  param([string]$JwtToken)
  $url = "$ApiUrl/api/v1/settings"
  Write-Host "==> Protected endpoint with token (expect 200): $url"
  if (-not $JwtToken) {
    Write-Host "JWT not provided; skipping token-auth check."
    return $true
  }
  $headers = @{ "Authorization" = "Bearer $JwtToken" }
  $res = Invoke-GetWithStatus -Url $url -Headers $headers
  if ($res.Code -eq 200) {
    Write-Pass "Authorized as expected (200)"
    return $true
  } else {
    Write-Fail "With JWT expected 200, got $($res.Code)"
    return $false
  }
}

Write-Host "API_URL=$ApiUrl"
if ($Jwt) { Write-Host "JWT provided: yes" } else { Write-Host "JWT provided: no" }

$ok = $true
if (-not (Health-Check)) { $ok = $false }
if (-not (Protected-Check-NoToken)) { $ok = $false }
$withTokenOk = Protected-Check-WithToken -JwtToken $Jwt
if (-not $withTokenOk) { $ok = $false }

if ($ok) {
  Write-Host "All checks completed." -ForegroundColor Green
  exit 0
} else {
  # Exit non-zero on failure cases; especially when JWT is present and 200 not returned.
  exit 1
}