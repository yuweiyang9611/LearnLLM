$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv. Run scripts\bootstrap.ps1 first."
}

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

Invoke-CheckedPython `
    -Arguments @("-m", "unittest", "discover", "-s", (Join-Path $ProjectRoot "tests"), "-v") `
    -Description "Unit tests"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\00_environment_check.py")) `
    -Description "Environment isolation check"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\09_tiny_rag.py")) `
    -Description "Tiny RAG experiment"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\10_sft_tiny_gpt.py"), "--quick") `
    -Description "TinyGPT SFT experiment"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\11_local_agent.py")) `
    -Description "Local Agent experiment"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "examples\capstone_attention\run.py")) `
    -Description "Capstone Attention example"
