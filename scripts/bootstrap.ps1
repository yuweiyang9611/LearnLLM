param(
    [string]$Python = "python",
    [switch]$WithDocs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$VenvDirectory = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDirectory "Scripts\python.exe"

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Checking Python 3.11 or 3.12: $Python"
Invoke-CheckedPython `
    -Executable $Python `
    -Arguments @(
        "-c",
        "import sys; assert sys.version_info[:2] in {(3, 11), (3, 12)}, f'Python 3.11 or 3.12 is required, got {sys.version.split()[0]}'; print(sys.version.split()[0])"
    ) `
    -Description "Python version check"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    Write-Host "Creating isolated environment: $VenvDirectory"
    Invoke-CheckedPython `
        -Executable $Python `
        -Arguments @("-m", "venv", $VenvDirectory) `
        -Description "Virtual environment creation"
}

Invoke-CheckedPython `
    -Executable $VenvPython `
    -Arguments @(
        "-c",
        "import sys; assert sys.version_info[:2] in {(3, 11), (3, 12)}, f'Existing .venv uses unsupported Python {sys.version.split()[0]}; recreate .venv with Python 3.11 or 3.12'"
    ) `
    -Description "Virtual environment version check"

Write-Host "Installing learning dependencies into .venv ..."
Invoke-CheckedPython `
    -Executable $VenvPython `
    -Arguments @("-m", "pip", "--disable-pip-version-check", "install", "--no-input", "-r", (Join-Path $ProjectRoot "requirements.txt")) `
    -Description "Learning dependency installation"
Invoke-CheckedPython `
    -Executable $VenvPython `
    -Arguments @("-m", "pip", "--disable-pip-version-check", "install", "--no-input", "--no-index", "--no-build-isolation", "--no-deps", "-e", $ProjectRoot) `
    -Description "Editable project installation"

if ($WithDocs) {
    Write-Host "Installing optional PDF dependencies into .venv ..."
    Invoke-CheckedPython `
        -Executable $VenvPython `
        -Arguments @("-m", "pip", "--disable-pip-version-check", "install", "--no-input", "-r", (Join-Path $ProjectRoot "requirements-docs.txt")) `
        -Description "PDF dependency installation"
}

Write-Host "Environment ready (all packages are installed only in .venv):"
Invoke-CheckedPython `
    -Executable $VenvPython `
    -Arguments @(
        "-c",
        "import sys, torch, numpy; print(sys.executable); print('torch', torch.__version__); print('numpy', numpy.__version__)"
    ) `
    -Description "Environment import check"

if ($WithDocs) {
    Invoke-CheckedPython `
        -Executable $VenvPython `
        -Arguments @(
            "-c",
            "import reportlab, pypdf; print('reportlab', reportlab.Version); print('pypdf', pypdf.__version__)"
        ) `
        -Description "PDF dependency import check"
}
