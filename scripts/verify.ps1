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
    -Arguments @((Join-Path $ProjectRoot "experiments\01_math_foundations.py")) `
    -Description "Math foundations experiment"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\02_tokenization.py")) `
    -Description "Tokenization experiment"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\03_bigram_language_model.py")) `
    -Description "Bigram language model experiment"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\04_attention.py")) `
    -Description "Attention experiment"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\05_transformer_block.py")) `
    -Description "Transformer block experiment"
Invoke-CheckedPython `
    -Arguments @(
        (Join-Path $ProjectRoot "experiments\06_train_tiny_gpt.py"),
        "--quick"
    ) `
    -Description "TinyGPT pretraining experiment"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\07_generate.py")) `
    -Description "TinyGPT generation experiment"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\08_lora.py")) `
    -Description "LoRA and SFT masking experiment"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\09_tiny_rag.py")) `
    -Description "Tiny RAG experiment"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\10_sft_tiny_gpt.py"), "--quick") `
    -Description "TinyGPT SFT experiment"
Invoke-CheckedPython `
    -Arguments @(
        (Join-Path $ProjectRoot "experiments\07_generate.py"),
        "--base-checkpoint",
        (Join-Path $ProjectRoot "checkpoints\tiny_gpt.pt"),
        "--adapter",
        (Join-Path $ProjectRoot "checkpoints\tiny_gpt_lora_adapter.pt"),
        "--prompt",
        "为什么需要因果掩码？",
        "--tokens",
        "8"
    ) `
    -Description "Independent LoRA adapter inference"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "experiments\11_local_agent.py")) `
    -Description "Local Agent experiment"
Invoke-CheckedPython `
    -Arguments @((Join-Path $ProjectRoot "examples\capstone_attention\run.py")) `
    -Description "Capstone Attention example"
