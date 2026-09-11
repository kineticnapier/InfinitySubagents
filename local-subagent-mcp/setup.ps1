$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    py -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install -U pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

if (-not (Test-Path "config.toml")) {
    Copy-Item "config.example.toml" "config.toml"
    Write-Host "Created config.toml."
}

Write-Host ""
Write-Host "Choose the LM Studio model without editing config.toml manually:"
Write-Host ".\.venv\Scripts\python.exe model_config.py select"
Write-Host ""
Write-Host "Codex config snippet:"
& ".\.venv\Scripts\python.exe" install_codex.py
Write-Host ""
Write-Host "After choosing a model, run this to append the Codex MCP entry:"
Write-Host ".\.venv\Scripts\python.exe install_codex.py --install"
