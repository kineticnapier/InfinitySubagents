$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    py -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install -U pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

if (-not (Test-Path "config.toml")) {
    Copy-Item "config.example.toml" "config.toml"
    Write-Host "Created config.toml. Set lmstudio.model before using the MCP."
}

Write-Host ""
Write-Host "Codex config snippet:"
& ".\.venv\Scripts\python.exe" install_codex.py
Write-Host ""
Write-Host "After editing config.toml, run this to append the Codex MCP entry:"
Write-Host ".\.venv\Scripts\python.exe install_codex.py --install"
