# Codex Free Subagents

This repository uses **plain Git tags for versions**. There is no `runtime.py`, version database, switchboard, or custom runtime manager.

Both MCP configurations point directly at this checkout. To change versions, stop/reload the MCP servers and use `git switch`.

## Layout

```text
codex-free-subagents/
├─ localdev-mcp/
└─ local-subagent-mcp/
```

Local configuration files are ignored by Git, so they survive tag/branch switches:

```text
localdev-mcp/config.toml
localdev-mcp/jobs.json
local-subagent-mcp/config.toml
.venv/
```

## Initial setup

```powershell
cd F:\dev\codex-free-subagents

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\localdev-mcp\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r .\local-subagent-mcp\requirements.txt

Copy-Item .\localdev-mcp\config.example.toml .\localdev-mcp\config.toml
Copy-Item .\local-subagent-mcp\config.example.toml .\local-subagent-mcp\config.toml
```

Set the local model in `local-subagent-mcp\config.toml`, for example:

```toml
model = "qwen/qwen3-14b"
```

Register MinoFlux with the LocalDev manager supplied by the checked-out version.

## LM Studio MCP

Point LM Studio directly at the checkout:

```json
{
  "mcpServers": {
    "localdev": {
      "command": "F:\\dev\\codex-free-subagents\\.venv\\Scripts\\python.exe",
      "args": [
        "F:\\dev\\codex-free-subagents\\localdev-mcp\\server.py"
      ]
    }
  }
}
```

## Codex MCP

Point Codex directly at the same checkout:

```toml
[mcp_servers.local_subagent]
command = "F:\\dev\\codex-free-subagents\\.venv\\Scripts\\python.exe"
args = ["F:\\dev\\codex-free-subagents\\local-subagent-mcp\\server.py"]
enabled = true
env_vars = ["LM_API_TOKEN"]
startup_timeout_sec = 30
tool_timeout_sec = 900
```

## Version switching

First stop/reload the LocalDev and Local Subagent MCP processes. Then:

```powershell
git status --short
git tag --list

git switch --detach v0.1.0
# or
git switch --detach v0.2.0
```

`git switch` refuses to overwrite conflicting tracked changes. Keep `git status` clean before switching.

If dependencies changed between versions, refresh the same venv:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\localdev-mcp\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r .\local-subagent-mcp\requirements.txt
```

Then restart/reload both MCP servers.

To return to development HEAD:

```powershell
git switch main
```

To return to the previously checked-out branch/commit:

```powershell
git switch -
```

No source checkout is copied elsewhere and no custom "active version" state exists.

## Releases

Develop on `main`, commit normally, run tests, then make an annotated tag:

```powershell
git switch main
git status
# edit / test
git add .
git commit -m "..."
git tag -a v0.3.0 -m "v0.3.0"
```

The tag itself is the version.
