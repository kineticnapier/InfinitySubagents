# Local Subagent MCP for Codex

This MCP server lets **Codex stay the parent agent** while delegating bounded work to a local model served by LM Studio.

```text
Codex (GPT parent)
    |
    | MCP: local_agent(...)
    v
Local Subagent MCP
    |
    | POST http://127.0.0.1:1234/api/v1/chat
    v
LM Studio / local model
    |
    | integration: mcp/localdev
    v
LocalDev MCP
    v
managed repo worktree / tests / benchmark
```

The local model's reasoning and raw MCP outputs are deliberately **not returned to Codex**. Codex gets only the worker's final result, compact tool-call counts, and token/runtime stats. This keeps the parent's context and paid usage small.

## Requirements

- Windows + Python 3.11+
- Codex CLI/Desktop using its normal OpenAI model
- LM Studio 0.4.0+ with the local API server enabled
- a local model capable of tool use
- LocalDev MCP configured in LM Studio as `mcp/localdev`

## 1. Configure LocalDev MCP in LM Studio

The bundle contains `../localdev-mcp`.

Create its venv/config and register the repos you want. Then add it to LM Studio's `mcp.json` under the server label `localdev`. An example is in `../examples/lmstudio-mcp.example.json`.

In LM Studio Server Settings enable **Allow calling servers from mcp.json**. The native `/api/v1/chat` endpoint can then give the model access to the configured MCP server.

## 2. Configure this server

```powershell
cd local-subagent-mcp
.\setup.ps1
```

You do not need to edit `config.toml` just to change models. With the LM Studio API server running, select one interactively:

```powershell
.\.venv\Scripts\python.exe model_config.py select
```

Other model-management commands:

```powershell
.\.venv\Scripts\python.exe model_config.py list
.\.venv\Scripts\python.exe model_config.py current
.\.venv\Scripts\python.exe model_config.py set qwen/qwen3-14b
```

`set` verifies the exact model id against `/api/v1/models` before changing the config. `--force` skips that verification for recovery/offline configuration.

Useful defaults for a single-GPU worker are already conservative:

- one local generation at a time
- 8192 context
- 4096 output-token cap
- localhost-only LM Studio endpoint

The output cap is intentional: if the local model gets stuck in a reasoning loop, it should terminate and report a blocker instead of consuming its whole context indefinitely.

## 3. Register it in Codex

After the venv exists and a model is selected:

```powershell
.\.venv\Scripts\python.exe install_codex.py --install
```

This only appends a new `[mcp_servers.local_subagent]` table if one does not already exist. It refuses to overwrite an existing one.

Equivalent `~/.codex/config.toml` form:

```toml
[mcp_servers.local_subagent]
command = "F:\\dev\\codex-free-subagents\\local-subagent-mcp\\.venv\\Scripts\\python.exe"
args = ["F:\\dev\\codex-free-subagents\\local-subagent-mcp\\server.py"]
enabled = true
```

Restart Codex after changing the MCP registration itself. Changing only the selected local model does **not** require a restart: `LocalAgentClient` reloads `config.toml` when it changes, so the next worker call uses the new model.

## MCP tools exposed to Codex

### `local_agent`

Starts a fresh local worker.

Arguments:

- `task`: task to delegate
- `mode`: `research` or `code`
- `repo`: optional LocalDev repo alias
- `job`: optional existing LocalDev job
- `max_output_tokens`: optional value up to the human-configured cap
- `context_length`: optional value up to the human-configured cap

`research` only exposes read-only LocalDev tools to the local model.

`code` additionally exposes controlled worktree operations: create job/checkpoint, apply patch, test, benchmark, revert and commit. `delete_job` is intentionally not delegated.

### `continue_local_agent`

Continue a stored LM Studio response by passing the returned `response_id`.

### `local_agent_health`

Checks LM Studio connectivity and reports the configured model/integration.

### `local_agent_models`

Lists the model ids currently returned by LM Studio and marks the model configured for the local worker.

### `local_agent_set_model`

Switches the local worker to an exact id returned by `local_agent_models`. It only rewrites `[lmstudio].model`; the next `local_agent` call reloads that setting automatically.

Example parent-agent workflow:

```text
Call local_agent_models, switch to the requested model with local_agent_set_model,
then run the same smoke test so I can compare tool accuracy and speed.
```

## Suggested Codex usage

Ask Codex something like:

```text
Use local_agent in research mode to inspect repo=minoflux and find the likely
performance bottleneck. Do not solve it yourself unless the local worker is
blocked. Return the local worker's conclusion and then decide whether a stronger
Codex pass is necessary.
```

For implementation:

```text
Delegate this implementation attempt to local_agent with mode=code and
repo=minoflux. Let the local worker create an isolated job, make one focused
change, run configured tests/benchmark, and report the result. Review its final
diff/result before deciding what to do next.
```

## Why this is not `codex --oss`

`codex --oss` points the **whole Codex harness** at a local model. This project instead keeps the strong OpenAI Codex model as the parent and exposes the local model as a callable subordinate worker.

## Security boundaries

- LM Studio URL and MCP integration remain local configuration.
- model switching is restricted to exact model ids currently returned by LM Studio unless the local CLI is explicitly run with `--force`.
- remote LM Studio hosts are rejected by default.
- research mode exposes only LocalDev read tools.
- code mode still writes only through LocalDev's managed job/worktree boundary.
- arbitrary shell is not exposed by this project.
- local reasoning and raw tool output are stripped before results return to Codex.
- local worker concurrency defaults to 1 to avoid one GPU being thrashed by parallel parent subagents.

## Codex environment forwarding

The installer adds `env_vars = ["LM_API_TOKEN"]` to the Codex MCP entry.
Set `LM_API_TOKEN` before launching/restarting Codex. The token is not stored in this repository.

If a local-agent request fails, the MCP tool returns the LM Studio HTTP error text in a structured `status=error` result instead of only surfacing a generic tool-execution failure.
