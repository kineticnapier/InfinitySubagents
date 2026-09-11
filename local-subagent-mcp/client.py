from __future__ import annotations

import json
import os
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.toml"

READ_TOOLS = [
    "list_repos",
    "repo_info",
    "list_jobs",
    "job_info",
    "list_files",
    "read_file",
    "search_text",
    "git_status",
    "git_diff",
]

CODE_TOOLS = READ_TOOLS + [
    "create_job",
    "create_checkpoint",
    "apply_patch",
    "run_tests",
    "run_benchmark",
    "revert_to_checkpoint",
    "commit_job",
]


@dataclass(frozen=True)
class Config:
    base_url: str
    model: str
    integration_id: str
    timeout_seconds: int
    context_length: int
    max_output_tokens: int
    temperature: float
    reasoning: str | None
    token_env: str | None
    max_parallel: int
    max_task_chars: int
    max_result_chars: int
    allow_remote_lmstudio: bool


def load_config(path: Path = CONFIG_PATH) -> Config:
    if not path.exists():
        raise RuntimeError(
            f"Missing {path.name}. Copy config.example.toml to config.toml and edit it."
        )
    with path.open("rb") as f:
        raw = tomllib.load(f)
    s = raw.get("lmstudio", {})
    model = str(s.get("model", "")).strip()
    if not model or model == "CHANGE_ME":
        raise RuntimeError("Set [lmstudio].model in config.toml to your LM Studio model id")

    reasoning = str(s.get("reasoning", "")).strip().lower() or None
    if reasoning not in {None, "off", "low", "medium", "high", "on"}:
        raise RuntimeError("reasoning must be off/low/medium/high/on or empty")

    cfg = Config(
        base_url=str(s.get("base_url", "http://127.0.0.1:1234")).rstrip("/"),
        model=model,
        integration_id=str(s.get("integration_id", "mcp/localdev")),
        timeout_seconds=int(s.get("timeout_seconds", 900)),
        context_length=int(s.get("context_length", 8192)),
        max_output_tokens=int(s.get("max_output_tokens", 4096)),
        temperature=float(s.get("temperature", 0.2)),
        reasoning=reasoning,
        token_env=(str(s.get("token_env", "LM_API_TOKEN")).strip() or None),
        max_parallel=max(1, int(s.get("max_parallel", 1))),
        max_task_chars=max(1000, int(s.get("max_task_chars", 50000))),
        max_result_chars=max(1000, int(s.get("max_result_chars", 20000))),
        allow_remote_lmstudio=bool(s.get("allow_remote_lmstudio", False)),
    )
    _validate_base_url(cfg.base_url, cfg.allow_remote_lmstudio)
    return cfg


def _validate_base_url(base_url: str, allow_remote: bool) -> None:
    u = urllib.parse.urlparse(base_url)
    if u.scheme not in {"http", "https"}:
        raise RuntimeError("LM Studio base_url must use http or https")
    if not u.hostname:
        raise RuntimeError("LM Studio base_url is missing a hostname")
    if not allow_remote and u.hostname.lower() not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            "Remote LM Studio hosts are disabled. Set allow_remote_lmstudio=true only if intended."
        )


def _request_json(
    cfg: Config,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{cfg.base_url}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cfg.token_env:
        token = os.getenv(cfg.token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio HTTP {e.code}: {payload[:4000]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach LM Studio at {cfg.base_url}: {e}") from e
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError("LM Studio returned invalid JSON") from e


def build_system_prompt(mode: str, repo: str | None, job: str | None) -> str:
    if mode not in {"research", "code"}:
        raise ValueError("mode must be 'research' or 'code'")
    target = []
    if repo:
        target.append(f"Repository alias: {repo}")
    if job:
        target.append(f"Existing managed job: {job}")
    target_text = "\n".join(target) if target else "No target alias was supplied; inspect available repositories if needed."

    if mode == "research":
        behavior = (
            "You are a read-only local subagent. Investigate the task using LocalDev MCP tools. "
            "Do not modify files, create jobs, commit, or attempt shell execution."
        )
    else:
        behavior = (
            "You are a bounded local coding subagent. Use LocalDev MCP for repository work. "
            "Write only in managed job worktrees. If no job is supplied and a change is needed, "
            "create a job first. Before meaningful edits, create a checkpoint when possible. "
            "After editing, run configured tests and benchmark when relevant, inspect the diff, "
            "and commit only when the task is complete and checks are acceptable."
        )

    return f"""{behavior}

{target_text}

Rules:
- Never invent tool results.
- Do not ask for an arbitrary shell tool; it is intentionally unavailable.
- If a tool or configured command is unavailable, report that as a blocker instead of pretending it ran.
- If the same approach fails twice, stop repeating it and reconsider assumptions or choose a different approach.
- Keep the final answer compact. Report: status, what you found/changed, tests, benchmark (if any), and remaining risks/blockers.
- The parent Codex agent receives your final answer and a compact tool trace; internal reasoning is intentionally not forwarded.
"""


def build_chat_request(
    cfg: Config,
    *,
    task: str,
    mode: str,
    repo: str | None = None,
    job: str | None = None,
    previous_response_id: str | None = None,
    max_output_tokens: int | None = None,
    context_length: int | None = None,
) -> dict[str, Any]:
    if len(task) > cfg.max_task_chars:
        raise ValueError(f"Task exceeds {cfg.max_task_chars} characters")
    if mode not in {"research", "code"}:
        raise ValueError("mode must be 'research' or 'code'")
    tools = _tools_for_mode(mode)
    out_cap = cfg.max_output_tokens if max_output_tokens is None else int(max_output_tokens)
    ctx = cfg.context_length if context_length is None else int(context_length)
    if out_cap < 1 or out_cap > cfg.max_output_tokens:
        raise ValueError(f"max_output_tokens must be between 1 and {cfg.max_output_tokens}")
    if ctx < 2048 or ctx > cfg.context_length:
        raise ValueError(f"context_length must be between 2048 and {cfg.context_length}")

    body: dict[str, Any] = {
        "model": cfg.model,
        "input": task,
        "system_prompt": build_system_prompt(mode, repo, job),
        "integrations": [
            {
                "type": "plugin",
                "id": cfg.integration_id,
                "allowed_tools": tools,
            }
        ],
        "context_length": ctx,
        "max_output_tokens": out_cap,
        "temperature": cfg.temperature,
        "store": True,
        "stream": False,
    }
    if cfg.reasoning:
        body["reasoning"] = cfg.reasoning
    if previous_response_id:
        if not previous_response_id.startswith("resp_"):
            raise ValueError("previous_response_id must start with 'resp_'")
        body["previous_response_id"] = previous_response_id
    return body


def compact_response(cfg: Config, payload: dict[str, Any]) -> dict[str, Any]:
    messages: list[str] = []
    tool_names: list[str] = []
    invalid_calls: list[dict[str, Any]] = []

    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        typ = item.get("type")
        if typ == "message":
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                messages.append(content.strip())
        elif typ == "tool_call":
            name = str(item.get("tool", "?"))
            tool_names.append(name)
        elif typ == "invalid_tool_call":
            invalid_calls.append(
                {
                    "tool": item.get("tool_name") or item.get("metadata", {}).get("tool_name"),
                    "reason": item.get("reason"),
                }
            )
        # Intentionally drop `reasoning` and tool outputs. They can be huge and the
        # parent agent only needs the worker's result plus a compact execution trace.

    final_message = messages[-1] if messages else ""
    if len(final_message) > cfg.max_result_chars:
        final_message = final_message[: cfg.max_result_chars] + "\n...[truncated by local-subagent MCP]"

    counts: dict[str, int] = {}
    for name in tool_names:
        counts[name] = counts.get(name, 0) + 1

    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    return {
        "status": "ok" if final_message else "no_final_message",
        "response_id": payload.get("response_id"),
        "model_instance_id": payload.get("model_instance_id"),
        "final": final_message,
        "tool_calls": counts,
        "invalid_tool_calls": invalid_calls[:20],
        "stats": {
            k: stats.get(k)
            for k in (
                "input_tokens",
                "total_output_tokens",
                "reasoning_output_tokens",
                "tokens_per_second",
                "time_to_first_token_seconds",
                "model_load_time_seconds",
            )
            if k in stats
        },
    }


class LocalAgentClient:
    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self._cfg = load_config(config_path)
        self._sem = threading.BoundedSemaphore(self._cfg.max_parallel)
        self._mtime_ns = config_path.stat().st_mtime_ns
        self._guard = threading.Lock()

    @property
    def config(self) -> Config:
        self._maybe_reload()
        return self._cfg

    def _maybe_reload(self) -> None:
        with self._guard:
            try:
                mtime = self.config_path.stat().st_mtime_ns
            except OSError:
                return
            if mtime != self._mtime_ns:
                cfg = load_config(self.config_path)
                self._cfg = cfg
                self._sem = threading.BoundedSemaphore(cfg.max_parallel)
                self._mtime_ns = mtime

    def health(self) -> dict[str, Any]:
        cfg = self.config
        started = time.monotonic()
        payload = _request_json(cfg, "GET", "/api/v1/models")
        return {
            "ok": True,
            "base_url": cfg.base_url,
            "configured_model": cfg.model,
            "integration_id": cfg.integration_id,
            "latency_seconds": round(time.monotonic() - started, 3),
            "models_payload": payload,
        }

    def run(
        self,
        *,
        task: str,
        mode: str,
        repo: str | None = None,
        job: str | None = None,
        previous_response_id: str | None = None,
        max_output_tokens: int | None = None,
        context_length: int | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        body = build_chat_request(
            cfg,
            task=task,
            mode=mode,
            repo=repo,
            job=job,
            previous_response_id=previous_response_id,
            max_output_tokens=max_output_tokens,
            context_length=context_length,
        )
        started = time.monotonic()
        with self._sem:
            payload = _request_json(cfg, "POST", "/api/v1/chat", body)
        result = compact_response(cfg, payload)
        result["wall_seconds"] = round(time.monotonic() - started, 3)
        result["mode"] = mode
        result["repo"] = repo
        result["job"] = job
        return result
