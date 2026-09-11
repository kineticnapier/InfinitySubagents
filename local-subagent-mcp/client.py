from __future__ import annotations

import hashlib
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
    "find_files",
    "read_file",
    "search_text",
    "git_status",
    "git_diff",
]

CODE_TOOLS = READ_TOOLS + [
    "create_job",
    "create_checkpoint",
    "write_text_file",
    "replace_text",
    "apply_patch",
    "run_tests",
    "run_benchmark",
    "revert_to_latest_checkpoint",
    "revert_to_checkpoint",
    "commit_job",
]

_TRACE_MAX_ITEMS = 64
_TRACE_STRING_LIMIT = 240
_TRACE_PREVIEW_LIMIT = 800


def _tools_for_mode(mode: str) -> list[str]:
    if mode == "research":
        return list(READ_TOOLS)
    if mode == "code":
        return list(CODE_TOOLS)
    raise ValueError("mode must be 'research' or 'code'")


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
- Before guessing a file or path, inspect it with list_files or find_files.
- Treat a tool result with ok=false as a real failure. Report its returned error_type/error; never invent an exception class.
- Do not ask for an arbitrary shell tool; it is intentionally unavailable.
- If a tool or configured command is unavailable, report that as a blocker instead of pretending it ran.
- For a simple new text file prefer write_text_file. For one exact replacement in an existing text file prefer replace_text. Reserve apply_patch for edits that actually need a unified diff.
- Prefer revert_to_latest_checkpoint unless an exact older checkpoint is explicitly required.
- After any write/edit tool error, inspect git_status and git_diff before retrying the same write.
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


def _short_string(value: Any, limit: int = _TRACE_STRING_LIMIT) -> str:
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def _hash_text(value: Any) -> tuple[str, str]:
    text = value if isinstance(value, str) else str(value)
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return text, digest


def _compact_tool_arguments(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}
    result: dict[str, Any] = {}
    safe_keys = {
        "repo",
        "job",
        "path",
        "name",
        "query",
        "pattern",
        "base_ref",
        "checkpoint",
        "start_line",
        "end_line",
        "recursive",
        "case_sensitive",
        "staged",
        "force",
        "overwrite",
    }
    for key, value in arguments.items():
        if key == "patch":
            text, digest = _hash_text(value)
            result["patch_chars"] = len(text)
            result["patch_lines"] = text.count("\n") + (1 if text else 0)
            result["patch_sha256"] = digest
        elif key in {"content", "old", "new"}:
            text, digest = _hash_text(value)
            result[f"{key}_chars"] = len(text)
            result[f"{key}_sha256"] = digest
            if key == "content":
                result["content_ends_with_newline"] = text.endswith("\n")
        elif key == "message":
            result["message"] = _short_string(value, 120)
        elif key in safe_keys:
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[key] = _short_string(value) if isinstance(value, str) else value
            elif isinstance(value, list):
                result[key] = [_short_string(v, 80) for v in value[:8]]
    return result


def _decode_tool_output(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text

    # MCP transports commonly wrap a JSON result in [{"type":"text","text":"..."}].
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        inner = parsed[0]
        if inner.get("type") == "text" and isinstance(inner.get("text"), str):
            nested = inner["text"].strip()
            try:
                return json.loads(nested)
            except json.JSONDecodeError:
                return nested
    return parsed


def _error_text(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("error", "message", "detail"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                low = candidate.lower()
                if any(token in low for token in ("error", "failed", "failure", "exception", "does not exist", "not found")):
                    return candidate
        if value.get("isError") is True or value.get("is_error") is True:
            return _short_string(value, 400)
        return None
    if isinstance(value, str):
        low = value.lower()
        markers = (
            "unexpectedtoolerror",
            "error executing tool",
            "tool call failed",
            "traceback",
            "exception:",
            "file does not exist",
            "path does not exist",
        )
        if any(marker in low for marker in markers):
            return value
    return None


def _preview(text: Any, limit: int = _TRACE_PREVIEW_LIMIT) -> str:
    if text is None:
        return ""
    return _short_string(text, limit)


def _compact_tool_output(tool: str, raw: Any) -> dict[str, Any]:
    decoded = _decode_tool_output(raw)

    if isinstance(decoded, dict) and decoded.get("ok") is False:
        summary: dict[str, Any] = {"outcome": "error"}
        if decoded.get("error_type") is not None:
            summary["error_type"] = _short_string(decoded.get("error_type"), 120)
        if decoded.get("error") is not None:
            summary["error"] = _preview(decoded.get("error"), 400)
        for key in ("repo", "job", "path", "checkpoint", "matches"):
            if key in decoded:
                value = decoded.get(key)
                summary[key] = _short_string(value, 240) if isinstance(value, str) else value
        return summary

    err = _error_text(decoded)
    if err is not None:
        return {"outcome": "error", "error": _preview(err, 400)}

    if not isinstance(decoded, dict):
        text = decoded if isinstance(decoded, str) else json.dumps(decoded, ensure_ascii=False, default=str)
        return {
            "outcome": "unknown",
            "output_chars": len(text),
            "output_preview": _preview(text, 300),
        }

    summary = {"outcome": "ok"}
    if "exit_code" in decoded:
        summary["exit_code"] = decoded.get("exit_code")
        summary["outcome"] = "ok" if decoded.get("exit_code") == 0 else "error"
    if "truncated" in decoded:
        summary["truncated"] = bool(decoded.get("truncated"))

    if tool == "list_repos":
        repos = decoded.get("repos")
        if isinstance(repos, list):
            summary["repos"] = [
                item.get("name") for item in repos[:12] if isinstance(item, dict) and item.get("name")
            ]
    elif tool in {
        "repo_info",
        "job_info",
        "create_job",
        "create_checkpoint",
        "revert_to_checkpoint",
        "revert_to_latest_checkpoint",
        "commit_job",
        "apply_patch",
        "write_text_file",
        "replace_text",
    }:
        for key in (
            "name",
            "job",
            "source_repo",
            "path",
            "branch",
            "base_ref",
            "head",
            "checkpoint",
            "commit",
            "paths",
            "diff_stat",
            "status",
            "chars",
            "bytes",
            "replacements",
            "ends_with_newline",
            "latest",
        ):
            if key in decoded:
                value = decoded.get(key)
                if isinstance(value, str):
                    summary[key] = _short_string(value, 300)
                elif isinstance(value, list):
                    summary[key] = [_short_string(v, 120) for v in value[:12]]
                elif isinstance(value, (int, float, bool)) or value is None:
                    summary[key] = value
    elif tool == "list_jobs":
        jobs = decoded.get("jobs")
        if isinstance(jobs, list):
            summary["jobs"] = [
                item.get("name") for item in jobs[:12] if isinstance(item, dict) and item.get("name")
            ]
    elif tool == "list_files":
        entries = decoded.get("entries")
        summary["target"] = decoded.get("target")
        summary["base"] = decoded.get("base")
        if isinstance(entries, list):
            summary["entry_count"] = len(entries)
            summary["entries"] = [
                {"path": _short_string(item.get("path"), 160), "type": item.get("type")}
                for item in entries[:12]
                if isinstance(item, dict)
            ]
    elif tool == "find_files":
        summary["target"] = decoded.get("target")
        summary["pattern"] = _short_string(decoded.get("pattern"), 180)
        results = decoded.get("results")
        if isinstance(results, list):
            summary["result_count"] = len(results)
            summary["matches"] = [
                _short_string(item.get("path"), 180)
                for item in results[:12]
                if isinstance(item, dict) and item.get("path")
            ]
    elif tool == "read_file":
        for key in (
            "target",
            "path",
            "start_line",
            "end_line",
            "total_lines",
            "ends_with_newline",
            "source_ends_with_newline",
        ):
            if key in decoded:
                summary[key] = decoded.get(key)
        content = decoded.get("content")
        if isinstance(content, str):
            summary["content_chars"] = len(content)
            summary["content_sha256"] = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
            summary["content_preview"] = _preview(content, 360)
    elif tool == "search_text":
        for key in ("target", "query"):
            if key in decoded:
                summary[key] = _short_string(decoded.get(key), 180)
        results = decoded.get("results")
        if isinstance(results, list):
            summary["result_count"] = len(results)
            summary["matches"] = [
                {"path": _short_string(item.get("path"), 160), "line": item.get("line")}
                for item in results[:12]
                if isinstance(item, dict)
            ]
    elif tool in {"git_status", "git_diff", "run_tests", "run_benchmark"}:
        stdout = decoded.get("stdout")
        stderr = decoded.get("stderr")
        if isinstance(stdout, str):
            summary["stdout_chars"] = len(stdout)
            summary["stdout_sha256"] = hashlib.sha256(stdout.encode("utf-8", errors="replace")).hexdigest()[:16]
            summary["stdout_preview"] = _preview(stdout, 1200 if tool == "git_diff" else 800)
        if isinstance(stderr, str) and stderr:
            summary["stderr_chars"] = len(stderr)
            summary["stderr_preview"] = _preview(stderr, 500)
    else:
        summary["keys"] = sorted(str(key) for key in decoded.keys())[:20]

    return summary


def compact_response(cfg: Config, payload: dict[str, Any]) -> dict[str, Any]:
    messages: list[str] = []
    tool_names: list[str] = []
    tool_trace: list[dict[str, Any]] = []
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
            if len(tool_trace) < _TRACE_MAX_ITEMS:
                tool_trace.append(
                    {
                        "tool": name,
                        "arguments": _compact_tool_arguments(item.get("arguments")),
                        **_compact_tool_output(name, item.get("output")),
                    }
                )
        elif typ == "invalid_tool_call":
            invalid_calls.append(
                {
                    "tool": item.get("tool_name") or item.get("metadata", {}).get("tool_name"),
                    "reason": item.get("reason"),
                }
            )
        # Intentionally drop hidden reasoning and full raw tool outputs. The parent
        # receives a bounded audit trace containing arguments and compact summaries.

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
        "tool_trace": tool_trace,
        "tool_trace_truncated": len(tool_names) > len(tool_trace),
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
