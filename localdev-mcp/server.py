from __future__ import annotations

import fnmatch
import time
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from core import (
    State,
    apply_patch as core_apply_patch,
    commit_job as core_commit_job,
    create_checkpoint as core_create_checkpoint,
    create_job as core_create_job,
    delete_job as core_delete_job,
    git,
    git_text,
    is_probably_binary,
    iter_files,
    rel,
    resolve_target,
    revert_to_checkpoint as core_revert_to_checkpoint,
    safe_path,
)


state = State()
mcp = MCPServer("LocalDev MCP")

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
LOCAL_EXEC = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
DESTRUCTIVE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=False,
)


def _error(error_type: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error_type": error_type, "error": message, **extra}


def _resolve_read_target(
    *, repo: str | None, job: str | None, path: str
) -> tuple[str, Path, Path] | dict[str, Any]:
    try:
        name, root, _, _ = resolve_target(state, repo=repo, job=job)
        target = safe_path(root, path)
    except (ValueError, RuntimeError) as e:
        return _error("invalid_target", str(e), path=path)
    return name, root, target


def _managed_text_target(job: str, path: str) -> tuple[Any, Path, Path] | dict[str, Any]:
    try:
        j = state.job(job)
        root = Path(j.path).resolve()
        target = safe_path(root, path)
    except (ValueError, RuntimeError) as e:
        return _error("invalid_target", str(e), job=job, path=path)

    parts = Path(path.replace("\\", "/")).parts
    if not parts or any(part.lower() == ".git" for part in parts):
        return _error("forbidden_path", ".git paths are not writable", job=job, path=path)
    return j, root, target


@mcp.tool(annotations=READ_ONLY)
def list_repos() -> dict[str, Any]:
    return {
        "ok": True,
        "repos": [
            {
                "name": repo.name,
                "path": str(repo.root),
                "exists": repo.root.is_dir(),
                "is_git_repo": (repo.root / ".git").exists(),
                "test_configured": repo.test_command is not None,
                "benchmark_configured": repo.benchmark_command is not None,
            }
            for repo in state.list_repos()
        ],
    }


@mcp.tool(annotations=READ_ONLY)
def repo_info(repo: str) -> dict[str, Any]:
    try:
        r = state.repo(repo)
    except (ValueError, RuntimeError) as e:
        return _error("unknown_repo", str(e), repo=repo)
    return {
        "ok": True,
        "name": r.name,
        "path": str(r.root),
        "exists": r.root.is_dir(),
        "is_git_repo": (r.root / ".git").exists(),
        "test_command": list(r.test_command) if r.test_command else None,
        "benchmark_command": list(r.benchmark_command) if r.benchmark_command else None,
    }


@mcp.tool(annotations=READ_ONLY)
def list_jobs(repo: str | None = None) -> dict[str, Any]:
    jobs = []
    for j in state.list_jobs():
        if repo is not None and j.source_repo != repo:
            continue
        p = Path(j.path)
        jobs.append(
            {
                "name": j.name,
                "source_repo": j.source_repo,
                "path": j.path,
                "branch": j.branch,
                "base_ref": j.base_ref,
                "exists": p.is_dir(),
                "checkpoints": list(j.checkpoints),
            }
        )
    return {"ok": True, "jobs": jobs}


@mcp.tool(annotations=READ_ONLY)
def job_info(job: str) -> dict[str, Any]:
    try:
        j = state.job(job)
    except ValueError as e:
        return _error("unknown_job", str(e), job=job)
    p = Path(j.path)
    head = git_text(p, ["rev-parse", "HEAD"]) if p.is_dir() else None
    return {
        "ok": True,
        "name": j.name,
        "source_repo": j.source_repo,
        "path": j.path,
        "branch": j.branch,
        "base_ref": j.base_ref,
        "head": head,
        "exists": p.is_dir(),
        "checkpoints": dict(j.checkpoints),
    }


@mcp.tool(annotations=WRITE)
def create_job(repo: str, name: str, base_ref: str | None = None) -> dict[str, Any]:
    """Create an isolated git worktree and dedicated localdev/* branch."""
    try:
        return {"ok": True, **core_create_job(state, repo, name, base_ref)}
    except (ValueError, RuntimeError) as e:
        return _error("create_job_failed", str(e), repo=repo, job=name)


@mcp.tool(annotations=DESTRUCTIVE)
def delete_job(job: str, force: bool = False) -> dict[str, Any]:
    """Remove a managed worktree. Source repository files are not deleted."""
    try:
        return {"ok": True, **core_delete_job(state, job, force)}
    except (ValueError, RuntimeError) as e:
        return _error("delete_job_failed", str(e), job=job)


@mcp.tool(annotations=READ_ONLY)
def list_files(
    repo: str | None = None,
    job: str | None = None,
    path: str = ".",
    recursive: bool = False,
) -> dict[str, Any]:
    resolved = _resolve_read_target(repo=repo, job=job, path=path)
    if isinstance(resolved, dict):
        return resolved
    name, root, base = resolved
    if not base.exists():
        return _error("path_not_found", "Path does not exist", target=name, path=path)
    if not base.is_dir():
        return _error("not_directory", "Path is not a directory", target=name, path=path)

    cfg = state.server
    entries: list[dict[str, str]] = []
    if recursive:
        for item in iter_files(root, base, cfg.ignore_dirs):
            entries.append({"path": rel(root, item), "type": "file"})
            if len(entries) >= cfg.max_list_entries:
                break
    else:
        for item in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                item.resolve().relative_to(root.resolve())
            except (OSError, ValueError):
                continue
            entries.append(
                {
                    "path": rel(root, item),
                    "type": "directory" if item.is_dir() else "file",
                }
            )
            if len(entries) >= cfg.max_list_entries:
                break

    return {
        "ok": True,
        "target": name,
        "base": rel(root, base),
        "entries": entries,
        "truncated": len(entries) >= cfg.max_list_entries,
    }


@mcp.tool(annotations=READ_ONLY)
def find_files(
    pattern: str,
    repo: str | None = None,
    job: str | None = None,
    path: str = ".",
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """Find file paths by glob-like pattern without guessing repository layout."""
    if not pattern:
        return _error("invalid_pattern", "pattern must not be empty")
    resolved = _resolve_read_target(repo=repo, job=job, path=path)
    if isinstance(resolved, dict):
        return resolved
    name, root, base = resolved
    if not base.exists():
        return _error("path_not_found", "Path does not exist", target=name, path=path)
    if not base.is_dir():
        return _error("not_directory", "Path is not a directory", target=name, path=path)

    cfg = state.server
    needle = pattern if case_sensitive else pattern.casefold()
    match_full_path = "/" in pattern or "\\" in pattern
    results: list[dict[str, str]] = []
    for file in iter_files(root, base, cfg.ignore_dirs):
        relative = rel(root, file)
        candidate = relative if match_full_path else file.name
        haystack = candidate if case_sensitive else candidate.casefold()
        if fnmatch.fnmatchcase(haystack, needle):
            results.append({"path": relative})
            if len(results) >= cfg.max_search_results:
                return {
                    "ok": True,
                    "target": name,
                    "pattern": pattern,
                    "results": results,
                    "truncated": True,
                }
    return {
        "ok": True,
        "target": name,
        "pattern": pattern,
        "results": results,
        "truncated": False,
    }


@mcp.tool(annotations=READ_ONLY)
def read_file(
    path: str,
    repo: str | None = None,
    job: str | None = None,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict[str, Any]:
    resolved = _resolve_read_target(repo=repo, job=job, path=path)
    if isinstance(resolved, dict):
        return resolved
    name, root, target = resolved
    if not target.exists():
        return _error("file_not_found", "File does not exist", target=name, path=path)
    if not target.is_file():
        return _error("not_file", "Path is not a file", target=name, path=path)

    cfg = state.server
    if target.stat().st_size > cfg.max_read_bytes:
        return _error("file_too_large", "File too large", target=name, path=path)
    if is_probably_binary(target):
        return _error("binary_file", "Binary file reading is disabled", target=name, path=path)

    raw_text = target.read_text(encoding="utf-8", errors="replace")
    lines = raw_text.splitlines()
    start_line = max(1, start_line)
    if end_line is not None and end_line < start_line:
        return _error(
            "invalid_range",
            "end_line must be >= start_line",
            target=name,
            path=path,
            start_line=start_line,
            end_line=end_line,
        )
    start = start_line - 1
    end = len(lines) if end_line is None else min(end_line, len(lines))
    selected = lines[start:end]
    content = "\n".join(selected)
    if selected and end == len(lines) and raw_text.endswith("\n"):
        content += "\n"
    return {
        "ok": True,
        "target": name,
        "path": rel(root, target),
        "start_line": start + 1,
        "end_line": start + len(selected),
        "total_lines": len(lines),
        "content": content,
        "ends_with_newline": content.endswith("\n"),
        "source_ends_with_newline": raw_text.endswith("\n"),
    }


@mcp.tool(annotations=READ_ONLY)
def search_text(
    query: str,
    repo: str | None = None,
    job: str | None = None,
    path: str = ".",
    case_sensitive: bool = False,
) -> dict[str, Any]:
    if not query:
        return _error("invalid_query", "query must not be empty")
    resolved = _resolve_read_target(repo=repo, job=job, path=path)
    if isinstance(resolved, dict):
        return resolved
    name, root, base = resolved
    if not base.exists():
        return _error("path_not_found", "Path does not exist", target=name, path=path)

    cfg = state.server
    needle = query if case_sensitive else query.casefold()
    results: list[dict[str, Any]] = []
    for file in iter_files(root, base, cfg.ignore_dirs):
        try:
            if file.stat().st_size > cfg.max_read_bytes or is_probably_binary(file):
                continue
            lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, 1):
            haystack = line if case_sensitive else line.casefold()
            if needle in haystack:
                results.append({"path": rel(root, file), "line": line_no, "text": line[:1000]})
                if len(results) >= cfg.max_search_results:
                    return {
                        "ok": True,
                        "target": name,
                        "query": query,
                        "results": results,
                        "truncated": True,
                    }
    return {
        "ok": True,
        "target": name,
        "query": query,
        "results": results,
        "truncated": False,
    }


@mcp.tool(annotations=READ_ONLY)
def git_status(repo: str | None = None, job: str | None = None) -> dict[str, Any]:
    try:
        _, root, _, j = resolve_target(state, repo=repo, job=job)
    except (ValueError, RuntimeError) as e:
        return _error("invalid_target", str(e))
    cfg = state.server
    lock = state.job_lock(j.name) if j else state.repo_lock(str(repo))
    with lock:
        result = git(
            root,
            ["status", "--short", "--branch"],
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
        )
    return {"ok": result.get("exit_code") == 0, **result}


@mcp.tool(annotations=READ_ONLY)
def git_diff(
    repo: str | None = None,
    job: str | None = None,
    staged: bool = False,
) -> dict[str, Any]:
    try:
        _, root, _, j = resolve_target(state, repo=repo, job=job)
    except (ValueError, RuntimeError) as e:
        return _error("invalid_target", str(e))
    cfg = state.server
    lock = state.job_lock(j.name) if j else state.repo_lock(str(repo))
    command = ["diff"]
    if staged:
        command.append("--cached")
    command.append("--")
    with lock:
        result = git(
            root,
            command,
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
        )
    return {"ok": result.get("exit_code") == 0, **result}


def _run_fixed(job: str, kind: str) -> dict[str, Any]:
    try:
        j = state.job(job)
        repo = state.repo(j.source_repo)
    except (ValueError, RuntimeError) as e:
        return _error("invalid_job", str(e), job=job)
    command = repo.test_command if kind == "test" else repo.benchmark_command
    if not command:
        return _error("command_not_configured", f"{kind} command is not configured", job=job)
    cfg = state.server
    started = time.monotonic()
    root = Path(j.path)
    with state.job_lock(job):
        before = git_text(root, ["rev-parse", "HEAD"])
        result = __import__("core").run_process(
            list(command),
            cwd=root,
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
        )
        after = git_text(root, ["rev-parse", "HEAD"])
    state.log(
        f"run_{kind}",
        repo=j.source_repo,
        job=job,
        success=result["exit_code"] == 0,
        duration=time.monotonic() - started,
        before_head=before,
        after_head=after,
    )
    return {"ok": result.get("exit_code") == 0, **result}


@mcp.tool(annotations=LOCAL_EXEC)
def run_tests(job: str) -> dict[str, Any]:
    """Run the human-configured test command in a job worktree."""
    return _run_fixed(job, "test")


@mcp.tool(annotations=LOCAL_EXEC)
def run_benchmark(job: str) -> dict[str, Any]:
    """Run the human-configured benchmark command in a job worktree."""
    return _run_fixed(job, "benchmark")


@mcp.tool(annotations=WRITE)
def write_text_file(
    job: str,
    path: str,
    content: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write exact UTF-8 text inside a managed job. Existing files require overwrite=true."""
    resolved = _managed_text_target(job, path)
    if isinstance(resolved, dict):
        return resolved
    j, root, target = resolved
    cfg = state.server
    encoded = content.encode("utf-8")
    if len(encoded) > cfg.max_patch_bytes:
        return _error("content_too_large", "Text content exceeds max_patch_bytes", job=job, path=path)

    started = time.monotonic()
    with state.job_lock(job):
        if not target.parent.is_dir():
            return _error("parent_not_found", "Parent directory does not exist", job=job, path=path)
        if target.exists():
            if not target.is_file():
                return _error("not_file", "Target path is not a file", job=job, path=path)
            if is_probably_binary(target):
                return _error("binary_file", "Binary files cannot be overwritten", job=job, path=path)
            if not overwrite:
                return _error("file_exists", "File already exists; set overwrite=true to replace it", job=job, path=path)

        before = git_text(root, ["rev-parse", "HEAD"])
        with target.open("w", encoding="utf-8", newline="") as f:
            f.write(content)
        relative = rel(root, target)
        status = git_text(root, ["status", "--short", "--", relative])
        state.log(
            "write_text_file",
            repo=j.source_repo,
            job=job,
            success=True,
            duration=time.monotonic() - started,
            before_head=before,
            after_head=before,
            detail=relative,
        )
    return {
        "ok": True,
        "job": job,
        "path": relative,
        "chars": len(content),
        "bytes": len(encoded),
        "ends_with_newline": content.endswith("\n"),
        "status": status,
    }


@mcp.tool(annotations=WRITE)
def replace_text(job: str, path: str, old: str, new: str) -> dict[str, Any]:
    """Replace exactly one occurrence in an existing UTF-8 text file inside a managed job."""
    if not old:
        return _error("invalid_old_text", "old text must not be empty", job=job, path=path)
    resolved = _managed_text_target(job, path)
    if isinstance(resolved, dict):
        return resolved
    j, root, target = resolved
    cfg = state.server

    started = time.monotonic()
    with state.job_lock(job):
        if not target.exists():
            return _error("file_not_found", "File does not exist", job=job, path=path)
        if not target.is_file():
            return _error("not_file", "Path is not a file", job=job, path=path)
        if target.stat().st_size > cfg.max_read_bytes:
            return _error("file_too_large", "File too large", job=job, path=path)
        if is_probably_binary(target):
            return _error("binary_file", "Binary file editing is disabled", job=job, path=path)

        text = target.read_text(encoding="utf-8", errors="strict")
        count = text.count(old)
        if count == 0:
            return _error("text_not_found", "old text was not found", job=job, path=path)
        if count != 1:
            return _error(
                "ambiguous_match",
                f"old text occurs {count} times; expected exactly one occurrence",
                job=job,
                path=path,
                matches=count,
            )
        updated = text.replace(old, new, 1)
        encoded = updated.encode("utf-8")
        if len(encoded) > cfg.max_patch_bytes:
            return _error("content_too_large", "Updated file exceeds max_patch_bytes", job=job, path=path)

        before = git_text(root, ["rev-parse", "HEAD"])
        with target.open("w", encoding="utf-8", newline="") as f:
            f.write(updated)
        relative = rel(root, target)
        diff_stat = git_text(root, ["diff", "--stat", "--", relative])
        status = git_text(root, ["status", "--short", "--", relative])
        state.log(
            "replace_text",
            repo=j.source_repo,
            job=job,
            success=True,
            duration=time.monotonic() - started,
            before_head=before,
            after_head=before,
            detail=relative,
        )
    return {
        "ok": True,
        "job": job,
        "path": relative,
        "replacements": 1,
        "chars": len(updated),
        "ends_with_newline": updated.endswith("\n"),
        "diff_stat": diff_stat,
        "status": status,
    }


@mcp.tool(annotations=WRITE)
def apply_patch(job: str, patch: str) -> dict[str, Any]:
    """Apply a checked unified diff. Prefer write_text_file/replace_text for simple edits."""
    try:
        return {"ok": True, **core_apply_patch(state, job, patch)}
    except (ValueError, RuntimeError) as e:
        return _error("apply_patch_failed", str(e), job=job)


@mcp.tool(annotations=WRITE)
def create_checkpoint(job: str) -> dict[str, Any]:
    """Create a clean-tree checkpoint for later rollback."""
    try:
        return {"ok": True, **core_create_checkpoint(state, job)}
    except (ValueError, RuntimeError) as e:
        return _error("checkpoint_failed", str(e), job=job)


@mcp.tool(annotations=DESTRUCTIVE)
def revert_to_checkpoint(job: str, checkpoint: str) -> dict[str, Any]:
    """Hard-reset a managed job to an explicit checkpoint and remove untracked files."""
    try:
        return {"ok": True, **core_revert_to_checkpoint(state, job, checkpoint)}
    except (ValueError, RuntimeError) as e:
        return _error("revert_failed", str(e), job=job, checkpoint=checkpoint)


@mcp.tool(annotations=DESTRUCTIVE)
def revert_to_latest_checkpoint(job: str) -> dict[str, Any]:
    """Hard-reset to the most recently created checkpoint; no checkpoint id memorization needed."""
    try:
        j = state.job(job)
    except ValueError as e:
        return _error("unknown_job", str(e), job=job)
    if not j.checkpoints:
        return _error("no_checkpoint", "Job has no checkpoints", job=job)
    checkpoint = next(reversed(j.checkpoints))
    try:
        result = core_revert_to_checkpoint(state, job, checkpoint)
    except (ValueError, RuntimeError) as e:
        return _error("revert_failed", str(e), job=job, checkpoint=checkpoint)
    return {"ok": True, "latest": True, **result}


@mcp.tool(annotations=WRITE)
def commit_job(job: str, message: str) -> dict[str, Any]:
    """Commit job changes on a managed localdev/* branch. Never pushes."""
    try:
        return {"ok": True, **core_commit_job(state, job, message)}
    except (ValueError, RuntimeError) as e:
        return _error("commit_failed", str(e), job=job)


if __name__ == "__main__":
    mcp.run()
