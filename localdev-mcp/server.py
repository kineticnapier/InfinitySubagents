from __future__ import annotations

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


@mcp.tool(annotations=READ_ONLY)
def list_repos() -> dict[str, Any]:
    return {
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
        ]
    }


@mcp.tool(annotations=READ_ONLY)
def repo_info(repo: str) -> dict[str, Any]:
    r = state.repo(repo)
    return {
        "name": r.name,
        "path": str(r.root),
        "exists": r.root.is_dir(),
        "is_git_repo": (r.root / ".git").exists(),
        "test_command": list(r.test_command) if r.test_command else None,
        "benchmark_command": list(r.benchmark_command)
        if r.benchmark_command
        else None,
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
    return {"jobs": jobs}


@mcp.tool(annotations=READ_ONLY)
def job_info(job: str) -> dict[str, Any]:
    j = state.job(job)
    p = Path(j.path)
    head = git_text(p, ["rev-parse", "HEAD"]) if p.is_dir() else None
    return {
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
    return core_create_job(state, repo, name, base_ref)


@mcp.tool(annotations=DESTRUCTIVE)
def delete_job(job: str, force: bool = False) -> dict[str, Any]:
    """Remove a managed worktree. Source repository files are not deleted."""
    return core_delete_job(state, job, force)


@mcp.tool(annotations=READ_ONLY)
def list_files(
    repo: str | None = None,
    job: str | None = None,
    path: str = ".",
    recursive: bool = False,
) -> dict[str, Any]:
    name, root, _, _ = resolve_target(state, repo=repo, job=job)
    base = safe_path(root, path)
    if not base.exists():
        raise ValueError("Path does not exist")
    if not base.is_dir():
        raise ValueError("Path is not a directory")

    cfg = state.server
    entries: list[dict[str, str]] = []
    if recursive:
        for item in iter_files(root, base, cfg.ignore_dirs):
            entries.append({"path": rel(root, item), "type": "file"})
            if len(entries) >= cfg.max_list_entries:
                break
    else:
        for item in sorted(
            base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        ):
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
        "target": name,
        "base": rel(root, base),
        "entries": entries,
        "truncated": len(entries) >= cfg.max_list_entries,
    }


@mcp.tool(annotations=READ_ONLY)
def read_file(
    path: str,
    repo: str | None = None,
    job: str | None = None,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict[str, Any]:
    name, root, _, _ = resolve_target(state, repo=repo, job=job)
    target = safe_path(root, path)
    if not target.is_file():
        raise ValueError("File does not exist")

    cfg = state.server
    if target.stat().st_size > cfg.max_read_bytes:
        raise ValueError("File too large")
    if is_probably_binary(target):
        raise ValueError("Binary file reading is disabled")

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start_line = max(1, start_line)
    if end_line is not None and end_line < start_line:
        raise ValueError("end_line must be >= start_line")
    start = start_line - 1
    end = len(lines) if end_line is None else min(end_line, len(lines))
    selected = lines[start:end]
    return {
        "target": name,
        "path": rel(root, target),
        "start_line": start + 1,
        "end_line": start + len(selected),
        "total_lines": len(lines),
        "content": "\n".join(selected),
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
        raise ValueError("query must not be empty")

    name, root, _, _ = resolve_target(state, repo=repo, job=job)
    base = safe_path(root, path)
    if not base.exists():
        raise ValueError("Path does not exist")

    cfg = state.server
    needle = query if case_sensitive else query.casefold()
    results: list[dict[str, Any]] = []

    for file in iter_files(root, base, cfg.ignore_dirs):
        try:
            if file.stat().st_size > cfg.max_read_bytes or is_probably_binary(file):
                continue
            lines = file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, 1):
            haystack = line if case_sensitive else line.casefold()
            if needle in haystack:
                results.append(
                    {"path": rel(root, file), "line": line_no, "text": line[:1000]}
                )
                if len(results) >= cfg.max_search_results:
                    return {
                        "target": name,
                        "query": query,
                        "results": results,
                        "truncated": True,
                    }

    return {
        "target": name,
        "query": query,
        "results": results,
        "truncated": False,
    }


@mcp.tool(annotations=READ_ONLY)
def git_status(
    repo: str | None = None,
    job: str | None = None,
) -> dict[str, Any]:
    _, root, _, j = resolve_target(state, repo=repo, job=job)
    cfg = state.server
    lock = state.job_lock(j.name) if j else state.repo_lock(str(repo))
    with lock:
        return git(
            root,
            ["status", "--short", "--branch"],
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
        )


@mcp.tool(annotations=READ_ONLY)
def git_diff(
    repo: str | None = None,
    job: str | None = None,
    staged: bool = False,
) -> dict[str, Any]:
    _, root, _, j = resolve_target(state, repo=repo, job=job)
    cfg = state.server
    lock = state.job_lock(j.name) if j else state.repo_lock(str(repo))
    command = ["diff"]
    if staged:
        command.append("--cached")
    command.append("--")
    with lock:
        return git(
            root,
            command,
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
        )


def _run_fixed(job: str, kind: str) -> dict[str, Any]:
    j = state.job(job)
    repo = state.repo(j.source_repo)
    command = repo.test_command if kind == "test" else repo.benchmark_command
    if not command:
        raise RuntimeError(f"{kind} command is not configured")
    cfg = state.server
    started = time.monotonic()
    root = Path(j.path)
    with state.job_lock(job):
        before = git_text(root, ["rev-parse", "HEAD"])
        result = git_process = __import__("core").run_process(
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
    return result


@mcp.tool(annotations=LOCAL_EXEC)
def run_tests(job: str) -> dict[str, Any]:
    """Run the human-configured test command in a job worktree."""
    return _run_fixed(job, "test")


@mcp.tool(annotations=LOCAL_EXEC)
def run_benchmark(job: str) -> dict[str, Any]:
    """Run the human-configured benchmark command in a job worktree."""
    return _run_fixed(job, "benchmark")


@mcp.tool(annotations=WRITE)
def apply_patch(job: str, patch: str) -> dict[str, Any]:
    """Apply a checked unified diff. Writes are allowed only inside managed jobs."""
    return core_apply_patch(state, job, patch)


@mcp.tool(annotations=WRITE)
def create_checkpoint(job: str) -> dict[str, Any]:
    """Create a clean-tree checkpoint for later rollback."""
    return core_create_checkpoint(state, job)


@mcp.tool(annotations=DESTRUCTIVE)
def revert_to_checkpoint(job: str, checkpoint: str) -> dict[str, Any]:
    """Hard-reset a managed job to a checkpoint and remove untracked files."""
    return core_revert_to_checkpoint(state, job, checkpoint)


@mcp.tool(annotations=WRITE)
def commit_job(job: str, message: str) -> dict[str, Any]:
    """Commit job changes on a managed localdev/* branch. Never pushes."""
    return core_commit_job(state, job, message)


if __name__ == "__main__":
    mcp.run()
