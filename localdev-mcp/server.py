from __future__ import annotations

import os
import subprocess
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

CONFIG_PATH = Path(__file__).with_name("config.toml")


@dataclass(frozen=True)
class ServerConfig:
    command_timeout_seconds: int = 300
    max_output_bytes: int = 65_536
    max_read_bytes: int = 256 * 1024
    max_list_entries: int = 1000
    max_search_results: int = 200
    ignore_dirs: frozenset[str] = frozenset({
        ".git", ".venv", "node_modules", "__pycache__",
        "bin", "obj", "target", "dist", "build",
    })


@dataclass(frozen=True)
class RepoConfig:
    name: str
    root: Path
    test_command: tuple[str, ...] | None
    benchmark_command: tuple[str, ...] | None


class Registry:
    """Thread-safe repository registry that reloads config.toml on change."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._guard = threading.RLock()
        self._mtime_ns: int | None = None
        self._server = ServerConfig()
        self._repos: dict[str, RepoConfig] = {}
        self._command_locks: dict[str, threading.Lock] = {}
        self._reload(force=True)

    def _reload(self, force: bool = False) -> None:
        with self._guard:
            if not self.config_path.exists():
                raise RuntimeError(
                    f"Missing {self.config_path.name}. "
                    "Run manage.py add <name> <path> first, or copy config.example.toml."
                )

            stat = self.config_path.stat()
            if not force and self._mtime_ns == stat.st_mtime_ns:
                return

            with self.config_path.open("rb") as f:
                raw = tomllib.load(f)

            s = raw.get("server", {})
            defaults = ServerConfig()
            self._server = ServerConfig(
                command_timeout_seconds=int(s.get("command_timeout_seconds", defaults.command_timeout_seconds)),
                max_output_bytes=int(s.get("max_output_bytes", defaults.max_output_bytes)),
                max_read_bytes=int(s.get("max_read_bytes", defaults.max_read_bytes)),
                max_list_entries=int(s.get("max_list_entries", defaults.max_list_entries)),
                max_search_results=int(s.get("max_search_results", defaults.max_search_results)),
                ignore_dirs=frozenset(s.get("ignore_dirs", list(defaults.ignore_dirs))),
            )

            repos: dict[str, RepoConfig] = {}
            for name, item in raw.get("repos", {}).items():
                path = str(item.get("path", "")).strip()
                if not path:
                    continue
                test = item.get("test_command")
                bench = item.get("benchmark_command")
                repos[name] = RepoConfig(
                    name=name,
                    root=Path(path).expanduser().resolve(),
                    test_command=tuple(str(x) for x in test) if test else None,
                    benchmark_command=tuple(str(x) for x in bench) if bench else None,
                )
                self._command_locks.setdefault(name, threading.Lock())

            self._repos = repos
            self._mtime_ns = stat.st_mtime_ns

    def refresh(self) -> None:
        self._reload(force=False)

    @property
    def server(self) -> ServerConfig:
        self.refresh()
        return self._server

    def list(self) -> list[RepoConfig]:
        self.refresh()
        return list(self._repos.values())

    def get(self, name: str) -> RepoConfig:
        self.refresh()
        repo = self._repos.get(name)
        if repo is None:
            known = ", ".join(sorted(self._repos)) or "(none)"
            raise ValueError(f"Unknown repo {name!r}. Registered repos: {known}")
        if not repo.root.is_dir():
            raise RuntimeError(f"Repository path does not exist for {name!r}: {repo.root}")
        return repo

    def command_lock(self, name: str) -> threading.Lock:
        self.refresh()
        with self._guard:
            return self._command_locks.setdefault(name, threading.Lock())


registry = Registry(CONFIG_PATH)
mcp = MCPServer("LocalDev MCP")
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
LOCAL_EXEC = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


def safe_path(repo: RepoConfig, path: str = ".") -> Path:
    root = repo.root.resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError("Path escapes the target repository") from e
    return candidate


def rel(repo: RepoConfig, path: Path) -> str:
    return path.resolve().relative_to(repo.root.resolve()).as_posix()


def is_probably_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return b"\x00" in f.read(4096)
    except OSError:
        return True


def iter_files(repo: RepoConfig, base: Path) -> Iterator[Path]:
    cfg = registry.server
    if base.is_file():
        yield base
        return

    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        current = Path(dirpath)
        kept: list[str] = []
        for d in dirnames:
            if d in cfg.ignore_dirs:
                continue
            p = current / d
            try:
                p.resolve().relative_to(repo.root.resolve())
            except (OSError, ValueError):
                continue
            kept.append(d)
        dirnames[:] = kept

        for filename in filenames:
            p = current / filename
            try:
                p.resolve().relative_to(repo.root.resolve())
            except (OSError, ValueError):
                continue
            yield p


def truncate_bytes(data: bytes, limit: int) -> tuple[str, bool]:
    return data[:limit].decode("utf-8", errors="replace"), len(data) > limit


def run_fixed_command(repo: RepoConfig, command: tuple[str, ...] | list[str] | None) -> dict[str, Any]:
    if not command:
        raise RuntimeError("This command is not configured for this repository")

    cfg = registry.server
    # Same repo: serialize build/test/benchmark to avoid collisions.
    # Different repos/worktrees: locks differ, so they can run concurrently.
    with registry.command_lock(repo.name):
        try:
            result = subprocess.run(
                list(command), cwd=repo.root, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=cfg.command_timeout_seconds, shell=False,
            )
        except subprocess.TimeoutExpired as e:
            stdout, a = truncate_bytes(e.stdout or b"", cfg.max_output_bytes)
            stderr, b = truncate_bytes(e.stderr or b"", cfg.max_output_bytes)
            return {"command": list(command), "timed_out": True, "exit_code": None,
                    "stdout": stdout, "stderr": stderr, "truncated": a or b}
        except FileNotFoundError as e:
            return {"command": list(command), "timed_out": False, "exit_code": None,
                    "stdout": "", "stderr": str(e), "truncated": False}

    stdout, a = truncate_bytes(result.stdout, cfg.max_output_bytes)
    stderr, b = truncate_bytes(result.stderr, cfg.max_output_bytes)
    return {"command": list(command), "timed_out": False, "exit_code": result.returncode,
            "stdout": stdout, "stderr": stderr, "truncated": a or b}


@mcp.tool(annotations=READ_ONLY)
def list_repos() -> dict[str, Any]:
    """List repositories/worktrees explicitly registered by the human operator."""
    return {"repos": [{
        "name": r.name,
        "path": str(r.root),
        "exists": r.root.is_dir(),
        "is_git_repo": (r.root / ".git").exists(),
        "test_configured": r.test_command is not None,
        "benchmark_configured": r.benchmark_command is not None,
    } for r in registry.list()]}


@mcp.tool(annotations=READ_ONLY)
def repo_info(repo: str) -> dict[str, Any]:
    """Return configuration and status for one registered repository."""
    r = registry.get(repo)
    return {
        "name": r.name, "path": str(r.root), "exists": r.root.is_dir(),
        "is_git_repo": (r.root / ".git").exists(),
        "test_command": list(r.test_command) if r.test_command else None,
        "benchmark_command": list(r.benchmark_command) if r.benchmark_command else None,
    }


@mcp.tool(annotations=READ_ONLY)
def list_files(repo: str, path: str = ".", recursive: bool = False) -> dict[str, Any]:
    """List files/directories inside one registered repository."""
    r = registry.get(repo)
    base = safe_path(r, path)
    if not base.exists():
        raise ValueError("Path does not exist")
    if not base.is_dir():
        raise ValueError("Path is not a directory")

    cfg = registry.server
    entries: list[dict[str, str]] = []
    if not recursive:
        for item in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                item.resolve().relative_to(r.root.resolve())
            except (OSError, ValueError):
                continue
            entries.append({"path": rel(r, item), "type": "directory" if item.is_dir() else "file"})
            if len(entries) >= cfg.max_list_entries:
                break
    else:
        for item in iter_files(r, base):
            entries.append({"path": rel(r, item), "type": "file"})
            if len(entries) >= cfg.max_list_entries:
                break

    return {"repo": r.name, "base": rel(r, base), "entries": entries,
            "truncated": len(entries) >= cfg.max_list_entries}


@mcp.tool(annotations=READ_ONLY)
def read_file(repo: str, path: str, start_line: int = 1, end_line: int | None = None) -> dict[str, Any]:
    """Read a text file inside a registered repo. Line numbers are 1-based."""
    r = registry.get(repo)
    target = safe_path(r, path)
    if not target.is_file():
        raise ValueError("File does not exist")

    cfg = registry.server
    size = target.stat().st_size
    if size > cfg.max_read_bytes:
        raise ValueError(f"File too large ({size} bytes)")
    if is_probably_binary(target):
        raise ValueError("Binary file reading is disabled")

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start_line = max(1, start_line)
    if end_line is not None and end_line < start_line:
        raise ValueError("end_line must be >= start_line")
    start = start_line - 1
    end = len(lines) if end_line is None else min(end_line, len(lines))
    selected = lines[start:end]
    return {"repo": r.name, "path": rel(r, target), "start_line": start + 1,
            "end_line": start + len(selected), "total_lines": len(lines),
            "content": "\n".join(selected)}


@mcp.tool(annotations=READ_ONLY)
def search_text(repo: str, query: str, path: str = ".", case_sensitive: bool = False) -> dict[str, Any]:
    """Search text files under a path in one registered repository."""
    if not query:
        raise ValueError("query must not be empty")
    r = registry.get(repo)
    base = safe_path(r, path)
    if not base.exists():
        raise ValueError("Path does not exist")

    cfg = registry.server
    needle = query if case_sensitive else query.casefold()
    results: list[dict[str, Any]] = []
    for file in iter_files(r, base):
        try:
            if file.stat().st_size > cfg.max_read_bytes or is_probably_binary(file):
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            haystack = line if case_sensitive else line.casefold()
            if needle in haystack:
                results.append({"path": rel(r, file), "line": line_no, "text": line[:1000]})
                if len(results) >= cfg.max_search_results:
                    return {"repo": r.name, "query": query, "results": results, "truncated": True}
    return {"repo": r.name, "query": query, "results": results, "truncated": False}


@mcp.tool(annotations=READ_ONLY)
def git_status(repo: str) -> dict[str, Any]:
    """Return git status for one registered repository/worktree."""
    return run_fixed_command(registry.get(repo), ["git", "status", "--short", "--branch"])


@mcp.tool(annotations=READ_ONLY)
def git_diff(repo: str, staged: bool = False) -> dict[str, Any]:
    """Return working-tree or staged git diff."""
    r = registry.get(repo)
    command = ["git", "diff"] + (["--cached"] if staged else []) + ["--"]
    return run_fixed_command(r, command)


@mcp.tool(annotations=LOCAL_EXEC)
def run_tests(repo: str) -> dict[str, Any]:
    """Run only the human-configured test command for a repository."""
    r = registry.get(repo)
    return run_fixed_command(r, r.test_command)


@mcp.tool(annotations=LOCAL_EXEC)
def run_benchmark(repo: str) -> dict[str, Any]:
    """Run only the human-configured benchmark command for a repository."""
    r = registry.get(repo)
    return run_fixed_command(r, r.benchmark_command)


if __name__ == "__main__":
    mcp.run()
