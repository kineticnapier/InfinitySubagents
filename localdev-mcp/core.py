from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


CONFIG_PATH = Path(__file__).with_name("config.toml")
JOBS_PATH = Path(__file__).with_name("jobs.json")
LOG_PATH = Path(__file__).with_name("localdev-mcp.log.jsonl")


@dataclass(frozen=True)
class ServerConfig:
    command_timeout_seconds: int = 300
    max_output_bytes: int = 65_536
    max_read_bytes: int = 256 * 1024
    max_patch_bytes: int = 256 * 1024
    max_list_entries: int = 1000
    max_search_results: int = 200
    jobs_root: Path = Path.cwd() / ".localdev-jobs"
    ignore_dirs: frozenset[str] = frozenset(
        {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            "bin",
            "obj",
            "target",
            "dist",
            "build",
        }
    )


@dataclass(frozen=True)
class RepoConfig:
    name: str
    root: Path
    test_command: tuple[str, ...] | None
    benchmark_command: tuple[str, ...] | None


@dataclass
class JobRecord:
    name: str
    source_repo: str
    path: str
    branch: str
    base_ref: str
    created_at: float
    checkpoints: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "JobRecord":
        return cls(
            name=str(obj["name"]),
            source_repo=str(obj["source_repo"]),
            path=str(obj["path"]),
            branch=str(obj["branch"]),
            base_ref=str(obj["base_ref"]),
            created_at=float(obj["created_at"]),
            checkpoints={
                str(k): str(v) for k, v in obj.get("checkpoints", {}).items()
            },
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_repo": self.source_repo,
            "path": self.path,
            "branch": self.branch,
            "base_ref": self.base_ref,
            "created_at": self.created_at,
            "checkpoints": dict(self.checkpoints),
        }


class State:
    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        jobs_path: Path = JOBS_PATH,
        log_path: Path = LOG_PATH,
    ):
        self.config_path = config_path
        self.jobs_path = jobs_path
        self.log_path = log_path
        self._guard = threading.RLock()
        self._config_mtime_ns: int | None = None
        self._server = ServerConfig()
        self._repos: dict[str, RepoConfig] = {}
        self._jobs: dict[str, JobRecord] = {}
        self._job_locks: dict[str, threading.RLock] = {}
        self._repo_command_locks: dict[str, threading.RLock] = {}
        self.reload_config(force=True)
        self.reload_jobs()

    def reload_config(self, force: bool = False) -> None:
        with self._guard:
            if not self.config_path.exists():
                raise RuntimeError(
                    f"Missing {self.config_path.name}. "
                    "Copy config.example.toml to config.toml or use manage.py."
                )
            stat = self.config_path.stat()
            if not force and stat.st_mtime_ns == self._config_mtime_ns:
                return
            with self.config_path.open("rb") as f:
                raw = tomllib.load(f)

            s = raw.get("server", {})
            defaults = ServerConfig()
            jobs_root_raw = s.get("jobs_root")
            jobs_root = (
                Path(str(jobs_root_raw)).expanduser().resolve()
                if jobs_root_raw
                else (self.config_path.parent / ".localdev-jobs").resolve()
            )
            self._server = ServerConfig(
                command_timeout_seconds=int(
                    s.get("command_timeout_seconds", defaults.command_timeout_seconds)
                ),
                max_output_bytes=int(
                    s.get("max_output_bytes", defaults.max_output_bytes)
                ),
                max_read_bytes=int(
                    s.get("max_read_bytes", defaults.max_read_bytes)
                ),
                max_patch_bytes=int(
                    s.get("max_patch_bytes", defaults.max_patch_bytes)
                ),
                max_list_entries=int(
                    s.get("max_list_entries", defaults.max_list_entries)
                ),
                max_search_results=int(
                    s.get("max_search_results", defaults.max_search_results)
                ),
                jobs_root=jobs_root,
                ignore_dirs=frozenset(
                    str(x)
                    for x in s.get("ignore_dirs", sorted(defaults.ignore_dirs))
                ),
            )

            repos: dict[str, RepoConfig] = {}
            for name, item in raw.get("repos", {}).items():
                raw_path = str(item.get("path", "")).strip()
                if not raw_path:
                    continue
                test = item.get("test_command")
                bench = item.get("benchmark_command")
                repos[name] = RepoConfig(
                    name=name,
                    root=Path(raw_path).expanduser().resolve(),
                    test_command=tuple(str(x) for x in test) if test else None,
                    benchmark_command=tuple(str(x) for x in bench) if bench else None,
                )
                self._repo_command_locks.setdefault(name, threading.RLock())

            self._repos = repos
            self._config_mtime_ns = stat.st_mtime_ns

    def refresh_config(self) -> None:
        self.reload_config(force=False)

    def reload_jobs(self) -> None:
        with self._guard:
            if not self.jobs_path.exists():
                self._jobs = {}
                return
            raw = json.loads(self.jobs_path.read_text(encoding="utf-8"))
            self._jobs = {
                name: JobRecord.from_json(obj)
                for name, obj in raw.get("jobs", {}).items()
            }
            for name in self._jobs:
                self._job_locks.setdefault(name, threading.RLock())

    def save_jobs(self) -> None:
        with self._guard:
            payload = {
                "version": 1,
                "jobs": {
                    name: job.to_json()
                    for name, job in sorted(self._jobs.items())
                },
            }
            tmp = self.jobs_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, self.jobs_path)

    @property
    def server(self) -> ServerConfig:
        self.refresh_config()
        return self._server

    def list_repos(self) -> list[RepoConfig]:
        self.refresh_config()
        return list(self._repos.values())

    def repo(self, name: str) -> RepoConfig:
        self.refresh_config()
        repo = self._repos.get(name)
        if repo is None:
            known = ", ".join(sorted(self._repos)) or "(none)"
            raise ValueError(f"Unknown repo {name!r}. Registered repos: {known}")
        if not repo.root.is_dir():
            raise RuntimeError(f"Repository path does not exist: {repo.root}")
        return repo

    def list_jobs(self) -> list[JobRecord]:
        with self._guard:
            return list(self._jobs.values())

    def job(self, name: str) -> JobRecord:
        with self._guard:
            job = self._jobs.get(name)
            if job is None:
                known = ", ".join(sorted(self._jobs)) or "(none)"
                raise ValueError(f"Unknown job {name!r}. Registered jobs: {known}")
            return job

    def set_job(self, job: JobRecord) -> None:
        with self._guard:
            self._jobs[job.name] = job
            self._job_locks.setdefault(job.name, threading.RLock())
            self.save_jobs()

    def remove_job_record(self, name: str) -> JobRecord:
        with self._guard:
            job = self.job(name)
            del self._jobs[name]
            self.save_jobs()
            return job

    def job_lock(self, name: str) -> threading.RLock:
        with self._guard:
            return self._job_locks.setdefault(name, threading.RLock())

    def repo_lock(self, name: str) -> threading.RLock:
        with self._guard:
            return self._repo_command_locks.setdefault(name, threading.RLock())

    def log(
        self,
        operation: str,
        *,
        job: str | None = None,
        repo: str | None = None,
        success: bool,
        duration: float,
        before_head: str | None = None,
        after_head: str | None = None,
        detail: str | None = None,
    ) -> None:
        event = {
            "timestamp": time.time(),
            "operation": operation,
            "job": job,
            "repo": repo,
            "success": success,
            "duration_seconds": round(duration, 6),
            "before_head": before_head,
            "after_head": after_head,
            "detail": detail,
        }
        line = json.dumps(event, ensure_ascii=False)
        with self._guard:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def validate_name(value: str, kind: str = "name") -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(
            f"{kind} may contain only letters, digits, '.', '_' and '-'"
        )
    return value


def safe_path(root: Path, path: str = ".") -> Path:
    root = root.resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError("Path escapes the target root") from e
    return candidate


def rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def is_probably_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return b"\x00" in f.read(4096)
    except OSError:
        return True


def iter_files(root: Path, base: Path, ignore_dirs: frozenset[str]) -> Iterator[Path]:
    if base.is_file():
        yield base
        return

    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        current = Path(dirpath)
        kept: list[str] = []
        for dirname in dirnames:
            if dirname in ignore_dirs:
                continue
            p = current / dirname
            try:
                p.resolve().relative_to(root.resolve())
            except (OSError, ValueError):
                continue
            kept.append(dirname)
        dirnames[:] = kept

        for filename in filenames:
            p = current / filename
            try:
                p.resolve().relative_to(root.resolve())
            except (OSError, ValueError):
                continue
            yield p


def truncate_bytes(data: bytes, limit: int) -> tuple[str, bool]:
    return (
        data[:limit].decode("utf-8", errors="replace"),
        len(data) > limit,
    )


def run_process(
    command: list[str] | tuple[str, ...],
    *,
    cwd: Path,
    timeout: int,
    max_output_bytes: int,
    input_bytes: bytes | None = None,
) -> dict[str, Any]:
    try:
        kwargs = {
            "cwd": cwd,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "timeout": timeout,
            "shell": False,
        }
        if input_bytes is None:
            kwargs["stdin"] = subprocess.DEVNULL
        else:
            kwargs["input"] = input_bytes

        result = subprocess.run(
            list(command),
            **kwargs,
        )
    except subprocess.TimeoutExpired as e:
        stdout, out_trunc = truncate_bytes(e.stdout or b"", max_output_bytes)
        stderr, err_trunc = truncate_bytes(e.stderr or b"", max_output_bytes)
        return {
            "command": list(command),
            "timed_out": True,
            "exit_code": None,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": out_trunc or err_trunc,
        }
    except FileNotFoundError as e:
        return {
            "command": list(command),
            "timed_out": False,
            "exit_code": None,
            "stdout": "",
            "stderr": str(e),
            "truncated": False,
        }

    stdout, out_trunc = truncate_bytes(result.stdout, max_output_bytes)
    stderr, err_trunc = truncate_bytes(result.stderr, max_output_bytes)
    return {
        "command": list(command),
        "timed_out": False,
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": out_trunc or err_trunc,
    }


def git(
    root: Path,
    args: list[str],
    *,
    timeout: int = 300,
    max_output_bytes: int = 65_536,
    input_bytes: bytes | None = None,
) -> dict[str, Any]:
    return run_process(
        ["git", *args],
        cwd=root,
        timeout=timeout,
        max_output_bytes=max_output_bytes,
        input_bytes=input_bytes,
    )


def git_text(
    root: Path,
    args: list[str],
    *,
    timeout: int = 300,
    max_output_bytes: int = 65_536,
) -> str:
    result = git(
        root,
        args,
        timeout=timeout,
        max_output_bytes=max_output_bytes,
    )
    if result["exit_code"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"] or "git failed")
    return str(result["stdout"]).strip()


def resolve_target(state: State, *, repo: str | None = None, job: str | None = None) -> tuple[str, Path, RepoConfig, JobRecord | None]:
    if bool(repo) == bool(job):
        raise ValueError("Specify exactly one of repo or job")
    if repo:
        r = state.repo(repo)
        return repo, r.root, r, None

    j = state.job(str(job))
    r = state.repo(j.source_repo)
    root = Path(j.path).resolve()
    if not root.is_dir():
        raise RuntimeError(f"Job worktree does not exist: {root}")
    return j.name, root, r, j


def validate_patch(patch: str, job_root: Path, max_patch_bytes: int) -> list[str]:
    encoded = patch.encode("utf-8")
    if not encoded:
        raise ValueError("Patch is empty")
    if len(encoded) > max_patch_bytes:
        raise ValueError(f"Patch is too large ({len(encoded)} bytes)")
    if "GIT binary patch" in patch or "Binary files " in patch:
        raise ValueError("Binary patches are not allowed")
    if re.search(r"(?m)^Submodule ", patch):
        raise ValueError("Submodule patches are not allowed")
    if re.search(r"(?m)^(old mode|new mode) 160000$", patch):
        raise ValueError("Submodule mode changes are not allowed")

    candidates: set[str] = set()

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            # Git quotes exotic paths; for safety reject quoted/space-containing header forms.
            m = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", line)
            if not m:
                raise ValueError("Unsupported/ambiguous diff --git header")
            candidates.update(m.groups())
        elif line.startswith("--- ") or line.startswith("+++ "):
            p = line[4:].split("\t", 1)[0]
            if p == "/dev/null":
                continue
            if p.startswith("a/") or p.startswith("b/"):
                p = p[2:]
            candidates.add(p)

    if not candidates:
        raise ValueError("No file paths found in patch")

    validated: list[str] = []
    for raw in sorted(candidates):
        normalized = raw.replace("\\", "/")
        pp = Path(normalized)
        if pp.is_absolute():
            raise ValueError(f"Absolute path is not allowed: {raw}")
        if ".." in pp.parts:
            raise ValueError(f"Parent traversal is not allowed: {raw}")
        if not pp.parts:
            raise ValueError("Empty patch path")
        if pp.parts[0].lower() == ".git" or ".git" in (p.lower() for p in pp.parts):
            raise ValueError(f".git paths are not allowed: {raw}")

        candidate = (job_root / pp).resolve()
        try:
            candidate.relative_to(job_root.resolve())
        except ValueError as e:
            raise ValueError(f"Patch path escapes job root: {raw}") from e

        # Existing path or its nearest existing parent must remain in the worktree.
        probe = candidate
        while not probe.exists() and probe != job_root:
            probe = probe.parent
        try:
            probe.resolve().relative_to(job_root.resolve())
        except ValueError as e:
            raise ValueError(f"Symlink escape detected: {raw}") from e

        validated.append(pp.as_posix())

    return validated


def create_job(state: State, repo_name: str, name: str, base_ref: str | None = None) -> dict[str, Any]:
    started = time.monotonic()
    repo = state.repo(repo_name)
    validate_name(name, "job name")

    if any(j.name == name for j in state.list_jobs()):
        raise ValueError(f"Job {name!r} already exists")

    cfg = state.server
    cfg.jobs_root.mkdir(parents=True, exist_ok=True)
    job_path = (cfg.jobs_root / name).resolve()
    try:
        job_path.relative_to(cfg.jobs_root.resolve())
    except ValueError as e:
        raise ValueError("Job path escaped jobs_root") from e
    if job_path.exists():
        raise ValueError(f"Job directory already exists: {job_path}")

    base = base_ref or "HEAD"
    safe_branch_name = re.sub(r"[^A-Za-z0-9._/-]+", "-", name)
    branch = f"localdev/{safe_branch_name}-{uuid.uuid4().hex[:8]}"

    result = git(
        repo.root,
        ["worktree", "add", "-b", branch, str(job_path), base],
        timeout=cfg.command_timeout_seconds,
        max_output_bytes=cfg.max_output_bytes,
    )
    if result["exit_code"] != 0:
        state.log(
            "create_job",
            repo=repo_name,
            job=name,
            success=False,
            duration=time.monotonic() - started,
            detail=result["stderr"],
        )
        raise RuntimeError(result["stderr"] or result["stdout"])

    head = git_text(job_path, ["rev-parse", "HEAD"])
    record = JobRecord(
        name=name,
        source_repo=repo_name,
        path=str(job_path),
        branch=branch,
        base_ref=base,
        created_at=time.time(),
    )
    state.set_job(record)
    state.log(
        "create_job",
        repo=repo_name,
        job=name,
        success=True,
        duration=time.monotonic() - started,
        after_head=head,
    )
    return {
        "job": name,
        "source_repo": repo_name,
        "path": str(job_path),
        "branch": branch,
        "base_ref": base,
        "head": head,
    }


def delete_job(state: State, name: str, force: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    job = state.job(name)
    repo = state.repo(job.source_repo)
    job_path = Path(job.path)
    cfg = state.server

    with state.job_lock(name):
        if job_path.exists() and not force:
            status = git_text(job_path, ["status", "--porcelain"])
            if status:
                raise RuntimeError("Job has uncommitted changes; use force=True to remove")

        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(job_path))
        result = git(
            repo.root,
            args,
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
        )
        if result["exit_code"] != 0 and job_path.exists():
            state.log(
                "delete_job",
                repo=job.source_repo,
                job=name,
                success=False,
                duration=time.monotonic() - started,
                detail=result["stderr"],
            )
            raise RuntimeError(result["stderr"] or result["stdout"])

        state.remove_job_record(name)
        state.log(
            "delete_job",
            repo=job.source_repo,
            job=name,
            success=True,
            duration=time.monotonic() - started,
        )
        return {
            "deleted": name,
            "worktree_removed": not job_path.exists(),
            "branch_preserved": job.branch,
        }


def apply_patch(state: State, job_name: str, patch: str) -> dict[str, Any]:
    started = time.monotonic()
    job = state.job(job_name)
    root = Path(job.path).resolve()
    cfg = state.server
    data = patch.encode("utf-8")

    with state.job_lock(job_name):
        before = git_text(root, ["rev-parse", "HEAD"])
        paths = validate_patch(patch, root, cfg.max_patch_bytes)

        check = git(
            root,
            ["apply", "--check", "--whitespace=nowarn", "-"],
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
            input_bytes=data,
        )
        if check["exit_code"] != 0:
            state.log(
                "apply_patch",
                repo=job.source_repo,
                job=job_name,
                success=False,
                duration=time.monotonic() - started,
                before_head=before,
                after_head=before,
                detail=check["stderr"],
            )
            raise RuntimeError(check["stderr"] or check["stdout"])

        applied = git(
            root,
            ["apply", "--whitespace=nowarn", "-"],
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
            input_bytes=data,
        )
        if applied["exit_code"] != 0:
            state.log(
                "apply_patch",
                repo=job.source_repo,
                job=job_name,
                success=False,
                duration=time.monotonic() - started,
                before_head=before,
                after_head=before,
                detail=applied["stderr"],
            )
            raise RuntimeError(applied["stderr"] or applied["stdout"])

        diff_stat = git_text(root, ["diff", "--stat", "--"])
        state.log(
            "apply_patch",
            repo=job.source_repo,
            job=job_name,
            success=True,
            duration=time.monotonic() - started,
            before_head=before,
            after_head=before,
            detail=", ".join(paths),
        )
        return {
            "job": job_name,
            "paths": paths,
            "head": before,
            "diff_stat": diff_stat,
        }


def create_checkpoint(state: State, job_name: str) -> dict[str, Any]:
    started = time.monotonic()
    job = state.job(job_name)
    root = Path(job.path).resolve()

    with state.job_lock(job_name):
        status = git_text(root, ["status", "--porcelain"])
        if status:
            raise RuntimeError(
                "v0.2 checkpoints require a clean working tree and index"
            )
        head = git_text(root, ["rev-parse", "HEAD"])
        checkpoint = f"cp-{uuid.uuid4().hex[:12]}"
        job.checkpoints[checkpoint] = head
        state.set_job(job)
        state.log(
            "create_checkpoint",
            repo=job.source_repo,
            job=job_name,
            success=True,
            duration=time.monotonic() - started,
            before_head=head,
            after_head=head,
        )
        return {"checkpoint": checkpoint, "head": head}


def revert_to_checkpoint(state: State, job_name: str, checkpoint: str) -> dict[str, Any]:
    started = time.monotonic()
    job = state.job(job_name)
    root = Path(job.path).resolve()
    cfg = state.server
    head = job.checkpoints.get(checkpoint)
    if not head:
        raise ValueError("Unknown checkpoint for this job")

    with state.job_lock(job_name):
        before = git_text(root, ["rev-parse", "HEAD"])
        reset = git(
            root,
            ["reset", "--hard", head],
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
        )
        if reset["exit_code"] != 0:
            raise RuntimeError(reset["stderr"] or reset["stdout"])
        clean = git(
            root,
            ["clean", "-fd"],
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
        )
        if clean["exit_code"] != 0:
            raise RuntimeError(clean["stderr"] or clean["stdout"])
        after = git_text(root, ["rev-parse", "HEAD"])
        state.log(
            "revert_to_checkpoint",
            repo=job.source_repo,
            job=job_name,
            success=True,
            duration=time.monotonic() - started,
            before_head=before,
            after_head=after,
        )
        return {
            "job": job_name,
            "checkpoint": checkpoint,
            "before_head": before,
            "after_head": after,
            "cleaned_untracked": clean["stdout"],
        }


def commit_job(state: State, job_name: str, message: str) -> dict[str, Any]:
    started = time.monotonic()
    if not message.strip():
        raise ValueError("Commit message must not be empty")

    job = state.job(job_name)
    root = Path(job.path).resolve()
    cfg = state.server

    with state.job_lock(job_name):
        branch = git_text(root, ["branch", "--show-current"])
        if branch in {"main", "master"} or not branch.startswith("localdev/"):
            raise RuntimeError(f"Refusing to commit on branch {branch!r}")

        before = git_text(root, ["rev-parse", "HEAD"])
        status = git_text(root, ["status", "--porcelain"])
        if not status:
            raise RuntimeError("Nothing to commit")

        add = git(
            root,
            ["add", "-A"],
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
        )
        if add["exit_code"] != 0:
            raise RuntimeError(add["stderr"] or add["stdout"])

        # Double-check that there is staged content.
        staged = git(
            root,
            ["diff", "--cached", "--quiet", "--exit-code"],
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
        )
        if staged["exit_code"] == 0:
            raise RuntimeError("Nothing staged to commit")
        if staged["exit_code"] not in (0, 1):
            raise RuntimeError(staged["stderr"] or staged["stdout"])

        commit = git(
            root,
            ["commit", "-m", message],
            timeout=cfg.command_timeout_seconds,
            max_output_bytes=cfg.max_output_bytes,
        )
        if commit["exit_code"] != 0:
            raise RuntimeError(commit["stderr"] or commit["stdout"])

        after = git_text(root, ["rev-parse", "HEAD"])
        state.log(
            "commit_job",
            repo=job.source_repo,
            job=job_name,
            success=True,
            duration=time.monotonic() - started,
            before_head=before,
            after_head=after,
        )
        return {
            "job": job_name,
            "branch": branch,
            "before_head": before,
            "after_head": after,
            "commit_output": commit["stdout"],
        }
