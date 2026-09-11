from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from core import State, delete_job, validate_name


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.toml"

DEFAULT_SERVER = {
    "command_timeout_seconds": 300,
    "max_output_bytes": 65_536,
    "max_read_bytes": 256 * 1024,
    "max_patch_bytes": 256 * 1024,
    "max_list_entries": 1000,
    "max_search_results": 200,
    "jobs_root": str((HERE / ".localdev-jobs").resolve()),
    "ignore_dirs": [
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
    ],
}


def normalize_path(value: str) -> str:
    value = value.strip().strip('"')
    if os.name == "nt":
        return str(Path(value).expanduser().resolve())

    m = re.fullmatch(r"([A-Za-z]):[\\/](.*)", value)
    if m:
        return f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}"
    return str(Path(value).expanduser().resolve())


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
        data.setdefault("server", dict(DEFAULT_SERVER))
        data.setdefault("repos", {})
        return data
    return {"server": dict(DEFAULT_SERVER), "repos": {}}


def save_config(data: dict[str, Any]) -> None:
    tmp = CONFIG_PATH.with_suffix(".toml.tmp")
    tmp.write_text(tomli_w.dumps(data), encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)


def parse_command(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return shlex.split(value, posix=(os.name != "nt"))


def require_repo(data: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return data["repos"][name]
    except KeyError:
        known = ", ".join(sorted(data["repos"])) or "(none)"
        raise SystemExit(f"Unknown repo {name!r}. Registered: {known}")


def check_one(name: str, repo: dict[str, Any]) -> bool:
    p = Path(repo["path"]).expanduser()
    exists = p.is_dir()
    git_meta = (p / ".git").exists()
    print(f"[{name}]")
    print(f"  path:                 {p}")
    print(f"  repository exists:    {'yes' if exists else 'NO'}")
    print(f"  git metadata:         {'yes' if git_meta else 'no'}")
    print(f"  test configured:      {'yes' if repo.get('test_command') else 'no'}")
    print(f"  benchmark configured: {'yes' if repo.get('benchmark_command') else 'no'}")
    return exists and git_meta


def cmd_init(_: argparse.Namespace) -> None:
    if CONFIG_PATH.exists():
        print(f"{CONFIG_PATH} already exists")
        return
    save_config({"server": dict(DEFAULT_SERVER), "repos": {}})
    print(f"Created {CONFIG_PATH}")


def cmd_list(_: argparse.Namespace) -> None:
    data = load_config()
    if not data["repos"]:
        print("No repositories registered.")
        return
    width = max(len(name) for name in data["repos"])
    for name, repo in sorted(data["repos"].items()):
        p = Path(repo["path"])
        print(
            f"{name:<{width}}  "
            f"{'OK' if p.is_dir() else 'MISSING':7}  "
            f"{'git' if (p / '.git').exists() else '-':3}  "
            f"{repo['path']}"
        )


def cmd_add(args: argparse.Namespace) -> None:
    data = load_config()
    validate_name(args.name, "repo name")
    if args.name in data["repos"] and not args.force:
        raise SystemExit("Repo already registered; use --force to replace")

    entry: dict[str, Any] = {"path": normalize_path(args.path)}
    test = parse_command(args.test)
    bench = parse_command(args.benchmark)
    if test:
        entry["test_command"] = test
    if bench:
        entry["benchmark_command"] = bench
    data["repos"][args.name] = entry
    save_config(data)
    print(f"Registered {args.name}: {entry['path']}")
    check_one(args.name, entry)


def cmd_remove(args: argparse.Namespace) -> None:
    data = load_config()
    require_repo(data, args.name)
    del data["repos"][args.name]
    save_config(data)
    print(f"Removed registration {args.name}; files were not deleted.")


def cmd_set_command(args: argparse.Namespace) -> None:
    data = load_config()
    repo = require_repo(data, args.name)
    cmd = parse_command(args.command)
    if not cmd:
        raise SystemExit("Command must not be empty")
    key = "test_command" if args.kind == "test" else "benchmark_command"
    repo[key] = cmd
    save_config(data)
    print(f"{args.name}.{key} = {cmd!r}")


def cmd_clear_command(args: argparse.Namespace) -> None:
    data = load_config()
    repo = require_repo(data, args.name)
    key = "test_command" if args.kind == "test" else "benchmark_command"
    repo.pop(key, None)
    save_config(data)
    print(f"Cleared {args.name}.{key}")


def cmd_check(args: argparse.Namespace) -> None:
    data = load_config()
    if args.name:
        ok = check_one(args.name, require_repo(data, args.name))
        raise SystemExit(0 if ok else 1)
    if not data["repos"]:
        print("No repositories registered.")
        raise SystemExit(1)
    ok = True
    for name, repo in sorted(data["repos"].items()):
        ok = check_one(name, repo) and ok
    raise SystemExit(0 if ok else 1)


def cmd_jobs(_: argparse.Namespace) -> None:
    state = State()
    jobs = state.list_jobs()
    if not jobs:
        print("No managed jobs.")
        return
    for job in sorted(jobs, key=lambda j: j.name):
        p = Path(job.path)
        print(
            f"{job.name:24} "
            f"{job.source_repo:16} "
            f"{'OK' if p.is_dir() else 'STALE':5} "
            f"{job.branch} "
            f"{job.path}"
        )


def cmd_job_remove(args: argparse.Namespace) -> None:
    result = delete_job(State(), args.name, force=args.force)
    print(result)


def cmd_cleanup_stale(args: argparse.Namespace) -> None:
    state = State()
    removed = []
    for job in list(state.list_jobs()):
        if not Path(job.path).exists():
            state.remove_job_record(job.name)
            removed.append(job.name)
    print(f"Removed {len(removed)} stale metadata record(s): {', '.join(removed) or '(none)'}")


def cmd_start(_: argparse.Namespace) -> None:
    os.execv(
        sys.executable,
        [sys.executable, str(HERE / "server.py")],
    )


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Human-side manager for LocalDev MCP")
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("init")
    q.set_defaults(func=cmd_init)

    q = sub.add_parser("list")
    q.set_defaults(func=cmd_list)

    q = sub.add_parser("add")
    q.add_argument("name")
    q.add_argument("path")
    q.add_argument("--test")
    q.add_argument("--benchmark")
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=cmd_add)

    q = sub.add_parser("remove")
    q.add_argument("name")
    q.set_defaults(func=cmd_remove)

    q = sub.add_parser("set-command")
    q.add_argument("name")
    q.add_argument("kind", choices=["test", "benchmark"])
    q.add_argument("command")
    q.set_defaults(func=cmd_set_command)

    q = sub.add_parser("clear-command")
    q.add_argument("name")
    q.add_argument("kind", choices=["test", "benchmark"])
    q.set_defaults(func=cmd_clear_command)

    q = sub.add_parser("check")
    q.add_argument("name", nargs="?")
    q.set_defaults(func=cmd_check)

    q = sub.add_parser("jobs")
    q.set_defaults(func=cmd_jobs)

    q = sub.add_parser("job-remove")
    q.add_argument("name")
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=cmd_job_remove)

    q = sub.add_parser("cleanup-stale")
    q.set_defaults(func=cmd_cleanup_stale)

    q = sub.add_parser("start")
    q.set_defaults(func=cmd_start)

    return p


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
