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

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.toml"

DEFAULT_SERVER = {
    "command_timeout_seconds": 300,
    "max_output_bytes": 65_536,
    "max_read_bytes": 256 * 1024,
    "max_list_entries": 1000,
    "max_search_results": 200,
    "ignore_dirs": [".git", ".venv", "node_modules", "__pycache__", "bin", "obj", "target", "dist", "build"],
}


def windows_to_wsl(value: str) -> str:
    value = value.strip().strip('"')
    m = re.fullmatch(r"([A-Za-z]):[\\/](.*)", value)
    if m:
        return f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}"
    m = re.fullmatch(r"\\\\(?:wsl\.localhost|wsl\$)\\[^\\]+\\(.*)", value, flags=re.IGNORECASE)
    if m:
        return "/" + m.group(1).replace("\\", "/")
    return value


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
    return shlex.split(value, posix=True)


def require_repo(data: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in data["repos"]:
        known = ", ".join(sorted(data["repos"])) or "(none)"
        raise SystemExit(f"Unknown repo {name!r}. Registered: {known}")
    return data["repos"][name]


def check_one(name: str, repo: dict[str, Any]) -> bool:
    p = Path(repo["path"]).expanduser()
    exists = p.is_dir()
    print(f"[{name}]")
    print(f"  path:                 {p}")
    print(f"  repository exists:    {'yes' if exists else 'NO'}")
    print(f"  git metadata:         {'yes' if (p / '.git').exists() else 'no'}")
    print(f"  test configured:      {'yes' if repo.get('test_command') else 'no'}")
    print(f"  benchmark configured: {'yes' if repo.get('benchmark_command') else 'no'}")
    return exists


def cmd_list(_: argparse.Namespace) -> None:
    data = load_config()
    if not data["repos"]:
        print("No repositories registered.")
        return
    width = max(len(n) for n in data["repos"])
    for name, repo in sorted(data["repos"].items()):
        p = Path(repo["path"]).expanduser()
        print(f"{name:<{width}}  {'OK' if p.is_dir() else 'MISSING':7}  {repo['path']}")


def cmd_add(args: argparse.Namespace) -> None:
    data = load_config()
    name = args.name.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise SystemExit("Repo name may contain only letters, digits, '.', '_' and '-'.")
    if name in data["repos"] and not args.force:
        raise SystemExit(f"{name!r} already exists. Use --force to replace it.")
    entry: dict[str, Any] = {"path": windows_to_wsl(args.path)}
    test = parse_command(args.test)
    bench = parse_command(args.benchmark)
    if test:
        entry["test_command"] = test
    if bench:
        entry["benchmark_command"] = bench
    data["repos"][name] = entry
    save_config(data)
    print(f"Registered {name}: {entry['path']}")
    check_one(name, entry)


def cmd_remove(args: argparse.Namespace) -> None:
    data = load_config()
    require_repo(data, args.name)
    del data["repos"][args.name]
    save_config(data)
    print(f"Removed registration: {args.name} (files untouched)")


def cmd_show(args: argparse.Namespace) -> None:
    data = load_config()
    repo = require_repo(data, args.name)
    print(tomli_w.dumps({"repos": {args.name: repo}}), end="")


def cmd_path(args: argparse.Namespace) -> None:
    data = load_config()
    repo = require_repo(data, args.name)
    repo["path"] = windows_to_wsl(args.path)
    save_config(data)
    check_one(args.name, repo)


def cmd_set_command(args: argparse.Namespace) -> None:
    data = load_config()
    repo = require_repo(data, args.name)
    key = "test_command" if args.kind == "test" else "benchmark_command"
    cmd = parse_command(args.command)
    if not cmd:
        raise SystemExit("Command must not be empty.")
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


def cmd_start(_: argparse.Namespace) -> None:
    server = HERE / "server.py"
    os.execv(sys.executable, [sys.executable, str(server)])


def cmd_worktree_add(args: argparse.Namespace) -> None:
    data = load_config()
    source = require_repo(data, args.source)
    source_path = Path(source["path"]).expanduser().resolve()
    dest = Path(windows_to_wsl(args.path)).expanduser()
    if args.name in data["repos"]:
        raise SystemExit(f"Registration {args.name!r} already exists.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "-C", str(source_path), "worktree", "add"]
    if args.new_branch:
        command += ["-b", args.new_branch]
    command.append(str(dest))
    if args.commitish:
        command.append(args.commitish)
    print("+", shlex.join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    entry: dict[str, Any] = {"path": str(dest.resolve())}
    for key in ("test_command", "benchmark_command"):
        if source.get(key):
            entry[key] = list(source[key])
    data["repos"][args.name] = entry
    save_config(data)
    print(f"Registered worktree {args.name}: {entry['path']}")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Human-side manager for LocalDev MCP")
    sub = p.add_subparsers(dest="subcommand", required=True)

    q = sub.add_parser("list"); q.set_defaults(func=cmd_list)
    q = sub.add_parser("add")
    q.add_argument("name"); q.add_argument("path")
    q.add_argument("--test"); q.add_argument("--benchmark"); q.add_argument("--force", action="store_true")
    q.set_defaults(func=cmd_add)
    q = sub.add_parser("remove"); q.add_argument("name"); q.set_defaults(func=cmd_remove)
    q = sub.add_parser("show"); q.add_argument("name"); q.set_defaults(func=cmd_show)
    q = sub.add_parser("path"); q.add_argument("name"); q.add_argument("path"); q.set_defaults(func=cmd_path)
    q = sub.add_parser("set-command")
    q.add_argument("name"); q.add_argument("kind", choices=["test", "benchmark"]); q.add_argument("command")
    q.set_defaults(func=cmd_set_command)
    q = sub.add_parser("clear-command")
    q.add_argument("name"); q.add_argument("kind", choices=["test", "benchmark"]); q.set_defaults(func=cmd_clear_command)
    q = sub.add_parser("check"); q.add_argument("name", nargs="?"); q.set_defaults(func=cmd_check)
    q = sub.add_parser("start"); q.set_defaults(func=cmd_start)
    q = sub.add_parser("worktree-add")
    q.add_argument("source"); q.add_argument("name"); q.add_argument("path")
    q.add_argument("--new-branch"); q.add_argument("--commitish")
    q.set_defaults(func=cmd_worktree_add)
    return p


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
