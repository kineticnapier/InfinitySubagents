from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from core import (
    State,
    apply_patch,
    commit_job,
    create_checkpoint,
    create_job,
    delete_job,
    revert_to_checkpoint,
    safe_path,
    validate_patch,
)


def run(cmd, cwd: Path):
    return subprocess.run(
        cmd, cwd=cwd, check=True, text=True, capture_output=True
    )


def make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    run(["git", "init", "-b", "main"], path)
    run(["git", "config", "user.email", "test@example.com"], path)
    run(["git", "config", "user.name", "LocalDev Test"], path)
    (path / "hello.txt").write_text("hello\n", encoding="utf-8")
    run(["git", "add", "hello.txt"], path)
    run(["git", "commit", "-m", "initial"], path)
    return path


def make_state(tmp_path: Path) -> tuple[State, Path]:
    repo = make_repo(tmp_path / "source")
    config = tmp_path / "config.toml"
    jobs = tmp_path / "jobs.json"
    log = tmp_path / "events.jsonl"
    config.write_text(
        f"""
[server]
jobs_root = {json.dumps(str(tmp_path / "jobs"))}
command_timeout_seconds = 30
max_output_bytes = 65536
max_read_bytes = 262144
max_patch_bytes = 262144

[repos.demo]
path = {json.dumps(str(repo))}
test_command = ["python", "-c", "print('ok')"]
benchmark_command = ["python", "-c", "print('1.0')"]
""",
        encoding="utf-8",
    )
    return State(config, jobs, log), repo


def test_safe_path_rejects_parent(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError):
        safe_path(root, "../escape.txt")


def test_validate_patch_rejects_parent(tmp_path: Path):
    root = tmp_path / "job"
    root.mkdir()
    patch = """diff --git a/../x b/../x
--- a/../x
+++ b/../x
@@ -0,0 +1 @@
+x
"""
    with pytest.raises(ValueError):
        validate_patch(patch, root, 10000)


@pytest.mark.skipif(
    not hasattr(os, "symlink"),
    reason="symlink unavailable",
)
def test_safe_path_rejects_symlink_escape(tmp_path: Path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")
    with pytest.raises(ValueError):
        safe_path(root, "link/file.txt")


def test_job_patch_checkpoint_revert_and_isolation(tmp_path: Path):
    state, source = make_state(tmp_path)
    j1 = create_job(state, "demo", "job1")
    j2 = create_job(state, "demo", "job2")

    cp = create_checkpoint(state, "job1")["checkpoint"]

    patch = """diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-hello
+changed
"""
    apply_patch(state, "job1", patch)

    assert (Path(j1["path"]) / "hello.txt").read_text().strip() == "changed"
    assert (Path(j2["path"]) / "hello.txt").read_text().strip() == "hello"
    assert (source / "hello.txt").read_text().strip() == "hello"

    revert_to_checkpoint(state, "job1", cp)
    assert (Path(j1["path"]) / "hello.txt").read_text().strip() == "hello"

    delete_job(state, "job1", force=True)
    delete_job(state, "job2", force=True)
    assert (source / "hello.txt").read_text().strip() == "hello"


def test_invalid_patch_does_not_modify_file(tmp_path: Path):
    state, _ = make_state(tmp_path)
    job = create_job(state, "demo", "job1")
    root = Path(job["path"])

    bad = """diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -999 +999 @@
-nope
+bad
"""
    with pytest.raises(RuntimeError):
        apply_patch(state, "job1", bad)
    assert (root / "hello.txt").read_text().strip() == "hello"
    delete_job(state, "job1", force=True)


def test_commit_is_job_branch_only(tmp_path: Path):
    state, _ = make_state(tmp_path)
    job = create_job(state, "demo", "job1")
    patch = """diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-hello
+committed
"""
    apply_patch(state, "job1", patch)
    result = commit_job(state, "job1", "change hello")
    assert result["branch"].startswith("localdev/")
    assert result["before_head"] != result["after_head"]
    delete_job(state, "job1", force=True)


def test_multiple_repos_load(tmp_path: Path):
    r1 = make_repo(tmp_path / "r1")
    r2 = make_repo(tmp_path / "r2")
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[server]
jobs_root = {json.dumps(str(tmp_path / "jobs"))}

[repos.one]
path = {json.dumps(str(r1))}

[repos.two]
path = {json.dumps(str(r2))}
""",
        encoding="utf-8",
    )
    state = State(config, tmp_path / "jobs.json", tmp_path / "log.jsonl")
    assert {r.name for r in state.list_repos()} == {"one", "two"}
