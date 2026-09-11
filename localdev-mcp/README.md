# LocalDev MCP v0.2

LocalDev MCP exposes a deliberately constrained local development surface to MCP-capable agents.

The v0.2 goal is **safe isolated coding jobs**:

```text
parent agent / local subagent
        ↓
LocalDev MCP
        ↓
managed git worktree
        ↓
read/search → patch → test → benchmark → diff
        ↓
commit or revert
```

## Safety model

The important boundary is enforced in code rather than in prompts:

- source repositories are read-only from write tools
- write operations require a managed `job`
- each job is an isolated Git worktree on a dedicated `localdev/*` branch
- no arbitrary shell MCP tool
- test/benchmark commands must be configured by the human
- `shell=False`
- path traversal and symlink escape are rejected
- patches are size-limited and checked with `git apply --check`
- `.git`, binary patches and submodule patches are rejected
- same-job write/test/benchmark/Git operations are serialized
- different jobs can run concurrently
- `commit_job` never pushes
- `revert_to_checkpoint` operates only inside the managed worktree

v0.2 checkpoints intentionally require a **clean working tree and index**.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy config.example.toml config.toml
```

Edit `config.toml` or register repos with the human-side manager.

```powershell
python manage.py add minoflux "F:\dev\MinoFlux"
python manage.py set-command minoflux test "python -m pytest -q"
python manage.py set-command minoflux benchmark "python tools/benchmark.py"
python manage.py check
```

Set a job root in `config.toml`:

```toml
[server]
jobs_root = "F:\\dev\\localdev-jobs"
```

`config.toml`, `jobs.json` and the JSONL operation log are ignored by Git.

## MCP tools

Repository/read tools:

- `list_repos`
- `repo_info`
- `list_files(repo=... | job=...)`
- `read_file(path, repo=... | job=...)`
- `search_text(query, repo=... | job=...)`
- `git_status(repo=... | job=...)`
- `git_diff(repo=... | job=...)`

Managed job tools:

- `create_job(repo, name, base_ref=None)`
- `list_jobs(repo=None)`
- `job_info(job)`
- `delete_job(job, force=False)`
- `create_checkpoint(job)`
- `apply_patch(job, patch)`
- `run_tests(job)`
- `run_benchmark(job)`
- `revert_to_checkpoint(job, checkpoint)`
- `commit_job(job, message)`

There is intentionally no `write_file()` and no `run_command()`.

## Example agent flow

```text
1. list_repos()
2. create_job(repo="minoflux", name="qwen-001")
3. create_checkpoint(job="qwen-001")
4. read/search inside job qwen-001
5. apply_patch(job="qwen-001", patch="...")
6. run_tests(job="qwen-001")
7. run_benchmark(job="qwen-001")
8. git_diff(job="qwen-001")
9a. commit_job(job="qwen-001", message="...")
or
9b. revert_to_checkpoint(job="qwen-001", checkpoint="...")
```

## Human-side commands

```powershell
python manage.py init
python manage.py add <name> <path>
python manage.py list
python manage.py check
python manage.py set-command <repo> test "<command>"
python manage.py set-command <repo> benchmark "<command>"
python manage.py jobs
python manage.py job-remove <job> [--force]
python manage.py cleanup-stale
```

LM Studio/Polaris should normally launch `server.py` as a stdio MCP process.

## Parallelism

A global active repo/job is deliberately not used.

```text
Qwen A → job-a
Qwen B → job-b
Qwen C → job-c
```

Different worktrees have separate locks, so they may run independently. One GPU may still make parallel model inference slower; this MCP only ensures the development workspaces themselves do not collide.

## Tests

```powershell
pytest -q
```

The tests create temporary Git repositories/worktrees and cover:

- traversal rejection
- symlink escape rejection where the OS permits creating symlinks
- patch isolation
- invalid patch safety
- checkpoint/revert
- job branch commits
- multiple repo loading

## Notes

`delete_job` preserves the job branch after removing the worktree. This is intentional: deleting a worktree should not silently destroy committed work. Branch cleanup can be done by the human after review.
