# LocalDev MCP v0.1

Multi-repository local MCP server for coding agents.

## Main design

- One MCP server exposes multiple repos/worktrees.
- Every MCP call explicitly names `repo`; there is no global `active_repo`.
- Human-side `manage.py` controls which paths are registered.
- Windows paths such as `F:\dev\MinoFlux` are converted to `/mnt/f/dev/MinoFlux`.
- Path traversal and symlink escape outside each repo root are rejected.
- No arbitrary shell tool is exposed.
- Tests/benchmarks can execute only commands configured by the human.
- Commands for the same repo are serialized; different repos/worktrees can run concurrently.
- `config.toml` is reloaded automatically when changed.
- v0.1 has no file-write tool yet.

## Install

```bash
cd ~/dev/localdev-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You do not have to edit TOML manually. Register a repo:

```bash
python manage.py add minoflux 'F:\dev\MinoFlux'
python manage.py list
python manage.py check
```

Optional fixed commands:

```bash
python manage.py set-command minoflux test "python -m pytest -q"
python manage.py set-command minoflux benchmark "python benchmark.py"
```

## Human-side path management

```bash
python manage.py list
python manage.py add NAME PATH
python manage.py remove NAME
python manage.py show NAME
python manage.py path NAME NEW_PATH
python manage.py check [NAME]
python manage.py set-command NAME test "..."
python manage.py set-command NAME benchmark "..."
python manage.py clear-command NAME test
python manage.py clear-command NAME benchmark
```

## Parallel worktrees

For independent jobs on the same Git repo, create a worktree and register it as another repo root:

```bash
python manage.py worktree-add \
  minoflux \
  minoflux-job-001 \
  'F:\dev\worktrees\MinoFlux-job-001' \
  --new-branch ai/job-001
```

This helper is human-only; the model does not get permission to create arbitrary worktrees.

## MCP tools

- `list_repos()`
- `repo_info(repo)`
- `list_files(repo, path=".", recursive=false)`
- `read_file(repo, path, start_line=1, end_line=null)`
- `search_text(repo, query, path=".", case_sensitive=false)`
- `git_status(repo)`
- `git_diff(repo, staged=false)`
- `run_tests(repo)`
- `run_benchmark(repo)`

## LM Studio

For a WSL project at `/home/kinetic_napier/dev/localdev-mcp`:

```json
{
  "mcpServers": {
    "localdev": {
      "command": "wsl.exe",
      "args": [
        "bash",
        "-lc",
        "cd /home/kinetic_napier/dev/localdev-mcp && exec .venv/bin/python server.py"
      ]
    }
  }
}
```

## First test prompt

```text
Use list_repos first. Inspect the minoflux repository using list_files and read_file as needed. Do not modify anything. Explain what the repository does and report the configured test/benchmark commands.
```

## Concurrency

Different registered repo roots have different execution locks, so `run_tests` / `run_benchmark` can run in parallel across repos or worktrees. The same repo root is serialized to avoid build-directory collisions.

For parallel AI experiments against one source repository, create separate Git worktrees and register them under different names.

## Next

v0.2: controlled `apply_patch`, checkpoint/revert, job lifecycle, benchmark acceptance rules, and automated candidate evaluation.
