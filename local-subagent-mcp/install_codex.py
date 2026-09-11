from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def toml_string(value: str) -> str:
    # JSON-style quoted strings are valid TOML basic strings.
    import json
    return json.dumps(value)


def snippet() -> str:
    python = HERE / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    server = HERE / "server.py"
    return (
        "[mcp_servers.local_subagent]\n"
        f"command = {toml_string(str(python))}\n"
        f"args = [{toml_string(str(server))}]\n"
        "enabled = true\n"
        "env_vars = [\"LM_API_TOKEN\"]\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Install/print the Codex MCP config for Local Subagent MCP")
    p.add_argument("--install", action="store_true", help="append the section to ~/.codex/config.toml")
    args = p.parse_args()
    block = snippet()
    if not args.install:
        print(block)
        return

    codex_dir = Path.home() / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    config = codex_dir / "config.toml"
    text = config.read_text(encoding="utf-8") if config.exists() else ""
    if "[mcp_servers.local_subagent]" in text:
        raise SystemExit(
            f"{config} already has [mcp_servers.local_subagent]. Edit it manually instead of duplicating the table."
        )
    if text and not text.endswith("\n"):
        text += "\n"
    text += "\n# Local Qwen/LM Studio worker exposed to Codex\n" + block
    config.write_text(text, encoding="utf-8")
    print(f"Appended Local Subagent MCP config to {config}")
    print("Restart Codex, then ask it to call local_agent_health().")


if __name__ == "__main__":
    main()
