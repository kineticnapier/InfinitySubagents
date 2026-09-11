from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from client import LocalAgentClient


mcp = MCPServer("Local Subagent MCP")
client = LocalAgentClient()

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
LOCAL_AGENT = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)


@mcp.tool(annotations=READ_ONLY)
def local_agent_health() -> dict[str, Any]:
    """Check that LM Studio is reachable and show the configured local model."""
    try:
        return client.health()
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@mcp.tool(annotations=LOCAL_AGENT)
def local_agent(
    task: str,
    mode: str = "research",
    repo: str | None = None,
    job: str | None = None,
    max_output_tokens: int | None = None,
    context_length: int | None = None,
) -> dict[str, Any]:
    """Delegate one bounded task to the local LM Studio model.

    mode='research' exposes only read-only LocalDev MCP tools.
    mode='code' also exposes managed-worktree edit/test/benchmark/commit tools.
    The local model's hidden reasoning and raw tool outputs are not returned to Codex;
    only its final result, compact tool-call counts, and runtime stats are returned.
    """
    try:
        return client.run(
            task=task,
            mode=mode,
            repo=repo,
            job=job,
            max_output_tokens=max_output_tokens,
            context_length=context_length,
        )
    except Exception as e:
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "mode": mode,
            "repo": repo,
            "job": job,
        }


@mcp.tool(annotations=LOCAL_AGENT)
def continue_local_agent(
    response_id: str,
    task: str,
    mode: str = "research",
    repo: str | None = None,
    job: str | None = None,
    max_output_tokens: int | None = None,
    context_length: int | None = None,
) -> dict[str, Any]:
    """Continue a stored LM Studio local-subagent response."""
    try:
        return client.run(
            task=task,
            mode=mode,
            repo=repo,
            job=job,
            previous_response_id=response_id,
            max_output_tokens=max_output_tokens,
            context_length=context_length,
        )
    except Exception as e:
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "mode": mode,
            "repo": repo,
            "job": job,
            "response_id": response_id,
        }


if __name__ == "__main__":
    mcp.run()
