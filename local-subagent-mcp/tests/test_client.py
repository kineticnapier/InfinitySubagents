from pathlib import Path

import pytest

from client import Config, build_chat_request, compact_response


def cfg() -> Config:
    return Config(
        base_url="http://127.0.0.1:1234",
        model="qwen-test",
        integration_id="mcp/localdev",
        timeout_seconds=60,
        context_length=8192,
        max_output_tokens=4096,
        temperature=0.2,
        reasoning=None,
        token_env=None,
        max_parallel=1,
        max_task_chars=50000,
        max_result_chars=20000,
        allow_remote_lmstudio=False,
    )


def test_research_mode_is_read_only():
    body = build_chat_request(cfg(), task="inspect repo", mode="research", repo="demo")
    allowed = body["integrations"][0]["allowed_tools"]
    assert "read_file" in allowed
    assert "apply_patch" not in allowed
    assert "commit_job" not in allowed


def test_code_mode_has_managed_write_tools_but_no_delete():
    body = build_chat_request(cfg(), task="fix bug", mode="code", repo="demo")
    allowed = body["integrations"][0]["allowed_tools"]
    assert "create_job" in allowed
    assert "apply_patch" in allowed
    assert "run_tests" in allowed
    assert "commit_job" in allowed
    assert "delete_job" not in allowed


def test_caps_are_enforced():
    with pytest.raises(ValueError):
        build_chat_request(cfg(), task="x", mode="code", max_output_tokens=4097)
    with pytest.raises(ValueError):
        build_chat_request(cfg(), task="x", mode="code", context_length=9000)


def test_response_drops_reasoning_and_raw_tool_outputs():
    payload = {
        "model_instance_id": "qwen",
        "response_id": "resp_123",
        "output": [
            {"type": "reasoning", "content": "very long private scratchpad"},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "a"}, "output": "huge"},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "b"}, "output": "huge2"},
            {"type": "message", "content": "done"},
        ],
        "stats": {"input_tokens": 100, "total_output_tokens": 200, "tokens_per_second": 25.5},
    }
    result = compact_response(cfg(), payload)
    assert result["final"] == "done"
    assert result["tool_calls"] == {"read_file": 2}
    assert "private scratchpad" not in str(result)
    assert "huge2" not in str(result)


def test_bad_response_id_rejected():
    with pytest.raises(ValueError):
        build_chat_request(
            cfg(), task="continue", mode="research", previous_response_id="bad"
        )
