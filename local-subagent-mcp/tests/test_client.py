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


def test_research_mode_is_read_only_and_can_find_files():
    body = build_chat_request(cfg(), task="inspect repo", mode="research", repo="demo")
    allowed = body["integrations"][0]["allowed_tools"]
    assert "read_file" in allowed
    assert "find_files" in allowed
    assert "apply_patch" not in allowed
    assert "write_text_file" not in allowed
    assert "commit_job" not in allowed


def test_code_mode_has_model_friendly_managed_write_tools_but_no_delete():
    body = build_chat_request(cfg(), task="fix bug", mode="code", repo="demo")
    allowed = body["integrations"][0]["allowed_tools"]
    assert "create_job" in allowed
    assert "write_text_file" in allowed
    assert "replace_text" in allowed
    assert "apply_patch" in allowed
    assert "run_tests" in allowed
    assert "revert_to_latest_checkpoint" in allowed
    assert "revert_to_checkpoint" in allowed
    assert "commit_job" in allowed
    assert "delete_job" not in allowed


def test_caps_are_enforced():
    with pytest.raises(ValueError):
        build_chat_request(cfg(), task="x", mode="code", max_output_tokens=4097)
    with pytest.raises(ValueError):
        build_chat_request(cfg(), task="x", mode="code", context_length=9000)


def test_response_drops_reasoning_and_raw_tool_outputs_but_keeps_audit_trace():
    payload = {
        "model_instance_id": "qwen",
        "response_id": "resp_123",
        "output": [
            {"type": "reasoning", "content": "very long private scratchpad"},
            {
                "type": "tool_call",
                "tool": "read_file",
                "arguments": {"path": "a.py", "repo": "demo"},
                "output": '{"ok":true,"target":"demo","path":"a.py","start_line":1,"end_line":2,"total_lines":2,"content":"hello\\nworld\\n","ends_with_newline":true,"source_ends_with_newline":true}',
            },
            {
                "type": "tool_call",
                "tool": "apply_patch",
                "arguments": {"job": "j1", "patch": "SECRET PATCH BODY\n" * 100},
                "output": '{"ok":true,"job":"j1","paths":["a.py"],"head":"abc","diff_stat":"1 file changed"}',
            },
            {"type": "message", "content": "done"},
        ],
        "stats": {"input_tokens": 100, "total_output_tokens": 200, "tokens_per_second": 25.5},
    }
    result = compact_response(cfg(), payload)
    assert result["final"] == "done"
    assert result["tool_calls"] == {"read_file": 1, "apply_patch": 1}
    assert len(result["tool_trace"]) == 2
    assert result["tool_trace"][0]["outcome"] == "ok"
    assert result["tool_trace"][0]["path"] == "a.py"
    assert result["tool_trace"][0]["ends_with_newline"] is True
    assert result["tool_trace"][0]["source_ends_with_newline"] is True
    assert result["tool_trace"][0]["content_preview"] == "hello\\nworld\\n"
    assert result["tool_trace"][1]["arguments"]["patch_chars"] > 0
    assert "SECRET PATCH BODY" not in str(result)
    assert "private scratchpad" not in str(result)


def test_structured_tool_error_is_visible_to_parent():
    payload = {
        "output": [
            {
                "type": "tool_call",
                "tool": "read_file",
                "arguments": {"path": "missing.py", "repo": "demo"},
                "output": '{"ok":false,"error_type":"file_not_found","error":"File does not exist","target":"demo","path":"missing.py"}',
            },
            {"type": "message", "content": "blocked"},
        ]
    }
    result = compact_response(cfg(), payload)
    trace = result["tool_trace"][0]
    assert trace["tool"] == "read_file"
    assert trace["arguments"]["path"] == "missing.py"
    assert trace["outcome"] == "error"
    assert trace["error_type"] == "file_not_found"
    assert trace["error"] == "File does not exist"
    assert trace["path"] == "missing.py"


def test_legacy_transport_tool_error_is_still_visible_to_parent():
    payload = {
        "output": [
            {
                "type": "tool_call",
                "tool": "read_file",
                "arguments": {"path": "missing.py", "repo": "demo"},
                "output": "UnexpectedToolError: Error executing tool read_file: File does not exist",
            },
            {"type": "message", "content": "blocked"},
        ]
    }
    result = compact_response(cfg(), payload)
    trace = result["tool_trace"][0]
    assert trace["outcome"] == "error"
    assert "does not exist" in trace["error"]


def test_find_files_trace_lists_observed_paths():
    payload = {
        "output": [
            {
                "type": "tool_call",
                "tool": "find_files",
                "arguments": {"repo": "demo", "pattern": "*.py", "path": "src"},
                "output": '{"ok":true,"target":"demo","pattern":"*.py","results":[{"path":"src/a.py"},{"path":"src/b.py"}],"truncated":false}',
            },
            {"type": "message", "content": "done"},
        ]
    }
    trace = compact_response(cfg(), payload)["tool_trace"][0]
    assert trace["outcome"] == "ok"
    assert trace["arguments"]["pattern"] == "*.py"
    assert trace["result_count"] == 2
    assert trace["matches"] == ["src/a.py", "src/b.py"]


def test_sensitive_simple_edit_arguments_are_hashed_not_forwarded():
    payload = {
        "output": [
            {
                "type": "tool_call",
                "tool": "write_text_file",
                "arguments": {
                    "job": "j1",
                    "path": "note.txt",
                    "content": "TOP SECRET CONTENT\n",
                    "overwrite": False,
                },
                "output": '{"ok":true,"job":"j1","path":"note.txt","chars":19,"bytes":19,"ends_with_newline":true,"status":"?? note.txt"}',
            },
            {
                "type": "tool_call",
                "tool": "replace_text",
                "arguments": {
                    "job": "j1",
                    "path": "README.md",
                    "old": "VERY SECRET OLD",
                    "new": "VERY SECRET NEW",
                },
                "output": '{"ok":true,"job":"j1","path":"README.md","replacements":1,"chars":100,"ends_with_newline":true,"diff_stat":" README.md | 2 +-"}',
            },
            {"type": "message", "content": "done"},
        ]
    }
    result = compact_response(cfg(), payload)
    write_args = result["tool_trace"][0]["arguments"]
    replace_args = result["tool_trace"][1]["arguments"]
    assert write_args["content_chars"] == len("TOP SECRET CONTENT\n")
    assert write_args["content_ends_with_newline"] is True
    assert "content_sha256" in write_args
    assert replace_args["old_chars"] == len("VERY SECRET OLD")
    assert replace_args["new_chars"] == len("VERY SECRET NEW")
    assert "TOP SECRET CONTENT" not in str(result)
    assert "VERY SECRET OLD" not in str(result)
    assert "VERY SECRET NEW" not in str(result)


def test_git_status_preview_is_bounded_and_visible():
    status = "## localdev/test\n?? LOCALDEV_SMOKE_TEST.txt\n"
    payload = {
        "output": [
            {
                "type": "tool_call",
                "tool": "git_status",
                "arguments": {"job": "smoke"},
                "output": '{"ok":true,"exit_code":0,"stdout":"' + status.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"') + '","stderr":"","truncated":false}',
            },
            {"type": "message", "content": "done"},
        ]
    }
    result = compact_response(cfg(), payload)
    trace = result["tool_trace"][0]
    assert trace["outcome"] == "ok"
    assert trace["exit_code"] == 0
    assert "LOCALDEV_SMOKE_TEST.txt" in trace["stdout_preview"]


def test_bad_response_id_rejected():
    with pytest.raises(ValueError):
        build_chat_request(
            cfg(), task="continue", mode="research", previous_response_id="bad"
        )
