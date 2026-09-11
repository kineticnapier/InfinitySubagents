from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.toml"
EXAMPLE_CONFIG_PATH = HERE / "config.example.toml"


@dataclass(frozen=True)
class ModelSettings:
    base_url: str
    current_model: str
    token_env: str | None
    timeout_seconds: int
    allow_remote_lmstudio: bool


def ensure_config(path: Path = CONFIG_PATH) -> Path:
    if path.exists():
        return path
    if not EXAMPLE_CONFIG_PATH.exists():
        raise RuntimeError(f"Missing {path.name} and {EXAMPLE_CONFIG_PATH.name}")
    shutil.copyfile(EXAMPLE_CONFIG_PATH, path)
    return path


def read_settings(path: Path = CONFIG_PATH) -> ModelSettings:
    ensure_config(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    section = raw.get("lmstudio", {})
    settings = ModelSettings(
        base_url=str(section.get("base_url", "http://127.0.0.1:1234")).rstrip("/"),
        current_model=str(section.get("model", "")).strip(),
        token_env=(str(section.get("token_env", "LM_API_TOKEN")).strip() or None),
        timeout_seconds=max(1, int(section.get("timeout_seconds", 30))),
        allow_remote_lmstudio=bool(section.get("allow_remote_lmstudio", False)),
    )
    _validate_base_url(settings.base_url, settings.allow_remote_lmstudio)
    return settings


def _validate_base_url(base_url: str, allow_remote: bool) -> None:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError("LM Studio base_url must use http or https")
    if not parsed.hostname:
        raise RuntimeError("LM Studio base_url is missing a hostname")
    if not allow_remote and parsed.hostname.lower() not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise RuntimeError(
            "Remote LM Studio hosts are disabled. "
            "Set allow_remote_lmstudio=true only if intended."
        )


def _request_models_payload(settings: ModelSettings) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if settings.token_env:
        token = os.getenv(settings.token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{settings.base_url}/api/v1/models",
        method="GET",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(
            request, timeout=min(settings.timeout_seconds, 30)
        ) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio HTTP {e.code}: {payload[:4000]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach LM Studio at {settings.base_url}: {e}"
        ) from e
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError("LM Studio returned invalid JSON") from e
    if not isinstance(parsed, dict):
        raise RuntimeError("LM Studio model list response was not a JSON object")
    return parsed


def _model_id(item: dict[str, Any]) -> str:
    for key in ("key", "id", "model", "name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_models_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        return []

    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_models:
        if isinstance(item, str):
            model_id = item.strip()
            source: dict[str, Any] = {}
        elif isinstance(item, dict):
            source = item
            model_id = _model_id(source)
        else:
            continue
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        entry: dict[str, Any] = {"id": model_id}
        for key in ("type", "quantization", "state", "arch", "publisher"):
            value = source.get(key)
            if value not in (None, ""):
                entry[key] = value
        models.append(entry)
    return models


def list_models(path: Path = CONFIG_PATH) -> dict[str, Any]:
    settings = read_settings(path)
    payload = _request_models_payload(settings)
    models = parse_models_payload(payload)
    return {
        "current_model": settings.current_model,
        "models": models,
        "count": len(models),
    }


_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(?:#.*)?(?:\r?\n)?$")
_MODEL_RE = re.compile(r"^(\s*)model\s*=")


def _rewrite_model_text(text: str, model: str) -> str:
    if not model or "\n" in model or "\r" in model:
        raise ValueError("model id must be a non-empty single line")

    lines = text.splitlines(keepends=True)
    section_start: int | None = None
    section_end = len(lines)
    for i, line in enumerate(lines):
        match = _SECTION_RE.match(line)
        if not match:
            continue
        if match.group(1).strip() == "lmstudio":
            section_start = i
            for j in range(i + 1, len(lines)):
                if _SECTION_RE.match(lines[j]):
                    section_end = j
                    break
            break

    if section_start is None:
        raise RuntimeError("config.toml is missing [lmstudio]")

    encoded = json.dumps(model, ensure_ascii=False)
    default_newline = "\r\n" if "\r\n" in text else "\n"
    for i in range(section_start + 1, section_end):
        match = _MODEL_RE.match(lines[i])
        if not match:
            continue
        ending = (
            "\r\n"
            if lines[i].endswith("\r\n")
            else ("\n" if lines[i].endswith("\n") else default_newline)
        )
        lines[i] = f"{match.group(1)}model = {encoded}{ending}"
        return "".join(lines)

    lines.insert(section_start + 1, f"model = {encoded}{default_newline}")
    return "".join(lines)


def set_model(
    model: str,
    path: Path = CONFIG_PATH,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    model = model.strip()
    settings = read_settings(path)

    if verify:
        available = list_models(path)["models"]
        ids = {str(item["id"]) for item in available}
        if model not in ids:
            shown = ", ".join(sorted(ids)[:20]) or "(none)"
            raise ValueError(
                f"Model {model!r} was not returned by LM Studio. Available: {shown}"
            )

    original = path.read_text(encoding="utf-8")
    updated = _rewrite_model_text(original, model)
    if updated != original:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(updated, encoding="utf-8", newline="")
        os.replace(tmp, path)

    return {
        "ok": True,
        "previous_model": settings.current_model,
        "model": model,
        "changed": updated != original,
        "restart_required": False,
    }


def _print_models(result: dict[str, Any]) -> None:
    current = str(result.get("current_model", ""))
    models = result.get("models", [])
    print(f"Current: {current or '(not set)'}")
    if not models:
        print("No models returned by LM Studio.")
        return
    for index, item in enumerate(models, 1):
        model_id = str(item["id"])
        suffix: list[str] = []
        if model_id == current:
            suffix.append("current")
        if item.get("quantization"):
            suffix.append(str(item["quantization"]))
        if item.get("state"):
            suffix.append(str(item["state"]))
        label = f" [{' | '.join(suffix)}]" if suffix else ""
        print(f"{index:>2}. {model_id}{label}")


def select_model(path: Path = CONFIG_PATH) -> dict[str, Any]:
    result = list_models(path)
    models = result["models"]
    _print_models(result)
    if not models:
        raise RuntimeError("LM Studio returned no selectable models")

    current = str(result.get("current_model", ""))
    default_index = next(
        (i for i, item in enumerate(models, 1) if item["id"] == current),
        None,
    )
    if default_index is None:
        prompt = f"Select model [1-{len(models)}]: "
    else:
        prompt = f"Select model [1-{len(models)}] (Enter keeps {default_index}): "

    choice = input(prompt).strip()
    if not choice and default_index is not None:
        choice = str(default_index)
    try:
        index = int(choice)
    except ValueError as e:
        raise ValueError("Selection must be a model number") from e
    if index < 1 or index > len(models):
        raise ValueError("Selection is out of range")
    return set_model(str(models[index - 1]["id"]), path, verify=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List and switch the LM Studio model used by Local Subagent MCP."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List models returned by LM Studio")
    subparsers.add_parser("current", help="Print the configured model")

    set_parser = subparsers.add_parser("set", help="Set an exact LM Studio model id")
    set_parser.add_argument("model")
    set_parser.add_argument(
        "--force",
        action="store_true",
        help="Set without checking the LM Studio model list",
    )

    subparsers.add_parser("select", help="Interactively select an LM Studio model")

    args = parser.parse_args(argv)
    command = args.command or "select"
    try:
        if command == "list":
            _print_models(list_models())
        elif command == "current":
            print(read_settings().current_model or "(not set)")
        elif command == "set":
            result = set_model(args.model, verify=not args.force)
            print(f"Model: {result['previous_model'] or '(not set)'} -> {result['model']}")
        elif command == "select":
            result = select_model()
            print(f"Model: {result['previous_model'] or '(not set)'} -> {result['model']}")
        else:
            parser.error(f"Unknown command: {command}")
    except (OSError, RuntimeError, ValueError) as e:
        parser.exit(1, f"error: {e}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
