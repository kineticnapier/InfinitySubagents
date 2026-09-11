import tomllib

from model_config import _rewrite_model_text, parse_models_payload, set_model


def test_parse_native_model_list():
    models = parse_models_payload(
        {
            "models": [
                {
                    "key": "qwen/qwen3-14b",
                    "quantization": "Q4_K_M",
                    "state": "loaded",
                },
                {"id": "glm-test"},
            ]
        }
    )
    assert [item["id"] for item in models] == ["qwen/qwen3-14b", "glm-test"]
    assert models[0]["quantization"] == "Q4_K_M"


def test_rewrite_only_lmstudio_model():
    original = """[lmstudio]\nbase_url = \"http://127.0.0.1:1234\"\nmodel = \"old\"\n\n[other]\nmodel = \"leave-me\"\n"""
    updated = _rewrite_model_text(original, "new/model")
    assert 'model = "new/model"' in updated
    assert '[other]\nmodel = "leave-me"' in updated


def test_set_model_without_network(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        """[lmstudio]\nbase_url = \"http://127.0.0.1:1234\"\nmodel = \"old\"\ntoken_env = \"LM_API_TOKEN\"\nallow_remote_lmstudio = false\n""",
        encoding="utf-8",
    )

    result = set_model("new/model", config, verify=False)
    parsed = tomllib.loads(config.read_text(encoding="utf-8"))

    assert result["previous_model"] == "old"
    assert result["model"] == "new/model"
    assert result["changed"] is True
    assert result["restart_required"] is False
    assert parsed["lmstudio"]["model"] == "new/model"
