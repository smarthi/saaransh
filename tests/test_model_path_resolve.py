"""Tests for the forgiving model-path resolver (cache_dir vs --colqwen-model mix-up)."""

from __future__ import annotations

from saaransh.embedders.colqwen2_backbone import _resolve_local_model


def test_cache_dir_pointing_at_model_folder(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    model, cache = _resolve_local_model("vidore/colqwen2-v1.0-hf", str(tmp_path))
    assert model == str(tmp_path)  # treated as the model path
    assert cache is None


def test_real_cache_dir_left_alone(tmp_path):
    # a hub-cache dir has no top-level config.json -> leave both as-is
    model, cache = _resolve_local_model("vidore/colqwen2-v1.0-hf", str(tmp_path))
    assert model == "vidore/colqwen2-v1.0-hf"
    assert cache == str(tmp_path)


def test_no_cache_dir():
    model, cache = _resolve_local_model("vidore/colqwen2-v1.0-hf", None)
    assert model == "vidore/colqwen2-v1.0-hf"
    assert cache is None
