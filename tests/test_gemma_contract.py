"""Locks the Gemma 4 config contract confirmed on device by probe_gemma.py:
hidden_size = 3840 lives at config.text_config.hidden_size (not config.hidden_size)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from saaransh.embedders.gemma4_pooled import _hidden_size


def test_hidden_size_on_text_config():
    # Gemma4UnifiedConfig: top-level hidden_size is absent; it's under text_config.
    cfg = SimpleNamespace(text_config=SimpleNamespace(hidden_size=3840))
    assert _hidden_size(cfg) == 3840


def test_hidden_size_top_level():
    assert _hidden_size(SimpleNamespace(hidden_size=2560)) == 2560


def test_hidden_size_missing_raises():
    with pytest.raises(AttributeError):
        _hidden_size(SimpleNamespace())
