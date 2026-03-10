"""Conftest for harness tests — skip markers based on availability."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    ollama_available = None
    minigrid_available = None
    gemini_available = None

    for item in items:
        if "ollama" in item.keywords:
            if ollama_available is None:
                ollama_available = _check_ollama()
            if not ollama_available:
                item.add_marker(
                    pytest.mark.skip(reason="Ollama server not available")
                )
        if "minigrid" in item.keywords:
            if minigrid_available is None:
                minigrid_available = _check_minigrid()
            if not minigrid_available:
                item.add_marker(
                    pytest.mark.skip(reason="minigrid not installed")
                )
        if "gemini" in item.keywords:
            if gemini_available is None:
                gemini_available = _check_gemini()
            if not gemini_available:
                item.add_marker(
                    pytest.mark.skip(reason="GEMINI_API_KEY not set")
                )


def _check_ollama() -> bool:
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _check_minigrid() -> bool:
    try:
        import minigrid  # noqa: F401

        return True
    except ImportError:
        return False


def _check_gemini() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))
