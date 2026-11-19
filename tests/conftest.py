"""Pytest configuration and common fixtures for NoticeWise."""

import os
import pytest

@pytest.fixture(autouse=True)
def disable_extract_cache_for_tests(monkeypatch: pytest.MonkeyPatch):
    """Ensure disk extraction cache is disabled during unit tests unless explicitly enabled."""
    monkeypatch.setenv("DISABLE_EXTRACT_CACHE", "1")
