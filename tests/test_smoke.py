"""Smoke test for the nuagent package."""

import nuagent


def test_version():
    assert nuagent.__version__ == "0.1.0"


def test_public_surface():
    for name in nuagent.__all__:
        assert hasattr(nuagent, name), name
