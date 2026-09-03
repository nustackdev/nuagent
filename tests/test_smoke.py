"""Smoke test for the nuagent package."""

import nuagent


def test_version():
    assert nuagent.__version__ == "0.1.0"


def test_public_surface():
    for name in nuagent.__all__:
        assert hasattr(nuagent, name), name


def test_system_prompt_builds():
    text = nuagent.system_prompt("Set the counter to 7.")
    assert "You are an agent" in text
    assert "## nu.core" in text
    assert "## nu.flows" in text
    assert "## nu.spans" in text
    assert "## nu.forms.primitives" in text
    assert "nu.info.Inspect" in text
    assert text.rstrip().endswith("Set the counter to 7.")
