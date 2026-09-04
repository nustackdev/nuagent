"""Smoke test for the nuagent package."""

import nuagent


def test_version():
    assert nuagent.__version__ == "0.1.0"


def test_public_surface():
    for name in nuagent.__all__:
        assert hasattr(nuagent, name), name


def test_system_prompt_builds():
    # The old assertions here checked the prompt's own wording ("You are an
    # agent") and its module headers. The wording now lives in text/*.md and
    # belongs to whoever writes the prose, so this asserts the structural
    # facts instead: the catalogue covers the modules the agent composes
    # against, the lookup call is spelled right, and the task lands last.
    # The prose itself is checked in test_prompt.py.
    text = nuagent.system_prompt("Set the counter to 7.")
    for module in nuagent.DEFAULT_MODULES:
        assert f"## {module.__name__}" in text
    assert "## nu.core" in text
    assert "## nu.core.flows" in text
    assert "## nu.core.spans" in text
    assert "## nu.forms.primitives" in text
    assert "nu.inspect.Inspect" in text
    assert text.rstrip().endswith("Set the counter to 7.")
