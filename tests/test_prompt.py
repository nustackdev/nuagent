"""Tests for nuagent.prompt: assembly, section surgery, prose rot, packaging."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import nuagent
from nuagent.prompt import (
    DEFAULT_MODULES,
    DEFAULT_SECTIONS,
    PROSE,
    Section,
    bytes_methods,
    catalogue_section,
    inserted,
    read_markdown,
    render_catalogue,
    replaced,
    system_prompt,
    without,
)
from nuagent.prompt.sections import text_files


TASK = "Set the counter to 7."
REPO = Path(__file__).resolve().parents[1]


# --- assembly -------------------------------------------------------------


def test_prompt_builds():
    text = system_prompt(TASK)
    assert text.strip()
    assert len(text) > 10_000


def test_every_section_appears_in_order():
    text = system_prompt(TASK)
    at = -1
    for section in DEFAULT_SECTIONS:
        found = text.find(section.render())
        assert found != -1, f"{section.name} missing from the prompt"
        assert found > at, f"{section.name} out of order"
        at = found


def test_section_order_is_the_agreed_order():
    assert [s.name for s in DEFAULT_SECTIONS] == [*PROSE, "catalogue"]
    assert list(PROSE) == [
        "role",
        "thesis",
        "fabrics",
        "crashcourse",
        "examples",
        "protocol",
        "inspect",
    ]


def test_task_goes_last():
    text = system_prompt(TASK)
    assert text.rstrip().endswith(TASK)
    assert text.index("# Task") > text.index(DEFAULT_SECTIONS[-1].render())


def test_task_is_stripped_not_formatted():
    # Braces in the task must survive; nothing here runs str.format.
    text = system_prompt("  write {'a': 1} into the store  ")
    assert text.rstrip().endswith("write {'a': 1} into the store")


def test_prose_is_never_run_through_format():
    # The prose carries dict literals; .format would raise on them. Assert the
    # braces arrive in the prompt exactly as written on disk.
    for name in PROSE:
        raw = read_markdown(f"{name}.md")
        if "{" in raw:
            assert raw.strip("\n") in system_prompt(TASK)


# --- section surgery ------------------------------------------------------


def test_a_section_can_be_dropped():
    trimmed = without(DEFAULT_SECTIONS, "crashcourse", "fabrics")
    assert [s.name for s in trimmed] == [
        "role",
        "thesis",
        "examples",
        "protocol",
        "inspect",
        "catalogue",
    ]
    text = system_prompt(TASK, sections=trimmed)
    assert read_markdown("crashcourse.md").strip("\n") not in text
    assert read_markdown("role.md").strip("\n") in text


def test_dropping_an_absent_name_is_a_no_op():
    assert without(DEFAULT_SECTIONS, "nope") == DEFAULT_SECTIONS


def test_a_section_can_be_replaced():
    swapped = replaced(DEFAULT_SECTIONS, Section("inspect", lambda: "LOOK IT UP YOURSELF"))
    assert [s.name for s in swapped] == [s.name for s in DEFAULT_SECTIONS]
    text = system_prompt(TASK, sections=swapped)
    assert "LOOK IT UP YOURSELF" in text
    assert read_markdown("inspect.md").strip("\n") not in text


def test_the_catalogue_is_replaceable_to_add_a_fabric():
    import nu.mem

    swapped = replaced(DEFAULT_SECTIONS, catalogue_section((*DEFAULT_MODULES, nu.mem)))
    text = system_prompt(TASK, sections=swapped)
    assert "## nu.mem" in text
    assert "IntRef" in text


def test_a_section_can_be_inserted():
    extra = Section("house-rules", lambda: "# House rules\n\nno bytes.")
    before = inserted(DEFAULT_SECTIONS, extra, before="catalogue")
    assert [s.name for s in before][-2:] == ["house-rules", "catalogue"]
    after = inserted(DEFAULT_SECTIONS, extra, after="role")
    assert [s.name for s in after][:3] == ["role", "house-rules", "thesis"]
    appended = inserted(DEFAULT_SECTIONS, extra)
    assert appended[-1].name == "house-rules"
    assert "no bytes." in system_prompt(TASK, sections=before)


def test_inserting_against_a_missing_anchor_raises():
    with pytest.raises(KeyError):
        inserted(DEFAULT_SECTIONS, Section("x", lambda: "x"), before="nope")


def test_sections_can_be_reordered():
    reversed_sections = tuple(reversed(DEFAULT_SECTIONS))
    text = system_prompt(TASK, sections=reversed_sections)
    assert text.index(DEFAULT_SECTIONS[-1].render()) < text.index(DEFAULT_SECTIONS[0].render())


# --- catalogue ------------------------------------------------------------


def test_catalogue_lists_the_atoms_that_have_burned_turn_budgets():
    text = render_catalogue()
    # Every one of these was a documented unwinnable-game failure or is named
    # by the prose sections. None may fall out of the module set unnoticed.
    for name in ("ForEachDo", "ForRangeDo", "WhileDo", "AttrRef", "Literal", "Len", "Inspect"):
        assert re.search(rf"^\s+{name}\s", text, re.MULTILINE), name


def test_forms_are_not_listed_twice():
    text = render_catalogue((__import__("nu.forms.collections", fromlist=["x"]),))
    for name in ("List", "Dict", "Set", "Tuple"):
        assert len(re.findall(rf"^\s+{name}\s", text, re.MULTILINE)) == 1, name


def test_nothing_is_omitted_by_default():
    assert "omitted" not in render_catalogue()
    assert "BytesUpper" in render_catalogue()


def test_an_omission_is_announced_not_silent():
    text = render_catalogue(omit=bytes_methods)
    assert "BytesUpper" not in text
    assert "38 more omitted" in text
    assert 'nu.inspect.Inspect("nu.forms.primitives")' in text


def test_default_modules_do_not_duplicate_each_other():
    from nu.inspect import catalogue_forms, catalogue_interactions, catalogue_refs

    seen: dict[str, str] = {}
    collisions = []
    for module in DEFAULT_MODULES:
        for group in (catalogue_forms, catalogue_refs, catalogue_interactions):
            for record in group(module):
                if record.name in seen:
                    collisions.append((record.name, seen[record.name], module.__name__))
                seen[record.name] = module.__name__
    # nu.core and nu.forms.primitives both export Hex. That is the only one.
    assert [c[0] for c in collisions] == ["Hex"]


# --- prose rot ------------------------------------------------------------


def _python_blocks(markdown: str) -> list[str]:
    """Every ```python / ```py fenced block, in order."""
    blocks = []
    current: list[str] | None = None
    for line in markdown.splitlines():
        stripped = line.strip()
        if current is None:
            if stripped in ("```python", "```py"):
                current = []
            continue
        if stripped == "```":
            blocks.append("\n".join(current))
            current = None
            continue
        current.append(line)
    return blocks


@pytest.mark.parametrize("filename", text_files())
def test_every_fenced_python_block_compiles(filename):
    blocks = _python_blocks(read_markdown(filename))
    for i, block in enumerate(blocks, start=1):
        try:
            compile(block, f"{filename}#{i}", "exec")
        except SyntaxError as exc:
            pytest.fail(f"{filename} block {i} does not compile: {exc}\n\n{block}")


def test_there_are_python_blocks_to_check():
    # Guard against the parametrize silently collapsing to nothing.
    assert text_files()
    total = sum(len(_python_blocks(read_markdown(f))) for f in text_files())
    assert total >= 5


# --- packaging ------------------------------------------------------------


def test_every_prose_section_has_a_file():
    shipped = set(text_files())
    missing = [f"{name}.md" for name in PROSE if f"{name}.md" not in shipped]
    assert not missing, f"missing prose: {missing}"


def test_prose_is_reachable_as_a_resource():
    # importlib.resources, not __file__ math: this is what has to work inside
    # a wheel and under an editable install both.
    for name in PROSE:
        assert read_markdown(f"{name}.md").strip()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_the_wheel_actually_contains_the_prose(tmp_path):
    subprocess.run(  # noqa: S603
        [shutil.which("uv"), "build", "--wheel", "-o", str(tmp_path)],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    (wheel,) = tmp_path.glob("nuagent-*.whl")
    names = set(zipfile.ZipFile(wheel).namelist())
    for name in PROSE:
        assert f"nuagent/prompt/text/{name}.md" in names


def test_the_public_surface_is_importable():
    assert sys.modules["nuagent"] is nuagent
    for name in nuagent.__all__:
        assert hasattr(nuagent, name), name
