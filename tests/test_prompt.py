"""Tests for nuagent.prompt: assembly, section surgery, prose rot, packaging."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import nu
import pytest

import nuagent
from nuagent.prompt import (
    DEFAULT_MODULES,
    DEFAULT_SECTIONS,
    PROSE,
    Section,
    app_records,
    bytes_methods,
    catalogue_section,
    inserted,
    read_markdown,
    render_catalogue,
    render_surface,
    replaced,
    surface_section,
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
        "finish",
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
        "finish",
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


# --- app surface ----------------------------------------------------------


class Widget(nu.Shape):
    """One widget on the wall.

    Notes:
        - `label` is what the UI prints, never the id.
    """

    label = nu.mem.StrRef.slot()
    weight = nu.mem.IntRef.slot()


class Wall(nu.Shape):
    """Every widget, by id."""

    widgets = nu.mem.ShapesDictRef.slot(Widget, str)
    total = nu.mem.IntRef.slot()


class Clock(nu.Service):
    """Wall clock, as a Service."""

    now = nu.service.QueryRef.method()


APP = (Wall, Widget, Clock)


def test_the_surface_renders_the_callers_shapes():
    text = render_surface(APP)
    assert "# Your app surface" in text
    for name in ("Wall", "Widget", "Clock"):
        assert f".{name}" in text, name


def test_every_entry_appears_with_its_type():
    text = render_surface(APP)
    for token in ("widgets", "ShapesDictRef", "total", "label", "weight", "now", "QueryRef"):
        assert token in text, token


def test_the_fabric_is_named_so_a_slot_can_be_redeclared():
    text = render_surface((Wall,))
    assert "(nu.mem)" in text
    assert "(nu.service)" in render_surface((Clock,))


def test_prose_and_notes_ride_along():
    text = render_surface((Widget,))
    assert "One widget on the wall." in text
    assert "- `label` is what the UI prints, never the id." in text


def test_no_method_table_leaks_in():
    # The whole point of the entry-summary level: a StrRef page is 83 lines
    # and six slots inlined is 400. If any of these show up, a Ref page got
    # expanded into the prompt.
    text = render_surface(APP)
    for leaked in (".set(...)", ".is_empty(...)", "CollectionResultT", "methods (", "-> Bool"):
        assert leaked not in text, leaked
    assert len(render_surface(APP, run=None).splitlines()) < 60


def test_a_module_contributes_every_shape_and_service_it_declares():
    import sys

    module = sys.modules[__name__]
    text = render_surface((module,))
    for name in ("Wall", "Widget", "Clock"):
        assert f".{name}" in text, name


def test_naming_a_class_twice_renders_it_once():
    import sys

    text = render_surface((sys.modules[__name__], Wall))
    # the caller's shapes, plus the Run block that always comes last
    assert text.count("## shape  ") == len(app_records((Wall, Widget))) + 1


def test_anything_that_is_not_a_declared_class_raises():
    with pytest.raises(TypeError):
        app_records((42,))


def test_an_empty_surface_says_so_rather_than_rendering_nothing():
    assert "nothing is bound" in render_surface(())


# --- the run shape rides on every surface ----------------------------------


def test_the_run_shape_is_on_the_surface_without_the_caller_naming_it():
    # The model cannot end the run without redeclaring it, so it is not a fact
    # the caller should have to remember at every call site.
    text = render_surface(APP)
    assert "shapes.Run" in text
    assert "  done               ref     BoolRef" in text
    assert text.index("shapes.Run") > text.index(".Clock")  # last, after the app


def test_the_run_shape_is_on_an_otherwise_empty_surface_too():
    assert "shapes.Run" in render_surface(())


def test_the_run_shape_can_be_swapped_or_dropped():
    assert "nu.kv" in render_surface((Wall,), run=nuagent.KVRun)
    assert "Run" not in render_surface((Wall,), run=None)


def test_the_section_carries_the_run_shape_through():
    text = system_prompt(TASK, sections=inserted(DEFAULT_SECTIONS, surface_section(APP)))
    assert "shapes.Run" in text
    assert "shapes.Run" not in system_prompt(
        TASK, sections=inserted(DEFAULT_SECTIONS, surface_section(APP, run=None))
    )


def test_the_surface_is_a_section_that_can_be_added_and_dropped():
    sections = inserted(DEFAULT_SECTIONS, surface_section(APP))
    assert [s.name for s in sections][-1] == "surface"
    text = system_prompt(TASK, sections=sections)
    assert "# Your app surface" in text
    assert text.index("# Your app surface") > text.index("# Catalogue")
    assert text.index("# Your app surface") < text.index("# Task")
    back = without(sections, "surface")
    assert [s.name for s in back] == [s.name for s in DEFAULT_SECTIONS]
    assert "# Your app surface" not in system_prompt(TASK, sections=back)


def test_the_surface_is_not_in_the_default_sections():
    # No caller, no app. The default prompt teaches the language only.
    assert "surface" not in [s.name for s in DEFAULT_SECTIONS]
