"""Extraction and one attempt at the program, run as Nu rather than as helpers.

Every case goes through ``nu.run``, because the point of writing these as
terms is that they evaluate inside the turn. Testing a python return value
would test something the agent never uses.
"""

from __future__ import annotations

import asyncio
import textwrap

import nu
import nustd
import pytest

from nuagent import attempted, fenced


class Out(nu.Shape):
    text = nustd.mem.StrRef.slot()


class World(nu.Shape):
    notes = nustd.mem.ListRef.slot(str)


class Loop(nu.Shape):
    draft = nustd.mem.ProgramRef.slot()
    outcome = nustd.mem.StrRef.slot()


def run(reply: str, **kw) -> str:
    ctx = nu.Context().bind(dict, {})
    nu.run(Out.text.set(fenced(reply, **kw)), ctx)
    return ctx.get(dict)["text"]


FENCED = "prose\n```python\nx = 1\n```\nmore prose"
TWO_BLOCKS = "broken:\n```py\nfirst\n```\nfixed:\n```python\nsecond\n```"


# --- the ordinary case ------------------------------------------------------


def test_it_takes_the_source_out_of_a_block():
    assert run(FENCED) == "x = 1"


def test_the_language_tag_goes_with_the_fence():
    assert run("```python\na\n```") == "a"
    assert run("```py\na\n```") == "a"
    assert run("```\na\n```") == "a"


def test_a_multi_line_module_survives_intact():
    src = "import nu\n\n\ndef out():\n    return nu.Int(1)"
    assert run(f"here:\n```python\n{src}\n```") == src


# --- first or last ----------------------------------------------------------


def test_first_is_the_default():
    assert run(TWO_BLOCKS) == "first"


def test_last_picks_the_correction():
    # A model quoting its broken version and then the fix is the case that
    # makes first-block lose a turn.
    assert run(TWO_BLOCKS, block="last") == "second"


def test_one_block_reads_the_same_either_way():
    assert run(FENCED, block="first") == run(FENCED, block="last") == "x = 1"


def test_three_blocks_pick_the_ends():
    reply = "```\na\n```\n```\nb\n```\n```\nc\n```"
    assert run(reply) == "a"
    assert run(reply, block="last") == "c"


def test_any_other_choice_is_refused():
    with pytest.raises(ValueError, match="first"):
        fenced("x", block="middle")


# --- no block ---------------------------------------------------------------


def test_a_bare_reply_is_used_whole():
    # Common enough to be worth handling, and the failure if it really was
    # prose is a Diagnostic, which is a turn the agent recovers from.
    assert run("  import nu  ") == "import nu"


def test_an_unclosed_fence_falls_back():
    assert run("here you go:\n```python\nx = 1") == "here you go:\n```python\nx = 1"


def test_an_empty_reply_stays_empty():
    assert run("") == ""


# --- it is a term, not a helper --------------------------------------------


def test_it_composes_over_a_ref():
    ctx = nu.Context().bind(dict, {})
    nu.run(Out.text.set(FENCED) >> Out.text.set(fenced(Out.text)), ctx)
    assert ctx.get(dict)["text"] == "x = 1"


def test_it_is_a_nu_term():
    assert isinstance(fenced("x"), nu.Nu)


# --- one attempt at the program --------------------------------------------


def src(text: str) -> str:
    return textwrap.dedent(text).lstrip()


APPENDS = src("""
    import nu
    import nustd


    class World(nu.Shape):
        notes = nustd.mem.ListRef.slot(str)


    def out():
        return World.notes.append("a")
""")

YIELDS = src("""
    import nu


    def out():
        return nu.Int(1) + 41
""")

BROKEN = src("""
    import nu


    def out():
        return nu.Len(nu.List([1, 2, 3])) >= 3
""")

# Builds perfectly and dies on evaluation: concatenating a str term with an
# int is a TypeError deep in the arithmetic thunk, not a construction error.
CRASHES = src("""
    import nu
    import nustd


    class World(nu.Shape):
        notes = nustd.mem.ListRef.slot(str)


    def out():
        return World.notes.append(nu.Str("n") + 1)
""")


def attempt(source: str) -> dict:
    app = nu.With(
        nu.Provide(nu.prog.PyBrace, {}),
        body=(
            World.notes.set([])
            >> Loop.draft.set(source)
            >> Loop.outcome.set(nu.Str(attempted(Loop.draft)))
        ),
    )
    ctx = nu.Context().bind(dict, {})
    asyncio.run(nu.arun(app, ctx))
    return ctx.get(dict)


def test_a_command_yields_nothing_so_state_is_the_evidence():
    out = attempt(APPENDS)
    assert out["outcome"] == "None"
    assert out["notes"] == ["a"]


def test_a_query_comes_back_rendered():
    assert attempt(YIELDS)["outcome"] == "42"


def test_a_construction_failure_is_a_value_not_a_raise():
    # `nu.Len(x) >= 3` is a real thing models write and it does not build.
    # The diagnostic returns through the same slot a yield does, so a failed
    # turn is just another turn.
    out = attempt(BROKEN)
    assert out["outcome"].startswith("CONSTRUCTION FAILED")
    assert not out["outcome"].startswith('"')  # not repr-quoted: the model reads this
    assert "line" in out["outcome"]


def test_a_program_that_builds_and_then_raises_is_also_a_value():
    # The failure that motivated the outer catch: a real model wrote a term
    # that constructed fine and raised at eval, and the traceback went up
    # through the turn and the loop and killed the agent.
    out = attempt(CRASHES)
    assert out["outcome"].startswith("RUNTIME FAILED")
    assert "concatenate" in out["outcome"]
    assert out["notes"] == []  # the world survived
