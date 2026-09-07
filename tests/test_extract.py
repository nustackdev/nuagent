"""Fenced-block extraction, run as Nu rather than asserted on a python helper.

Every case here goes through ``nu.run``, because the point of writing the
extractor as a Query is that it evaluates inside the turn. Testing a python
return value would test something the agent never uses.
"""

from __future__ import annotations

import nu
import pytest

from nuagent import fenced, fenceless


class Out(nu.Shape):
    text = nu.mem.StrRef.slot()
    flag = nu.mem.BoolRef.slot()


def run(reply: str, **kw) -> str:
    ctx = nu.Context().bind(dict, {})
    nu.run(Out.text.set(fenced(reply, **kw)), ctx)
    return ctx.get(dict)["text"]


def asks(reply: str) -> bool:
    ctx = nu.Context().bind(dict, {})
    nu.run(Out.flag.set(fenceless(reply)), ctx)
    return ctx.get(dict)["flag"]


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


# --- was there a block at all ----------------------------------------------


def test_fenceless_separates_a_block_from_prose():
    # The fallback cannot tell bare source from prose and does not try. This
    # is what lets the caller say "no code block" instead of reporting a
    # syntax error in an English sentence.
    assert not asks(FENCED)
    assert not asks(TWO_BLOCKS)
    assert asks("Done. Wrote the summaries and set the count.")
    assert asks("import nu")
    assert asks("here you go:\n```python\nx = 1")
    assert asks("")


def test_fenceless_is_a_nu_term():
    assert isinstance(fenceless("x"), nu.Nu)


# --- it is a term, not a helper --------------------------------------------


def test_it_composes_over_a_ref():
    ctx = nu.Context().bind(dict, {})
    nu.run(Out.text.set(FENCED) >> Out.text.set(fenced(Out.text)), ctx)
    assert ctx.get(dict)["text"] == "x = 1"


def test_it_is_a_nu_term():
    assert isinstance(fenced("x"), nu.Nu)
