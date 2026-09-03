"""What the model reads back, including the case where its module did not build.

No LLM here. The turn's second half is pure Nu, so a stored source string is
enough to exercise the whole path a model's reply takes after extraction.
"""

from __future__ import annotations

import asyncio
import textwrap

import nu

from nuagent.observation import attempted, observation


class World(nu.Shape):
    notes = nu.mem.ListRef.slot(str)


class Loop(nu.Shape):
    draft = nu.mem.ProgramRef.slot()
    outcome = nu.mem.StrRef.slot()
    observed = nu.mem.StrRef.slot()


def src(text: str) -> str:
    return textwrap.dedent(text).lstrip()


APPENDS = src("""
    import nu


    class World(nu.Shape):
        notes = nu.mem.ListRef.slot(str)


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


    class World(nu.Shape):
        notes = nu.mem.ListRef.slot(str)


    def out():
        return World.notes.append(nu.Str("n") + 1)
""")


def drive(source: str, state=None, verdict=None) -> dict:
    result = attempted(Loop.draft)
    app = nu.With(
        nu.Provide(nu.prog.PyBrace, {}),
        body=(
            World.notes.set([])
            >> Loop.draft.set(source)
            >> Loop.outcome.set(nu.Str(result))
            >> Loop.observed.set(observation(Loop.outcome, state, verdict=verdict))
        ),
    )
    ctx = nu.Context().bind(dict, {})
    asyncio.run(nu.arun(app, ctx))
    return ctx.get(dict)


def test_a_command_yields_nothing_so_state_is_the_evidence():
    out = drive(APPENDS, nu.Dict.of(notes=World.notes))
    assert out["outcome"] == "None"
    assert out["observed"] == "outcome: None\nstate: {'notes': ['a']}"


def test_a_query_needs_no_state_at_all():
    out = drive(YIELDS)
    assert out["outcome"] == "42"
    assert out["observed"] == "outcome: 42"


def test_a_failure_comes_back_as_an_observation_not_a_raise():
    # `nu.Len(x) >= 3` is a real thing models write and it does not build.
    # The diagnostic returns through the same slot a yield does, so a
    # failed turn is just another turn.
    out = drive(BROKEN)
    assert out["outcome"].startswith("CONSTRUCTION FAILED")
    assert not out["outcome"].startswith('"')  # not repr-quoted: the model reads this
    assert "line" in out["outcome"]
    assert out["observed"].startswith("outcome: CONSTRUCTION FAILED")


def test_a_program_that_builds_and_then_raises_is_also_just_an_observation():
    # The failure that motivated the outer catch: a real model wrote a term
    # that constructed fine and raised at eval, and the traceback went up
    # through the turn and the loop and killed the agent. A model cannot fix
    # what it never sees, so a runtime error has to come back like any other.
    out = drive(CRASHES, nu.Dict.of(notes=World.notes))
    assert out["outcome"].startswith("RUNTIME FAILED")
    assert "concatenate" in out["outcome"]
    assert out["observed"].startswith("outcome: RUNTIME FAILED")
    assert "'notes': []" in out["observed"]  # the world survived


def test_the_verdict_tells_the_model_it_was_wrong():
    # Working-but-wrong code is otherwise indistinguishable from success.
    out = drive(APPENDS, nu.Dict.of(notes=World.notes), verdict=nu.Str("NOT met"))
    assert out["observed"].endswith("goal: NOT met")


def test_the_yield_is_truncated_and_the_state_is_not():
    long = nu.Str("y" * 900)
    text = observation(long, nu.Str("s" * 900), limit=10)
    ctx = nu.Context().bind(dict, {})
    nu.run(Loop.observed.set(text), ctx)
    got = ctx.get(dict)["observed"]
    assert got.startswith("outcome: " + "y" * 10 + "\nstate:")
    assert "s" * 900 in got
