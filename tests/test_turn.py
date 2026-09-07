"""A whole turn, driven with a canned reply instead of a model.

``chat`` is a callable returning a term, so a constant term is a model that
always says the same thing. That makes the turn testable end to end with no
network: extraction, construction, the outcome, the verdict and the
observation are all the real ones.

The case these cover is a model that decides it is finished and answers with
prose. The old harness fed that prose to the python parser and reported a
syntax error in an English sentence, then told the model to look at what had
changed, and nothing had.
"""

from __future__ import annotations

import textwrap

import nu

from nuagent import Turn


class Loop(nu.Shape):
    messages = nu.mem.ListRef.slot(dict)
    reply = nu.mem.StrRef.slot()
    draft = nu.mem.ProgramRef.slot()
    outcome = nu.mem.StrRef.slot()
    observed = nu.mem.StrRef.slot()


class World(nu.Shape):
    notes = nu.mem.ListRef.slot(str)


def src(text: str) -> str:
    return textwrap.dedent(text).lstrip()


APPENDS = src("""
    import nu


    class World(nu.Shape):
        notes = nu.mem.ListRef.slot(str)


    def out():
        return World.notes.append("a")
""")

PROSE = (
    "Done. The program:\n"
    '- Wrote summaries for t2 ("Backfill the metrics table") and t3 — the two '
    "high-priority tasks\n"
    "- Set open_count to 3\n"
)


def canned(text: str):
    """A model that always answers with ``text``. Turn only needs {"text": ...}."""

    def chat(*, messages: nu.Nu) -> nu.Nu:
        del messages  # a canned model does not read the conversation
        return nu.Dict.of(text=nu.Str(text))

    return chat


def drive(reply: str, *, goal: nu.Nu | None = None) -> dict:
    turn = Turn(
        chat=canned(reply),
        messages=Loop.messages,
        reply=Loop.reply,
        draft=Loop.draft,
        outcome=Loop.outcome,
        observation=Loop.observed,
        state=nu.Dict.of(notes=World.notes),
        goal=goal,
        echo=False,
    )
    app = nu.With(
        nu.Provide(nu.prog.PyBrace, {}),
        body=Loop.messages.set([]) >> World.notes.set([]) >> turn,
    )
    ctx = nu.Context().bind(dict, {})
    nu.run(app, ctx)
    return ctx.get(dict)


# --- prose is not source ----------------------------------------------------


def test_prose_reports_no_code_block_not_a_syntax_error():
    out = drive(PROSE)
    assert out["outcome"].startswith("NO CODE BLOCK")
    assert "nothing ran" in out["outcome"]
    # The old failure: a Diagnostic about the em-dash in the model's own prose.
    assert "CONSTRUCTION FAILED" not in out["outcome"]
    assert "U+2014" not in out["outcome"]
    assert "line" not in out["outcome"]


def test_the_no_code_diagnostic_reaches_the_observation():
    out = drive(PROSE)
    assert out["observed"].startswith("outcome: NO CODE BLOCK")
    assert "state: {'notes': []}" in out["observed"]


# --- the bare-source fallback stays ----------------------------------------


def test_a_bare_unfenced_module_still_runs():
    # No fence, but it is source, so it constructs and the fallback earns
    # its keep. Nothing about the no-code path fires here.
    out = drive(APPENDS)
    assert out["outcome"] == "None"
    assert out["observed"] == "outcome: None\nstate: {'notes': ['a']}"


def test_a_fenced_reply_is_unaffected():
    out = drive(f"here you go:\n```python\n{APPENDS}```\nthat should do it")
    assert out["outcome"] == "None"
    assert out["observed"] == "outcome: None\nstate: {'notes': ['a']}"


def test_a_fenced_reply_that_does_not_parse_still_gets_its_diagnostic():
    out = drive("```python\ndef out(:\n```")
    assert out["outcome"].startswith("CONSTRUCTION FAILED")
    assert "line" in out["outcome"]


# --- never-ran is not the same as ran-and-wrong ----------------------------


def test_the_never_ran_verdict_differs_from_the_unmet_one():
    unmet = drive(APPENDS, goal=nu.Int(nu.Len(World.notes)) > 5)
    unran = drive(PROSE, goal=nu.Int(nu.Len(World.notes)) > 5)

    assert unmet["observed"].endswith(
        "goal: NOT met yet; the task is unfinished, look at what changed and correct it"
    )
    assert unran["observed"].endswith(
        "goal: NOT met, and nothing ran this turn; the world is unchanged, "
        "so read the outcome above, fix it, and send a program"
    )
    assert "look at what changed" not in unran["observed"]


def test_a_construction_failure_is_also_never_ran():
    out = drive("```python\ndef out(:\n```", goal=nu.Int(nu.Len(World.notes)) > 5)
    assert "nothing ran this turn" in out["observed"]


def test_a_met_goal_is_unchanged():
    out = drive(APPENDS, goal=nu.Int(nu.Len(World.notes)) == 1)
    assert out["observed"].endswith("goal: met")


def test_the_verdicts_are_overridable():
    turn = Turn(
        chat=canned(PROSE),
        messages=Loop.messages,
        reply=Loop.reply,
        draft=Loop.draft,
        outcome=Loop.outcome,
        observation=Loop.observed,
        goal=nu.Bool(False),
        unran="no program",
        echo=False,
    )
    ctx = nu.Context().bind(dict, {})
    nu.run(
        nu.With(nu.Provide(nu.prog.PyBrace, {}), body=Loop.messages.set([]) >> turn),
        ctx,
    )
    assert ctx.get(dict)["observed"].endswith("goal: no program")
