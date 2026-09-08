"""A whole turn, and a whole loop, driven with a canned reply instead of a model.

``chat`` is a callable returning a term, so a constant term is a model that
always says the same thing. That makes the turn testable end to end with no
network: extraction, construction, the outcome and the observation are all
the real ones, and so is the termination, because a canned reply can set
``Run.done`` exactly as a model would.
"""

from __future__ import annotations

import textwrap

import nu

import nuagent


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

#: What a finishing reply looks like: the work and the flag, one program.
FINISHES = src("""
    import nu


    class World(nu.Shape):
        notes = nu.mem.ListRef.slot(str)


    class Run(nu.Shape):
        done = nu.mem.BoolRef.slot()


    def out():
        return World.notes.append("a") >> Run.done.set(True)
""")

YIELDS = src("""
    import nu


    def out():
        return nu.Str("y" * 900)
""")

PROSE = (
    "Done. The program:\n"
    '- Wrote summaries for t2 ("Backfill the metrics table") and t3 — the two '
    "high-priority tasks\n"
    "- Set open_count to 3\n"
)


def canned(text: str):
    """A model that always answers with ``text``. turn only needs {"text": ...}."""

    def chat(*, messages: nu.Nu) -> nu.Nu:
        del messages  # a canned model does not read the conversation
        return nu.Dict.of(text=nu.Str(text))

    return chat


#: "you did not pass state", as distinct from "pass no state".
DEFAULT = object()


def drive(reply: str, *, state: object = DEFAULT, **kw) -> dict:
    if state is DEFAULT:
        state = nu.Dict.of(notes=World.notes)
    app = nu.With(
        nu.Provide(nu.prog.PyBrace, {}),
        body=nuagent.MemSession.messages.set([])
        >> World.notes.set([])
        >> nuagent.turn(
            session=nuagent.MemSession,
            chat=canned(reply),
            state=state,
            echo=False,
            **kw,
        ),
    )
    ctx = nu.Context().bind(dict, {})
    nu.run(app, ctx)
    return ctx.get(dict)


# --- the session is a shape, not twelve arguments ---------------------------


def test_the_turn_reads_every_slot_off_the_session():
    out = drive(f"```python\n{APPENDS}```")
    assert out["reply"].startswith("```python")
    assert out["draft"].startswith("import nu")
    assert out["outcome"] == "None"
    assert out["observation"] == "outcome: None\nstate: {'notes': ['a']}"
    # the conversation grew by the assistant reply and the user observation
    assert [m["role"] for m in out["messages"]] == ["assistant", "user"]
    assert out["messages"][-1]["content"] == out["observation"]


SLOTS = ["messages", "reply", "draft", "outcome", "observation", "turns"]


def test_the_mem_session_holds_the_six_slots_in_order():
    assert [n for n in vars(nuagent.MemSession) if not n.startswith("_")] == SLOTS


def test_the_kv_session_has_the_same_slots_in_the_same_order():
    assert [n for n in vars(nuagent.KVSession) if not n.startswith("_")] == SLOTS


def test_the_two_sessions_differ_only_in_the_fabric():
    kv = nuagent.KVSession
    for slot in SLOTS:
        mem_ref = getattr(nuagent.MemSession, slot)
        kv_ref = getattr(kv, slot)
        assert type(mem_ref).__name__ == type(kv_ref).__name__
        assert type(mem_ref).__module__.startswith("nu.mem")
        assert type(kv_ref).__module__.startswith("nu.kv")


# --- the run shape: the model's one lever ----------------------------------


def test_done_is_not_a_session_slot():
    # The session is bound tagged, out of the model's reach. A done slot on it
    # could never be the slot the model writes.
    assert "done" not in vars(nuagent.MemSession)
    assert "done" not in vars(nuagent.KVSession)


def test_the_run_shape_carries_exactly_one_slot():
    assert [n for n in vars(nuagent.Run) if not n.startswith("_")] == ["done"]
    assert [n for n in vars(nuagent.KVRun) if not n.startswith("_")] == ["done"]


def test_the_two_run_shapes_differ_only_in_the_fabric():
    assert type(nuagent.Run.done).__name__ == type(nuagent.KVRun.done).__name__
    assert type(nuagent.Run.done).__module__.startswith("nu.mem")
    assert type(nuagent.KVRun.done).__module__.startswith("nu.kv")


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
    assert out["observation"].startswith("outcome: NO CODE BLOCK")
    assert "state: {'notes': []}" in out["observation"]


def test_an_unclosed_fence_of_prose_is_also_no_code():
    # No complete fence, and what falls through does not construct, so it was
    # never code. A line number into an English sentence is the wrong answer.
    out = drive("here you go — the summaries are written")
    assert out["outcome"].startswith("NO CODE BLOCK")


# --- the bare-source fallback stays ----------------------------------------


def test_a_bare_unfenced_module_still_runs():
    # No fence, but it is source, so it constructs and the fallback earns its
    # keep. Nothing about the no-code path fires here.
    out = drive(APPENDS)
    assert out["outcome"] == "None"
    assert out["observation"] == "outcome: None\nstate: {'notes': ['a']}"


def test_a_fenced_reply_is_unaffected():
    out = drive(f"here you go:\n```python\n{APPENDS}```\nthat should do it")
    assert out["outcome"] == "None"
    assert out["observation"] == "outcome: None\nstate: {'notes': ['a']}"


def test_a_fenced_reply_that_does_not_parse_still_gets_its_diagnostic():
    out = drive("```python\ndef out(:\n```")
    assert out["outcome"].startswith("CONSTRUCTION FAILED")
    assert "line" in out["outcome"]


# --- the observation -------------------------------------------------------


def test_a_read_only_turn_needs_no_state_at_all():
    out = drive("```python\nimport nu\n\n\ndef out():\n    return nu.Int(1) + 41\n```", state=None)
    assert out["observation"] == "outcome: 42"


def test_the_yield_is_not_truncated():
    # A lookup turn's whole answer is the yield. Cutting it throws the turn
    # away.
    out = drive(f"```python\n{YIELDS}```", state=None)
    assert out["observation"] == "outcome: " + repr("y" * 900)


def test_the_state_is_not_truncated_either():
    # A model shown half a list it just wrote cannot tell a landed append from
    # a failed one, so it appends again and the observation grows.
    long = src("""
        import nu


        class World(nu.Shape):
            notes = nu.mem.ListRef.slot(str)


        def out():
            return World.notes.set(["n"] * 200)
    """)
    out = drive(f"```python\n{long}```")
    assert out["observation"].count("'n'") == 200


def test_the_program_runs_exactly_once():
    # Two Eval terms in the tree would append twice and silently corrupt the
    # world. The outcome slot is what makes it one.
    out = drive(f"```python\n{APPENDS}```")
    assert out["notes"] == ["a"]


def test_the_turn_passes_no_judgement():
    # The observation is what happened, never a verdict on it. Nothing here
    # tells the model whether the work was right; the world is in `state` and
    # it reads that for itself.
    assert "goal" not in drive(APPENDS)["observation"]
    assert "goal" not in drive(PROSE)["observation"]


# --- the loop --------------------------------------------------------------


def loop(reply: str, *, max_turns: int = 3) -> dict:
    agent = nuagent.agent(
        session=nuagent.MemSession,
        chat=canned(reply),
        state=nu.Dict.of(notes=World.notes),
        max_turns=max_turns,
        start=nuagent.MemSession.messages.set([]) >> World.notes.set([]),
        echo=False,
    )
    ctx = nu.Context().bind(dict, {})
    nu.run(nu.With(nu.Provide(nu.prog.PyBrace, {}), body=agent), ctx)
    return ctx.get(dict)


def test_the_loop_stops_the_turn_the_model_sets_the_ref():
    out = loop(FINISHES)
    assert out["turns"] == 1
    assert out["done"] is True
    assert out["notes"] == ["a"]  # one turn, one append


def test_the_model_writes_the_slot_the_loop_reads():
    # Not two slots that are supposed to agree: the model's own Run
    # declaration addresses by slot name into the same store the loop reads.
    out = loop(FINISHES)
    read_back = nu.run(nuagent.Run.done, nu.Context().bind(dict, out))[0]
    assert read_back is True
    assert out["done"] is True


def test_working_code_that_never_finishes_spends_the_budget():
    # The program builds, runs and changes the world every turn. Nothing about
    # that ends the run, so it goes to the wall.
    out = loop(APPENDS, max_turns=3)
    assert out["turns"] == 3
    assert out["done"] is False
    assert out["notes"] == ["a", "a", "a"]


def test_the_loop_spends_its_budget_and_gives_up():
    out = loop(PROSE, max_turns=3)
    assert out["turns"] == 3
    assert out["done"] is False


def test_max_turns_takes_a_ref():
    # A running agent's budget can be raised from outside.
    class Budget(nu.Shape):
        limit = nu.mem.IntRef.slot()

    agent = nuagent.agent(
        session=nuagent.MemSession,
        chat=canned(PROSE),
        max_turns=Budget.limit,
        start=Budget.limit.set(2) >> nuagent.MemSession.messages.set([]),
        echo=False,
    )
    ctx = nu.Context().bind(dict, {})
    nu.run(nu.With(nu.Provide(nu.prog.PyBrace, {}), body=agent), ctx)
    assert ctx.get(dict)["turns"] == 2


def test_start_runs_before_the_first_turn():
    # Without it messages is unset, which yields EMPTY, which collapses the
    # concatenations to INVALID, and the whole run fails silently.
    out = loop(FINISHES)
    assert out["messages"][0]["role"] == "assistant"
