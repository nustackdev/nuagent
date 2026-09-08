"""An agent embedded in a task tracker, with the tracker in its prompt.

The point of this example is the second half of nuagent's bet. A prompt that
carries only the nucore catalogue teaches the *language*; it says nothing
about the app, so the agent's first turns go on discovering that a Board
exists. Here the caller hands its own Shapes to ``surface_section`` and the
model opens the run already knowing the slot names, the types and the paths
to inspect.

The task is chosen so no single call can express it:

    for every task above priority 2 whose summary is empty, write one, then
    set open_count to the number of tasks still not done.

That is an iteration, a condition per element, a write per element, and an
aggregate over the same collection. A tool-calling agent needs one round trip
per task plus one to count. Here it is one term, and the model has to compose
``ForEachDo`` over ``Board.tasks.keys()`` with an ``IfDo`` inside it. It also
has to look ``keys`` up: the surface section lists ``tasks`` as a
``ShapesDictRef`` and no verbs at all, and ``ShapesDictRef`` is a nu.mem Ref,
which the nucore catalogue does not carry.

The model ends the run itself: it writes ``Run.done.set(True)`` into the
program that finishes the work, and the loop stops that turn. Nothing here
judges the board. ``main`` still checks it afterwards, but that is the example
verifying itself, not the agent's stopping rule.

Two dict fabrics, and the split matters. The agent's own working memory is
bound tagged to ``MemSession``, so nothing the model writes can reach it. The
board is bound untagged on the Context, for two reasons: nu.mem addresses by
slot name and the model's module declares its own ``Board`` class, where a
tagged binding is keyed on the host's class object and would not resolve;
and ``nu.Provide`` binds a copy of the dict it is given, so a board bound
that way is unreadable by the caller once the run ends. ``Run`` rides in that
same untagged store, which is exactly why the model can reach it too.

Run::

    .venv/bin/python examples/tracker.py

The default model is ``claude-opus-4-5``; override it with ``CC_MODEL``.
``DRY=1`` renders the prompt, prints it, and exits without calling a model.

The ``__main__`` guard re-imports this file under the name ``tracker`` and
runs that copy. Inspect resolves a dotted path by importing it, and telling a
model to inspect ``__main__.Board`` is telling it something that is only true
inside this process.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import nu
from nu.lang import ScalarQuery
from nu.lang.sentinels import EMPTY, INVALID, UNSET

import nuagent


if TYPE_CHECKING:
    from collections.abc import Callable

    from nu.lang import Nu
    from nu.lang.runtime import Runtime


MODEL = os.environ.get("CC_MODEL", "claude-opus-4-5")


# --- the app --------------------------------------------------------------


class Task(nu.Shape):
    """One unit of work on the board.

    Notes:
        - `summary` is one sentence saying what the task is about. An empty
          string means nobody has written it yet.
        - `priority` runs 1 (whenever) to 5 (now).
        - `done` is written by whoever finishes the work, never by a summary
          pass.
    """

    title = nu.mem.StrRef.slot()
    summary = nu.mem.StrRef.slot()
    priority = nu.mem.IntRef.slot()
    done = nu.mem.BoolRef.slot()
    owner = nu.mem.StrRef.slot()


class Board(nu.Shape):
    """The whole tracker: tasks by id, plus the counter the UI reads.

    Notes:
        - `open_count` is derived, not authoritative. It is the number of
          tasks whose `done` is false and it is recomputed, never adjusted.
        - Task ids are strings, and `tasks` is keyed by them.
    """

    name = nu.mem.StrRef.slot()
    open_count = nu.mem.IntRef.slot()
    tasks = nu.mem.ShapesDictRef.slot(Task, str)


#: The world before the agent touches it. Three of the five tasks are above
#: priority 2, two of those have no summary, and three are not done.
SEED: dict[str, dict[str, object]] = {
    "t1": {
        "title": "Rotate the signing keys",
        "summary": "The release keys expire in April and every artifact after that fails to verify.",
        "priority": 5,
        "done": False,
        "owner": "gor",
    },
    "t2": {
        "title": "Backfill the metrics table",
        "summary": "",
        "priority": 4,
        "done": False,
        "owner": "david",
    },
    "t3": {
        "title": "Drop the legacy /v1 endpoints",
        "summary": "",
        "priority": 3,
        "done": False,
        "owner": "gor",
    },
    "t4": {
        "title": "Rename the staging bucket",
        "summary": "",
        "priority": 1,
        "done": True,
        "owner": "david",
    },
    "t5": {
        "title": "Pin the CI python to 3.12",
        "summary": "",
        "priority": 2,
        "done": True,
        "owner": "gor",
    },
}

THRESHOLD = 2

#: Computed from the seed at construction time, on the host, where a python
#: loop is legal. NEEDS is what the model has to write; OPEN is what the
#: counter has to end up at.
NEEDS = tuple(tid for tid, t in SEED.items() if t["priority"] > THRESHOLD and not t["summary"])
OPEN = sum(1 for t in SEED.values() if not t["done"])


TASK = f"""\
Every task on the board with priority above {THRESHOLD} must have a summary. Some do not:
their `summary` slot holds the empty string. Write one for each of those, a single
sentence saying what the task is about, taken from its title.

Then set `open_count` to the number of tasks that are not done.

Do it as a program over the whole collection, not by hardcoding ids: iterate the
tasks, test each one, write what is missing, and derive the count. You do not know
the verbs of a ShapesDictRef yet, so look them up before you write.\
"""


# --- the agent's own memory -----------------------------------------------


class Bot(nu.Service):
    """Claude Code prompt surface for one agent run."""

    ask = nu.cc.PromptRef.method()


class FormatMessages(ScalarQuery):
    """Flatten a list of {role, content} dicts into one prompt string.

    Each call to Claude Code is a fresh session, so the whole conversation has
    to ride along in the prompt. Renders as role-tagged blocks.
    """

    def _compile(self, nid: int, children: tuple[Callable, ...]) -> Callable:
        (msgs,) = children

        def thunk(rt: Runtime) -> object:
            m = msgs(rt)
            if m is EMPTY or m is INVALID:
                return INVALID
            return _render(m)

        return thunk

    def _acompile(self, nid: int, children: tuple[Callable, ...]) -> Callable:
        (msgs,) = children

        async def athunk(rt: Runtime) -> object:
            m = await msgs(rt)
            if m is EMPTY or m is INVALID:
                return INVALID
            return _render(m)

        return athunk


def _render(msgs: list[dict]) -> str:
    parts: list[str] = []
    for d in msgs:
        role = str(d.get("role", "user")).upper()
        content = str(d.get("content", ""))
        parts.append(f"### {role}\n\n{content}")
    return "\n\n".join(parts)


def chat(*, messages: Nu) -> Nu:
    """Adapter matching turn's chat signature. Bot.ask takes prompt=."""
    return Bot.ask(prompt=FormatMessages(messages))


# --- prompt, world, state -------------------------------------------------


def prompt() -> str:
    """The system prompt: the stock sections plus this app's surface.

    ``inserted`` with no anchor appends, which puts the surface after the
    catalogue and immediately before the task. The order that leaves is: what
    Nu is, how to write it, how to look things up, what the language holds,
    what your world holds, what to do. The world sits next to the task
    because it is the half of the task that is not prose.

    The two app Shapes are named one by one rather than handing over this
    module, which also declares ``Bot``. That is the
    harness, not the agent's world, and a surface is exactly the set of Refs
    the caller chose to bind.
    """
    sections = nuagent.inserted(nuagent.DEFAULT_SECTIONS, nuagent.surface_section((Board, Task)))
    return nuagent.system_prompt(TASK, sections=sections)


def seed_world() -> Nu:
    """Write the board into the fabric. Every slot exists before anything reads it."""
    step = Board.name.set("arkkln") >> Board.open_count.set(0)
    for tid, task in SEED.items():
        step = step >> Board.tasks[tid].set(dict(task))
    return step


def state() -> Nu:
    """What the model sees of the world each turn. The whole board, it is small."""
    return nu.Dict.of(open_count=Board.open_count, tasks=Board.tasks)


def agent() -> Nu:
    """The whole run as one term: memory and model bound around the loop.

    The tracker fabric is not bound here. ``nu.Provide`` binds a copy, so
    writes through it are invisible to the caller afterwards, and the caller
    checking the board at the end is the whole point of an example. The board
    is bound on the Context instead, where the host keeps the dict it handed
    over.
    """
    system = prompt()
    seed = (
        nuagent.MemSession.messages.set(
            nu.List.of(nu.Dict.of(role="system", content=nu.Str(system))),
        )
        >> seed_world()
    )

    loop = nuagent.agent(
        session=nuagent.MemSession,
        chat=chat,
        state=state(),
        max_turns=8,
        start=seed,
        brace=UNSET,
        echo=True,
        report=nu.print(nu.Str("\n=== board ===\n") + nu.ToStr(nu.Repr(state()))),
    )
    return nu.With(
        nu.Provide(dict, {}, tag=nuagent.MemSession),
        nu.cc.bind(Bot, model=MODEL, allowed_tools=[], permission_mode="default"),
        body=loop,
    )


def main() -> int:
    if os.environ.get("DRY"):
        print(prompt())
        return 0
    world: dict = {}
    print(f"{'#' * 70}\n# TASK\n{'#' * 70}\n{TASK}\n")
    nu.run(agent(), nu.Context().bind(dict, world))
    summarised = all(world["tasks"][tid]["summary"] for tid in NEEDS)
    counted = world["open_count"] == OPEN
    print(f"\nsummarised: {summarised}   open_count correct: {counted}")
    return 0 if summarised and counted else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(importlib.import_module("tracker").main())
