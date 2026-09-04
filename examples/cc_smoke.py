"""Smoke test the agent loop against Claude Code as the model backend.

Uses ``nu.cc`` with a fresh session per call and no tools, so Claude Code
behaves like a plain chat model (prompt in, text out) rather than an agent.

Run::

    .venv/bin/python examples/cc_smoke.py

Override the model with ``CC_MODEL=claude-opus-4-5``.

The full message history is flattened into the prompt each turn because a
fresh session has no memory. Tools are disabled via ``allowed_tools=[]`` and
``max_turns=1`` so the SDK stops after one text emission.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import nu
from nu.lang import ScalarQuery
from nu.lang.sentinels import EMPTY, INVALID, UNSET

import nuagent


if TYPE_CHECKING:
    from collections.abc import Callable

    from nu.lang import Nu
    from nu.lang.runtime import Runtime


MODEL = os.environ.get("CC_MODEL", "claude-sonnet-4-5")


class Bot(nu.Service):
    """Claude Code prompt surface for one agent run."""

    ask = nu.cc.PromptRef.method()


class Ephemeral(nu.Shape):
    """Working memory for one agent run. All slots live in nu.mem."""

    messages = nu.mem.ListRef.slot(dict)
    reply = nu.mem.StrRef.slot()
    draft = nu.mem.ProgramRef.slot()
    outcome = nu.mem.StrRef.slot()
    observation = nu.mem.StrRef.slot()
    turns = nu.mem.IntRef.slot()
    done = nu.mem.BoolRef.slot()


class FormatMessages(ScalarQuery):
    """Flatten a list of {role, content} dicts into one prompt string.

    Each call to Claude Code is a fresh session, so the whole conversation
    has to ride along in the prompt. Renders as role-tagged blocks.
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
    """Adapter matching Turn's chat signature. Bot.ask takes prompt=."""
    return Bot.ask(prompt=FormatMessages(messages))


EXPECTED = "385"

TASKS = [
    (
        "Compute the sum of squares of the integers 1 through 10 as an Int "
        f"term. The correct answer is {EXPECTED}; your `out()` must yield a "
        "term whose rendered value equals that exactly.\n\n"
        "How to work: this is your first time seeing Nu. Do not guess. "
        "Turn 1 and 2 should be exploration only: pick the handful of atoms "
        "you think you'll need from the catalogue and run "
        "`nu.info.Inspect(\"<path>\")` on each to learn the real "
        "signatures and semantics. Only once you've read what you need, "
        "start composing. If a compose attempt fails, inspect what confused "
        "you before rewriting."
    ),
]


def run_task(task: str) -> None:
    system = nuagent.system_prompt(task)
    seed = Ephemeral.messages.set(
        nu.List.of(nu.Dict.of(role="system", content=nu.Str(system))),
    )

    agent = nuagent.Agent(
        chat=chat,
        messages=Ephemeral.messages,
        reply=Ephemeral.reply,
        draft=Ephemeral.draft,
        outcome=Ephemeral.outcome,
        observation=Ephemeral.observation,
        goal=Ephemeral.outcome == nu.Str(EXPECTED),
        done=Ephemeral.done,
        turns=Ephemeral.turns,
        max_turns=50,
        start=seed,
        brace=UNSET,
        echo=True,
        report=nu.print(
            nu.Str("\n=== final outcome ===\n") + Ephemeral.outcome,
        ),
    )

    print(f"\n{'#' * 70}\n# TASK: {task}\n{'#' * 70}")
    nu.run(
        nu.With(
            nu.Provide(dict, {}, tag=Ephemeral),
            nu.cc.bind(
                Bot,
                model=MODEL,
                allowed_tools=[],
                permission_mode="default",
            ),
            body=agent,
        )
    )


def main() -> int:
    for task in TASKS:
        run_task(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
