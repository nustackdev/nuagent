"""Smoke test the agent loop against OpenRouter's free Nemotron.

Run::

    export OPENROUTER_API_KEY=sk-or-...
    .venv/bin/python examples/openrouter_smoke.py

The agent has no bound Shape from the caller's world; the task is a pure
query. It writes a Nu term that returns a value, we read the yield from the
outcome slot. This exercises the whole path (prompt -> chat -> extract ->
construct -> eval -> observe) with the smallest tasks that still need real
composition, so a run that succeeds proves the pipe is clean.
"""

from __future__ import annotations

import os
import sys

import nu
from nu.lang.sentinels import UNSET

import nuagent


MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")


class Bot(nu.Service):
    """Chat surface bound to OpenRouter for this run."""

    chat = nu.llm.ChatRef.method(temperature=0.0)


class Ephemeral(nu.Shape):
    """Working memory for one agent run. All slots live in nu.mem."""

    messages = nu.mem.ListRef.slot(dict)
    reply = nu.mem.StrRef.slot()
    draft = nu.mem.ProgramRef.slot()
    outcome = nu.mem.StrRef.slot()
    observation = nu.mem.StrRef.slot()
    turns = nu.mem.IntRef.slot()
    done = nu.mem.BoolRef.slot()


TASKS = [
    "Return the sum of the first five positive integers as an Int term.",
    "Return the product 6 * 7 as an Int term.",
    "Return a List containing the ints 1, 2, 3 in order.",
]


def run_task(api_key: str, task: str) -> None:
    system = nuagent.system_prompt(task)
    seed = Ephemeral.messages.set(
        nu.List.of(nu.Dict.of(role="system", content=nu.Str(system))),
    )

    goal = Ephemeral.outcome.not_empty()

    agent = nuagent.Agent(
        chat=Bot.chat,
        messages=Ephemeral.messages,
        reply=Ephemeral.reply,
        draft=Ephemeral.draft,
        outcome=Ephemeral.outcome,
        observation=Ephemeral.observation,
        goal=goal,
        done=Ephemeral.done,
        turns=Ephemeral.turns,
        max_turns=4,
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
            nu.llm.openrouter(Bot, api_key=api_key, model=MODEL),
            body=agent,
        )
    )


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2
    for task in TASKS:
        run_task(api_key, task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
