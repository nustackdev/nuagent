"""Smoke test the agent loop against Ollama on red.

Run::

    .venv/bin/python examples/ollama_smoke.py

Override the model with ``OLLAMA_MODEL=qwen2.5:32b-instruct-q4_K_M`` or the
host with ``OLLAMA_HOST=localhost``.

The agent has no bound Shape from the caller's world; the task is a pure
query. It writes a Nu term that returns a value, we read the yield from the
outcome slot. This exercises the whole path (prompt -> chat -> extract ->
construct -> eval -> observe) with the smallest tasks that still need real
composition, so a run that succeeds proves the pipe is clean.
"""

from __future__ import annotations

import os

import nu
from nu.lang.sentinels import UNSET

import nuagent


MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
HOST = os.environ.get("OLLAMA_HOST", "red")


class Bot(nu.Service):
    """Chat surface bound to Ollama for this run."""

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
    "You have the whole core catalogue of NU. think about some interesting Nu expression and write it. on the first place state it then write code and iterate unless it reutrns expected output"
    # "Return the sum of the first five positive integers as an Int term.",
    # "Return the product 6 * 7 as an Int term.",
    # "Return a List containing the ints 1, 2, 3 in order.",
]


def run_task(task: str) -> None:
    system = nuagent.system_prompt(task)
    seed = Ephemeral.messages.set(
        nu.List.of(nu.Dict.of(role="system", content=nu.Str(system))),
    )

    # Success: the outcome slot holds a value AND it isn't a failure prefix.
    outcome = Ephemeral.outcome
    failed_run = outcome.startswith("RUNTIME FAILED").or_(
        outcome.startswith("CONSTRUCTION FAILED"),
    )
    goal = outcome.not_empty().and_(failed_run.not_())

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
            nu.llm.ollama(Bot, host=HOST, model=MODEL),
            body=agent,
        )
    )


def main() -> int:
    for task in TASKS:
        run_task(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
