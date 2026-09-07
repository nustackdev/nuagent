"""One turn: ask, extract, run, observe.

The turn is the unit. A model is sent the conversation, answers with a
module, the module is extracted into a ``ProgramRef``, ``Eval`` drives what it
constructs, and the yield plus the state come back as the next user message.
There is no tool schema anywhere in that, because the model's action *is* the
program: JSON tool calling is an IR that exists for hosts that cannot run code
the model wrote, and Nu can.

A Turn is a Nu term like any other, so it is useful on its own. Run one and
you have a single-shot agent: one question, one program, one observation.
:mod:`nuagent.loop` is that same term under a ``WhileDo``.

The program runs exactly once
-----------------------------

``outcome`` is a Ref the rendered yield lands in, and it is required rather
than optional. Everything downstream reads it from there: the observation
text, and the goal predicate in a loop. Composing the yield into both places
directly would put two ``Eval`` terms in the tree and run the model's program
twice, which for a program that appends to a list is a silently wrong world.

Nothing here declares a Shape. Every Ref is the caller's, so the same Turn
over ``nu.mem`` refs is ephemeral and over ``nu.kv`` refs is a durable,
queryable history, with no change to this module and none to the prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nu
from nu.lang.sentinels import UNSET

from .extract import fenced
from .observation import attempted, failed, never_ran
from .observation import observation as observe_default


if TYPE_CHECKING:
    from collections.abc import Callable

    from nu.lang import Nu


__all__ = ["Turn"]


def Turn(  # noqa: N802 -- a term constructor, named like the Flows it composes
    *,
    chat: Callable[..., Nu],
    messages: Nu,
    reply: Nu,
    draft: Nu,
    outcome: Nu,
    observation: Nu,
    state: Nu | object | None = None,
    goal: Nu | None = None,
    met: str = "met",
    unmet: str = "NOT met yet; the task is unfinished, look at what changed and correct it",
    unran: str = (
        "NOT met, and nothing ran this turn; the world is unchanged, "
        "so read the outcome above, fix it, and send a program"
    ),
    extract: Callable[[Nu], Nu] = fenced,
    observe: Callable[..., Nu] = observe_default,
    on_error: Nu | None = None,
    on_crash: Nu | None = None,
    brace: object = UNSET,
    echo: bool = True,
    label: Nu | str | None = None,
    after: Nu | None = None,
) -> Nu:
    """Compose one turn of an agent.

    Args:
        chat: the bound chat method, e.g. ``Bot.chat``. Called with
            ``messages=``.
        messages: Ref holding the conversation, a list of role/content
            dicts. Initialise it before the first turn: an unset slot yields
            EMPTY, EMPTY collapses the string concatenations here to INVALID,
            and a Command handed INVALID writes nothing and says nothing.
        reply: Ref the raw model text lands in.
        draft: ``ProgramRef`` the extracted source lands in.
        outcome: Ref the rendered yield (or the construction diagnostic)
            lands in. Read it in a goal predicate for an agent that
            terminates on what it produced rather than on what it changed.
        observation: Ref the next user message lands in.
        state: term describing the world after the program ran, appended to
            the observation. Omit for a read-only agent.
        goal: the same Bool-yielding term the loop stops on. Passed here it
            also reaches the model, so a program that runs but does not
            satisfy the goal reads as a failure rather than as silence.
            ``Agent`` forwards its own goal automatically.
        met: verdict text when the goal holds.
        unmet: verdict text when it does not, and the program ran. This is
            the only signal a model gets that working-but-wrong code is
            wrong, so it is worth phrasing as an instruction rather than a
            status.
        unran: verdict text when the goal does not hold and nothing ran, so
            the world is exactly as the turn found it. Separate from
            ``unmet`` because a model told to inspect changes that do not
            exist invents them.
        extract: source out of the reply. Defaults to the first fenced
            block; pass ``partial(fenced, block="last")`` for the last.
        observe: builds the observation from the outcome and the state.
        on_error: the branch when the module does not construct. Defaults
            to the rendered ``Diagnostic``, which is what lets a model fix
            its own code.
        on_crash: the branch when the module constructs and then raises
            while running. Defaults to the rendered exception. Without it
            the error leaves the turn and takes the whole agent with it.
        brace: tag of the ``PyBrace`` to construct in.
        echo: print the reply and the observation as they happen.
        label: prefix for the echoed reply, e.g. a turn counter.
        after: term run at the end of the turn, after the observation is
            recorded. Where a loop puts its counter and its goal check.

    Returns:
        A Flow: one turn, ready to run once or to iterate.
    """
    text = nu.Str(nu.dict(chat(messages=messages))["text"])
    source = extract(reply)
    caught = failed(reply=reply) if on_error is None else on_error
    result = attempted(draft, brace=brace, on_error=caught, on_crash=on_crash)

    banner = nu.Str("\n=== model ===\n") if label is None else nu.Str("\n=== ") + label + " ===\n"

    steps: list[Nu] = []
    if echo:
        asking = (
            nu.Str("\n>>> asking model... <<<")
            if label is None
            else nu.Str("\n>>> ") + label + nu.Str(" | asking model... <<<")
        )
        steps.append(nu.print(asking))
    steps.append(reply.set(text))
    if echo:
        steps.append(nu.print(banner + reply))
    steps += [
        messages.append(nu.Dict.of(role="assistant", content=reply)),
        draft.set(source),
    ]
    if echo:
        steps.append(nu.print(nu.Str("--- extracted source ---\n") + draft))
    steps.append(outcome.set(nu.Str(result)))
    if echo:
        steps.append(nu.print(nu.Str("--- outcome ---\n") + outcome))
    if goal is None:
        steps.append(observation.set(observe(outcome, state)))
    else:
        missed = nu.Str(nu.If(never_ran(outcome), nu.Str(unran), nu.Str(unmet)))
        verdict = nu.If(goal, nu.Str(met), missed)
        steps.append(observation.set(observe(outcome, state, verdict=verdict)))
    if echo:
        steps.append(nu.print(nu.Str("--- observation ---\n") + observation))
    steps.append(messages.append(nu.Dict.of(role="user", content=observation)))
    if after is not None:
        steps.append(after)

    step = steps[0]
    for nxt in steps[1:]:
        step = step >> nxt
    return step
