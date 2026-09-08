"""The agent: one turn, and that turn under a ``WhileDo``.

A turn is ask, extract, run, observe. It is a Nu term like any other, so run
one and it is a single-shot agent; put it under ``agent`` and it iterates
until a goal holds. There is no python ``while`` anywhere, and no tool schema:
the model's action *is* the program.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nu
from nu.lang import UNSET

from .utils import FAILED_LABEL, NO_CODE_LABEL, attempted, failed, fenced


if TYPE_CHECKING:
    from collections.abc import Callable

    from nu import IntArg, Nu

    from .shapes import KVSession, MemSession


__all__ = ["agent", "turn"]


UNMET = "NOT met yet; the task is unfinished, look at what changed and correct it"
UNRAN = (
    "NOT met, and nothing ran this turn; the world is unchanged, "
    "so read the outcome above, fix it, and send a program"
)


def turn(
    *,
    session: type[KVSession | MemSession],
    chat: Callable[..., Nu],
    state: Nu | object | None = None,
    goal: Nu | None = None,
    met: str = "met",
    unmet: str = UNMET,
    unran: str = UNRAN,
    extract: Callable[[Nu], Nu] = fenced,
    on_error: Nu | None = None,
    on_crash: Nu | None = None,
    brace: object = UNSET,
    echo: bool = True,
    label: Nu | str | None = None,
    after: Nu | None = None,
) -> Nu:
    """Compose one turn of an agent.

    Args:
        session: the Shape class holding the run's slots,
            :class:`~.shapes.MemSession` or :class:`~.shapes.KVSession` or one
            shaped like them.
        chat: the bound chat method, e.g. ``Bot.chat``. Called with
            ``messages=``.
        state: term describing the world after the program ran, appended to
            the observation. Omit for a read-only agent.
        goal: the Bool term the loop stops on. Passed here it also reaches the
            model, so a program that runs but does not satisfy it reads as a
            failure rather than as silence. ``agent`` forwards its own.
        met: verdict text when the goal holds.
        unmet: verdict text when it does not, and the program ran. The only
            signal a model gets that working-but-wrong code is wrong, so it is
            phrased as an instruction rather than a status.
        unran: verdict text when the goal does not hold and nothing ran.
            Separate from ``unmet`` because a model told to inspect changes
            that do not exist invents them.
        extract: source out of the reply. Pass
            ``partial(fenced, block="last")`` for the last block.
        on_error: the branch when the module does not construct. Defaults to
            the rendered ``Diagnostic``, which is what lets a model fix itself.
        on_crash: the branch when the module constructs and then raises.
        brace: tag of the ``PyBrace`` to construct in.
        echo: print the reply and the observation as they happen.
        label: prefix for the echoed reply, e.g. a turn counter.
        after: term run at the end of the turn. Where a loop puts its counter
            and its goal check.

    Returns:
        A Flow: one turn, ready to run once or to iterate.

    Notes:
        - The program runs exactly once. Its yield lands in ``session.outcome``
          and both the observation and the goal read it from there; composing
          it into each directly would put two ``Eval`` terms in the tree and
          run an appending program twice.
        - Nothing is truncated. A model shown half a list it just wrote cannot
          tell a landed append from a failed one, so it appends again and the
          observation grows. The yield is not cut either: for a lookup turn it
          is the whole answer.
    """
    messages = session.messages  # type: ignore[attr-defined]
    reply = session.reply  # type: ignore[attr-defined]
    draft = session.draft  # type: ignore[attr-defined]
    outcome = session.outcome  # type: ignore[attr-defined]
    observation = session.observation  # type: ignore[attr-defined]

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

    observed = nu.Str("outcome: ") + outcome
    if state is not None:
        observed = observed + "\nstate: " + nu.ToStr(nu.Repr(state))
    if goal is not None:
        ran = nu.Str(outcome)
        never_ran = nu.Or(ran.startswith(FAILED_LABEL), ran.startswith(NO_CODE_LABEL))
        missed = nu.Str(nu.If(never_ran, nu.Str(unran), nu.Str(unmet)))
        observed = observed + "\ngoal: " + nu.Str(nu.If(goal, nu.Str(met), missed))
    steps.append(observation.set(nu.Str(observed)))

    if echo:
        steps.append(nu.print(nu.Str("--- observation ---\n") + observation))

    steps.append(messages.append(nu.Dict.of(role="user", content=observation)))

    if after is not None:
        steps.append(after)

    step = steps[0]
    for nxt in steps[1:]:
        step = step >> nxt
    return step


def agent(
    *,
    session: type[KVSession | MemSession],
    chat: Callable[..., Nu],
    goal: Nu,
    max_turns: IntArg = 6,
    start: Nu | None = None,
    report: Nu | None = None,
    **turn_args: object,
) -> Nu:
    """Iterate a :func:`turn` until the goal holds or the budget runs out.

    Args:
        session: the Shape class holding the run's slots. ``turns`` and
            ``done`` come off it.
        chat: the bound chat method.
        goal: Bool term checked after each turn. It may read the outcome as
            easily as the world, which is what lets a read-only agent (answer
            a question, draft a plan) terminate at all.
        max_turns: the budget. A Ref works as well as an int, so a running
            agent's budget can be raised from outside.
        start: term run before the first turn. Not optional in practice: the
            conversation and the world both have to be seeded, and an unset
            slot yields EMPTY, which collapses every string composed with it
            to INVALID, which writes nothing and raises nothing.
        report: term run after the loop.
        turn_args: forwarded to :func:`turn`.

    Returns:
        A Flow: initialise, seed, iterate, report.

    Notes:
        - Nothing here inspects the world. Read ``session.done`` afterwards to
          tell success from exhaustion.
    """
    turns = session.turns
    done = session.done

    label = nu.Str("turn ") + nu.ToStr(turns) + " | model"
    body = turn(
        session=session,
        chat=chat,
        label=label,
        goal=goal,
        after=turns.inc() >> nu.IfDo(goal, done.set(True)),
        **turn_args,  # type: ignore[arg-type]
    )

    step = turns.set(0) >> done.set(False)
    if start is not None:
        step = step >> start
    step = step >> nu.WhileDo(nu.And(turns < max_turns, done.not_()), body)
    if report is not None:
        step = step >> report
    return step
