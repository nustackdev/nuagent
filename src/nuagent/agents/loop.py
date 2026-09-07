"""The loop: a Turn under a ``WhileDo``, stopping on a goal.

There is no python ``while`` here, and that is the point. The turn is a
subtree, ``WhileDo`` iterates it, and the bound on turns is a Ref, so the
whole agent is one term: walkable, rewritable, and runnable anywhere a Nu
term runs.

The goal is a term, not a hardcoded predicate
---------------------------------------------

``goal`` is any Bool-yielding term the caller writes, and it can read the
*outcome* as easily as the state. That distinction is the difference between
a toolkit and a state-mutation toolkit wearing a generic name.

A goal that can only test mutated state fails twice. A task the predicate was
not written for never terminates early and always burns the full budget. And
a model that has already succeeded is still asked for another turn, so it
edits working code to have something to say, which is how a finished agent
un-finishes itself. Meanwhile a whole class of agents -- answer a question,
draft a plan, review something -- writes nothing at all, and its entire
result is the yield.

So :func:`Agent` never looks at the world. It runs turns until ``goal``
holds or the budget runs out, and what ``goal`` reads is the caller's
business.

The goal is also told to the model. It is evaluated once a turn to decide
whether to keep going, and that answer is worth more in the observation than
in the loop condition alone: without it, code that constructs and runs and is
simply wrong looks exactly like code that worked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nu

from .turn import Turn


if TYPE_CHECKING:
    from nu.lang import IntArg, Nu


__all__ = ["Agent"]


def Agent(  # noqa: N802 -- a term constructor, named like the Flows it composes
    *,
    goal: Nu,
    done: Nu,
    turns: Nu,
    max_turns: IntArg = 6,
    start: Nu | None = None,
    report: Nu | None = None,
    **turn_args: object,
) -> Nu:
    """Iterate a :func:`~nuagent.agents.turn.Turn` until the goal holds.

    Args:
        goal: Bool-yielding term checked after each turn. Read the outcome
            Ref for an agent that terminates on what it produced, the world
            for one that terminates on what it changed, or both.
        done: Ref set to True once ``goal`` holds. Also the loop's exit
            condition, and worth reading afterwards to tell success from
            exhaustion.
        turns: Ref counting completed turns.
        max_turns: the budget. A Ref works here as well as an int, so a
            running agent's budget can be raised from outside.
        start: term run before the first turn, after the loop's own control
            Refs are initialised. Where the conversation gets seeded and the
            world gets its initial values, both of which must happen: an
            unset slot yields EMPTY and every string this composes with it
            collapses to INVALID, which writes nothing and raises nothing.
        report: term run after the loop.
        turn_args: forwarded to :func:`~nuagent.agents.turn.Turn`.

    Returns:
        A Flow: initialise, seed, iterate, report.
    """
    label = nu.Str("turn ") + nu.ToStr(turns) + " | model"
    body = Turn(
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
