"""What the model reads after its program ran.

One turn produces two things worth reporting: what the program *yielded*, and
what the world looks like *now*. Both are Nu terms, and this module composes
them into the text that becomes the next user message.

Yield and state are not symmetric
---------------------------------

Most programs a model writes are Command chains, and a Command yields
nothing. So the yield is often ``None`` and the state is the only evidence
anything happened at all. That asymmetry decides the truncation policy: the
yield is truncated, the state never is.

Truncating state is what makes a loop diverge. A model that is shown half of
a list it just wrote cannot tell a successful append from a failed one, so it
writes the append again, and the observation gets longer, and it truncates
harder. Whatever the caller passes as ``state`` is therefore its own problem
to keep small -- a count, a slice, a few slots -- and this module reproduces
it whole.

Being wrong is also an outcome
------------------------------

A construction failure comes back with a Diagnostic, so a model that writes
code which does not compile can fix it. A model that writes code which
compiles, runs, and is *wrong* used to get nothing: the yield, the state, and
no word about whether any of it was what was asked for. So it either declared
victory or edited working code at random. ``verdict`` closes that: the goal is
already being evaluated to decide whether to loop, and this puts its answer
where the model can read it.

Construction failure is an outcome
----------------------------------

:func:`failed` is the branch for ``ProgramRef.run(on_error=...)``. The caught
``ConstructionError`` sits on the attrs fabric, its ``Diagnostic`` renders as
"message (line N)", and that is exactly what a model needs to fix its own
code. It comes back through the same slot a successful yield does, so a
failure is just another observation rather than a separate control path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nu
from nu.lang.sentinels import UNSET


if TYPE_CHECKING:
    from nu.lang import Nu, StrArg


__all__ = ["DEFAULT_LIMIT", "attempted", "crashed", "failed", "observation", "rendered"]


DEFAULT_LIMIT = 400
"""Characters of yield kept. Long enough for a real value, short enough that
a runaway repr cannot push the state out of the model's attention."""


def rendered(term: Nu | object) -> Nu:
    """A term as text, whatever it yields.

    ``Repr`` first, because a yield can be anything (a dict, a list, None)
    and only the repr survives all of them, then ``ToStr`` because ``Repr``
    is not itself a Str and will not concatenate.
    """
    return nu.ToStr(nu.Repr(term))


def failed(*, attr: str = "error", label: str = "CONSTRUCTION FAILED") -> Nu:
    """The catch branch for ``run(on_error=...)``: the Diagnostic, as text.

    Args:
        attr: attrs key the caught error is bound under. ``TryCatch`` uses
            "error"; change it only if the caller re-tagged it.
        label: prefix the model is told to look for.

    Returns:
        A ``Str`` term yielding "LABEL: message (line N)".
    """
    error = nu.GetAttr(nu.AttrRef(attr), "exception")
    return nu.Str(f"{label}: ") + nu.ToStr(nu.GetAttr(error, "diagnostic"))


def crashed(*, attr: str = "error", label: str = "RUNTIME FAILED") -> Nu:
    """The catch branch for a program that constructed but raised while running.

    ``TryCatch`` binds a ``CaughtError`` at ``attrs[attr]``, which is a ``str``
    subclass carrying the message, so the text is readable without unwrapping.

    Args:
        attr: attrs key the caught error is bound under.
        label: prefix the model is told to look for.

    Returns:
        A ``Str`` term yielding "LABEL: message".
    """
    return nu.Str(f"{label}: ") + nu.ToStr(nu.AttrRef(attr))


def observation(
    outcome: StrArg,
    state: Nu | object | None = None,
    *,
    verdict: Nu | object | None = None,
    limit: int = DEFAULT_LIMIT,
    outcome_label: str = "outcome",
    state_label: str = "state",
    verdict_label: str = "goal",
) -> Nu:
    """Compose the next turn's input from the yield, the world and the verdict.

    Args:
        outcome: the program's yield, already rendered as text. Usually the
            slot :func:`nuagent.turn.Turn` stored it in, so the program runs
            once and both the observation and the goal read the same value.
        state: any term describing the world afterwards, commonly a
            ``nu.Dict.of(...)`` over the slots that matter. Omitted for a
            read-only agent, where the yield is the whole result.
        verdict: how the goal stands, as text. Without it a program that
            constructs and runs but does not satisfy the goal produces an
            observation indistinguishable from success, and the model has
            nothing to correct against.
        limit: characters of ``outcome`` kept. State and verdict are never cut.
        outcome_label: prefix for the yield line.
        state_label: prefix for the state line.
        verdict_label: prefix for the verdict line.

    Returns:
        A ``Str`` term yielding the observation text.
    """
    text = nu.Str(f"{outcome_label}: ") + nu.Str(outcome)[:limit]
    if state is not None:
        text = text + f"\n{state_label}: " + rendered(state)
    if verdict is not None:
        text = text + f"\n{verdict_label}: " + nu.Str(verdict)
    return nu.Str(text)


def attempted(
    draft: Nu,
    *,
    brace: object = UNSET,
    on_error: Nu | None = None,
    on_crash: Nu | None = None,
) -> Nu:
    """One attempt at the model's program, as text however it goes.

    This is ``ProgramRef.run(on_error=...)`` with one thing moved: the render
    sits *inside* the ``TryCatch`` rather than around it. Wrapping the whole
    thing would repr the catch branch too, and the catch branch is already
    text -- the diagnostic would reach the model quoted, with its inner
    quotes escaped, and it is the one message in the loop worth keeping
    readable.

    The same three atoms ``run`` composes, in the same order, wrapped in a
    second ``TryCatch`` for everything that is not a construction failure.

    Both catches exist for the same reason: a mistake by the model is input,
    not a crash. Source that does not construct is the obvious case, but code
    that constructs and then raises while running is just as much the model's
    error, and without the outer catch it propagates out of the turn, out of
    the loop, and kills the agent. A model cannot fix a traceback it never
    sees. Catching ``Exception`` here is deliberately broad because everything
    under it descends from evaluating a term the model wrote; it is scoped to
    that evaluation and nothing wider.

    Args:
        draft: the ``ProgramRef`` holding the source.
        brace: tag of the ``PyBrace`` to construct in.
        on_error: the branch when construction fails. Defaults to
            :func:`failed`.
        on_crash: the branch when a constructed program raises while running.
            Defaults to :func:`crashed`.

    Returns:
        A ``Str`` term: the rendered yield, the rendered Diagnostic, or the
        runtime error, whichever happened.
    """
    running = nu.Eval(nu.LoadNu(draft, brace=brace))
    constructed = nu.TryCatch(
        nu.Str(rendered(running)),
        catch=failed() if on_error is None else on_error,
        errors=nu.prog.ConstructionError,
    )
    return nu.TryCatch(
        constructed,
        catch=crashed() if on_crash is None else on_crash,
        errors=Exception,
    )
