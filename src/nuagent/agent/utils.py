"""The two pieces of the turn with logic of their own: extraction and the attempt.

Both are Nu terms, not python helpers. They evaluate *inside* the turn, and a
python hole in that tree would make it unwalkable and unrunnable anywhere but
the host process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import nu
from nu.lang.sentinels import UNSET


if TYPE_CHECKING:
    from nu.lang import Nu, StrArg


__all__ = [
    "FAILED_LABEL",
    "FENCE",
    "NO_CODE",
    "NO_CODE_LABEL",
    "attempted",
    "failed",
    "fenced",
]


FENCE = "```"

# ```python / ```py / bare ```. The tag heads the chunk, so stripping is a
# chain of removeprefix, each a no-op when it does not match.
_TAGS = ("python", "py")

#: Prefix on the diagnostic from a module that did not build.
FAILED_LABEL = "CONSTRUCTION FAILED"

#: Prefix on the diagnostic for a reply that carried no code at all.
NO_CODE_LABEL = "NO CODE BLOCK"

#: What a model is told when it answered with prose. No line number, because
#: there is no source to have a line 2.
NO_CODE = (
    f"{NO_CODE_LABEL}: your reply contained no fenced code block, so nothing ran. "
    "Prose is not an action here. Reply with one fenced python block defining "
    "out(); to finish, send a program that sets Run.done."
)


def fenced(text: StrArg, *, block: Literal["first", "last"] = "first") -> Nu:
    """The source inside a fenced block, as a Str-yielding term.

    Args:
        text: the reply, as a str or any Str-yielding term.
        block: which fenced block to take, ``"first"`` (default) or ``"last"``.

    Returns:
        A ``Str`` term yielding the extracted source.

    Raises:
        ValueError: for a ``block`` other than "first" or "last".

    Notes:
        - Splitting on the fence puts code at the odd indices, so the first
          block is index 1 and the last is the second-to-last chunk, computed
          at runtime because the count is unknown when the tree is built.
        - Neither choice is right always. A model correcting itself quotes the
          broken version first; a model appending an illustration breaks
          last-block. First is the safer default: re-reading broken code costs
          a turn, running an illustration does something nobody asked for.
        - Fewer than three chunks means no complete block, and the whole reply
          is used stripped. That is right for bare source and wrong for prose,
          and the two are not distinguishable here; :func:`failed` resolves it.
    """
    if block not in ("first", "last"):
        msg = f"block must be 'first' or 'last', not {block!r}"
        raise ValueError(msg)

    body = nu.Str(text)
    chunks = body.split(FENCE)
    count = nu.Int(nu.Len(chunks))
    index = nu.Int(1) if block == "first" else count - 2

    picked = nu.Str(nu.List(chunks)[index])
    for tag in _TAGS:
        picked = picked.removeprefix(tag)

    return nu.Str(nu.If(count >= 3, picked.strip(), body.strip()))


def failed(*, reply: Nu | None = None) -> Nu:
    """The catch branch for a module that did not construct: the Diagnostic, as text.

    Args:
        reply: Ref holding the raw model text. Passed, a construction failure
            on a reply with no complete fence reports :data:`NO_CODE` instead.
            Omitted, the Diagnostic is reported either way.

    Returns:
        A ``Str`` term yielding "CONSTRUCTION FAILED: message (line N)", or
        :data:`NO_CODE`.

    Notes:
        - The fenceless fork is what keeps a model from trying to fix its own
          English. Without it, prose reaches the python parser and the model
          is handed "invalid character '-' (U+2014) (line 2)".
    """
    error = nu.GetAttr(nu.AttrRef("error"), "exception")
    diagnostic = nu.Str(f"{FAILED_LABEL}: ") + nu.ToStr(nu.GetAttr(error, "diagnostic"))
    if reply is None:
        return diagnostic
    fenceless = nu.Int(nu.Len(nu.Str(reply).split(FENCE))) < 3
    return nu.Str(nu.If(fenceless, nu.Str(NO_CODE), diagnostic))


def attempted(
    draft: Nu,
    *,
    brace: object = UNSET,
    on_error: Nu | None = None,
    on_crash: Nu | None = None,
) -> Nu:
    """One attempt at the model's program, as text however it goes.

    Args:
        draft: the ``ProgramRef`` holding the source.
        brace: tag of the ``PyBrace`` to construct in.
        on_error: the branch when construction fails. Defaults to :func:`failed`.
        on_crash: the branch when a constructed program raises while running.

    Returns:
        A ``Str`` term: the rendered yield, the Diagnostic, or the runtime
        error, whichever happened.

    Notes:
        - The render sits *inside* the inner ``TryCatch``, not around it.
          Wrapping the whole thing reprs the catch branch, and the model then
          reads its diagnostic quoted with the inner quotes escaped.
        - Both catches exist because a mistake by the model is input, not a
          crash. Source that constructs and then raises is just as much the
          model's error, and uncaught it leaves the turn, leaves the loop, and
          kills the agent. ``Exception`` is deliberately broad here: everything
          under it comes from evaluating a term the model wrote.
    """
    running = nu.Eval(nu.LoadNu(draft, brace=brace))
    crashed = nu.Str("RUNTIME FAILED: ") + nu.ToStr(nu.AttrRef("error"))
    constructed = nu.TryCatch(
        nu.Str(nu.ToStr(nu.Repr(running))),
        catch=failed() if on_error is None else on_error,
        errors=nu.prog.ConstructionError,
    )
    return nu.TryCatch(
        constructed,
        catch=crashed if on_crash is None else on_crash,
        errors=Exception,
    )
