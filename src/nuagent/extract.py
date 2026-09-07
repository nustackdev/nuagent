"""Pull the source out of a chat reply, as a Nu Query.

A model answers with a fenced block, sometimes with prose around it. Getting
the source back out is three string operations, and they are written here as
Nu (``Str.split``, a ``List`` index, ``removeprefix``, ``strip``, ``If``)
rather than as a python helper.

That is not decoration. The extraction happens *inside* the turn, between the
chat call and the ``ProgramRef`` the source lands in, and everything else in
that turn is a Nu term. A python helper would make the turn a tree with a
hole in it: not walkable, not attributable, not rewritable, and impossible to
run anywhere but the host process. Keeping it in the tree costs nothing.

First block or last block
-------------------------

Both are wrong sometimes and it is not close enough to pick one for everyone.
A model correcting itself commonly quotes the broken version first and the
fix second, and first-block loses that turn. A model that answers and then
appends an illustrative snippet breaks last-block. So it is a parameter,
defaulting to first, which is the safer failure: an agent that re-reads its
own broken code gets a diagnostic and another turn, where one that runs an
illustration can do something the task never asked for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import nu


if TYPE_CHECKING:
    from nu.lang import Nu, StrArg


__all__ = ["FENCE", "fenced", "fenceless"]


FENCE = "```"

# ```python / ```py / bare ```. The tag sits at the head of the chunk, so
# stripping is a chain of removeprefix, each a no-op when it does not match.
_TAGS = ("python", "py")


def fenceless(text: StrArg) -> Nu:
    """True when the reply holds no complete fenced block, as a Bool term.

    Splitting on the fence yields at least three chunks when a block opened
    and closed, so fewer than three means there is nothing fenced to take and
    :func:`fenced` is about to fall back to the whole reply.

    That fallback is right for bare source and wrong for prose, and neither
    case is distinguishable here. The caller decides: it is the thing that
    also knows whether the fallback then failed to construct, and a
    construction failure on a reply with no fence is prose, not code.

    Args:
        text: the reply, as a str or any Str-yielding term.

    Returns:
        A ``Bool`` term.
    """
    return nu.Int(nu.Len(nu.Str(text).split(FENCE))) < 3


def fenced(text: StrArg, *, block: Literal["first", "last"] = "first") -> Nu:
    """The source inside a fenced block, as a Str-yielding term.

    Splitting on the fence puts code at the odd indices: ``[before, code,
    between, code, after]``. The first block is index 1, the last is the
    second-to-last chunk, computed at runtime because the count is not known
    when the tree is built.

    When there is no complete block (fewer than three chunks) the whole reply
    is used, stripped. A model that answered with bare source and no fence is
    common enough to be worth handling, and the failure mode if it was really
    prose is a construction Diagnostic, which is a turn the agent recovers
    from.

    Args:
        text: the reply, as a str or any Str-yielding term.
        block: which fenced block to take, ``"first"`` (default) or
            ``"last"``.

    Returns:
        A ``Str`` term yielding the extracted source.

    Raises:
        ValueError: for a ``block`` other than "first" or "last".
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
