"""The agent: the session Shape, the turn, and the loop over it.

- :mod:`shapes` -- the slots one run needs, one Shape per fabric.
- :mod:`utils` -- fenced-block extraction and one attempt at the program.
- :mod:`agent` -- ``Turn``, and ``Agent`` as a Turn under a ``WhileDo``.
"""

from __future__ import annotations

from .agent import agent, turn
from .shapes import KVSession, MemSession
from .utils import FAILED_LABEL, FENCE, NO_CODE, NO_CODE_LABEL, attempted, failed, fenced


__all__ = [
    "FAILED_LABEL",
    "FENCE",
    "NO_CODE",
    "NO_CODE_LABEL",
    "KVSession",
    "MemSession",
    "agent",
    "attempted",
    "failed",
    "fenced",
    "turn",
]
