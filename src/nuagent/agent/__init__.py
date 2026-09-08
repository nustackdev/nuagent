"""The agent: the session Shape, the run Shape, the turn, and the loop over it.

- :mod:`shapes` -- the slots one run needs, one Shape per fabric, plus the
  one-slot ``Run`` the model writes to end the loop.
- :mod:`utils` -- fenced-block extraction and one attempt at the program.
- :mod:`agent` -- ``turn``, and ``agent`` as a turn under a ``WhileDo``.
"""

from __future__ import annotations

from .agent import agent, turn
from .shapes import KVRun, KVSession, MemSession, Run
from .utils import FAILED_LABEL, FENCE, NO_CODE, NO_CODE_LABEL, attempted, failed, fenced


__all__ = [
    "FAILED_LABEL",
    "FENCE",
    "NO_CODE",
    "NO_CODE_LABEL",
    "KVRun",
    "KVSession",
    "MemSession",
    "Run",
    "agent",
    "attempted",
    "failed",
    "fenced",
    "turn",
]
