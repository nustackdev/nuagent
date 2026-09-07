"""Agents: the turn machinery and one term-composition per kind.

- :mod:`extract` — fenced block out of a reply, as a Nu Query.
- :mod:`observation` — the yield and the state, as the next message.
- :mod:`turn` — one turn, usable alone as a single-shot agent.
- :func:`loop.Agent` — Turn under a WhileDo, stops on a goal or a budget.
"""

from __future__ import annotations

from .extract import fenced, fenceless
from .loop import Agent
from .observation import NO_CODE, attempted, crashed, failed, never_ran, observation, rendered
from .turn import Turn


__all__ = [
    "NO_CODE",
    "Agent",
    "Turn",
    "attempted",
    "crashed",
    "failed",
    "fenced",
    "fenceless",
    "never_ran",
    "observation",
    "rendered",
]
