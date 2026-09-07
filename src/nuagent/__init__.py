"""nuagent: an agent as a Nu term.

The thesis
----------

The model does not emit JSON tool calls. It writes python source defining
``out()`` that returns a Nu term. The source lands in a ``ProgramRef``,
``LoadNu`` constructs it inside a ``PyBrace``, ``Eval`` drives it. When
construction fails the ``Diagnostic`` becomes the next turn's input and the
model fixes its own code.

So there is no tool schema, and nothing to register. What the model may
compose against is whatever ``nu.inspect`` can describe.

The modules
-----------

- :mod:`~nuagent.prompt` -- system prompt as ordered sections: prose shipped
  as package data, vocabulary generated from ``nu.inspect`` catalogues, and
  the caller's own Shapes and Services rendered as the app surface. Drop or
  replace a section per agent.
- :mod:`~nuagent.agent` -- the session Shape (the same slots on ``nu.mem`` or
  ``nu.kv``, which is all that separates an ephemeral run from a durable one),
  the fenced block out of a reply, one attempt at the program, ``Turn`` usable
  alone as a single-shot agent, and ``Agent``: a Turn under a WhileDo.

Prefer ``nu.arun``: an LLM call is network-bound and blocks the loop under
sync.
"""

from __future__ import annotations

from .agent import (
    FAILED_LABEL,
    FENCE,
    NO_CODE,
    NO_CODE_LABEL,
    KVSession,
    MemSession,
    agent,
    attempted,
    failed,
    fenced,
    turn,
)
from .prompt import DEFAULT_MODULES, DEFAULT_SECTIONS, inserted, surface_section, system_prompt


__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODULES",
    "DEFAULT_SECTIONS",
    "FAILED_LABEL",
    "FENCE",
    "NO_CODE",
    "NO_CODE_LABEL",
    "KVSession",
    "MemSession",
    "__version__",
    "agent",
    "attempted",
    "failed",
    "fenced",
    "inserted",
    "surface_section",
    "system_prompt",
    "turn",
]
