"""nuagent: an agent as a Nu term.

Not a fabric. ``kv``, ``mem``, ``http``, ``llm`` and ``cc`` each own a
substrate -- a store, a connection, a process -- and exist to make it
addressable. ``nuagent`` owns nothing external. Its substrate is whatever
the caller bound, and it holds no Shape, no state, no registry of its own.
The right analogue is :mod:`nu.core.flows`: composition over what already exists.

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
- :mod:`~nuagent.agents` -- the turn machinery: the fenced block out of a
  reply, the yield and the state as the next message, one turn usable alone
  as a single-shot agent, and the composed agents. One so far: ``Agent`` (a
  Turn under a WhileDo).

Prefer ``nu.arun``: an LLM call is network-bound and blocks the loop under
sync.
"""

from __future__ import annotations

from .agents import (
    NO_CODE,
    Agent,
    Turn,
    attempted,
    crashed,
    failed,
    fenced,
    fenceless,
    never_ran,
    observation,
    rendered,
)
from .prompt import DEFAULT_MODULES, DEFAULT_SECTIONS, inserted, surface_section, system_prompt


__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODULES",
    "DEFAULT_SECTIONS",
    "NO_CODE",
    "Agent",
    "Turn",
    "__version__",
    "attempted",
    "crashed",
    "failed",
    "fenced",
    "fenceless",
    "inserted",
    "never_ran",
    "observation",
    "rendered",
    "surface_section",
    "system_prompt",
]
