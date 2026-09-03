"""Prompt assembly from nu.info catalogues.

The old vocabulary lived here as hand-written text. It's gone. The prompt
now reads the same records the docs site reads, so a rename in the core
turns into a rendered rename in the next prompt instead of a stale line a
model treats as authoritative.

Four modules are hardcoded as the standard vocabulary an agent needs to
compose against: ``nu.core``, ``nu.flows``, ``nu.spans``, and
``nu.forms.primitives``. The prompt lists each subject with a one-line
summary; the model uses ``nu.info.Inspect(path)`` to pull the full docs
when it needs them. Sending everything up front would blow past what the
model can hold; sending nothing would force blind composition. This is the
middle path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nu.core
import nu.flows
import nu.forms.collections
import nu.forms.primitives
import nu.spans
from nu.info import catalogue_forms, catalogue_interactions, catalogue_refs


if TYPE_CHECKING:
    from types import ModuleType


__all__ = ["DEFAULT_MODULES", "system_prompt"]


DEFAULT_MODULES: tuple[ModuleType, ...] = (
    nu.core,
    nu.flows,
    nu.spans,
    nu.forms.primitives,
    nu.forms.collections,
)


_ORIENTATION = """\
You are an agent that solves tasks by composing Nu terms.

Nu at a glance
- A Nu program is a tree. You build it by calling constructors; the host
  runs the finished tree.
- Interactions are the atoms of the tree (`Add`, `IfDo`, `WhileDo`, `Print`).
  They take Nu children and yield a value when the tree runs.
- Forms are typed interfaces over values (`Int`, `Str`, `List`, `Dict`).
  Their methods build more terms (`Int(2) + Int(3)` builds an Add term).
- Refs name locations in a Fabric. They read as a value and write with
  methods like `.set(...)`. What Refs exist depends on what the caller
  bound.
- `>>` sequences terms. The left runs, then the right. Use it to chain
  Commands. Any Nu constructor call is itself a term you can compose.

How you reply
- One fenced ```python``` block per turn, defining `out()` that returns a
  Nu term.

      import nu


      def out():
          return nu.<term>

- Never call `nu.run`; the host does. Never handle exceptions inside
  `out()` - if construction fails the diagnostic comes back to you as the
  next turn's input and you fix your code.

Learning what exists
- The catalogue below lists every atom you can compose against, one line
  each. It tells you the atom's name and what it does at a glance.
- When you need details - args, notes, examples, semantics - compose
  `nu.info.Inspect("<dotted.path>")` and read the returned string. It
  works on a whole module (`nu.info.Inspect("nu.core.arithmetic")`) or one
  atom (`nu.info.Inspect("nu.core.arithmetic.Add")`, or on a Form
  `nu.info.Inspect("nu.forms.primitives.Int")`).
- Inspect is itself a Nu atom that yields a string. To read it in the
  current turn, use `nu.print(nu.info.Inspect("..."))` so it lands in the
  observation; then compose the real answer next turn.

Modules on your surface:
"""


def system_prompt(
    task: str,
    *,
    modules: tuple[ModuleType, ...] = DEFAULT_MODULES,
) -> str:
    """Build the system prompt.

    Args:
        task: what the agent should do. Appended last.
        modules: the modules whose catalogues become the vocabulary.
            Defaults to nu.core, nu.flows, nu.spans, nu.forms.primitives.

    Returns:
        The prompt text.
    """
    parts: list[str] = [_ORIENTATION]
    parts.extend(f"  - {m.__name__}" for m in modules)
    parts.extend(["", "CATALOGUE (name — one-line summary; use nu.info.Inspect for details)"])
    for module in modules:
        parts.append("")
        parts.append(_thin(module))
    parts.extend(["", "TASK", "", task])
    return "\n".join(parts)


def _thin(module: ModuleType) -> str:
    """A short catalogue: module name, then one line per subject."""
    lines: list[str] = [f"## {module.__name__}"]
    forms = catalogue_forms(module)
    refs = catalogue_refs(module)
    interactions = catalogue_interactions(module)
    for group_name, group in (
        ("forms", forms),
        ("refs", refs),
        ("interactions", interactions),
    ):
        if not group:
            continue
        lines.append(f"  {group_name}:")
        for record in group:
            summary = (record.summary or "-").rstrip(".")
            lines.append(f"    {record.name:<22} {summary}")
    return "\n".join(lines)
