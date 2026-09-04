"""Ordering the sections and appending the task.

The order is the argument: who you are, what the model is, where values
live, how to write it, what it looks like done, what comes back, how to look
things up, what exists, and only then what to do. The task goes last because
it is the only part that changes per call and a model attends hardest to the
end of what it is given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .catalogue import DEFAULT_MODULES, catalogue_section
from .sections import PROSE, markdown_section


if TYPE_CHECKING:
    from .sections import Section


__all__ = ["DEFAULT_MODULES", "DEFAULT_SECTIONS", "system_prompt"]


DEFAULT_SECTIONS: tuple[Section, ...] = (
    *(markdown_section(name) for name in PROSE),
    catalogue_section(),
)


def system_prompt(task: str, *, sections: tuple[Section, ...] = DEFAULT_SECTIONS) -> str:
    """Build the system prompt.

    Args:
        task: what the agent should do. Appended last, under a ``# Task``
            heading.
        sections: the sections to render, in order. Use ``without``,
            ``replaced`` and ``inserted`` from ``nuagent.prompt.sections`` to
            derive a set from ``DEFAULT_SECTIONS``.

    Returns:
        The prompt text.
    """
    parts = [section.render() for section in sections]
    parts.append("# Task\n\n" + task.strip())
    return "\n\n".join(parts) + "\n"
