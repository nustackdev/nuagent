"""The prompt as an ordered list of addressable pieces.

A system prompt is not one string, it is a sequence of things a model needs
to know, and which of them a given agent needs depends on the agent. An
agent with no bound fabric has no use for the fabrics section; an agent
without ``Inspect`` on its surface is being taught a call it cannot make.
Both cases are a caller dropping one name from a tuple, which is why the
prompt is a tuple of named sections rather than a template.

The prose lives in ``text/*.md``, shipped as package data and read through
``importlib.resources`` so it resolves the same from a wheel and from an
editable checkout. Nothing here formats that text. The markdown contains
Python samples full of dict and set literals, so ``str.format`` would raise
on the braces and an f-string would swallow them; sections compose by
concatenation only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    "PROSE",
    "Section",
    "inserted",
    "markdown_section",
    "read_markdown",
    "replaced",
    "text_files",
    "without",
]


TEXT_DIR = "text"

#: The static prose, in the order it is presented to the model. Each name is
#: the section name and ``<name>.md`` is the file it reads.
PROSE: tuple[str, ...] = (
    "role",
    "thesis",
    "fabrics",
    "crashcourse",
    "examples",
    "protocol",
    "finish",
    "inspect",
)


def read_markdown(filename: str) -> str:
    """Read one file out of the shipped ``text/`` directory."""
    return resources.files(__package__).joinpath(TEXT_DIR).joinpath(filename).read_text("utf-8")


def text_files() -> tuple[str, ...]:
    """Every ``.md`` shipped in ``text/``, sorted. Used by the rot tests."""
    directory = resources.files(__package__).joinpath(TEXT_DIR)
    if not directory.is_dir():
        return ()
    return tuple(sorted(entry.name for entry in directory.iterdir() if entry.name.endswith(".md")))


@dataclass(frozen=True)
class Section:
    """One named piece of the prompt.

    ``body`` is a callable rather than a string so a section can be generated
    at build time (the catalogue reads the live ``nu.inspect`` records) and so
    a file-backed section does not touch the filesystem until someone asks
    for a prompt.
    """

    name: str
    body: Callable[[], str] = field(repr=False)

    def render(self) -> str:
        """The section's text, stripped of surrounding blank lines."""
        return self.body().strip("\n")


def markdown_section(name: str, filename: str | None = None) -> Section:
    """A section backed by ``text/<name>.md``."""
    path = filename or f"{name}.md"
    return Section(name, lambda: read_markdown(path))


def without(sections: tuple[Section, ...], *names: str) -> tuple[Section, ...]:
    """``sections`` minus every section named in ``names``."""
    drop = set(names)
    return tuple(section for section in sections if section.name not in drop)


def replaced(sections: tuple[Section, ...], section: Section) -> tuple[Section, ...]:
    """``sections`` with the same-named section swapped for ``section``."""
    return tuple(section if existing.name == section.name else existing for existing in sections)


def inserted(
    sections: tuple[Section, ...],
    section: Section,
    *,
    before: str | None = None,
    after: str | None = None,
) -> tuple[Section, ...]:
    """``sections`` with ``section`` spliced in, or appended when neither anchor is given."""
    anchor = before or after
    if anchor is None:
        return (*sections, section)
    names = [existing.name for existing in sections]
    if anchor not in names:
        raise KeyError(f"no section named {anchor!r}")
    at = names.index(anchor) + (0 if before else 1)
    return (*sections[:at], section, *sections[at:])
