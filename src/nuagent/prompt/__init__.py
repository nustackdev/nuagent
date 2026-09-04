"""Prompt assembly: ordered sections, prose from disk, vocabulary from nu.inspect."""

from __future__ import annotations

from .assemble import DEFAULT_SECTIONS, system_prompt
from .catalogue import DEFAULT_MODULES, bytes_methods, catalogue_section, render_catalogue
from .sections import PROSE, Section, inserted, markdown_section, read_markdown, replaced, without


__all__ = [
    "DEFAULT_MODULES",
    "DEFAULT_SECTIONS",
    "PROSE",
    "Section",
    "bytes_methods",
    "catalogue_section",
    "inserted",
    "markdown_section",
    "read_markdown",
    "render_catalogue",
    "replaced",
    "system_prompt",
    "without",
]
