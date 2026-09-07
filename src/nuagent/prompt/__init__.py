"""Prompt assembly: ordered sections, prose from disk, vocabulary and surface from nu.inspect."""

from __future__ import annotations

from .assemble import DEFAULT_SECTIONS, system_prompt
from .catalogue import DEFAULT_MODULES, bytes_methods, catalogue_section, render_catalogue
from .sections import PROSE, Section, inserted, markdown_section, read_markdown, replaced, without
from .surface import app_records, render_surface, surface_section


__all__ = [
    "DEFAULT_MODULES",
    "DEFAULT_SECTIONS",
    "PROSE",
    "Section",
    "app_records",
    "bytes_methods",
    "catalogue_section",
    "inserted",
    "markdown_section",
    "read_markdown",
    "render_catalogue",
    "render_surface",
    "replaced",
    "surface_section",
    "system_prompt",
    "without",
]
