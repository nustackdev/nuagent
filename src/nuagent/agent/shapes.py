"""The session: every slot a turn reads or writes, one Shape per fabric.

The same seven slots fill the same roles in every turn, so they are a Shape
rather than a dozen Ref arguments. ``turn`` and ``agent`` take one ``session``
and read the rest off it.
"""

from __future__ import annotations

import nu


__all__ = ["KVSession", "MemSession"]


class MemSession(nu.Shape):
    """One agent run's slots, on nu.mem."""

    messages = nu.mem.ListRef.slot(dict)
    reply = nu.mem.StrRef.slot()
    draft = nu.mem.ProgramRef.slot()
    outcome = nu.mem.StrRef.slot()
    observation = nu.mem.StrRef.slot()
    turns = nu.mem.IntRef.slot()
    done = nu.mem.BoolRef.slot()


class KVSession(nu.Shape):
    """The same slots on nu.kv: durable, and readable after the run ends."""

    messages = nu.kv.ListRef.slot(dict)
    reply = nu.kv.StrRef.slot()
    draft = nu.kv.ProgramRef.slot()
    outcome = nu.kv.StrRef.slot()
    observation = nu.kv.StrRef.slot()
    turns = nu.kv.IntRef.slot()
    done = nu.kv.BoolRef.slot()
