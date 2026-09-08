"""The session and the run: the host's slots, and the one slot the model owns.

The same six slots fill the same roles in every turn, so they are a Shape
rather than a dozen Ref arguments. ``turn`` and ``agent`` take one ``session``
and read the rest off it.

``Run`` is the other half, and it is deliberately not part of the session. The
session is the host's working memory and is normally bound *tagged*, out of
the model's reach, so nothing the model writes can touch the conversation.
``Run`` is bound where the app is bound, untagged, because the model has to
write it: it redeclares the Shape in its own module and sets ``done`` when the
task is finished, and the loop stops that turn. One slot, one meaning, written
by the model and read by the loop.
"""

from __future__ import annotations

import nu


__all__ = ["KVRun", "KVSession", "MemSession", "Run"]


class MemSession(nu.Shape):
    """One agent run's slots, on nu.mem."""

    messages = nu.mem.ListRef.slot(dict)
    reply = nu.mem.StrRef.slot()
    draft = nu.mem.ProgramRef.slot()
    outcome = nu.mem.StrRef.slot()
    observation = nu.mem.StrRef.slot()
    turns = nu.mem.IntRef.slot()


class KVSession(nu.Shape):
    """The same slots on nu.kv: durable, and readable after the run ends."""

    messages = nu.kv.ListRef.slot(dict)
    reply = nu.kv.StrRef.slot()
    draft = nu.kv.ProgramRef.slot()
    outcome = nu.kv.StrRef.slot()
    observation = nu.kv.StrRef.slot()
    turns = nu.kv.IntRef.slot()


class Run(nu.Shape):
    """The run itself: set `done` to True and the agent stops.

    Notes:
        - `done` is yours to write. Set it in the same program that finishes
          the work, once the work is actually finished.
        - The loop reads this exact slot after every turn. Nothing else ends
          a run except the turn budget.
        - It rides on the app's own fabric, so it needs no binding of its own.
    """

    done = nu.mem.BoolRef.slot()


class KVRun(nu.Shape):
    """The same one slot on nu.kv, for an app whose world is durable.

    Notes:
        - Declare it as `Run`: addressing is by slot name, and the class name
          is not part of the address.
    """

    done = nu.kv.BoolRef.slot()
