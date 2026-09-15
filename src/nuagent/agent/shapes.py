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
import nustd


__all__ = ["KVRun", "KVSession", "MemSession", "Run"]


class MemSession(nu.Shape):
    """One agent run's slots, on nustd.mem."""

    messages = nustd.mem.ListRef.slot(dict)
    reply = nustd.mem.StrRef.slot()
    draft = nustd.mem.ProgramRef.slot()
    outcome = nustd.mem.StrRef.slot()
    observation = nustd.mem.StrRef.slot()
    turns = nustd.mem.IntRef.slot()


class KVSession(nu.Shape):
    """The same slots on nustd.kv: durable, and readable after the run ends."""

    messages = nustd.kv.ListRef.slot(dict)
    reply = nustd.kv.StrRef.slot()
    draft = nustd.kv.ProgramRef.slot()
    outcome = nustd.kv.StrRef.slot()
    observation = nustd.kv.StrRef.slot()
    turns = nustd.kv.IntRef.slot()


class Run(nu.Shape):
    """The run itself: set `done` to True and the agent stops.

    Notes:
        - `done` is yours to write. Set it in the same program that finishes
          the work, once the work is actually finished.
        - The loop reads this exact slot after every turn. Nothing else ends
          a run except the turn budget.
        - It rides on the app's own fabric, so it needs no binding of its own.
    """

    done = nustd.mem.BoolRef.slot()


class KVRun(nu.Shape):
    """The same one slot on nustd.kv, for an app whose world is durable.

    Notes:
        - Declare it as `Run`: addressing is by slot name, and the class name
          is not part of the address.
    """

    done = nustd.kv.BoolRef.slot()
