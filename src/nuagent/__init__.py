"""nuagent -- AI agents whose action is python source, not a JSON tool call.

The model writes a module defining ``out()`` that returns a Nu term. The
source lands in a ``ProgramRef``, ``LoadNu`` constructs it inside a
``PyBrace``, ``Eval`` drives it, and a construction ``Diagnostic`` comes
back as the next turn's input so the model repairs its own code.

JSON tool calling is an IR that exists because most hosts cannot run code
the model wrote. Nu can, so the IR is unnecessary.
"""

from __future__ import annotations


__version__ = "0.1.0"

__all__ = [
    "__version__",
]
