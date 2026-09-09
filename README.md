<div align="center">
  <h1>nuagent</h1>
  <h3>An AI agent that speaks <a href="https://github.com/nustackdev/nu">Nu</a></h3>

  [![Powered by Nu](https://img.shields.io/badge/powered%20by-Nu-5865F2)](https://github.com/nustackdev/nu)
  [![PyPI - Python Version](https://img.shields.io/badge/python-%3E%3D%203.10-blue)](https://pypi.org/project/nuagent/)
  [![PyPI Package](https://img.shields.io/pypi/v/nuagent?color=yellow)](https://pypi.org/project/nuagent/)
</div>

<br/>

A tool-calling agent picks one registered function per turn and answers with JSON. Six steps, six round trips, every branch routed back through the model.

Here the action is a program. The model writes one Nu term, an immutable expression tree, that reads, branches and writes against the Refs you bound. nuagent runs it and feeds back what it yielded plus the state it left behind.

```python
def out():
    return nu.ForEachDo(
        nu.Iter(nu.Literal(["mon", "tue", "wed"])),
        Notes.items.append(nu.AttrRef("item")),
    ) >> Notes.count.set(nu.Len(Notes.items)) >> Run.done.set(True)
```

A loop, three writes, a derived count and the run ending itself. One reply.

- Nothing to register, no schema to keep in sync. Capability is the set of Refs bound around the loop, so bind a different set and it is a different agent with the same code.
- Reach is whatever Nu reaches: a dict, a `nu.kv` store on disk, an http service, anything a fabric backs. The shipped catalogue is nucore only, so finding a fabric's verbs costs the model a turn.
- The loop is a term as well: a `WhileDo` over the turn, no python driver. Swap `MemSession` for `KVSession` and the same run is durable.

## Installation

Requires Python 3.10+.

```bash
pip install nuagent
```

## Usage

An agent over a notes list. The app's Shape goes into the prompt, so the model
opens the run already knowing the slot names and their types.

```python
import nu, nuagent


class Notes(nu.Shape):
    items = nu.mem.ListRef.slot(str)
    count = nu.mem.IntRef.slot()


class Bot(nu.Service):
    chat = nu.llm.ChatRef.method(temperature=0)


TASK = """\
Append one note per weekday to `items`, then set `count` to how many notes are
in the list. Do it as one program, and derive the count."""

system = nuagent.system_prompt(
    TASK,
    sections=nuagent.inserted(nuagent.DEFAULT_SECTIONS, nuagent.surface_section((Notes,))),
)

loop = nuagent.agent(
    session=nuagent.MemSession,
    chat=Bot.chat,
    state=nu.Dict.of(items=Notes.items, count=Notes.count),
    start=nuagent.MemSession.messages.set(
        nu.List.of(nu.Dict.of(role="system", content=nu.Str(system))),
    )
    >> Notes.items.set([])
    >> Notes.count.set(0),
    report=nu.print(nu.Str("\nnotes: ") + nu.ToStr(nu.Repr(Notes.items))),
)

nu.run(
    nu.With(
        nu.Provide(dict, {}),
        nu.llm.ollama(Bot, host="localhost", model="qwen2.5:7b-instruct"),
        body=loop,
    )
)
```

## Status

Early. The API is not settled yet.
