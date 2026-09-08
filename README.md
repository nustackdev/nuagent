<div align="center">
  <h1>nuagent</h1>
  <h3>An AI agent that speaks <a href="https://github.com/nustackdev/nu">Nu</a></h3>

  [![Powered by Nu](https://img.shields.io/badge/powered%20by-Nu-5865F2)](https://github.com/nustackdev/nu)
  [![PyPI - Python Version](https://img.shields.io/badge/python-%3E%3D%203.10-blue)](https://pypi.org/project/nuagent/)
  [![PyPI Package](https://img.shields.io/pypi/v/nuagent?color=yellow)](https://pypi.org/project/nuagent/)
</div>

<br/>

Agents are usually bolted onto a stack that was never one thing to begin with, so you hand-write a tool for every corner of it and the agent only ever sees the corners you remembered. Nu apps are already coherent: one primitive over data, UI, compute and services. So an agent that speaks Nu inherits all of it. It answers with a program instead of a call, which means one turn can sequence, branch and loop over your real state. It queries a billion-row `nu.kv` store the same way it reads a counter. Bind `nu.cluster` and what it writes runs on the cluster. Nothing to register, nothing to keep in sync, and no part of your app it cannot reach.

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
