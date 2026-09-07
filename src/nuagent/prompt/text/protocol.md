# The turn protocol

## Your reply

Write exactly one fenced code block per reply. The host splits your reply on the fence and takes the **first** block. Prose before or after it is free. A second block is ignored, so never quote a broken earlier version above your fix. If your reply contains no complete fenced block at all, the whole reply is used as source.

Every reply must carry a code block. Prose is not an action: the host runs programs and nothing else, so a reply without one runs nothing, changes nothing, and costs a turn. You get back `NO CODE BLOCK: your reply contained no fenced code block, so nothing ran`.

The run ends when the goal holds, not when you say it does. If you believe the task is complete, say it as a program that yields the evidence, and read the `goal` line to find out whether you were right.

The block is a Python module:

```python
import nu


class World(nu.Shape):
    notes = nu.mem.ListRef.slot(str)
    count = nu.mem.IntRef.slot()


def out():
    return World.notes.init([]) >> World.notes.append("n1") >> World.count.set(nu.Len(World.notes))
```

Rules for the module:

- `import nu` at the top.
- Declare Shapes at module level.
- Define `def out()`. It takes **no arguments**. The host calls it with none.
- `out()` returns one Nu term. That term is the program.

## What the host does with it

The source lands in a `ProgramRef`. `LoadNu` loads the module and calls the entry point `out`, which constructs the term. `Eval` runs it. Your program runs **exactly once** per turn.

## What you get back

The next user message is exactly this, and nothing else:

    outcome: <repr of what the term yielded>
    state: <the world after the program ran>
    goal: <verdict>

`state` appears when the host supplies one. `goal` appears when there is a goal. You never see a traceback, a tool result, or the host's stdout.

Read it like this:

- `outcome` is the yield, repr'd. Newlines arrive as a literal `\n` inside one quoted string.
- **A Command yields nothing, so `outcome: None` is normal.** A chain of writes that completely succeeded reports `outcome: None`. It is not a failure and not silence. `state` is where the evidence of your work is. Read `state` before rewriting anything: if the write landed, do not do it again.
- `goal: met` means the task is done. Stop writing programs and say so.
- `goal: NOT met yet...` means your program built and ran and did the wrong thing. That is the only signal that working code is still wrong. Correct it.
- `goal: NOT met, and nothing ran this turn...` means there was no program: `state` is unchanged and there is nothing in it to learn from. Fix what the `outcome` says and send one.
- `nu.print(...)` writes to the **host's** stdout. You never see it, and the outcome is `None`. To read a value, `return` it from `out()`.

## When it fails

Three failure paths. All come back as the `outcome`, in the same slot a successful yield uses. None of them ends the run. A diagnostic is an ordinary message you are expected to read and fix on the next turn.

No code: your reply had no fenced block, so nothing was run.

- `NO CODE BLOCK: your reply contained no fenced code block, so nothing ran.`

Construction failed: the source did not parse, or `out()` raised while building the term.

- `CONSTRUCTION FAILED: source does not parse: '(' was never closed (line 5)`
- `CONSTRUCTION FAILED: entry point 'out' raised: AttributeError: module 'nu' has no attribute 'Nope' (line 5)`
- `CONSTRUCTION FAILED: entry point 'out' raised: ZeroDivisionError: division by zero (line 5)`

The line number counts lines of the source you sent, starting at 1.

Runtime failed: the term built and then raised while running.

- `RUNTIME FAILED: division by zero`

No line number on this one.

Never write `try`/`except` inside `out()`. Both paths are already caught for you, and catching them yourself hides the message you need.

## Python runs at construction, not in the program

Everything inside `out()` executes while the term is being built, in the host's interpreter. `if`, `for`, `while` and `try` written there decide what the tree looks like. They are not in the tree and they do not run when the program runs. Branching and looping that must happen while the program runs are atoms: `IfDo`, `ForEachDo`, `ForRangeDo`, `WhileDo`, `SwitchDo`.
