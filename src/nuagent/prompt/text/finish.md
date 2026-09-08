# Finishing

You end the run, and the only way to end it is a write. One extra Shape is on your surface for exactly that:

```python
class Run(nu.Shape):
    done = nu.mem.BoolRef.slot()
```

Redeclare it in your module like any other Shape, taking the fabric off its heading in your app surface, and set `done` in the same program that finishes the work:

```python
def out():
    return (
        Board.tasks["t2"].summary.set("Backfill the metrics table.")
        >> Board.open_count.set(3)
        >> Run.done.set(True)
    )
```

- Set `Run.done` to True only once the work is actually done. The turn finishes normally and then there is no next turn, so anything you were still going to do does not happen.
- Leave it alone to keep going. Every turn you do not set it, you get another one, up to the turn budget.
- A turn spent looking something up is a turn where you do not set it. Never set it in the same breath as a guess.
- Saying "done" in prose ends nothing. The Ref is the only signal the host reads.
