# Writing Nu

## Shapes

A Shape declares slots. Each slot is a Ref: an address in a fabric.

```python
import nu


class Movie(nu.Shape):
    title: nu.mem.StrRef
    year: nu.mem.IntRef


class Library(nu.Shape):
    name: nu.mem.StrRef
    tags = nu.mem.ListRef.slot(str)
    scores = nu.mem.DictRef.slot(int)
    featured = nu.mem.ShapeRef.slot(Movie)
```

Two slot forms, both correct, mixable in one class. Annotation (`title: nu.mem.StrRef`) when the Ref class alone is the whole declaration. `.slot(...)` when the slot takes an argument (`str`, `int`, a Shape) or config (`view=`, `size=`, `capacity=`). Never both on one slot. Every shipped example in the nu repo uses `.slot()` for every slot, so that is the form you will see in the wild.

**Shapes are never instantiated.** `Library()` is never written. Refs are class attributes used unbound:

```python
Library.tags.append("dune")  # a term
Library.featured.year.set(1984)  # nested slot, a term
Library.scores["x"].set(4)  # keyed slot, a term
```

Slot verbs: `set init erase exists missing is_empty not_empty is_invalid not_invalid`, plus per-type verbs (`inc dec` on Int; `append extend insert pop remove clear len contains index first_elem last_elem slice` on List).

## Services

A Service declares method Refs over a plain Python object. Pick the Ref by kind.

```python
class Calc(nu.Service):
    add = nu.service.QueryRef.method()  # yields, no mutation
    bump = nu.service.ActionRef.method()  # mutates and yields
    wipe = nu.service.CommandRef.method(name="reset")  # mutates, yields nothing
    squares = nu.service.StreamQueryRef.method(name="range")  # generator
```

Method Refs are called with kwargs: `Calc.add(a=2, b=3)`. `name=` maps to a differently named attribute on the target. The Service class is never instantiated; the target is.

## Binding a fabric

From outside the tree:

```python
ctx = nu.Context().bind(dict, {})  # nu.mem, every Shape
ctx = nu.Context().bind(dict, state, Library)  # nu.mem, scoped to one Shape
```

`Context.bind` returns a new Context. Inside the tree:

```python
nu.Provide(dict, {}, Library.name.set("x"))  # one fabric, one body
nu.With(nu.kv.memory_navigator(), body=Library.name.set("x"))  # several brackets, flat
```

Brackets come from fabric helpers, each returning a `Provide` or a `With`:

```python
nu.service.bind(Calc, target=object())
nu.http.bind(Calc, base_url="https://api.example.com")
nu.llm.ollama(Calc, model="qwen3")
nu.kv.memory_navigator()
nu.kv.rocksdb_navigator(".db")
```

`nu.mem` ships no bind helper; use `Context.bind(dict, ...)` or `nu.Provide(dict, {}, body)`. `nu.kv` writes need `nu.kv.auto_flow_atomic(body)` around the flow.

Nothing in the tree reaches a fabric that was not provided around it. `nu.run(Library.name.set("x"))` with no Context raises `LookupError: No binding for: dict[Library]`.

`nu.mem` addresses by slot name only, so two Shapes with the same slot name collide under one unscoped `bind(dict, {})`. Tag with the Shape to keep them apart.

## Running

```python
value, ctx = nu.run(term)  # sync
# asyncio.run(nu.arun(term))         # async
```

`nu.run` raises `ValueError` on Nu law validation, and `RuntimeError: eval: program contains an async-only atom` when the tree needs `arun`. Use `arun` for `nu.ui.server`, long polls, `ParallelAsync`, `Watch`. Everything else runs under `nu.run`.

**The host runs the tree.** You build and return a term. Do not write `nu.run`, `asyncio.run`, or `if __name__ == "__main__":` unless asked for a standalone script.

## Sentinels

An unset slot yields `EMPTY`. `EMPTY` in a concat or arithmetic collapses the whole term to `INVALID`. What a Command then does with `INVALID` depends on the verb: `.set(...)` raises `ValueError: cannot store sentinel value`, while `.append(...)` writes nothing and raises nothing. The silent half is the dangerous one, because the program runs, changes nothing, and reports no error.

Seed every slot before reading it. `.init(v)` writes only when the slot is missing. Test with `.missing()`, `.exists()`, `nu.IsEmpty(ref)`, `nu.IsInvalid(term)`.
