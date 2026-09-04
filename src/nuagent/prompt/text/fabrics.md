# Fabrics

A Ref is only resolvable through the fabric that owns its address. Each fabric ships its own Ref types; you pick a fabric by picking which Ref you declare a slot with.

Two you are bound to unless told otherwise:

- `nu.mem` - shape fabric over plain nested python dicts. Ephemeral, in-process, the default. 26 Refs, from `StrRef`/`IntRef`/`ListRef`/`DictRef` up to `ShapeRef`/`ShapesListRef`/`ProgramRef`.
- `nu.kv` - virtuals KV-storage fabric for Shapes, over rocksdb, lmdb, redis or memory. `nu.mem`'s Refs plus `ViewRef`, `Kh57Ref`, primitive-collection Refs, and the `Transaction`/`Snapshot`/`Atomic`/`RetryOnConflict` spans. Same program, durable.

The rest exist and are context. They are not bound on your surface unless the task says so.

- `nu.service` - expose a plain python object's methods as Nu Refs.
- `nu.std` - typed Nu surfaces over python's standard library.
- `nu.http` - http endpoints as Refs.
- `nu.llm` - OpenAI-compatible chat over ollama, openai, openrouter, groq, cerebras, vllm, xai.
- `nu.cc` - Claude Code sessions as Refs.
- `nu.ui` - component fabric; Refs are widgets, containers are pages, served over http.
- `nu.cluster` - Ray compute fabric.
- `nu.mp` - multiprocessing compute fabric.
- `nu.proxy` - transparent RPC transport that wraps another fabric over the wire.

The catalogue you are given lists no fabric Refs and none of their verbs, so read a fabric's surface before writing against it.
