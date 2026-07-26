# Findings

A running log of JAX rough edges turned up while writing the examples in this
directory. Examples exercise the public API the way a new user does, in
complete programs rather than snippets, so they tend to hit things before a
user reports them. This is the list of those things: nothing here is fixed by
the examples themselves, and each entry is a question for someone who owns the
relevant area.

Append new entries at the top. Strike through or delete entries once they're
resolved upstream.

---

## 2026-07-26 — from the ZeRO-2 rewrite of `nanolm.py` and from `fsdp_pipeline.py`

### 5. Converting an unreduced array to NumPy fails, and the error asks for a bug report

`reduced`/`unreduced` works beautifully — a one-line cast in the forward pass
really does move the reduction to where you want it in the backward pass. But
an unreduced array can't leave the device:

```python
g = jax.grad(loss)(params, batch)   # f32[...]{U:data}
np.asarray(g['up'])
# NotImplementedError: device_indices_map doesn't work with unreduced.
# Please file a bug at https://github.com/jax-ml/jax/issues
```

Filing that bug here. The natural reading of `np.asarray` on an unreduced array
is "perform the pending reduction, then give me the value", which is what
`jax.reshard(g, ...)` would have done anyway. Failing is defensible, but the
message should then say *"unreduced arrays must be resharded before their value
can be read; use `jax.reshard`"* rather than pointing at an internal helper and
asking for a bug report. This is easy to hit: it's what happens the first time
you print or assert on a gradient.

### 6. `jax.reshard` with a wrong-rank spec raises a bare `AssertionError`

Passing a spec whose rank doesn't match the array's:

```python
jax.reshard(x, jax.P(None, None, None, 'data'))   # x has rank 3
# AssertionError: (3, P(None, None, None, 'data'))
```

The tuple in the message is `(ndim, spec)`, which is the right information with
none of the words. This is a very easy mistake to make when a stacked parameter
`[L, ...]` and a single layer's slice share a spec table, which is exactly the
situation in `fsdp_pipeline.py`. A `TypeError` naming both ranks would have
saved a debugging cycle.

### 7. `PartitionSpec` is a tuple subclass, so `tree.map` silently destructures it

```python
jax.tree.map(lambda x, s: jax.reshard(x, s), weights, specs)
```

does not do what it looks like: `PartitionSpec` subclasses `tuple`, so
`tree.map` treats each spec as a container and pairs array leaves with
individual *axis names*. There's no error at the tree level — you get a
confusing failure further in, from whatever receives a string where it wanted a
spec. Registering `PartitionSpec` as a leaf, or documenting the hazard next to
`jax.P`, would help; specs and pytrees-of-arrays are natural to zip together.

### 8. XLA:CPU doesn't fuse all-reduce + dynamic-slice into reduce-scatter

`jax.reshard(unreduced_grad, P('data', ...))` should be a reduce-scatter, and on
TPU/GPU the all-reduce + dynamic-slice pair it lowers to gets rewritten into
one. On CPU it doesn't, so the optimized HLO shows `all-reduce` followed by
`dynamic-slice` and `reduce-scatter(` never appears.

Not a correctness problem, and CPU performance isn't the point. It matters here
only because these examples *print their collective counts as output* — so the
CPU run under-reports the pattern the reader is meant to see, and `nanolm.py`
has to spend three lines explaining that. Worth knowing if anyone else builds
teaching material around inspecting CPU HLO.

---

## 2026-07-25 — from `nanolm.py`, `sample.py`, `moe.py`

### 1. Explicit sharding catches genuine ambiguities in ordinary model code

Not a bug — the opposite — but worth recording because it's the strongest
evidence for the design and it showed up unprompted.

Two places in a twenty-line transformer body raise `ShardingTypeError` and
require an annotation:

- the embedding lookup, `params['embed'][tokens]`, which needs
  `.at[tokens].get(out_sharding=...)`;
- the attention output projection, where the contracted head axis is sharded
  on both operands, which needs `out_sharding=` on the `einsum`.

Both errors are correct, both are decisions the author should be making, and
both appear in the first model anyone would write. `nanolm.py` leaves them in
rather than working around them. The error message that names
`.at[...].get(out_sharding=)` explicitly is doing a lot of work here; more
messages like it would be good.

### 2. Sharded refs can't be indexed by integers, and `jax.ds` is undocumented

Writing a KV cache entry the obvious way fails:

```python
k_cache[i, :, pos] = k
# TypeError: sharded ref (array reference) can only be indexed by slices,
# not integers
```

Switching to a slice with a traced bound fails differently:

```python
k_cache[i:i+1, :, pos:pos+seq] = k[None]
# jax.errors.TracerBoolConversionError, raised from
# jax._src.core.canonicalize_slice
```

The form that works is:

```python
k_cache[i:i+1, :, jax.ds(start, seq)] = k[None]
```

`jax.ds` is public (exported from `jax/__init__.py`) and is the right tool. But
`docs/new_docs/101/state.md` doesn't mention it, and in-place update of a
sharded buffer at a dynamic offset is *the* motivating use case for refs — it's
what a KV cache is. Two suggestions: put a `jax.ds` example in the refs doc,
and make the integer-indexing `TypeError` point at it, since a reader who hits
that error has no way to guess `jax.ds` exists.

The `TracerBoolConversionError` from `canonicalize_slice` is also worth a
look — it surfaces an internal function name for what is really "slice bounds
must be static; use `jax.ds` for a dynamic offset".

### 3. `jax.get_mesh()` doesn't exist

`docs/new_docs/201/sharding.md` says:

> At the top level only, the concrete mesh can be queried using `jax.get_mesh()
> -> jax.sharding.Mesh`.

`jax.get_mesh` raises `AttributeError`. The accessor is
`jax.sharding.get_mesh()`. Either the doc or the export should change; given
that `jax.set_mesh`, `jax.make_mesh`, `jax.P`, and `jax.typeof` are all
top-level, exporting `jax.get_mesh` seems more consistent than fixing the doc.

### 4. Async dispatch can deadlock CPU collectives, and the failure is an abort

A training loop that never waits on its output runs ahead of the device and
queues many steps' worth of collectives. Every simulated CPU device is backed
by the same thread pool, so past some depth the rendezvous can't be satisfied,
and the process aborts:

```
Termination timeout for `all gather RendezvousKey{run_id=..., global_devices=[1, 3],
num_local_participants=2, collective_op_kind=cross_module, op_id=28}` of 40 seconds
exceeded. Exiting to ensure a consistent program state. Expected 2 threads to join
the rendezvous, but only 1 of them arrived on time.
```

Reproduced with `jax_num_cpu_devices` at both 8 and 4 on a 4-core machine. It
only shows up on longer runs: a loop that prints its loss every few steps syncs
often enough to stay under the limit, so the same script works at 150 steps and
dies at 1200, which makes it look nondeterministic. It is not affected by
`jax_cpu_collectives_implementation` (that setting only covers cross-process
collectives) and there's no knob to grow the pool —
`--xla_force_host_platform_device_count` says outright that "all of these host
devices are backed by the same threadpool".

The workaround is one `float(loss)` per step, which is what the examples here
do and what `README.md` warns about. But the path to hitting this is just
"write an ordinary training loop, don't inspect the loss every step, run it on
CPU" — and the outcome is a hard abort with a message that describes the
symptom rather than the cause. Worth considering either bounding in-flight
executions per device on the CPU backend, or naming the likely cause in the
message.
