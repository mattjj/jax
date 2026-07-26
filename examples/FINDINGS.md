# Findings

A running log of JAX rough edges turned up while writing the examples in this
directory. Examples exercise the public API the way a new user does, in
complete programs rather than snippets, so they tend to hit things before a
user reports them. This is the list of those things: nothing here is fixed by
the examples themselves, and each entry is a question for someone who owns the
relevant area.

Append new entries at the top. Strike through or delete entries once they're
resolved upstream.

Re-validated against jax-ml/jax main at `704b65fe` (2026-07-25) on 2026-07-26.

---

## 2026-07-26 (later still) — from `diffsim.py`

### 9. Scalar accumulators in a scan carry break grad-of-vmap over a sharded axis

Differentiating a `vmap`-over-a-sharded-axis of a `scan` whose carry contains
a scalar accumulator fails while stamping the batch sharding onto an aval
whose mesh is empty:

```python
jax.set_mesh(jax.make_mesh((4,), ('data',)))
gs = jax.device_put(jnp.arange(8.), jax.P('data'))

def roll(g):
  def body(c, _):
    x, acc = c
    x = jnp.tanh(x + g)
    return (x, acc + x * x), None
  (x, acc), _ = jax.lax.scan(body, (0.0 * g, 0.0), None, length=5)
  return x + acc

jax.grad(lambda gs: jnp.sum(jax.vmap(roll)(gs)))(gs)
# ValueError: Resource axis: data of P('data',) is not found in mesh: ().
# (raised from batching.py batch_jaxpr_axes -> NamedSharding.update)
```

In this 12-line form, replacing the Python `0.0` with `jnp.zeros(())` fixes
it, implicating the weak-typed literal's empty-mesh aval. But the minimal fix
is not sufficient in general: `diffsim.py`'s fuel accumulator triggered the
same error *with* an array-typed init once the body was a full physics step
(unminimized; the extra ingredient was not isolated). The robust workaround
used there is structural — keep per-step scalars out of the carry, emit them
as scan outputs, and sum afterwards. Forward-only vmap-of-scan is fine; it
needs `grad` around it.

Found on `0.11.0.dev20260711` (`4f484c50`), the newest tree this environment
can execute; not yet checked against current main, which needs a jaxlib this
environment cannot reach.

---

## 2026-07-26 (later) — from `hmc.py` and `flow_matching.py`

### 8. `jax.random.split` has no `out_sharding`, unlike its siblings

`jax.random.normal`, `uniform`, etc. all take `out_sharding=`, so a sharded
program can create sharded randomness directly. `split` does not:

```python
split sig:  (key, num=2)
normal sig: (key, shape=(), dtype=None, *, out_sharding=None)
```

The place this bites is per-chain/per-example keys under a `vmap` over a
sharded axis, which is exactly where you want them:

```python
x = jax.random.normal(k, (chains, DIM), out_sharding=P('data', None))
keys = jax.random.split(key, chains)          # replicated
jax.vmap(step)(x, keys)
# ValueError: Mapped away dimension of inputs passed to vmap should be
# sharded the same. Got inconsistent axis specs: data vs None
```

(The error itself is good — clear, and it points at the actual mismatch.)
The workaround in `hmc.py` is `jax.reshard(jax.random.split(key, n), P('data'))`,
which works but describes a *slice* of a replicated computation; an
`out_sharding` on `split` could instead let each device compute only its own
keys. Same asymmetry, same fix, in `flow_matching.py`'s label-dropout keys.

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

Still present at `704b65fe` (`jax/_src/sharding.py:51`). Filing that bug here.
The natural reading of `np.asarray` on an unreduced array
is "perform the pending reduction, then give me the value", which is what
`jax.reshard(g, ...)` would have done anyway. Failing is defensible, but the
message should then say *"unreduced arrays must be resharded before their value
can be read; use `jax.reshard`"* rather than pointing at an internal helper and
asking for a bug report. This is easy to hit: it's what happens the first time
you print or assert on a gradient.

### 6. `jax.reshard` with a wrong-rank spec raises a bare `AssertionError`

Minimal repro:

```python
import jax, jax.numpy as jnp
jax.config.update('jax_num_cpu_devices', 4)
jax.set_mesh(jax.make_mesh((4,), ('data',)))

x = jnp.zeros((8, 4, 2))                 # rank 3
spec = jax.P(None, None, None, 'data')   # rank 4
jax.reshard(x, spec)
# AssertionError: (3, P(None, None, None, 'data'))
```

The assertion is `jax/_src/core.py:2266` (still present at `704b65fe`):

```python
assert all(s is None for s in pspec.partitions[ndim:]), (ndim, pspec)
```

The tuple in the message is `(ndim, spec)` — the right information with none of
the words, and an `AssertionError` rather than a `TypeError`. This is easy to
hit whenever a stacked parameter `[L, ...]` and a single layer's slice share a
spec table, which is exactly the situation in `fsdp_pipeline.py`; it cost a
debugging cycle there, and it also sent an earlier version of this log chasing
the wrong culprit (see below). A `TypeError` naming both ranks would fix it.

### 7. XLA:CPU doesn't fuse all-reduce + dynamic-slice into reduce-scatter

`jax.reshard(unreduced_grad, P('data', ...))` should be a reduce-scatter, and on
TPU/GPU the all-reduce + dynamic-slice pair it lowers to gets rewritten into
one. On CPU it doesn't, so the optimized HLO shows `all-reduce` followed by
`dynamic-slice` and `reduce-scatter(` never appears.

Not a correctness problem, and CPU performance isn't the point. It matters here
only because these examples *print their collective counts as output* — so the
CPU run under-reports the pattern the reader is meant to see, and `nanolm.py`
has to spend three lines explaining that. Worth knowing if anyone else builds
teaching material around inspecting CPU HLO.

### Retracted: "`PartitionSpec` is destructured by `tree.map`"

An earlier revision of this log claimed that `jax.tree.map(f, arrays, specs)`
silently flattens each `PartitionSpec` into its axis names, because
`PartitionSpec` subclasses `tuple`. **That is wrong on current main** —
`PartitionSpec.__mro__` is `(PartitionSpec, object)`, and `tree.map` correctly
treats a spec as a leaf:

```python
jax.tree.map(lambda x, s: (x.shape, s), (a, b), (P('data', None), P(None, 'data')))
# (((8, 4), P('data', None)), ((4, 8), P(None, 'data')))
```

What actually happened: a `tree.map` over specs failed with finding 6's bare
`AssertionError`, and the rank mismatch was misread as destructuring. Rewriting
it as an explicit `zip` failed identically — which should have settled it, and
didn't. No bug here; kept as a note because the misdiagnosis is exactly the kind
an unhelpful error message invites.

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

### 3. ~~`jax.get_mesh()` doesn't exist~~ — RESOLVED upstream

`docs/new_docs/201/sharding.md` documented `jax.get_mesh()`, but the name
raised `AttributeError`; only `jax.sharding.get_mesh()` worked. As of
`704b65fe`, `jax/__init__.py` exports `get_mesh`, so the doc is now correct and
the top-level spelling matches `jax.set_mesh` / `jax.make_mesh` / `jax.P`.
`jax.sharding.get_mesh` still exists too, so both spellings work. Nothing to do.

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
