# Draft closing comments (7 issues)

All verified against head at `4f484c5` (jax 0.10.2.dev), CPU, with
`XLA_FLAGS=--xla_force_host_platform_device_count=N` where multiple devices
were needed. Close reason: "completed" for all seven.

---

## #9374 — `jax.custom_jvp(nondiff_argnums=...)` crashes inside of `lax.cond`

> Verified at head: the repro from the OP now runs cleanly and returns the
> correct gradient (1.0) — no more escaped-tracer error. Closing as fixed.
> Please reopen if you hit a variant of this that still fails.

## #16000 — `jax.custom_jvp` promotes tangent-of-integer to integer (not float0)

> Verified at head: the jvp rule now receives a float0 tangent for the integer
> primal (dtype `[('float0', 'V')]`), and `jax.jvp` on the repro returns the
> expected `(2.0, 1.0)`. Closing as fixed.

## #15905 — JAX internal assertion fails with vmap(shard_map(...)) when setting axis_name and spmd_axis_name

> Verified at head (with the repro updated to `jax.shard_map`): the internal
> `get_frame` assertion is gone and
> `jax.vmap(..., axis_name='x', spmd_axis_name='x')` returns the correct
> result. Closing as fixed.

## #13931 — pmap inside jit causes assert not ragged assertion

> Verified at head: the internal `assert not ragged` failure is gone. The
> single-pmap-inside-jit cases from the repro now work. Mixing two pmaps over
> *different* device subsets inside one jit now raises an intentional, clear
> error ("Received incompatible devices for jitted computation...") rather
> than an internal assertion — that pattern isn't supported (pmap is
> implemented via shard_map these days; use shard_map directly for anything
> fancy). Closing since the internal-assertion bug is fixed.

## #26361 — Writing function input to global mutable array fails inside `jax.grad`

> Verified at head (repro updated from the old `mutable_array` to
> `jax.new_ref`): the internal `TypeError: Argument 'Zero(Ref{float32[]})'
> ... is not a valid JAX type` is gone. Writing a differentiated value into a
> ref that isn't an argument of the differentiated function now raises a
> deliberate, actionable error: "performing a set/swap operation with a
> differentiated value on a non-differentiated array reference ... Move the
> array reference to be an argument of the differentiated function?" — which
> is the intended semantics (and the suggested workaround works). Closing.

---

## #27877 — `jnp.ravel()` fails with dynamic shapes in a jit context

> The experimental `jax_dynamic_shapes` config this bug depends on has been
> removed from JAX, so the reported failure mode is no longer reachable
> (`jax.config.update("jax_dynamic_shapes", True)` now raises "Unrecognized
> config option"). Closing as obsolete.

## #23782 — Issue with `jax_getattr` inside `jax.scan` when the PyTree has multiple leaves

> The experimental attrs API (`jax.experimental.attrs`, `jax_getattr` /
> `jax_setattr`) this bug is about has been removed from JAX, superseded by
> array refs (`jax.new_ref` etc.). Closing as obsolete — if you hit a similar
> problem with refs inside `scan`, please open a new issue.
