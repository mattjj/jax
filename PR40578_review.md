# Review of jax-ml/jax#40578: `lax.max` / `lax.min` JVP rule

PR: https://github.com/jax-ml/jax/pull/40578 (merged as `1462726`, fixes
https://github.com/jax-ml/jax/issues/40564). Author: jakevdp. Reviewer: mattjj.

This branch contains three commits on top of the fork's `main`:

1. a cherry-pick of the merged PR commit, for context;
2. a follow-up that replaces the PR's `_balanced_cmp` with a variant that keeps
   the bug fix but removes four regressions (details in section 6), plus tests
   that pin every behavior discussed below;
3. this document.

**How this was validated.** No jaxlib >= 0.11 is reachable from the sandbox
this review was written in, so the repository's own test suite could not be run
against `main`. Instead, released jax 0.10.2 (which ships the identical pre-PR
rule) was installed in a venv, and the old rule, the PR's rule, and the
follow-up rule were monkeypatched into `max_p` / `min_p`, then diffed across a
broad battery of probes. Every mechanism behind a finding was confirmed by
reading the corresponding code on `main`. The follow-up's test methods were run
in that venv under all three rules (section 6).

## 1. TL;DR

- The PR fixes the reported bug by construction and, for real dtypes, reproduces
  the old gradient semantics bitwise: ties split 0.5/0.5, NaN gives 0/0.
- It introduced four regressions, all in corners: forward mode with Python
  scalar primals crashes, complex inputs raise a `TypeError`, tangents and
  gradients of weakly typed inputs become strongly typed, and eager reverse
  mode under an explicit mesh raises a `ShardingTypeError`.
- The existing tie behavior (average the tangents, split the cotangent evenly)
  is good and worth keeping. The NaN behavior (0/0) is an accident of the old
  implementation but harmless, and preserving it was the right call.
- The implementation is at least as efficient as before. The follow-up variant
  is cheaper still and has none of the regressions.

## 2. What the PR changed

Old rule (`_balanced_eq`), for `max`, the mask on `x`'s tangent:

```python
div(select(eq(x, ans), 1, 0), select(eq(y, ans), 2, 1))   # ans = max(x, y)
```

New rule (`_balanced_cmp`):

```python
select(gt(x, y), 1, select(eq(x, y), 0.5, 0))
```

The old mask depended on `ans` (the saved forward output) through a bitwise
equality test. The new mask depends only on `x` and `y`. `min` uses the same
function with the arguments swapped. Both rules multiply the tangent by the
mask, so the transpose is a multiply by the same mask.

## 3. Does it recover existing behavior?

**Identical to the old rule, verified bitwise in the venv:**

- the full 7x7 grid of `{-inf, -1, -0.0, 0.0, 1, inf, nan}` for both operands,
  both ops, both arguments, jvp and grad, eager and jit;
- f16, bf16, f32, f64, float8_e4m3fn, float8_e5m2 at ties and with NaN;
- lax-level broadcasting (rank-0 operand, size-1 dims) including the
  `_unbroadcast` in the transpose;
- `vmap` of grad and jvp, with batched and unbatched operands;
- second and third derivatives, jvp-of-grad and grad-of-jvp;
- `shard_map` with `check_vma=True`, varying and invariant operands;
- `jit` under an explicit mesh, with sharded, replicated, and Python-scalar
  operands;
- residuals under `jax.linearize`: two float masks, as before;
- the `testReluGradientConstants` regression test in `lax_numpy_test.py`.

**Regressions.** Each was reproduced with the PR's exact `_balanced_cmp`, and
each is fixed by the follow-up in this branch.

```python
# 1. Forward mode with Python-scalar primals passed straight to lax.max/lax.min.
#    jax.vjp canonicalizes its inputs; jax.jvp, jacfwd and hessian do not, and
#    JVPTracer.primal is then a raw float, so `x.dtype` raises.
jax.jvp(lax.max, (1.0, 2.0), (1.0, 1.0))
# AttributeError: 'float' object has no attribute 'dtype'
jax.jvp(lambda x: lax.max(x, jnp.float32(2.)), (1.0,), (1.0,))       # same
# jnp.maximum is unaffected: dtype promotion re-binds the tracer first.

# 2. Complex inputs. lax.max documents lexicographic complex support and it
#    differentiated before the PR (via eq); gt rejects complex dtypes.
jax.grad(lambda z: jnp.sum(jnp.abs(jnp.maximum(z, w))))(z_c64)
# TypeError: gt does not accept dtype complex64 at position 0.

# 3. Weak types. The mask is always strongly typed now, so gradients of weakly
#    typed inputs come back strong. Every other lax JVP checked (abs, clamp,
#    sin, mul, where, relu) preserves weak type.
jax.grad(lambda x: jnp.clip(x, 0., 1.))(0.5)        # was weak f32, now strong
jax.grad(lambda x: jnp.maximum(x, 0.))(1.) * jnp.ones(2, jnp.bfloat16)
# was bf16, now f32

# 4. Eager reverse mode under an explicit mesh.
with jax.set_mesh(explicit_mesh):
  jax.grad(lambda a, b: jnp.sum(lax.max(a, b)))(x_sharded, y_replicated)
# ShardingTypeError: select cases must have the same shardings, got
# [NamedSharding(..., spec=P('i',)), NamedSharding(..., spec=P(None,))]
```

Number 4 is the subtle one. Under `jax.grad`, the JVP rule runs inside
`linearize_from_jvp`'s `JaxprTrace`. The integer fills `0` and `1` in
`full_like` go through a real `convert_element_type` bind on a `TypedNdArray`,
which evaluates eagerly to a concrete array, so `lax.full` takes its
`make_array_from_callback` branch and keeps the concrete sharding. The float
fill `0.5` instead takes the scalar branch of `_convert_element_type`, which
calls `stage`, and under a `JaxprTrace` that yields a known `JaxprTracer`.
`lax.full` then falls through to a plain `broadcast` and the sharding is
dropped, so `half` is replicated while `ones` and `zeros` are sharded. Under
`jit` everything is abstract and it works. The root cause is arguably in
`lax.full` (a known tracer fill value with a concrete sharding silently loses
the sharding), but this rule is the first to hit it.

## 4. Does it fix the bug?

Yes, by construction. The old mask asked "is my (possibly recomputed) `x`
bitwise equal to the saved `ans`?", so any 1-ulp difference between the forward
and backward copies of `x` zeroed both masks and the gradient vanished silently.
The new mask asks "`x > y`, `x == y`, or `x < y`?", a trichotomy: for non-NaN
inputs the two masks sum to exactly 1 no matter how `x` or `y` were rounded.
Disagreement between passes now only matters when `x` and `y` are within an
ulp of each other, where the derivative is ill-defined anyway.

A numpy simulation makes the difference concrete. With random `x`, `y` and the
backward-pass copy of `x` perturbed by one ulp:

| rule | entries with wrong `x` mask | entries with both masks zero |
|---|---|---|
| old | 49.8% (every entry where `x` won) | 49.8% |
| new | 0% | 0% |

The GPU failure itself could not be reproduced in this sandbox (CPU only). The
author confirmed the regression test fails on A100 without the fix, and the
mechanism above explains why the fix is sufficient.

One side effect worth knowing: on a backend run with fast, non-NaN-propagating
min/max, `(nan, 1.0)` used to yield gradients `(0, 1)` under the old rule and
now yields `(0, 0)` everywhere, which is more consistent across backends.

## 5. Is the existing behavior good?

**The tie rule is good and worth keeping.** Averaging the tangents at a tie is
symmetric under swapping the arguments, sums to one, makes `max(x, x)`
differentiate to 1, makes `abs` written as `max(x, -x)` give 0 at 0, and
matches PyTorch's `maximum` backward. TensorFlow uses an `x >= y` mask, which
is also a valid subgradient but asymmetric. The true directional derivative at
a tie is `max(tx, ty)`, which no linear rule can reproduce, so the midpoint is
the sensible linear proxy.

**The NaN rule (0/0) is arbitrary but harmless.** It was never designed; it fell
out of `eq(nan, ans)` being false. PyTorch sends the full gradient to both
inputs, TensorFlow sends it to `y`. In practice a NaN forward value usually
poisons the cotangent through downstream derivatives anyway. Preserving and
testing it was the right call for a bug-fix PR. Note that the old rule's "both
masks zero" outcome was also exactly the silent failure mode of the bug; the new
formulation only produces it for NaN.

**Same hazard elsewhere.** `_reduce_chooser_jvp_rule` (so `jnp.max`, `jnp.min`,
`reduce_max`, `reduce_min`) and the corresponding rules in
`jax/experimental/jet.py` still compare `operand` against `ans` bitwise. There a
mismatch yields 0/0 counts, so it fails loudly with NaN rather than silently,
but it is the same GPU rematerialization exposure and a candidate follow-up.

## 6. Efficiency, and the follow-up on this branch

Under `jit` all variants fuse into the same two fusions on CPU and the
difference is noise. Eager mode is where op count shows:

| variant | grad jaxpr eqns | StableHLO lines | divides | eager ms/call (200k elems, CPU) |
|---|---|---|---|---|
| old | 23 | 36 | 2 | 3.8 to 5.3 |
| PR | 19 | 29 | 0 | 3.2 to 3.6 |
| follow-up | 17 | 25 | 0 | 2.5 to 2.7 |

Residual memory is unchanged: two float masks under `jax.linearize` / `jax.vjp`
in every variant.

The follow-up `_balanced_cmp`:

```python
def _balanced_cmp(x, y):
  # 1.0 if x > y, 0.5 if x == y, 0.0 if x < y or if either is NaN. This
  # compares x against y directly rather than against max(x, y), so the result
  # doesn't depend on the forward and backward passes computing bitwise
  # identical values (https://github.com/jax-ml/jax/issues/40564).
  dtype = _dtype(x)  # unlike x.dtype, also works for Python scalar primals
  # The mask is weakly typed iff both inputs are, like the primal output.
  weak_type = dtypes.is_weakly_typed(x) and dtypes.is_weakly_typed(y)
  if dtypes.issubdtype(dtype, np.complexfloating):
    # Lexicographic (real, imag) order, matching the max/min lowering rules.
    xr, yr = real(x), real(y)
    gt_mask = bitwise_or(gt(xr, yr),
                         bitwise_and(eq(xr, yr), gt(imag(x), imag(y))))
  else:
    gt_mask = gt(x, y)
  gt_f = _convert_element_type(gt_mask, dtype, weak_type=weak_type)
  eq_f = _convert_element_type(eq(x, y), dtype, weak_type=weak_type)
  # Use a rank-0 constant rather than full_like(eq_f, 0.5): in eager mode under
  # an explicit mesh, a full-shape float fill value can come back replicated
  # while eq_f is sharded.
  half = full_like(eq_f, 0.5, shape=())
  return add(gt_f, mul(eq_f, half))
```

Why each piece:

- `_dtype(x)` instead of `x.dtype` handles Python scalar primals (regression 1).
- The complex branch reproduces the lexicographic order used by
  `mlir.max_hlo` / `mlir.min_hlo` (regression 2). It only adds ops for complex
  dtypes.
- `weak_type` is computed the way the old mask inherited it from `ans`: weak iff
  both inputs are weak (regression 3). `_const` alone would not do here, since it
  only preserves weak types for rank-0 examples.
- The 0.5 factor is a rank-0 `full_like(..., shape=())`, the same idiom
  `lax.py` already uses for its scalar constants, so it carries no sharding and
  multiplies cleanly against a sharded `eq_f` in eager explicit mode
  (regression 4), while still inheriting `eq_f`'s weak type.
- No `select`, no `div`: `gt_f + 0.5 * eq_f` is exactly `1 / 0.5 / 0`.

**Tests added** (`tests/lax_autodiff_test.py`), each parameterized over
`lax.max` and `lax.min` where it makes sense:

- `test_max_min_jvp_python_scalar_primals`: `jax.jvp` and `jax.jacfwd` with
  Python scalar primals, including a scalar broadcast against a vector;
- `test_max_min_grad_weak_type`: weak type of grads and tangents for weak
  scalar, weak rank-1, and strong inputs;
- `test_max_min_complex_grad`: lexicographic ordering, tie splitting, jvp and
  grad for complex64, checked against the primal's choice;
- `test_max_min_grad_eager_explicit_sharding`: eager `jax.grad` under a
  one-device explicit mesh with a sharded and a replicated operand;
- `test_max_min_nan_gradient` now also runs the reverse-mode check under `jit`.

**Validation of the follow-up** (venv, jax 0.10.2, rules monkeypatched):

| rule installed | `-k test_max_min` result |
|---|---|
| old (pre-PR) | 9 passed, 1 skipped (the x64-only regression test) |
| PR | 7 failed, 2 passed, 1 skipped: the seven failures are exactly the new regression tests |
| follow-up | 9 passed, 1 skipped |

The follow-up also matched the old rule output-for-output across the whole
probe battery of section 3, with x64 both off and on.

## 7. Nits on the PR's own tests

- `test_max_min_jvp` passes on CPU with or without the fix (relative error
  1.6e-7 either way), so it only guards the bug on GPU, and it exercises
  `jit(grad)` rather than jvp.
- `@skipIf(not jax.config.x64_enabled, ...)` is evaluated at import time. CI
  sets `JAX_ENABLE_X64` through the environment so it works, but
  `Config.x64_enabled` is a legacy property with a removal TODO, and the file's
  convention is a runtime `self.skipTest` on `config.enable_x64.value`.
- The file uses camelCase for 51 of its 55 test names; the new ones are
  snake_case. The follow-up's tests follow the PR's tests since they sit next
  to them.

## Appendix: how the numbers were produced

- Value grids, dtype sweeps, weak-type, scalar-primal, complex, vmap,
  higher-order and shard_map probes: a script that registers one of the rules
  with `ad.defjvp` / `ad.defjvp2`, prints every result, and is diffed across
  rules and across `JAX_ENABLE_X64` settings.
- Explicit sharding: a 4-device CPU mesh via
  `--xla_force_host_platform_device_count=4`, `jax.make_mesh(...,
  axis_types=(AxisType.Explicit,))`, `jax.set_mesh`, eager and jit.
- Efficiency: `jax.make_jaxpr` equation counts, StableHLO and compiled HLO
  line and op counts on CPU, and wall-clock per call for eager and jit grad on
  200k float32 elements. Timings are from a shared 4-core container and are
  only meaningful as rough ratios.
- Mechanism simulation: numpy, 100k random float32 pairs, backward copy of `x`
  perturbed with `np.nextafter`.
