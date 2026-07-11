# Triage of open jax-ml/jax issues assigned to mattjj

**Date:** 2026-07-11
**Method:** Of the 151 open issues assigned to mattjj, ~30 bug-shaped issues with
self-contained repros were selected and their reproductions were actually run
against JAX at upstream head (`4f484c5`, jax `0.10.2.dev`). Environment: CPU-only
Linux container; multi-device repros used
`XLA_FLAGS=--xla_force_host_platform_device_count=N`. Repros were minimally
adapted where APIs had moved (e.g. `jax.experimental.shard_map` → `jax.shard_map`);
adaptations are noted per issue. GPU-only issues could not be verified.

---

## 1. Verified FIXED at head — close now (5)

| Issue | Title | Evidence at head |
|---|---|---|
| #9374 | `custom_jvp(nondiff_argnums=...)` crashes inside `lax.cond` | Repro runs cleanly, returns correct gradient 1.0. No escaped-tracer error. |
| #16000 | `custom_jvp` promotes tangent-of-integer to integer (not float0) | JVP rule now receives float0 tangent (`[('float0','V')]`); `jax.jvp` returns expected `(2.0, 1.0)`. |
| #15905 | Internal assertion with `vmap(shard_map(...))` + `axis_name` + `spmd_axis_name` | `get_frame` AssertionError gone; repro (updated to `jax.shard_map`, 2x2 mesh) returns correct output. |
| #13931 | pmap inside jit causes `assert not ragged` | Internal assertion gone. Single-pmap-in-jit cases work. Two pmaps over *different* device subsets in one jit now raise a clear, intentional `ValueError` ("Received incompatible devices for jitted computation…") — unsupported by design now that pmap is implemented via shard_map. |
| #26361 | Writing function input to global mutable array fails inside `jax.grad` | (Repro updated `mutable_array` → `jax.new_ref`.) Internal `TypeError: Argument 'Zero(Ref{float32[]})' … not a valid JAX type` replaced by a deliberate, actionable error: "performing a set/swap operation with a differentiated value on a non-differentiated array reference … Move the array reference to be an argument of the differentiated function?" Closing requires accepting that erroring is intended semantics. |

## 2. Obsolete — the feature the bug lives in was deleted (2)

| Issue | Title | Why obsolete |
|---|---|---|
| #27877 | `jnp.ravel()` fails with dynamic shapes under jit | `jax_dynamic_shapes` config no longer exists (`jax.config.update("jax_dynamic_shapes", True)` → "Unrecognized config option"); the failure mode is unreachable. |
| #23782 | `jax_getattr` inside `jax.scan` with multi-leaf pytree | `jax.experimental.attrs` (`jax_getattr`/`jax_setattr`) removed entirely, superseded by array refs (`jax.new_ref`). |

## 3. Trivial code fixes — a few lines each (3)

- **#25659 — tree_util error handling throws error.** `str(k.key)` at
  `jax/_src/tree_util.py:1332,1335,1338` crashes with
  `AttributeError: 'GetAttrKey' object has no attribute 'key'` when building the
  prefix-mismatch error message (GetAttrKey has `.name`, DictKey `.key`,
  SequenceKey `.idx`). Reproduced with a custom pytree using `GetAttrKey`.
  Fix: `str(k)` on those three lines (+ note: line 796 uses `repr(k.key)` but is
  shielded by a bare `except`). A few-line PR with a small test.
- **#16218 — stax Dropout cannot switch train → test mode.** In
  `jax/example_libraries/stax.py:265-273`, `apply_fun` ignores a `mode` kwarg
  passed at call time and only reads the constructor-time closure. Reproduced:
  `apply(params, x, rng=rng, mode='test')` still applies dropout. One-line fix:
  `mode = kwargs.get('mode', mode)`.
- **#22874 — `jax.tree_util.equality_errors` not exported.** Still private at
  head (`jax._src.tree_util.equality_errors` exists; not in `jax.tree_util` or
  `jax.tree`). Mechanical export + docs entry — or close as explicit wontfix.

## 4. Closable with an explanation — working as intended (3)

- **#14666 — jit-within-jit turns static values into tracers.** Behavior
  unchanged and inherent to jit semantics: a jitted function's outputs are
  traced values, so static-ness cannot survive being returned from an inner
  jit. The issue's own `g1` workaround (call the un-jitted inner function) is
  the answer.
- **#11990 — `floor_divide` int division by zero mismatches NumPy.** Unchanged:
  `jnp.floor_divide(2, 0)` → `-2` vs NumPy `0` — but both values are arbitrary
  (int div-by-zero is implementation-defined in C/XLA; NumPy's 0 is its own
  convention). JAX now has the opt-in check for exactly this:
  `jnp.floor_divide` calls the divide-by-zero error hook
  (`jax/_src/numpy/ufuncs.py:2530`; config `jnp.error_checking_behavior`).
  Defensible close: implementation-defined + pointer to error-checking mode.
- **#32452 — Unable to create side effects in `custom_vjp`.** Repro still
  raises `UnexpectedTracerError` — but it deliberately leaks a tracer out of
  the fwd function via a closed-over dict, which the purity model forbids by
  design. The actionable remainder (auxiliary outputs from `custom_vjp`) is an
  enhancement; relabel or close with the workaround (return stats as explicit
  outputs).

## 5. Still broken — verified to reproduce at head; needs real work (13)

- **#5202 — custom_jvp transpose rule can fail.** Identical `AssertionError`
  in `_select_transpose_rule` (now `jax/_src/lax/lax.py:8107`,
  `assert not ad.is_undefined_primal(which)`). The fix sketched in the issue
  (partial-eval nonlinear eqns before transposition) was never implemented.
- **#5309 / #5552 — reverse-mode AD through gmres/cg in custom JVP /
  custom_root tangent_solve.** Same root cause: `_linear_solve_transpose_rule`
  (`jax/_src/lax/control_flow/solves.py:408`) explicitly raises
  `NotImplementedError` when the linear operator's closed-over params contain
  undefined primals. #5309 verified with the issue's verbatim repro (jacfwd
  works, jacrev fails); #5552 verified with a faithful reconstruction (its
  original repro was a dead Colab link). **Suggest deduping #5552 into #5309.**
- **#8783 — grad + vmap + odeint AssertionError.** Original assertion gone but
  the combination still crashes, now `IndexError: tuple index out of range` in
  `_split_shape_rule` (`lax.py:7443`) via `flatten_util`/`lax.split` batching in
  `_odeint_wrapper`. Isolated: fwd-only, vmap-only, grad-only all work;
  grad+vmap with complex y0 fails. Worth updating the issue with the new
  traceback. (Adaptations: modern config API; float `t` — odeint now rejects
  complex time.)
- **#10621 — scan unroll + nested scan + vmap "dead loop".** No longer hangs
  outright, but the pathological compile cliff persists: trigger config
  `n=2, m=95` compiles in **79 s** vs **1.6 s** for `m=96` (one element larger)
  — a ~50x discontinuity, essentially all XLA compile time. (Shift from the
  report: `n=1` no longer avoids it.) An XLA compile-time investigation.
- **#11373 — `Jaxpr` and `JaxprEqn` render same variables differently.**
  Still reproduces: a standalone `eqn` repr assigns fresh names from a new
  pretty-print context, so variables appear swapped vs the parent jaxpr.
  Cosmetic, but a real fix means sharing naming context in the pretty-printer.
- **#12339 — float0 should support +, −, scalar ×.** All three ops still fail;
  the current error message says float0 supports no operations *by design*.
  Needs either implementation or an explicit design wontfix.
- **#12795 — gradient of eigh does not respect `symmetrize_input`.** Verified
  numerically: gradients from lower-triangular vs symmetrized input disagree,
  and finite differences confirm the lower-triangular gradient is wrong (it
  corresponds to the symmetrized input). Cause unchanged: `_eigh_jvp_rule`
  (`jax/_src/lax/linalg.py:~1343`) symmetrizes and uses the full tangent;
  `symmetrize_input` isn't a parameter of `eigh_p` so the JVP can't see it.
  Related old PR: #12696.
- **#14776 — `ensure_compile_time_eval` puts trace stack in a bad state.**
  Repro still fails (`ConcretizationTypeError`). Notably, the pattern that the
  *current* `jax.checkpoint` docstring promises
  (`jax/_src/ad_checkpoint.py:337-350`: `static_argnums` +
  `ensure_compile_time_eval`) fails with `TracerBoolConversionError` when the
  static arg is a grad tracer — exactly the case the docstring advertises.
  Either fix the behavior or fix the docstring.
- **#15759 — "No constant handler" error after jit/pjit merge.** Verbatim
  repro: `jit(grad(...))` over vmap+switch+odeint still fails with
  `TypeError: No constant handler for type: DynamicJaxprTracer`
  (`jax/experimental/ode.py:182`); un-jitted grad works (309.2759).
- **#15728 — shard_map cryptic error on wrong number of arguments.** Calling a
  2-arg shard_mapped function with 1 arg still yields the pytree-structure
  ValueError instead of a `TypeError: missing argument` like jit gives.
  Moderate: contained error-message fix in spec matching.
- **#16303 — Cannot bind to primitive `Zero(AbstractToken())`.** Core probe:
  `ad_util.instantiate(Zero(core.abstract_token))` still `KeyError`s — no zeros
  handler for tokens; a stand-in token-threading primitive (mpi4jax-style)
  fails under `jit(linear_transpose(...))` with the modern wording. Small core
  fix (register a token zeros handler, materialize via `create_token`) — or
  close as stale since modern mpi4jax dropped explicit tokens for effects.
  (Caveat: verified with a stand-in, not real mpi4jax — no MPI in container.)
- **#16374 — crash from `jvp(jit(vmap(custom_jvp(in-place-update-on-size-zero-array))))`.**
  Fails identically: `TypeError: broadcast_in_dim broadcast_dimensions must
  have length equal to operand ndim…`. Root cause: batching-tracer
  non-propagation through size-zero in-place updates → primal/tangent shape
  divergence. Subtle.
- **#17874 / #23476 — closed-over values under `vmap(..., spmd_axis_name)` +
  shard_map.** Same family. #17874: explicit-arg case correct `(8,)`;
  closed-over case still wrong `(64,)` (the reported n² blowup). #23476: plain
  vmap works; `spmd_axis_name='x'` now fails with a *different* internal error
  (`ValueError: Axes mentioned in 'manual_axis_type' … should be of type
  'Manual'. Got … AxisType.Auto`) — failure mode morphed with the
  varying-manual-axes machinery. **Suggest linking the two.**
- **#34139 — `jnp.sinc` gradient has large errors.** Confirmed quantitatively
  vs a longdouble/Maclaurin reference: float64 max rel err **2.2e7** at
  x≈1.3e-12 (JAX 9.59e-05 vs true −4.34e-12); float32 max rel err **1.1e4** at
  x≈1.3e-6. Grad at exactly 0 is correct; `_sinc_maclaurin` custom-JVP
  (`jax/_src/numpy/ufuncs.py:3849-3866`) only special-cases x == 0. Well-scoped
  real fix: Taylor branch near zero with dtype-dependent threshold + tests.

## 6. Could not be verified here — GPU required (2)

- **#36703 — XLA topk-decomposer crash in pmap(vmap(...)).** CPU-adapted repro
  passes, but the crash is in XLA:GPU's `topk-decomposer` HLO pass, which the
  CPU pipeline never runs. Needs a 2-GPU CUDA machine at current jaxlib; likely
  an XLA-side fix.
- **#36958 — unexpected result with scan + vmap.** Reporter says CPU is
  unaffected, and indeed on CPU at head all three variants agree to 3.6e-07.
  Reported as a 0.9.2 → 0.10.0 GPU regression (plausible codegen/fusion
  miscompile) — deserves triage on CUDA hardware.

## 7. Not tested (~121 issues)

The remaining assigned issues are enhancements, documentation requests,
questions, and performance investigations — nothing that closes by
verification. Notable subsets:

- **Questions old enough to close with an answer:** e.g. #12158
  (custom-jvp backprop question, 2022), #4996 (complex differentiation, 2020),
  #803, #3297, #3514, #3567 (2019-2020 era).
- **Docs requests** (e.g. #390, #8726, #9348, #15587, #16363, #20243, #31892)
  — each closable by a small docs PR.
- **Performance issues** (e.g. #10197, #24411, #13543, #2160) and **large
  feature requests** (e.g. #17863 ragged arrays, #11319 chunked vmap,
  #10828 cache clearing) — real work, not triage-closable.

## Summary counts (of the 30 tested)

| Category | Count |
|---|---|
| Fixed at head — close now | 5 |
| Obsolete (feature deleted) — close now | 2 |
| Trivial fix available | 3 |
| Close with explanation (working as intended) | 3 |
| Still broken, needs real work | 15 |
| GPU-required, unverifiable here | 2 |

---

## Appendix: draft closing comments for the 7 close-now issues

Written in first person, neutral voice; close reason "completed".

**#9374:** Verified at head: the repro from the OP now runs cleanly and returns
the correct gradient (1.0) — no more escaped-tracer error. Closing as fixed.
Please reopen if you hit a variant of this that still fails.

**#16000:** Verified at head: the jvp rule now receives a float0 tangent for
the integer primal (dtype `[('float0', 'V')]`), and `jax.jvp` on the repro
returns the expected `(2.0, 1.0)`. Closing as fixed.

**#15905:** Verified at head (with the repro updated to `jax.shard_map`): the
internal `get_frame` assertion is gone and
`jax.vmap(..., axis_name='x', spmd_axis_name='x')` returns the correct result.
Closing as fixed.

**#13931:** Verified at head: the internal `assert not ragged` failure is gone.
The single-pmap-inside-jit cases from the repro now work. Mixing two pmaps over
*different* device subsets inside one jit now raises an intentional, clear
error ("Received incompatible devices for jitted computation...") rather than
an internal assertion — that pattern isn't supported (pmap is implemented via
shard_map these days; use shard_map directly for anything fancy). Closing since
the internal-assertion bug is fixed.

**#26361:** Verified at head (repro updated from the old `mutable_array` to
`jax.new_ref`): the internal `TypeError: Argument 'Zero(Ref{float32[]})' … is
not a valid JAX type` is gone. Writing a differentiated value into a ref that
isn't an argument of the differentiated function now raises a deliberate,
actionable error suggesting the workaround — which is the intended semantics.
Closing.

**#27877:** The experimental `jax_dynamic_shapes` config this bug depends on
has been removed from JAX, so the reported failure mode is no longer reachable
(`jax.config.update("jax_dynamic_shapes", True)` now raises "Unrecognized
config option"). Closing as obsolete.

**#23782:** The experimental attrs API (`jax.experimental.attrs`,
`jax_getattr`/`jax_setattr`) this bug is about has been removed from JAX,
superseded by array refs (`jax.new_ref` etc.). Closing as obsolete — if you hit
a similar problem with refs inside `scan`, please open a new issue.
