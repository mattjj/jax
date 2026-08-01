# Review: jax-ml/jax#39653 — "Enable full NumPy indexing compliance for jax.Ref"

Review of https://github.com/jax-ml/jax/pull/39653 (commit `31a8770`, branch
`test_957374261`), which adds `None`/newaxis support to `jax.Ref` indexing by
delegating index parsing to `jax._src.numpy.indexing` and appending a
`ReshapeTransform` after the physical `NDIndexer` when newaxes are present.

Review goals, per the request: (1) does it support the newaxis cases in the
tests, (2) is the added complexity minimal, and (3) does it avoid perturbing
existing indexing — i.e. no change to the code generated for indexing
patterns that work today.

## TL;DR

The core design is right and satisfies (2) and (3) well: strip the `None`s
out, build the physical state `NDIndexer` exactly as before, and append a
`ReshapeTransform` only when `None`s were present. Across a battery of 58
existing indexing programs (get/set/addupdate over slices, ints, negative
ints, integer arrays, contiguous and non-contiguous advanced indexing,
`ds`/dynamic slices, traced scalar/array indices, and `.at` views), **base
and PR produce byte-identical jaxprs**. The five new tests pass, and simple
newaxis patterns match NumPy for get/set/addupdate, including on views, 0-d
refs, and combined with `ds`.

However, it is not landable as-is. There is one real correctness bug (newaxis
between advanced indices → wrong layout), one internal-caller breakage
(`sp.ref_set(ref, None, val)` in the Mosaic DMA discharge path), a
wrong-old-value bug in `swap` through a reshape, and the two edits to
`jax/_src/numpy/indexing.py` are broader than needed, degrading error paths
for all `jnp` array indexing.

## Methodology

- Fetched the PR and upstream `main` (`381953b`) into separate worktrees.
- Ran on CPU with jaxlib 0.10.2 plus two small test-only shims for missing
  0.11 APIs (`weakref_lru_cache.strong_lru_cache`, `sdy.ReductionOp`). The
  same environment was used on both sides, so comparative results are valid.
- Diffed `jax.make_jaxpr` output for 58 indexing programs (base vs PR).
- Compared ~20 newaxis patterns × {get, set, addupdate} against NumPy ground
  truth, plus swap/vmap/eager/0-d/`ds`/bool-mask probes.
- Ran the full `tests/state_test.py` under both trees: identical failure sets
  (18 failures on both, all environment artifacts of the old-jaxlib sdy
  bindings; they also blocked grad/eager probes equally on both trees). The
  PR's 139-test run shows no regressions relative to base, and its 5 new
  tests pass.
- Pallas could not be exercised directly (`jax/_src/pallas/core.py` uses
  py3.12 `type` statements; test box is py3.11). Pallas-relevant conclusions
  below are established at the shared `state` layer.

## Blocking findings

### 1. `None` between advanced indices produces the wrong layout

`get_transforms_from_indices` computes advanced-index contiguity on the
*None-stripped physical* indices. NumPy (and `jnp` — verified) treats a
`newaxis` as a separator that triggers the broadcast-dims-to-front rule:

```python
x = np.zeros((3, 4, 5)); arr, arr2 = np.array([0, 2]), np.array([1, 3])
x[:, arr, None, arr2].shape   # NumPy and jnp: (2, 3, 1)
# PR, via ref:                # (3, 2, 1)  — wrong shape AND transposed data
```

- `ref[...]` get: silently wrong shape/layout.
- set: `ValueError: Invalid shape for swap ... Expected shape: (3, 2, 1). Value shape: (2, 3, 1)`.
- addupdate: `TypeError: add got incompatible shapes for broadcasting`.

A trailing reshape fundamentally cannot express this case — it requires a
transpose. Given the minimal-complexity goal, the pragmatic fix is to detect
it (advanced indices adjacent in physical space but separated by a `None` in
the original index expression, with at least one slice-produced dim before
the advanced block) and raise `NotImplementedError` rather than implement the
front rule. All other placements come out correct, including non-contiguous
advanced indices (slice-separated) combined with `None`s, scalar-int-only
advanced indices, and 2-D index arrays.

### 2. Explicit `idx=None` callers break (Pallas Mosaic DMA discharge)

The PR correctly stops mapping bare `None` → "read/write everything" in
`get_ref_and_transforms` (that mapping is what made `ref[None]` silently
wrong). But `jax/_src/pallas/mosaic/primitives.py:541-545` still calls:

```python
sp.ref_set(dst_ref, None, do_discharge_dst(...))   # and dst_sem, src_sem
```

Under the PR, `idx=None` means newaxis, so this raises
`ValueError: Invalid shape for swap. Ref shape: (...). Expected shape: (1, ...)`
(failure mode confirmed at the state level; Mosaic itself not runnable here).
These call sites must be updated to `...` (or `()`).

More broadly, any external caller of `jax.ref.get/set/swap(ref, None)`
changes meaning silently. The signature-default change (`idx=None` →
`idx=()`) is the right move, but it deserves a CHANGELOG entry, and the
`ref_get`/`ref_swap` docstrings should document that `None` is now a
meaningful (newaxis) index.

### 3. `swap` through a newaxis returns a wrong-shaped old value

In `transform_swap_array` (jax/_src/state/discharge.py), the forward/read
loop appends the reshaped value to `intermediates` for `ReshapeTransform` and
`BitcastTransform` **without updating `new_val`**:

```python
case ReshapeTransform():
    intermediates.append(new_val.reshape(transform.shape))   # new_val stays stale
```

This staleness predates the PR but was unreachable: the write phase raised
`NotImplementedError` for these transforms, so no swap through a reshape ever
completed. The PR unlocks the write phase, making it reachable:

- `jax.ref.swap(ref, (None,), val)` returns the old value with shape
  `(3, 4, 5)` while its trace-time aval says `(1, 3, 4, 5)` — silently wrong
  if returned; if used downstream, a confusing
  `TypeError: add: arrays must have the same number of dimensions` at
  discharge time.
- The write itself lands correctly, and `ref_set` discards the swap result —
  which is why the PR's tests don't catch it.

Fix is two lines: assign `new_val` in both cases.

## The `jax/_src/numpy/indexing.py` edits: load-bearing but too broad

Reverting only these two hunks and rerunning shows they are **not needed for
the newaxis tests** but **are needed to keep ref-as-index working**:
`ref[int_ref]` regresses to `IndexError` without them (the Mosaic
"ref indexer" feature; ref tracers with `AbstractRef` avals do not satisfy
`isinstance(idx, (Array, np.ndarray))`).

But the implementation —
`hasattr(idx, "dtype") and hasattr(idx, "shape") and not isinstance(idx, type)`
— now catches *any* array-like object in **all** `jnp` array indexing, and
error paths degrade:

| index object | base | PR |
|---|---|---|
| object with int dtype+shape | clean `IndexError` (valid-indices message) | bare `AssertionError` |
| object with junk `dtype` | clean `IndexError` | `TypeError: data type 'not-a-dtype' not understood` |
| object with bool dtype+shape | clean `IndexError` | `TypeError: ... must have integer or boolean type, got indexer with type bool` |

The `not isinstance(idx, type)` guard (needed because e.g. `np.int64` the
*class* has a `dtype` descriptor) is itself a sign the duck-typing is
fragile. Recommendation: narrow to the actual target — check
`isinstance(core.typeof(idx), AbstractRef)` / `isinstance(idx, TransformedRef)`
via a local import of state types (the PR already uses local imports in the
other direction).

Related: `_is_boolean_index` wraps `core.typeof(i)` in `except TypeError`
only, but `TransformedRefAvalError` subclasses plain `Exception`. So indexing
by a `TransformedRef` now dies at parse time
(`TransformedRefs cannot be abstractified`) *before* reaching state's
ref-indexer branch in `NDIndexer.from_indices_shape`, which deliberately
checks `isinstance(i, TransformedRef)` before ever calling `typeof`. Plain
`Ref` indices still work under the PR (verified); whether
`TransformedRef`-as-index has live Mosaic/sparsecore coverage should be
checked internally before landing.

## Behavior the PR quietly fixes (worth stating in the description)

- On main today, bare `ref[None]` and `ref.at[1:3][None]` **silently return
  unexpanded results** (e.g. `ref.at[1:3][None]` gives shape `(2,)`), because
  `get_ref_and_transforms` maps `None` → `()`. Tuple forms (`ref[None, :]`,
  `ref[1, None]`) raise `ValueError: not enough values to unpack`. So the new
  support fixes silent wrongness, and no working code could have depended on
  the tuple forms.
- Writes and addupdates through explicit `.reshape` views now work
  (previously `NotImplementedError: Unsupported transform: ReshapeTransform`);
  the write-phase discharge additions are correct for reshape chains over an
  `NDIndexer` or the raw buffer.
- Error paths now match `jnp`: too-many-indices raises `IndexError` (was
  `ValueError` — error-type-sensitive tests may notice); `ref[0.5]` is
  rejected loudly (on main it silently passes trace time!); list indices get
  jnp's informative non-tuple-sequence `TypeError` (identical to
  `jnp.zeros(4)[[0, 2]]`).

## Non-blocking gaps and notes

- **vmap**: `vmap` over `ref[None]` raises
  `NotImplementedError: Batching with multiple indexers not supported`
  (batching rules see the `(NDIndexer, ReshapeTransform)` pair). Fine to
  punt, but it will be the first thing users hit; deserves a
  test-with-expected-error or batching support.
- **Advanced arrays + `None` cannot execute end-to-end anyway**: integer-array
  indexing of refs through the jit/discharge path is already broken on
  upstream main *without* this PR (`TracerArrayConversionError` from
  `_convert_to_gather_arrays` in `_index_array`; reproduces on a base from
  three weeks earlier). So the only end-to-end-testable newaxis support today
  is with slices and scalar ints — consistent with the tests the PR adds, but
  "full NumPy indexing compliance" oversells the current state.
- **Scalar bools** remain unsupported: `ref[True]` errors loudly but with a
  confusing message (`indices must not be longer than shape`), where NumPy
  gives mask-plus-newaxis semantics. Either support via
  `expand_scalar_bool_indices` or raise a clean `NotImplementedError`.
- **`_addupdate_discharge` preamble**: the `broadcast_to(val, broadcast_shape)`
  is dead code — the abstract eval enforces exact shape equality. And
  `target_shape = ... else x.shape` is wrong if the popped reshapes sit over
  a `BitcastTransform` (that combination was and remains broken; a clean
  error would beat the eventual `AttributeError` in `_is_trivial_indexer`).
- **Pallas kernels**: `ref[None]` inside kernels will stage
  `(NDIndexer, ReshapeTransform)` into mosaic/triton get/swap lowerings.
  Untested here (py3.12 needed). If kernel-side newaxis isn't intended yet,
  backends without ReshapeTransform-on-get support will fail at lowering —
  acceptable, but worth a test either way. Existing kernel patterns are
  unaffected (jaxpr-identical; the reshape only appears when `None` is used).
- **`BitcastTransform` write-phase support** is a drive-by (nothing newaxis
  needs it), is untested, and enables the same stale-`new_val` bug for
  bitcast swaps. Either test it or leave it out of this PR.
- Minor: `get_transforms_from_indices` re-derives int-indexer contiguity that
  `NDIndexer.get_indexer_shape` also computes (duplicated logic, but fine);
  no CHANGELOG entry.

## Suggested changes, concretely

1. `transform_swap_array` forward loop: update `new_val` for
   `ReshapeTransform`/`BitcastTransform` (2 lines).
2. Detect `None` separating advanced indices (when it changes NumPy's
   placement rule) and raise `NotImplementedError` — or implement the front
   rule with a transpose.
3. Update the three `sp.ref_set(..., None, ...)` call sites in
   `jax/_src/pallas/mosaic/primitives.py` to `...`.
4. Narrow the `from_index` duck-typing to `AbstractRef`/`TransformedRef`, and
   make the `TransformedRef` parse path not raise
   `TransformedRefAvalError` out of `_is_boolean_index`.
5. Drop the dead `broadcast_to` in `_addupdate_discharge`; consider a clean
   error for reshape-over-bitcast.
6. CHANGELOG entry + docstring updates for the `idx=None` semantic change;
   a vmap test with expected error (or batching support).

With those, this is a clean, well-contained change: the bridge function is
~50 self-contained lines reusing the jnp parser, and the no-`None` fast path
provably leaves every existing pattern's generated code untouched.
