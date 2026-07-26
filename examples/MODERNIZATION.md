# Modernizing `examples/`

The plan and running status for replacing the 2018-era `examples/` directory.
[`README.md`](README.md) indexes what exists today; rough edges found while
building it are logged in [`FINDINGS.md`](FINDINGS.md). This document is the
*why*: the audit that started it, the design principles, the decisions made
along the way (including two reversals), and what remains.

## 1. Where we started

As of mid-2026, `examples/` was nine Python files, 1109 lines, essentially
frozen since 2018–19:

| File | What it showed | Outcome |
|---|---|---|
| `mnist_classifier.py` | MLP on MNIST via `stax` + `optimizers` | deleted — built on `jax.example_libraries`, whose own docstring says "You likely do not mean to import this module!" |
| `mnist_classifier_fromscratch.py` | same, hand-rolled | deleted — subsumed by `nanolm.py` |
| `spmd_mnist_classifier_fromscratch.py` | data parallelism on MNIST | deleted — 8-way DP on an MLP is not a parallelism story worth telling now |
| `mnist_vae.py` | VAE, reparameterization | deleted — replaced by `flow_matching.py` |
| `advi.py` | black-box VI on a 2-D posterior | deleted — absorbed into `hmc.py` |
| `onnx2xla.py` | ONNX graph → XLA | deleted — dead path, 2018 opset |
| `differentially_private_sgd.py` | per-example grads via `vmap` | rewritten — same DP recipe, no `example_libraries`, sharded example axis |
| `datasets.py` | MNIST downloader | kept — still the MNIST loader for dp-sgd |
| `ffi/`, `jax_cpp/`, `k8s/` | FFI, AOT-to-C++, multi-host k8s | kept — current and load-bearing |

The common thread: every deleted file predated `jit`-of-`grad` being boring,
predated `shard_map`, predated sharding-in-types entirely. None would have
changed if `jax.set_mesh` had never been written.

## 2. What other projects do

- **[pytorch/examples](https://github.com/pytorch/examples)** is a museum in
  the same way ours was: MNIST, DCGAN, VAE, word-level RNN LM, CartPole. Not a
  model to copy — but instructive as the failure mode of an examples directory
  that accretes rather than gets curated.
- **[ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)** is
  much healthier: `llms/`, `lora/`, `stable_diffusion/`, `whisper/`, organized
  by *what you'd want to build*, with MNIST explicitly demoted to on-ramp.
- **[karpathy/nanoGPT](https://github.com/karpathy/nanogpt)** and
  **[KellerJordan/modded-nanogpt](https://github.com/kellerjordan/modded-nanogpt)**
  set the bar for what a reference implementation feels like: near-single-file,
  no framework dependency, readable top to bottom, a real result at the end.

Two lessons. The interesting axis moved from *architecture* to *scale and
systems* — mlx's differentiator is unified memory, ours is the compiler and
the sharding system, and examples should be built around the differentiator.
And terse-and-complete beats broad-and-shallow.

## 3. Design principles

1. **Prefer examples that show something JAX does that the alternatives
   don't**, and give sharding pride of place — it is the thing JAX is most
   distinctively good at. But this is a preference, not a gate: `jit`, the
   autodiff APIs, `vmap`, hijax, and Pallas earn their own examples, and a
   plain good demonstration of how to use JAX is worth having even where
   another library would do as well. What the directory must avoid is the
   2018 failure mode — a set of examples that would read identically if the
   last eight years had not happened.
2. **Runs on a laptop CPU, unchanged on a pod.** Simulated CPU devices by
   default (as many as the machine has cores, capped at 8 — see `util.py`),
   so the sharding is real, inspectable, and testable with no accelerator.
3. **The interesting thing is visible in the output, not just the comments.**
   `jax.typeof` for the key arrays, HLO collective counts, the scan carry, an
   adapter-by-task loss grid, ASCII samples — every file prints the evidence
   for its own claim.
4. **No dependencies beyond `jax` + `numpy`.** Adam is six lines. These
   examples teach JAX, not a framework.
5. **≤ 250 lines of code, one file, readable top to bottom.** Comments and
   the header docstring don't count against the budget — in a teaching
   example they are the product.
6. **A `--check` mode** asserting a falsifiable claim — parity with an
   unsharded reference, exact moments, "each adapter wins its own task",
   "QAT beats PTQ" — wired into `examples_test.py` so the directory cannot
   rot silently again.
7. **A header docstring saying which JAX features the file demonstrates and
   how long it takes to run.**

## 4. The examples

Landed, in suggested reading order:

| | demonstrates | claim `--check` verifies |
|---|---|---|
| `nanolm.py` | tensor parallelism + ZeRO-2 via `reduced`/`unreduced` types; `jit`, `grad`, `scan`, `remat` | sharded loss/grads match a replicated run, on every mesh factorization |
| `sample.py` | refs (mutable arrays) as a sharded KV cache; `jax.ds` dynamic slices | cached greedy decoding ≡ the uncached model |
| `lora.py` | `vmap` over a *sharded* adapter axis: train and serve N fine-tunes in one call | each adapter beats the others on its own task |
| `fsdp_pipeline.py` | FSDP with explicit software pipelining: `custom_vjp` as the backward hook, `unroll=2` as the double buffer | naive and pipelined schedules compute the same gradients |
| `moe.py` | `shard_map` + `all_to_all` expert parallelism, GShard-style fixed-capacity routing | matches a sequential reference when nothing is dropped |
| `quantized.py` | a new array type via hijax (`VJPHiPrimitive`), whose *tangent type* is f32; refs for the train loop | PTQ degrades as bits shrink; QAT recovers (int2: 2.1×) |
| `flow_matching.py` | flow matching + classifier-free guidance; sampler `vmap`ed over guidance strengths | unconditional covers all modes; guidance hits its target |
| `hmc.py` | `grad` in the integrator, `vmap` over sharded chains, `scan` twice | sampled moments match closed-form moments |
| `diffsim.py` | `grad` through a `scan` simulator; `remat` with its memory saving *measured* by XLA (18×) | goals reached; remat leaves gradients unchanged |
| `differentially_private_sgd.py` | `vmap` of `grad` for per-example gradients, sharded example axis | clipping bound holds; the private model learns |

Still to build:

- **`flash_attention.py`** — a minimal Pallas attention kernel, checked
  against `jax.nn.dot_product_attention`. Deferred, not dropped: Pallas
  interpreter mode requires Python 3.12+ (PEP 695 generics in
  `pallas/mosaic/interpret`), and the development environment this directory
  was built in runs 3.11 with no accelerator — so the file could not be run,
  and unverified examples are against the house rules. CI's Python 3.12
  runners can.

Notes:

- `datasets.py` survives as the MNIST loader for
  `differentially_private_sgd.py`, which falls back to synthetic data
  offline.
- **CI is already wired**: the main build job runs
  `pytest ... tests examples`, and `examples_test.py` matches pytest's
  `*_test.py` pattern, so the `--check` suite runs on every PR with no
  workflow change.

## 5. Decisions and reversals

Recorded because they changed the plan, and would otherwise look like drift.

**FSDP moved out of `nanolm.py` (2026-07-26).** The original flagship sharded
parameters over 'data' and let the compiler all-gather them. Memory-correct,
but the gather for layer *i* sits in the same `scan` iteration as the matmul
consuming it, so nothing can hide the communication — and a survey of the
references found none of them shard parameters in the forward pass at this
scale (nanoGPT is DDP; modded-nanogpt is ZeRO-2 with hand-ordered optimizer
collectives; FSDP2 prefetches from runtime hooks, which a `scan` doesn't
have). `nanolm.py` now does TP + ZeRO-2 — the gradient reduce-scatter falling
out of the `reduced` type is its headline — and FSDP lives in
`fsdp_pipeline.py`, where the pipelining it requires is the lesson.

**Design principle 1 was relaxed (2026-07-26).** Originally "every example
must be a bad example if you delete the sharding." Too strong: it would have
excluded `diffsim.py`, `flash_attention.py`, and `quantized.py`, none of
which is improved by bolting a mesh on.

**QAT is fine-tuning, not training from scratch (2026-07-26).** The first
version of `quantized.py` trained with fake-quant from initialization and
*lost to PTQ* at every bit width once the float baseline was trained long
enough. The vetted recipe (torchao) fine-tunes a pretrained model, which wins
as published. Both results are kept in the file — the negative one is a
comment, and is arguably the more instructive of the two.

**Refs over donation where state branches.** `nanolm.py` and `lora.py` use
`donate_argnums` in the classic linear-training-loop shape. `quantized.py`
branches fine-tunes off a float model that must survive, which is exactly
where donation bites; its loop holds params and Adam state in refs, updated
in place, with `jax.new_ref(r[...])` as the explicit copy. The directory
deliberately shows both idioms.

## 6. What building these turned up

Every file surfaced something — genuine sharp edges, missing docs, one
already-fixed-upstream bug, and one hard abort reachable from an ordinary
training loop. They live in [`FINDINGS.md`](FINDINGS.md), which should keep
accumulating as the remaining examples get written.
