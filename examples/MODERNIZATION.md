# Modernizing `examples/`

A proposal to replace the current `examples/` directory. Status: **draft for
discussion**, not implemented.

## 1. Where we are

`examples/` is nine Python files, 1109 lines, essentially frozen since 2018–19.

| File | Lines | What it shows | Verdict |
|---|---|---|---|
| `mnist_classifier.py` | 97 | MLP on MNIST via `stax` + `optimizers` | **Delete.** Depends on `jax.example_libraries`, whose own docstring says "You likely do not mean to import this module!" |
| `mnist_classifier_fromscratch.py` | 95 | Same, hand-rolled | **Delete.** Subsumed by the new flagship. |
| `spmd_mnist_classifier_fromscratch.py` | 145 | Data parallelism on MNIST | **Delete.** Already partially modernized (uses `AxisType`/`reshard`), but 8-way DP on an MLP is not a parallelism story worth telling in 2026. |
| `mnist_vae.py` | 137 | VAE, reparameterization trick | **Replace** with flow matching. |
| `advi.py` | 139 | Black-box VI on a 2-D posterior | **Keep the idea, rewrite.** Best non-DL demo we have; `vmap`+`grad` shine. |
| `differentially_private_sgd.py` | 256 | Per-example grads via `vmap` | **Keep, modernize.** Per-example gradients remain the canonical "why `vmap`" argument. Drop `stax`/`optimizers`. |
| `onnx2xla.py` | 134 | ONNX graph → XLA | **Delete.** Dead path; hardcodes an ONNX opset from 2018. |
| `datasets.py` | 93 | MNIST downloader | **Replace** with a byte-level text loader. |
| `ffi/`, `jax_cpp/`, `k8s/` | — | FFI, AOT-to-C++, multi-host on k8s | **Keep.** These are current and load-bearing. |

The common thread: every one of these predates `jit`-of-`grad` being boring,
predates `shard_map`, predates sharding-in-types entirely. None of them would
change if `jax.set_mesh` had never been written.

## 2. What other projects do

Surveying the neighbors:

- **[pytorch/examples](https://github.com/pytorch/examples)** is a museum in the
  same way ours is: MNIST, DCGAN, VAE, word-level RNN LM, SNLI, actor-critic on
  CartPole. The distributed story is a thin `ddp/` and an `rpc/` directory. Not
  a model to copy — but instructive, because it shows the failure mode of an
  examples directory that accretes rather than gets curated.
- **[ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)** is
  much healthier: `llms/` (LLaMA/Mistral/Mixtral generation), `lora/`,
  `transformer_lm/`, `flux/` and `stable_diffusion/`, `whisper/`, `encodec/`,
  `musicgen/`, `clip/`, `llava/`, `segment_anything/`, `normalizing_flow/`,
  `gcn/`. MNIST is present but explicitly framed as the beginner on-ramp, not the
  centerpiece. Their examples are organized by *what you'd want to build*.
- **[karpathy/nanoGPT](https://github.com/karpathy/nanogpt)**,
  **[nanochat](https://github.com/karpathy/nanochat)**, and
  **[KellerJordan/modded-nanogpt](https://github.com/kellerjordan/modded-nanogpt)**
  set the current bar for what a reference implementation feels like: single
  file or near it, no framework dependency, readable top to bottom in one
  sitting, and a real result at the end.

Two lessons. First, the interesting axis moved from *architecture* to *scale and
systems* — mlx's differentiator is unified memory, ours is the compiler and the
sharding system, and examples should be built around the differentiator.
Second, terse-and-complete beats broad-and-shallow.

## 3. Design principles for the replacement set

1. **Every example must be a bad example if you delete the sharding.** If the
   file reads the same with one device and eight, it belongs in `docs/`, not
   `examples/`.
2. **Runs on a laptop CPU, unchanged on a pod.** Start each file with
   `jax.config.update('jax_num_cpu_devices', 8)`. The sharding is then real,
   inspectable, and reviewable in CI with no accelerator. Swapping in a real
   mesh is a one-line diff, and *that one-line diff is the pedagogy*.
3. **Sharding is visible in the output, not just the comments.** Print
   `jax.typeof(x)` for the key arrays (`float32[128@data,512@model]`), and grep
   the compiled HLO for `all-gather(` / `all-reduce(` so the reader sees which
   collectives their annotations bought them.
4. **No dependencies beyond `jax` + `numpy`.** No Flax, no Optax, no
   `jax.example_libraries`. Adam is six lines. These examples teach JAX, not a
   framework; anyone who wants a framework will find one.
5. **≤ 250 lines, one file, readable top to bottom.**
6. **A `--check` mode** that asserts parity against an unsharded reference
   computation, in the style of `docs/new_docs/201/shard-map.md`. It doubles as
   the test, so `examples/` stops rotting silently.
7. **A header docstring stating which JAX features the file demonstrates and
   how long it takes to run.** Makes the directory browsable as an index.

## 4. Proposed examples

### Tier 1 — the core set

**`nanolm.py` — decoder-only transformer, FSDP + tensor parallel.**
The flagship, replacing all three MNIST classifiers. Byte-level LM on
TinyShakespeare. Stacked layer params scanned over with `jax.lax.scan`, params
sharded on a 2-D `('data', 'model')` mesh so that FSDP and TP fall out of the
sharding annotations alone — no collectives written by hand, the compiler
inserts them and the example prints the count. Changing `(4, 2)` to `(8, 1)`
turns it into pure FSDP; `(1, 8)` into pure TP. Demonstrates: **explicit
sharding, FSDP, TP, `jit`, `grad`, `scan`, `remat`, donation.**

**`sample.py` — autoregressive decoding with a KV cache.**
The natural sequel, and something no JAX example currently covers. Demonstrates
`jax.lax.while_loop` decoding, buffer donation, and the new **mutable-array
refs** (`jax.new_ref`, per `docs/new_docs/101/state.md`) for the cache — which is
the honest way to write a cache and a good advertisement for a new API. Add
`vmap` over sampling temperatures to get a batched sweep for free.

**`moe.py` — mixture of experts with expert parallelism.**
The best real-world motivation for `shard_map` that exists: experts sharded over
a mesh axis, top-k routing, and an `all_to_all` to dispatch tokens to their
expert and gather results back. Demonstrates **`shard_map`, `all_to_all`,
`psum_scatter`, and mixing manual and automatic modes in one program** — which
is precisely the composition story `docs/new_docs/201/shard-map.md` develops but
that no runnable example currently exercises.

**`lora.py` — LoRA fine-tuning, and batched multi-adapter serving.**
Demonstrates pytree surgery (differentiating w.r.t. a subset of a pytree),
`remat`, and — the fun part — **`vmap` over a stack of adapters** to serve N
fine-tunes in a single batched forward pass. That is a genuinely JAX-shaped
trick that is awkward in every other framework, and it makes the case for `vmap`
far better than per-example gradients do.

### Tier 2 — JAX is not only a deep learning framework

This is where we differentiate from mlx and PyTorch, and where the old `advi.py`
was pointing before it aged out.

**`flow_matching.py`** — rectified flow / flow matching on a small image set,
replacing `mnist_vae.py`. Same "generative model in 100 lines" slot, current
paradigm. `vmap` over timesteps, `grad`, classifier-free guidance.

**`hmc.py`** — Hamiltonian Monte Carlo, replacing/absorbing `advi.py`.
`grad` for the leapfrog integrator, `vmap` over chains, `scan` over steps,
sharding the chains across devices. Roughly 60 lines and it makes JAX look like
what it actually is.

**`diffsim.py`** — gradients through an ODE/physics integrator: optimize a
control input by differentiating a `scan`-based simulation. This is the example
where **`jax.remat` is not optional**, so it's the only honest way to teach
rematerialization.

**`flash_attention.py`** — a Pallas attention kernel with a correctness check
against `jax.nn.dot_product_attention` and a benchmark. Covers the
kernel-authoring layer, which `examples/` says nothing about today.

### Feature coverage

| | jit | grad | vmap | scan | explicit sharding | shard_map | remat | refs |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `nanolm.py` | ✅ | ✅ | | ✅ | ✅ FSDP+TP | | ✅ | |
| `sample.py` | ✅ | | ✅ | ✅ | ✅ | | | ✅ |
| `moe.py` | ✅ | ✅ | | | ✅ | ✅ | | |
| `lora.py` | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ | |
| `flow_matching.py` | ✅ | ✅ | ✅ | | ✅ | | | |
| `hmc.py` | ✅ | ✅ | ✅ | ✅ | ✅ | | | |
| `diffsim.py` | ✅ | ✅ | ✅ | ✅ | | | ✅ | |
| `flash_attention.py` | ✅ | ✅ | | | | | | |
| `dpsgd.py` (kept) | ✅ | ✅ | ✅ | | ✅ | | | |


## 5. Status

Landed (see [`README.md`](README.md)):

| | | |
|---|---|---|
| `nanolm.py` | new | transformer with FSDP + TP from sharding annotations alone |
| `sample.py` | new | KV-cache decoding on `jax.new_ref` mutable arrays |
| `moe.py` | new | expert parallelism with `shard_map` + `all_to_all` |
| `data.py` | new | byte-level text, replacing the MNIST downloader |
| `util.py` | new | simulated-device defaults |
| `examples_test.py` | new | runs every `--check` mode |
| `mnist_classifier.py` | deleted | |
| `mnist_classifier_fromscratch.py` | deleted | |
| `spmd_mnist_classifier_fromscratch.py` | deleted | |
| `onnx2xla.py` | deleted | |

Still to do, in the order proposed above: `lora.py`, then `flow_matching.py`
(retiring `mnist_vae.py`), `hmc.py` (absorbing `advi.py`), `diffsim.py`,
`flash_attention.py`, and modernizing `differentially_private_sgd.py` off
`jax.example_libraries`. `datasets.py` stays until the last MNIST consumer is
gone. Nothing in CI runs `examples_test.py` yet; wiring it into
`.github/workflows/ci-build.yaml` next to the existing `examples/ffi` step is
the obvious follow-up.

## 6. What writing these turned up

Four things worth flagging, since they are all cases where the examples ran
into the sharp edges before a user would.

**Explicit sharding catches genuine ambiguities in ordinary model code.** The
embedding lookup `params['embed'][tokens]` raises `ShardingTypeError` and
requires `.at[tokens].get(out_sharding=...)`; so does the attention output
projection, where the contracted head axis is sharded on both operands. Both
errors are *correct* and both land inside a twenty-line model body. This is the
best advertisement the feature has, and it is why `nanolm.py` leaves them in
rather than working around them.

**Sharded refs can't be indexed by integers.** Writing a KV cache entry as
`k_cache[i, :, pos] = k` fails with "sharded ref (array reference) can only be
indexed by slices, not integers", and a Python slice with a traced bound fails
with a `TracerBoolConversionError` from `canonicalize_slice`. The working form
is `k_cache[i:i+1, :, jax.ds(start, seq)] = k[None]`. `jax.ds` is public and is
the right tool, but nothing in `docs/new_docs/101/state.md` mentions it, and the
in-place update of a sharded buffer at a dynamic offset is *the* motivating use
case for refs. Worth an example in the refs docs.

**`jax.get_mesh()` doesn't exist.** `docs/new_docs/201/sharding.md` says "the
concrete mesh can be queried using `jax.get_mesh() -> jax.sharding.Mesh`". The
accessor is `jax.sharding.get_mesh()`; `jax.get_mesh` raises `AttributeError`.

**Async dispatch can deadlock CPU collectives.** A training loop that never
waits on its output runs ahead and queues hundreds of steps' worth of
collectives; because every simulated CPU device is backed by the same thread
pool, the rendezvous then times out and aborts the process:

```
Termination timeout for `all gather RendezvousKey{...}` of 40 seconds
exceeded. Exiting to ensure a consistent program state. Expected 4 threads
to join the rendezvous, but only 3 of them arrived on time.
```

This reproduced at both 8 and 4 simulated devices on a 4-core machine, and only
on longer runs — short ones sync often enough to stay under the limit, which
makes it look nondeterministic. The fix in these examples is a `float(loss)`
per step, but a hard abort is a rough failure mode for something a user hits by
writing an ordinary un-synchronized training loop on CPU. It may be worth
either bounding in-flight executions per device or making the message name the
cause.
