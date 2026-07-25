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
sharding, FSDP, TP, `jit`, `grad`, `scan`, `remat`, donation.** A validated
prototype is in §5 below.

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

## 5. Worked prototype

The following runs today on CPU against this checkout (jax
`0.11.0.dev20260711`), on eight simulated devices. It is ~80 lines and is the
whole of `nanolm.py` minus data loading, sampling, and CLI.

```python
import jax, jax.numpy as jnp, numpy as np

jax.config.update('jax_num_cpu_devices', 8)

V, L, D, F, H, N, T, B = 256, 4, 128, 512, 32, 4, 64, 16
jax.set_mesh(jax.make_mesh((4, 2), ('data', 'model')))

# 'data' shards give FSDP, 'model' shards give tensor parallelism.
SPECS = dict(
    embed = jax.P('data', 'model'),              # [V, D]
    qkv   = jax.P(None, 'data', 'model', None),  # [L, D, N, 3H]
    proj  = jax.P(None, 'model', None, 'data'),  # [L, N, H, D]
    up    = jax.P(None, 'data', 'model'),        # [L, D, F]
    down  = jax.P(None, 'model', 'data'),        # [L, F, D]
    unemb = jax.P('data', 'model'),              # [D, V]
)
SHAPES = dict(embed=(V, D), qkv=(L, D, N, 3 * H), proj=(L, N, H, D),
              up=(L, D, F), down=(L, F, D), unemb=(D, V))

def init(key):
  keys = jax.random.split(key, len(SHAPES))
  return {k: jax.random.normal(kk, s, out_sharding=SPECS[k]) * (s[-2] ** -0.5)
          for kk, (k, s) in zip(keys, SHAPES.items())}

def rmsnorm(x):
  return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), -1, keepdims=True) + 1e-6)

def layer(x, p):
  q, k, v = jnp.split(jnp.einsum('btd,dnh->btnh', rmsnorm(x), p['qkv']), 3, -1)
  a = jax.nn.dot_product_attention(q, k, v, is_causal=True)
  x += jnp.einsum('btnh,nhd->btd', a, p['proj'], out_sharding=jax.P('data', None, None))
  h = jax.nn.gelu(jnp.einsum('btd,df->btf', rmsnorm(x), p['up']))
  x += jnp.einsum('btf,fd->btd', h, p['down'], out_sharding=jax.P('data', None, None))
  return x, None

def logits(params, tokens):
  x = params['embed'].at[tokens].get(out_sharding=jax.P('data', None, None))
  x, _ = jax.lax.scan(layer, x, {k: params[k] for k in ('qkv', 'proj', 'up', 'down')})
  return jnp.einsum('btd,dv->btv', rmsnorm(x), params['unemb'],
                    out_sharding=jax.P('data', None, None))

def loss(params, tokens):
  lg = logits(params, tokens[:, :-1])
  return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(lg), tokens[:, 1:, None], -1))

@jax.jit
def step(params, opt, tokens, lr=1e-3):
  g = jax.grad(loss)(params, tokens)
  m = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, opt['m'], g)
  v = jax.tree.map(lambda v, g: 0.99 * v + 0.01 * g * g, opt['v'], g)
  params = jax.tree.map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + 1e-8), params, m, v)
  return params, dict(m=m, v=v)
```

Output:

```
embed  float32[256@data,128@model]
qkv    float32[4,128@data,4@model,96]
proj   float32[4,4@model,32,128@data]
up     float32[4,128@data,512@model]
down   float32[4,512@model,128@data]
unemb  float32[128@data,256@model]
tokens int32[16@data,65]
loss    6.043725      <- before 20 steps
loss    4.114750      <- after
all-gather      8     <- FSDP: params gathered just in time
all-reduce      3     <- TP: partial sums combined
reduce-scatter  0
```

Three things worth noting, all of which are the *point* of the example:

- **The collectives are output, not commentary.** The reader annotates six
  arrays and gets FSDP and tensor parallelism; the HLO counts prove it.
- **Changing the mesh changes the strategy.** `(8, 1)` drops to 6 all-gathers
  and 1 all-reduce — pure FSDP. `(2, 4)` keeps 8 and 3. All verified. (`(1, 8)`
  needs `N ≥ 8` heads; the toy config only has 4. Real config sizes should be
  picked so every factorization of 8 works, which is itself a lesson worth
  putting in a comment.)
- **Explicit sharding caught a real ambiguity while writing this.** The
  embedding lookup `params['embed'][tokens]` raised `ShardingTypeError` because
  the gather's output sharding was ambiguous, forcing the explicit
  `.at[tokens].get(out_sharding=...)`. That error message *is* the sales pitch
  for sharding-in-types, and having it appear in a 20-line model body is worth
  more than a paragraph of docs.

## 6. Suggested sequencing

1. Land `nanolm.py` + a byte-level `data.py`; delete the three MNIST
   classifiers and `onnx2xla.py`.
2. Add `sample.py` and `moe.py`. Delete `mnist_vae.py` when `flow_matching.py`
   lands.
3. Modernize `differentially_private_sgd.py` (drop `example_libraries`) and fold
   `advi.py` into `hmc.py`.
4. Add an `examples/README.md` indexing the set by JAX feature, and wire the
   `--check` modes into CI so the directory can't rot again.

Steps 1 and 2 alone would replace 471 of the current 1109 lines with something
that argues for JAX as it exists now.
