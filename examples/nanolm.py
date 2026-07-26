# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training a byte-level transformer with tensor and data parallelism.

Demonstrates: explicit sharding, `reduced`/`unreduced` types, `jit`, `grad`,
`scan`, `remat`, donation.

The model is a small decoder-only transformer trained on Shakespeare. The
interesting part is the two tables below, `PARAM_SPECS` and `OPT_SPECS`: they
are the *only* place parallelism is expressed. There are no collectives in the
model code.

`PARAM_SPECS` shards the parameters over the 'model' mesh axis, which is
textbook tensor parallelism: each layer's two "row-parallel" matmuls contract
a sharded axis and the compiler all-reduces the partial sums. Over the 'data'
axis the parameters are *replicated*, so the forward pass needs no
communication for them at all.

`OPT_SPECS` additionally shards the gradients and the optimizer state over
'data'. This is the ZeRO-2 arrangement: full parameters everywhere, but each
device stores and updates only 1/N of the Adam state, so the memory that
actually dominates at this scale is sharded without touching the forward pass.
It is what modded-nanogpt does, and it is a good default.

THE TRICK: the ZeRO-2 gradient reduction falls out of the *type system*, not
out of any collective anyone writes. Parameters are stored with a
`reduced={'data'}` sharding -- bit-identical to replicated, no data moves --
and the only thing that changes is what autodiff does with them:

    param    float32[4,128,512@model]{R:data}   stored `reduced`, so...
    grad     float32[4,128,512@model]{U:data}   ...`grad` leaves it unreduced,
    sharded  float32[4,128@data,512@model]      ...so this reshard is a
                                                   reduce-scatter, not an
                                                   all-reduce we then discard
                                                   7/8ths of.

A one-word change to a type moves a collective and cuts its cost by a factor
of N. The program prints those three lines when you run it; `train_step` is
where the middle one becomes the bottom one, and
docs/new_docs/301/sharding-ad.md has the full story.

What this file deliberately does *not* do is shard parameters during the
forward pass (FSDP / ZeRO-3). Doing that well needs the all-gather for layer
i+1 to overlap the matmuls of layer i, which a `scan` will not give you for
free -- it needs explicit software pipelining. That is its own lesson, and it
lives in `fsdp_pipeline.py`.

By default this runs on *simulated* CPU devices -- as many as the machine has
cores, up to eight -- so the sharding is real and inspectable on a laptop. To
run it on real hardware, pass `--devices 0` and a mesh matching your machine.

    python examples/nanolm.py                         # ~1 min on CPU
    python examples/nanolm.py --devices 8 --mesh 8,1  # pure data parallelism
    python examples/nanolm.py --devices 8 --mesh 1,8  # pure tensor parallelism
    python examples/nanolm.py --check                 # verify against replicated

The 200 steps this runs by default are enough to watch the loss fall, not to
train a model. `--steps 1200 --save params.npz` takes about six minutes on a
laptop and gets to roughly 2.4 nats/byte, at which point `sample.py` produces
text with recognizable words and line breaks in it.
"""

import argparse
import functools
import time

import numpy as np

import jax
import jax.numpy as jnp

import data
import util


# -- configuration -----------------------------------------------------------

V = data.VOCAB_SIZE  # vocabulary (bytes)
L = 4                # layers
D = 128              # model dimension
F = 512              # feed-forward dimension
N = 8                # attention heads
H = 16               # head dimension (N * H == D)
T = 128              # sequence length
B = 32               # batch size

SHAPES = dict(
    embed=(V, D),
    qkv=(L, D, N, 3 * H),
    proj=(L, N, H, D),
    up=(L, D, F),
    down=(L, F, D),
    unemb=(D, V),
)

# How the parameters are laid out during the forward and backward passes.
# Only the 'model' axis appears: this is tensor parallelism, and over 'data'
# every parameter is replicated. `qkv` and `up` are "column parallel" (their
# output axis is sharded, so no communication is needed); `proj` and `down`
# are "row parallel" (they contract a sharded axis, so their partial sums have
# to be all-reduced). Every axis a mesh axis shards must divide evenly, which
# is why N == 8 rather than something smaller: it lets `--mesh 1,8` work.
PARAM_SPECS = dict(
    embed=jax.P(),                             # [V, D]
    qkv=jax.P(None, None, 'model', None),      # [L, D, N, 3H]  column parallel
    proj=jax.P(None, 'model', None, None),     # [L, N, H, D]   row parallel
    up=jax.P(None, None, 'model'),             # [L, D, F]      column parallel
    down=jax.P(None, 'model', None),           # [L, F, D]      row parallel
    unemb=jax.P(),                             # [D, V]
)

# How the gradients and the Adam state are laid out: the same, plus a shard
# over 'data'. This is the only place 'data' shards a parameter-shaped array,
# and it is what makes this ZeRO-2 rather than plain data parallelism.
OPT_SPECS = dict(
    embed=jax.P('data', None),                 # [V, D]
    qkv=jax.P(None, 'data', 'model', None),    # [L, D, N, 3H]
    proj=jax.P(None, 'model', None, 'data'),   # [L, N, H, D]
    up=jax.P(None, 'data', 'model'),           # [L, D, F]
    down=jax.P(None, 'model', 'data'),         # [L, F, D]
    unemb=jax.P('data', None),                 # [D, V]
)

# Activations keep the batch sharded over 'data' and everything else replicated.
ACTS = jax.P('data', None, None)

LAYER_KEYS = ('qkv', 'proj', 'up', 'down')


def reduced(spec):
  """Same layout, but autodiff hands back *unreduced* gradients.

  `reduced={'data'}` is physically identical to replicated over 'data' -- the
  cast below moves no data. It only changes the cotangent type, so that
  `jax.grad` stops before the cross-device sum and leaves per-device partial
  sums for `train_step` to reduce-scatter where it wants to.
  """
  return jax.P(*spec, reduced={'data'})


# -- model -------------------------------------------------------------------

def init(key, specs=PARAM_SPECS, as_reduced=True):
  keys = jax.random.split(key, len(SHAPES))
  params = {k: jax.random.normal(kk, s, out_sharding=specs[k]) * (s[-2] ** -0.5)
            for kk, (k, s) in zip(keys, SHAPES.items())}
  if not as_reduced:  # for the unsharded reference in `check`
    return params
  return {k: jax.reshard(v, reduced(specs[k])) for k, v in params.items()}


def rmsnorm(x):
  return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), -1, keepdims=True) + 1e-6)


def layer(x, p):
  q, k, v = jnp.split(jnp.einsum('btd,dnh->btnh', rmsnorm(x), p['qkv']), 3, -1)
  a = jax.nn.dot_product_attention(q, k, v, is_causal=True)
  # Both `x` and `proj` are sharded over 'model' along the contracted head
  # axis, so the output sharding is ambiguous and JAX makes us say what we
  # want. Asking for a batch-sharded result is what turns into an all-reduce.
  x += jnp.einsum('btnh,nhd->btd', a, p['proj'], out_sharding=ACTS)
  h = jax.nn.gelu(jnp.einsum('btd,df->btf', rmsnorm(x), p['up']))
  x += jnp.einsum('btf,fd->btd', h, p['down'], out_sharding=ACTS)
  return x, None


def logits(params, tokens):
  # A gather's output sharding is ambiguous too, hence `.at[].get()`.
  x = params['embed'].at[tokens].get(out_sharding=ACTS)
  layers = {k: params[k] for k in LAYER_KEYS}
  x, _ = jax.lax.scan(jax.remat(layer), x, layers)
  return jnp.einsum('btd,dv->btv', rmsnorm(x), params['unemb'], out_sharding=ACTS)


def loss(params, batch):
  logprobs = jax.nn.log_softmax(logits(params, batch[:, :-1]))
  return -jnp.mean(jnp.take_along_axis(logprobs, batch[:, 1:, None], -1))


# -- optimizer ---------------------------------------------------------------

def adam_init(params):
  """Adam state lives sharded over 'data' -- 1/N of it per device."""
  shards = {k: jax.reshard(v, OPT_SPECS[k]) for k, v in params.items()}
  return dict(m=jax.tree.map(jnp.zeros_like, shards),
              v=jax.tree.map(jnp.zeros_like, shards), t=jnp.zeros((), jnp.int32))


def schedule(t, steps, peak=3e-3, warmup=50):
  return peak * jnp.minimum((t + 1) / warmup,
                            0.5 * (1 + jnp.cos(jnp.pi * t / steps)))


@functools.partial(jax.jit, donate_argnums=(0, 1), static_argnums=3)
def train_step(params, opt, batch, steps, b1=0.9, b2=0.99, eps=1e-8):
  l, g = jax.value_and_grad(loss)(params, batch)
  t = opt['t'] + 1

  # `g` is unreduced over 'data': each device holds a partial sum, and the
  # cross-device reduction has not happened yet. Resharding it to a
  # 'data'-sharded layout does that reduction *and* the split in one step --
  # a reduce-scatter, where a replicated gradient would have cost a full
  # all-reduce followed by throwing away all but 1/N of the result.
  g = {k: jax.reshard(v, OPT_SPECS[k]) for k, v in g.items()}
  # Slicing the replicated parameters down to this device's shard is free.
  shards = {k: jax.reshard(v, OPT_SPECS[k]) for k, v in params.items()}

  m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt['m'], g)
  v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, opt['v'], g)
  lr = schedule(t, steps)
  shards = jax.tree.map(
      lambda p, m, v: p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps),
      shards, m, v)

  # Each device updated its own slice; all-gather them back into full
  # parameters for the next forward pass.
  params = {k: jax.reshard(v, reduced(PARAM_SPECS[k])) for k, v in shards.items()}
  return params, dict(m=m, v=v, t=t), l


# -- driver ------------------------------------------------------------------

def collectives(fn, *args):
  """Counts the collectives XLA inserted, which is the whole point here."""
  hlo = jax.jit(fn).lower(*args).compile().as_text()
  return {c: hlo.count(c + '(')
          for c in ('all-gather', 'all-reduce', 'reduce-scatter', 'all-to-all')}


def train(params, batch_iter, steps, log=print):
  """Runs `steps` Adam steps and returns the trained parameters."""
  opt = adam_init(params)
  start = time.perf_counter()
  for step in range(steps):
    batch = jax.device_put(next(batch_iter).astype(np.int32), jax.P('data', None))
    params, opt, l = train_step(params, opt, batch, steps)
    # Waiting on the loss each step bounds how much work is in flight. JAX
    # dispatches asynchronously, so without this the loop would run ahead and
    # queue hundreds of steps' worth of collectives at once -- which on the
    # CPU backend, where every device shares one thread pool, can deadlock.
    l = float(l)
    if log and (step % max(1, steps // 10) == 0 or step == steps - 1):
      log(f'  step {step:4d}  loss {l:.4f}  ({time.perf_counter() - start:.1f}s)')
  return params


def check(key, batch):
  """Asserts the sharded computation matches a fully replicated one."""
  replicated = {k: jax.P() for k in PARAM_SPECS}
  ref_params = init(key, replicated, as_reduced=False)
  params = {k: jax.reshard(v, reduced(PARAM_SPECS[k]))
            for k, v in ref_params.items()}
  ref_batch = jax.device_put(batch, jax.P())
  np.testing.assert_allclose(jax.jit(loss)(params, batch),
                             jax.jit(loss)(ref_params, ref_batch), rtol=1e-4)
  # Reduce the unreduced gradients so they are comparable to the reference.
  g = jax.jit(lambda p, b: {k: jax.reshard(v, OPT_SPECS[k])
                            for k, v in jax.grad(loss)(p, b).items()})(params, batch)
  ref_g = jax.jit(jax.grad(loss))(ref_params, ref_batch)
  # Compared against the magnitude of the gradient rather than elementwise:
  # partitioning reassociates the reductions, so the two computations agree in
  # exact arithmetic but not bit-for-bit in float32.
  for k in g:
    a, b = np.asarray(g[k]), np.asarray(ref_g[k])
    err = np.abs(a - b).max() / np.abs(b).max()
    assert err < 1e-3, f'{k}: relative error {err:.2e}'
    print(f'  {k:6s} relative error {err:.1e}')
  print('check: sharded loss and gradients match the replicated reference')


def main(args):
  mesh_shape = tuple(int(x) for x in args.mesh.split(','))
  jax.set_mesh(jax.make_mesh(mesh_shape, ('data', 'model')))
  print(f'mesh {dict(zip(("data", "model"), mesh_shape))} '
        f'on {jax.device_count()} {jax.devices()[0].platform} devices')

  key = jax.random.key(args.seed)
  batch_iter = data.batches(data.load(args.offline), B, T, seed=args.seed)
  batch = jax.device_put(next(batch_iter).astype(np.int32), jax.P('data', None))

  if args.check:
    return check(key, batch)

  params = init(key)
  print('\n'.join(f'  {k:6s} {jax.typeof(p)}' for k, p in params.items()))
  nparams = sum(np.prod(s) for s in SHAPES.values())
  print(f'  {nparams / 1e6:.2f}M parameters, batch {jax.typeof(batch)}')

  # The point of the file, in three types. `.trace(...)` stops after tracing,
  # so `out_avals` costs nothing to look at.
  g = jax.jit(jax.grad(loss)).trace(params, batch).out_avals
  g_sharded = jax.jit(
      lambda p, b: {k: jax.reshard(v, OPT_SPECS[k])
                    for k, v in jax.grad(loss)(p, b).items()}
      ).trace(params, batch).out_avals
  k = 'up'
  print(f'\n  the ZeRO-2 reduction, as types:\n'
        f'    parameter        {jax.typeof(params[k])}   stored `reduced`, so...\n'
        f'    its gradient     {g[k]}   ...`grad` leaves it unreduced,\n'
        f'    after resharding {g_sharded[k]}      ...so this is a '
        f'reduce-scatter\n')

  fwd = collectives(loss, params, batch)
  print('  forward+backward: '
        + ', '.join(f'{k}={v}' for k, v in fwd.items() if v)
        + '   (tensor parallelism only; no parameter gathers)')
  step = collectives(functools.partial(train_step, steps=args.steps),
                     params, adam_init(params), batch)
  print('  whole step:       '
        + ', '.join(f'{k}={v}' for k, v in step.items() if v)
        + '   (plus the optimizer gradient reduction and gather)')
  if jax.devices()[0].platform == 'cpu':
    print('  note: the gradient reduce-scatters show up above as all-reduce +'
          ' dynamic-slice;\n        XLA rewrites that pair into a'
          ' reduce-scatter on TPU and GPU, but not on CPU.')

  params = train(params, batch_iter, args.steps)

  if args.save:
    np.savez(args.save, **{k: np.asarray(v) for k, v in params.items()})
    print(f'  wrote {args.save}')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--devices', type=int, default=util.default_devices(),
                 help='simulated CPU devices; 0 to use real hardware')
  p.add_argument('--mesh', default=None, help='"data,model" mesh shape')
  p.add_argument('--steps', type=int, default=200)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--offline', action='store_true', help='skip the download')
  p.add_argument('--check', action='store_true')
  p.add_argument('--save', default=None, help='write params to this .npz')
  args = p.parse_args()
  if args.devices:
    jax.config.update('jax_num_cpu_devices', args.devices)
  if args.mesh is None:
    args.mesh = util.default_mesh(args.devices or jax.device_count())
  main(args)
