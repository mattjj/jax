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

"""Training a byte-level transformer with FSDP and tensor parallelism.

Demonstrates: explicit sharding, `jit`, `grad`, `scan`, `remat`, donation.

The model is a small decoder-only transformer trained on Shakespeare. The
interesting part is the `SPECS` table below: it is the *only* place
parallelism is expressed. There are no collectives in the model code. Sharding
the parameters over the 'data' mesh axis gives fully sharded data parallelism
(FSDP), because the compiler all-gathers each parameter just before it is
used; sharding them over the 'model' axis gives tensor parallelism, because
the compiler all-reduces the partial sums. The example prints the collectives
it ended up with so you can see which annotation bought which.

By default this runs on *simulated* CPU devices -- as many as the machine has
cores, up to eight -- so the sharding is real and inspectable on a laptop. To
run it on real hardware, pass `--devices 0` and a mesh matching your machine.

    python examples/nanolm.py                        # ~1 min on CPU
    python examples/nanolm.py --devices 8 --mesh 8,1  # pure FSDP
    python examples/nanolm.py --devices 8 --mesh 1,8  # pure tensor parallelism
    python examples/nanolm.py --check                # verify against replicated

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

# The whole parallelization strategy. 'data' shards give FSDP, 'model' shards
# give tensor parallelism, and every parameter is sharded over both. Note that
# every array axis mentioned here (V, D, N, F) must be divisible by the
# corresponding mesh axis size, which is why N == 8 rather than something
# smaller: it lets `--mesh 1,8` work.
SPECS = dict(
    embed=jax.P('data', 'model'),              # [V, D]
    qkv=jax.P(None, 'data', 'model', None),    # [L, D, N, 3H]
    proj=jax.P(None, 'model', None, 'data'),   # [L, N, H, D]
    up=jax.P(None, 'data', 'model'),           # [L, D, F]
    down=jax.P(None, 'model', 'data'),         # [L, F, D]
    unemb=jax.P('data', 'model'),              # [D, V]
)

# Activations keep the batch sharded over 'data' and everything else replicated.
ACTS = jax.P('data', None, None)

LAYER_KEYS = ('qkv', 'proj', 'up', 'down')


# -- model -------------------------------------------------------------------

def init(key, specs=SPECS):
  keys = jax.random.split(key, len(SHAPES))
  return {k: jax.random.normal(kk, s, out_sharding=specs[k]) * (s[-2] ** -0.5)
          for kk, (k, s) in zip(keys, SHAPES.items())}


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
  return dict(m=jax.tree.map(jnp.zeros_like, params),
              v=jax.tree.map(jnp.zeros_like, params), t=jnp.zeros((), jnp.int32))


def schedule(t, steps, peak=3e-3, warmup=50):
  return peak * jnp.minimum((t + 1) / warmup,
                            0.5 * (1 + jnp.cos(jnp.pi * t / steps)))


@functools.partial(jax.jit, donate_argnums=(0, 1), static_argnums=3)
def train_step(params, opt, batch, steps, b1=0.9, b2=0.99, eps=1e-8):
  l, g = jax.value_and_grad(loss)(params, batch)
  t = opt['t'] + 1
  m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt['m'], g)
  v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, opt['v'], g)
  lr = schedule(t, steps)
  params = jax.tree.map(
      lambda p, m, v: p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps),
      params, m, v)
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
  replicated = {k: jax.P() for k in SPECS}
  ref_params = init(key, replicated)
  params = jax.tree.map(lambda x, s: jax.device_put(x, s), ref_params, SPECS)
  ref_batch = jax.device_put(batch, jax.P())
  np.testing.assert_allclose(jax.jit(loss)(params, batch),
                             jax.jit(loss)(ref_params, ref_batch), rtol=1e-4)
  g = jax.jit(jax.grad(loss))(params, batch)
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

  counts = collectives(loss, params, batch)
  print('  collectives: ' + ', '.join(f'{k}={v}' for k, v in counts.items() if v))

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
