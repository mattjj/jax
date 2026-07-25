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

"""A mixture-of-experts layer with expert parallelism.

Demonstrates: `shard_map`, `all_to_all`, explicit sharding, `jit`, `grad`.

Every device holds a slice of the tokens *and* a slice of the experts, so a
token usually needs to be computed on a different device than the one holding
it. Moving them is one `all_to_all` there and one back -- the collective that
expert parallelism is built on, and the clearest reason to reach for
`shard_map` instead of letting the compiler decide.

The routing is the GShard formulation: instead of gathering ragged lists of
tokens per expert, build a fixed-shape one-hot `dispatch` tensor of shape
`[tokens, experts, capacity]` and let two einsums do the scatter and gather.
Everything stays a dense matmul, which is what makes it fast and what makes it
fit on a page. Tokens beyond an expert's capacity are dropped, exactly as in a
real implementation.

The expert-usage histogram printed at the end will show some experts unused:
with nothing pushing back, a top-1 router collapses onto a subset of experts.
Fixing that needs an auxiliary load-balancing loss, which is deliberately left
out here -- the point of this file is the parallelism, not the recipe.

    python examples/moe.py            # route, train, report expert usage
    python examples/moe.py --check    # verify against a sequential reference
"""

import argparse
import functools

import numpy as np

import jax
import jax.numpy as jnp

import util

E = 16     # experts (must be divisible by the mesh size)
D = 64     # model dimension
F = 128    # expert hidden dimension
TOK = 256  # tokens per step
FACTOR = 4  # capacity factor: how much above the average load to allow for


def capacity(devices, factor=FACTOR):
  """Tokens each expert accepts from each device, per the usual formula.

  A device holds `TOK // devices` tokens and would send `1 / E` of them to
  each expert if routing were perfectly balanced; `factor` is the headroom for
  the fact that it isn't.
  """
  return max(1, int(np.ceil(TOK / devices / E * factor)))


def init(key, experts=E):
  k1, k2, k3 = jax.random.split(key, 3)
  return dict(
      gate=jax.random.normal(k1, (D, experts), out_sharding=jax.P()) * D ** -0.5,
      up=jax.random.normal(k2, (experts, D, F), out_sharding=jax.P('expert')) * D ** -0.5,
      down=jax.random.normal(k3, (experts, F, D), out_sharding=jax.P('expert')) * F ** -0.5,
  )


def route(x, gate, experts, capacity):
  """Top-1 routing. Returns `[tokens, experts, capacity]` dispatch/combine."""
  probs = jax.nn.softmax(x @ gate, axis=-1)
  chosen = jnp.argmax(probs, -1)
  onehot = jax.nn.one_hot(chosen, experts, dtype=x.dtype)
  # Position of each token in its expert's queue: how many earlier tokens on
  # this device already chose the same expert.
  rank = jnp.max(jnp.cumsum(onehot, 0) * onehot - 1, -1).astype(jnp.int32)
  kept = onehot * (rank < capacity)[:, None]
  dispatch = kept[:, :, None] * jax.nn.one_hot(rank, capacity, dtype=x.dtype)[:, None, :]
  return dispatch, dispatch * jnp.max(probs, -1)[:, None, None]


def moe_layer(x, gate, up, down, cap):
  """Runs the MoE layer on this device's shard of the tokens.

  Shapes below are per device: `n` local experts out of `E`, `t` local tokens.
  """
  dispatch, combine = route(x, gate, E, cap)                     # [t, E, CAP]

  # Pack each expert's tokens into a fixed-size buffer, then send buffer `e`
  # to whichever device owns expert `e`. After the all_to_all this device
  # holds, for each of its own experts, the tokens every device sent it.
  buffers = jnp.einsum('td,tec->ecd', x, dispatch)               # [E, CAP, D]
  buffers = jax.lax.all_to_all(buffers, 'expert', split_axis=0, concat_axis=1,
                               tiled=True)                       # [n, CAP*P, D]

  h = jax.nn.gelu(jnp.einsum('ecd,edf->ecf', buffers, up))
  out = jnp.einsum('ecf,efd->ecd', h, down)                      # [n, CAP*P, D]

  # ... and send the results back where they came from.
  out = jax.lax.all_to_all(out, 'expert', split_axis=1, concat_axis=0,
                           tiled=True)                           # [E, CAP, D]
  return (jnp.einsum('ecd,tec->td', out, combine),
          jax.lax.psum(dispatch.sum((0, 2)), 'expert'))  # tokens per expert


def reference(x, params):
  """Sequential, unsharded version of the same computation.

  Expects replicated inputs; it gathers a whole expert's weights per token,
  which is exactly what expert parallelism exists to avoid.
  """
  probs = jax.nn.softmax(x @ params['gate'], -1)
  chosen = jnp.argmax(probs, -1)
  h = jax.nn.gelu(jnp.einsum('td,tdf->tf', x, params['up'][chosen]))
  y = jnp.einsum('tf,tfd->td', h, params['down'][chosen])
  return y * jnp.max(probs, -1)[:, None]


def build(cap):
  """Builds the expert-parallel layer, its loss, and a training step.

  The capacity has to be baked in -- it is a shape -- and it depends on how
  many devices the tokens are spread over, so this is a function rather than a
  module-level definition.
  """
  moe = jax.shard_map(
      functools.partial(moe_layer, cap=cap),
      in_specs=(jax.P('expert'), jax.P(), jax.P('expert'), jax.P('expert')),
      out_specs=(jax.P('expert'), jax.P()))

  def loss(params, x, targets):
    y, _ = moe(x, params['gate'], params['up'], params['down'])
    return jnp.mean(jnp.square(y - targets))

  @jax.jit
  def train_step(params, x, targets, lr=0.02):
    l, g = jax.value_and_grad(loss)(params, x, targets)
    return jax.tree.map(lambda p, g: p - lr * g, params, g), l

  return moe, loss, train_step


def check(key, moe):
  """With capacity high enough that nothing is dropped, the two must agree."""
  x = jax.random.normal(key, (TOK, D), out_sharding=jax.P('expert', None))
  params = init(key)
  y, counts = jax.jit(moe)(x, params['gate'], params['up'], params['down'])
  ref = jax.jit(reference)(jax.device_put(x, jax.P()),
                           jax.tree.map(lambda v: jax.device_put(v, jax.P()), params))
  dropped = TOK - int(counts.sum())
  err = np.abs(np.asarray(y) - np.asarray(ref)).max() / np.abs(np.asarray(ref)).max()
  assert dropped == 0, f'{dropped} tokens dropped; raise FACTOR to compare'
  assert err < 1e-5, f'relative error {err:.2e}'
  print(f'  relative error vs sequential reference {err:.1e}')
  print('check: expert-parallel MoE matches the sequential reference')


def main(args):
  jax.set_mesh(jax.make_mesh((args.mesh,), ('expert',)))
  key = jax.random.key(args.seed)
  cap = capacity(args.mesh)
  moe, loss, train_step = build(cap)
  print(f'{E} experts over {args.mesh} devices '
        f'({E // args.mesh} per device), capacity {cap}')

  if args.check:
    return check(key, moe)

  params = init(key)
  print('\n'.join(f'  {k:5s} {jax.typeof(v)}' for k, v in params.items()))
  hlo = jax.jit(loss).lower(
      params, jax.random.normal(key, (TOK, D), out_sharding=jax.P('expert', None)),
      jnp.zeros((TOK, D), out_sharding=jax.P('expert', None))).compile().as_text()
  print('  collectives: ' + ', '.join(
      f'{c}={hlo.count(c + "(")}' for c in ('all-to-all', 'all-gather', 'all-reduce')
      if hlo.count(c + '(')))

  # A task that rewards specialization: each token must be mapped by one of
  # `E` different random linear maps, chosen by a cluster label the router has
  # to discover from the input.
  key, k_task, k_centers = jax.random.split(key, 3)
  centers = jax.random.normal(k_centers, (E, D), out_sharding=jax.P()) * 3
  maps = jax.random.normal(k_task, (E, D, D), out_sharding=jax.P()) * D ** -0.5

  for step in range(args.steps):
    key, k1, k2 = jax.random.split(key, 3)
    label = jax.random.randint(k1, (TOK,), 0, E)
    x = centers[label] + jax.random.normal(k2, (TOK, D))
    x = jax.device_put(x, jax.P('expert', None))
    targets = jax.device_put(jnp.einsum('td,tde->te', x, maps[label]),
                             jax.P('expert', None))
    params, l = train_step(params, x, targets)
    l = float(l)  # bounds in-flight work; see the note in nanolm.train
    if step % max(1, args.steps // 8) == 0 or step == args.steps - 1:
      print(f'  step {step:4d}  loss {l:.4f}')

  _, counts = jax.jit(moe)(x, params['gate'], params['up'], params['down'])
  counts = np.asarray(counts)
  print(f'  tokens per expert: {counts.astype(int).tolist()}'
        f'  ({TOK - int(counts.sum())} dropped)')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--devices', type=int, default=util.default_devices(),
                 help='simulated CPU devices; 0 to use real hardware')
  p.add_argument('--mesh', type=int, default=None,
                 help='devices to spread the experts over')
  p.add_argument('--steps', type=int, default=400)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--check', action='store_true')
  args = p.parse_args()
  if args.devices:
    jax.config.update('jax_num_cpu_devices', args.devices)
  if args.mesh is None:
    args.mesh = args.devices or jax.device_count()
  main(args)
