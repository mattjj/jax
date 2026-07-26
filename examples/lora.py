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

"""Training and serving many LoRA adapters at once with `vmap`.

Demonstrates: `vmap` over a *sharded* axis, explicit sharding, `jit`, `grad`
with respect to part of a model, `scan`, `remat`.

LoRA freezes a pretrained model and learns a low-rank correction `a @ b` to a
few of its weight matrices. The adapters are tiny -- about 2% of the base model
here -- which means a whole stack of them fits alongside one copy of the base.
That is the interesting part, and it is what this file is about:

    jax.vmap(step, in_axes=(0, None, 0))(adapters, base, batches)

One base model, `in_axes=None`, shared. A stack of adapters, `in_axes=0`,
each with its own data. The *same* `vmap` trains all of them in one call and
later serves all of them in one call -- and because the adapter axis is
sharded over the 'data' mesh axis, each device holds and runs its own
adapters. That is multi-tenant serving in one line, and it is genuinely
awkward to express in a framework without `vmap`.

Each adapter learns a different byte-level transform of Shakespeare (leave it
alone, upper-case it, lower-case it, replace its spaces), so afterwards you
can check
that adapter i really is the best one at task i. `--check` asserts exactly
that.

    python examples/lora.py            # train the base, then four adapters
    python examples/lora.py --check    # assert each adapter wins its own task
"""

import argparse
import functools

import numpy as np

import jax
import jax.numpy as jnp

import data
import nanolm
import util

RANK = 4       # the "low rank" in low-rank adaptation
LORA_B = 16    # batch and sequence length for adapter training: smaller than
LORA_T = 64    # the base model's, since the adapters have far less to learn

# LoRA patches two matrices, chosen to show both sharding cases. `up` is
# column parallel -- sharded over 'model' along its output axis -- so the `b`
# factor carries that same shard and `a @ b` needs no communication. `unemb`
# is replicated, so its factors are too. In both cases the rule is the same:
# the correction must be sharded like the weight it is added to.
LORA_SHAPES = dict(up_a=(nanolm.L, nanolm.D, RANK),
                   up_b=(nanolm.L, RANK, nanolm.F),
                   unemb_a=(nanolm.D, RANK), unemb_b=(RANK, nanolm.V))
LORA_SPECS = dict(up_a=jax.P(None, None, None),
                  up_b=jax.P(None, None, 'model'),
                  unemb_a=jax.P(None, None),
                  unemb_b=jax.P(None, None))

# Inside the `vmap`, the mapped adapter axis is the one sharded over 'data',
# so nothing *within* an adapter may claim 'data' as well.
INNER_ACTS = jax.P(None, None, None)


# -- byte-level tasks, one per adapter ---------------------------------------

def _upper(b):
  return np.where((b >= 97) & (b <= 122), b - 32, b).astype(np.uint8)


def _lower(b):
  return np.where((b >= 65) & (b <= 90), b + 32, b).astype(np.uint8)


# Four transforms, chosen to be clearly distinguishable from one another.
# Two near-misses worth knowing about, because the check below rightly fails on
# both: a bijection on letters like rot13 is only a *relabeling*, so on text
# with a flat letter distribution no adapter can tell it from the identity; and
# `swapcase` is nearly `upper` on text that is mostly lower-case already.
TASKS = {
    'identity': lambda b: b,
    'upper': _upper,
    'lower': _lower,
    'underscore': lambda b: np.where(b == ord(' '), ord('_'), b).astype(np.uint8),
}


# -- the adapters -------------------------------------------------------------

def init_lora(key, adapters):
  """A stack of `adapters` LoRA adapters, sharded over the 'data' mesh axis."""
  keys = jax.random.split(key, len(LORA_SHAPES))
  out = {}
  for kk, (name, shape) in zip(keys, LORA_SHAPES.items()):
    spec = jax.P('data', *LORA_SPECS[name])
    if name.endswith('_b'):
      # b starts at zero, so a freshly initialized adapter is exactly a no-op.
      out[name] = jnp.zeros((adapters, *shape), out_sharding=spec)
    else:
      out[name] = jax.random.normal(kk, (adapters, *shape),
                                    out_sharding=spec) * nanolm.D ** -0.5
  return out


def patch(base, lora):
  """The base model with one adapter's low-rank correction added in."""
  return {**base,
          'up': base['up'] + jnp.einsum('ldr,lrf->ldf', lora['up_a'],
                                        lora['up_b']),
          'unemb': base['unemb'] + lora['unemb_a'] @ lora['unemb_b']}


def loss(lora, base, batch):
  """Loss for ONE adapter on ONE batch. `vmap` supplies the stack."""
  params = patch(base, lora)
  x = params['embed'].at[batch[:, :-1]].get(out_sharding=INNER_ACTS)
  layers = {k: params[k] for k in nanolm.LAYER_KEYS}
  body = functools.partial(nanolm.layer, acts=INNER_ACTS)
  x, _ = jax.lax.scan(jax.remat(body), x, layers)
  lg = jnp.einsum('btd,dv->btv', nanolm.rmsnorm(x), params['unemb'],
                  out_sharding=INNER_ACTS)
  lp = jax.nn.log_softmax(lg)
  return -jnp.mean(jnp.take_along_axis(lp, batch[:, 1:, None], -1))


# The whole point, twice: `grad` for training, plain evaluation for serving.
# `in_axes=(0, None, 0)` means "one adapter and one batch each, base shared".
batched_grad = jax.vmap(jax.value_and_grad(loss), in_axes=(0, None, 0))
batched_loss = jax.vmap(loss, in_axes=(0, None, 0))


@functools.partial(jax.jit, donate_argnums=(0, 1))
def train_step(lora, opt, base, batches, lr=1e-2, b1=0.9, b2=0.99, eps=1e-8):
  losses, g = batched_grad(lora, base, batches)
  t = opt['t'] + 1
  m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt['m'], g)
  v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, opt['v'], g)
  lora = jax.tree.map(
      lambda p, m, v: p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps),
      lora, m, v)
  return lora, dict(m=m, v=v, t=t), losses


# -- driver -------------------------------------------------------------------

def task_batches(raw, seed, adapters):
  """One independent batch stream per task, transformed bytes."""
  streams = [data.batches(fn(raw), LORA_B, LORA_T, seed=seed + i)
             for i, (_, fn) in enumerate(list(TASKS.items())[:adapters])]
  while True:
    yield jax.device_put(
        np.stack([next(s) for s in streams]).astype(np.int32),
        jax.P('data', None, None))


def main(args):
  jax.set_mesh(jax.make_mesh(tuple(int(x) for x in args.mesh.split(',')),
                             ('data', 'model')))
  names = list(TASKS)[:args.adapters]
  n_data = jax.sharding.get_mesh().shape['data']
  assert len(names) % n_data == 0, (
      f'{len(names)} adapters do not divide over a "data" axis of {n_data}; '
      'pass --adapters or --mesh so that they do')
  print(f'{len(names)} adapters ({", ".join(names)}) over '
        f'{jax.device_count()} devices, {len(names) // n_data} per device')

  key = jax.random.key(args.seed)
  raw = data.load(args.offline)

  key, subkey = jax.random.split(key)
  if args.params:
    base = {k: jax.device_put(jnp.asarray(v), nanolm.PARAM_SPECS[k])
            for k, v in np.load(args.params).items()}
  else:
    print(f'  pretraining the base model for {args.base_steps} steps')
    base = nanolm.train(nanolm.init(subkey),
                        data.batches(raw, nanolm.B, nanolm.T, seed=args.seed), args.base_steps)
  base = {k: jax.reshard(v, nanolm.PARAM_SPECS[k]) for k, v in base.items()}

  key, subkey = jax.random.split(key)
  lora = init_lora(subkey, len(names))
  n_base = sum(np.prod(s) for s in nanolm.SHAPES.values())
  n_lora = sum(np.prod(s) for s in LORA_SHAPES.values())
  print('\n'.join(f'  {k:6s} {jax.typeof(v)}' for k, v in lora.items()))
  print(f'  {n_lora / 1e3:.1f}k parameters per adapter, '
        f'{100 * n_lora / n_base:.1f}% of the {n_base / 1e6:.2f}M base\n')

  opt = dict(m=jax.tree.map(jnp.zeros_like, lora),
             v=jax.tree.map(jnp.zeros_like, lora), t=jnp.zeros((), jnp.int32))
  batches = task_batches(raw, args.seed, len(names))
  for step in range(args.steps):
    lora, opt, losses = train_step(lora, opt, base, next(batches))
    losses = np.asarray(losses)  # also bounds in-flight work; see nanolm.train
    if step % max(1, args.steps // 8) == 0 or step == args.steps - 1:
      print(f'  step {step:4d}  ' +
            '  '.join(f'{n}={l:.3f}' for n, l in zip(names, losses)))

  # Serving: every adapter against every task's data, in one batched call per
  # task. Adapter i should win column i.
  print('\n  loss of each adapter (rows) on each task (columns):')
  print(' ' * 14 + ''.join(f'{n:>11s}' for n in names))
  grid = []
  for j, name in enumerate(names):
    b = np.stack([next(data.batches(TASKS[name](raw), LORA_B, LORA_T,
                                    seed=args.seed + 99))]
                 * len(names)).astype(np.int32)
    grid.append(np.asarray(jax.jit(batched_loss)(
        lora, base, jax.device_put(b, jax.P('data', None, None)))))
  grid = np.stack(grid, axis=1)  # [adapter, task]
  for i, name in enumerate(names):
    print(f'  {name:>10s}  ' + ''.join(f'{v:11.4f}' for v in grid[i]))

  wins = [int(grid[:, j].argmin()) == j for j in range(len(names))]
  if args.check:
    assert all(wins), f'adapters did not win their own tasks: {wins}'
    print('\ncheck: each adapter is the best of the stack on its own task')
  else:
    print(f'\n  {sum(wins)}/{len(names)} tasks won by their own adapter')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--devices', type=int, default=util.default_devices(),
                 help='simulated CPU devices; 0 to use real hardware')
  p.add_argument('--mesh', default=None, help='"data,model" mesh shape')
  p.add_argument('--adapters', type=int, default=4)
  p.add_argument('--params', default=None, help='.npz written by nanolm.py --save')
  p.add_argument('--base-steps', type=int, default=300)
  p.add_argument('--steps', type=int, default=300)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--offline', action='store_true')
  p.add_argument('--check', action='store_true')
  args = p.parse_args()
  if args.devices:
    jax.config.update('jax_num_cpu_devices', args.devices)
  if args.mesh is None:
    args.mesh = util.default_mesh(args.devices or jax.device_count())
  main(args)
