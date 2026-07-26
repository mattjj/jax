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

"""Flow matching: a small generative model, with classifier-free guidance.

Demonstrates: `jit`, `grad`, `vmap` (over guidance strengths), `scan` (the
ODE integrator), explicit sharding of the sample batch.

Flow matching trains a generative model with nothing but regression. Draw
noise `x0`, data `x1`, and a time `t`; put yourself at the straight-line
point `x_t = (1-t) x0 + t x1`; regress the model's velocity `v(x_t, t)`
onto the line's direction `x1 - x0`. That is the whole training objective
(conditional flow matching / rectified flow). To sample, integrate
`dx/dt = v(x, t)` from noise at `t=0` to data at `t=1` -- here with 32 Euler
steps in a `jax.lax.scan`.

The data is an eight-mode Gaussian mixture arranged on a ring, so everything
is visible in the terminal: the model is conditioned on the mode label with
label dropout, giving classifier-free guidance (CFG) for free --

    v_guided = v_uncond + w * (v_cond - v_uncond)

and the sampler is `vmap`ed over the guidance strength `w`, producing all
panels below in one call. `w=0` ignores the label; larger `w` concentrates
samples on the requested mode.

`--check` asserts the unconditional samples cover all eight modes and the
guided samples land on the requested one.

    python examples/flow_matching.py            # ~1 min on CPU
    python examples/flow_matching.py --check
"""

import argparse
import functools

import numpy as np

import jax
import jax.numpy as jnp

import util

MODES = 8          # mixture components on the ring
RADIUS = 2.0
STD = 0.15         # per-mode standard deviation
HIDDEN = 128
STEPS_ODE = 32     # Euler steps from noise to data
DROP = 0.2         # label-dropout rate: trains the unconditional model


def mode_centers():
  th = 2 * jnp.pi * jnp.arange(MODES) / MODES
  return RADIUS * jnp.stack([jnp.cos(th), jnp.sin(th)], axis=-1)  # [MODES, 2]


def sample_data(key, n):
  """Mixture samples and their mode labels."""
  k1, k2 = jax.random.split(key)
  label = jax.random.randint(k1, (n,), 0, MODES)
  x = mode_centers()[label] + STD * jax.random.normal(k2, (n, 2))
  return x, label


# -- the velocity model -------------------------------------------------------

def init(key):
  # input: x (2) + sinusoidal time features (8) + label embedding (4)
  sizes = [2 + 8 + 4, HIDDEN, HIDDEN, 2]
  keys = jax.random.split(key, len(sizes))
  params = dict(embed=jax.random.normal(keys[0], (MODES + 1, 4)) * 0.1)
  for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
    params[f'w{i}'] = jax.random.normal(keys[i + 1], (m, n)) * m ** -0.5
    params[f'b{i}'] = jnp.zeros(n)
  return params


def time_features(t):
  freqs = 2.0 ** jnp.arange(4)
  ang = t[..., None] * freqs * jnp.pi
  return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], -1)  # [..., 8]


def velocity(params, x, t, label):
  """v(x, t, label). `label == MODES` is the unconditional 'null' label."""
  # The batch is sharded over 'data'; everything concatenated onto it must be
  # too, including the label embeddings this gather produces.
  emb = params['embed'].at[label].get(out_sharding=jax.typeof(x).sharding.spec)
  h = jnp.concatenate([x, time_features(t), emb], axis=-1)
  for i in range(3):
    h = h @ params[f'w{i}'] + params[f'b{i}']
    if i < 2:
      h = jax.nn.gelu(h)
  return h


# -- training: regression, nothing else ---------------------------------------

def loss(params, key, x1, label):
  n = x1.shape[0]
  batch_spec = jax.typeof(x1).sharding.spec
  k_t, k_noise, k_drop = jax.random.split(key, 3)
  t = jax.random.uniform(k_t, (n,), out_sharding=jax.P(*batch_spec[:1]))
  x0 = jax.random.normal(k_noise, x1.shape, out_sharding=batch_spec)
  # label dropout: with prob DROP, train the null (unconditional) label
  drop = jax.random.uniform(k_drop, (n,),
                            out_sharding=jax.P(*batch_spec[:1])) < DROP
  label = jnp.where(drop, MODES, label)

  x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
  target = x1 - x0                     # the straight line's velocity
  pred = velocity(params, x_t, t, label)
  return jnp.mean(jnp.sum(jnp.square(pred - target), -1))


@functools.partial(jax.jit, donate_argnums=(0, 1))
def train_step(params, opt, key, x1, label, lr=3e-3, b1=0.9, b2=0.99,
               eps=1e-8):
  l, g = jax.value_and_grad(loss)(params, key, x1, label)
  t = opt['t'] + 1
  m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt['m'], g)
  v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, opt['v'], g)
  params = jax.tree.map(
      lambda p, m, v: p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps),
      params, m, v)
  return params, dict(m=m, v=v, t=t), l


# -- sampling: integrate the ODE, vmapped over guidance strengths -------------

def generate(params, key, n, label, w):
  """Integrate dx/dt = v from t=0 (noise) to t=1 (data), one guidance `w`."""
  x = jax.random.normal(key, (n, 2), out_sharding=jax.P('data', None))
  null = jnp.full((n,), MODES, out_sharding=jax.P('data'))

  def step(x, t):
    ts = jnp.full((n,), t, out_sharding=jax.P('data'))
    v_cond = velocity(params, x, ts, label)
    v_unc = velocity(params, x, ts, null)
    v = v_unc + w * (v_cond - v_unc)   # classifier-free guidance
    return x + v / STEPS_ODE, None

  x, _ = jax.lax.scan(step, x, jnp.arange(STEPS_ODE) / STEPS_ODE)
  return x


# All guidance strengths in one call: `vmap` over the scalar `w`.
generate_all = jax.jit(jax.vmap(generate, in_axes=(None, None, None, None, 0)),
                       static_argnums=2)


# -- terminal rendering -------------------------------------------------------

def ascii_panel(pts, size=6.0, w=33, h=17):
  grid = np.zeros((h, w), int)
  ij = np.clip(((np.asarray(pts) + size / 2) / size * [w, h]).astype(int),
               0, [w - 1, h - 1])
  np.add.at(grid, (ij[:, 1], ij[:, 0]), 1)
  chars = np.array(list(' .:+*#'))
  lvl = np.clip((np.log1p(grid) * 2).astype(int), 0, len(chars) - 1)
  return [''.join(row) for row in chars[lvl][::-1]]


def show(panels, titles):
  print(''.join(t.center(35) for t in titles))
  for rows in zip(*[ascii_panel(p) for p in panels]):
    print('  '.join(rows))


# -- driver -------------------------------------------------------------------

def nearest_mode(x):
  d = np.linalg.norm(np.asarray(x)[:, None, :] - np.asarray(mode_centers()),
                     axis=-1)
  return d.argmin(-1)


def hit_rate(x, target_mode):
  """Fraction of samples whose nearest mode is the requested one."""
  return float(np.mean(nearest_mode(x) == target_mode))


def coverage(x):
  """How many of the 8 modes received at least 2% of the samples."""
  counts = np.bincount(nearest_mode(x), minlength=MODES)
  return int(np.sum(counts > 0.02 * x.shape[0]))


def main(args):
  jax.set_mesh(jax.make_mesh((jax.device_count(),), ('data',)))
  key = jax.random.key(args.seed)
  key, k_init, k_data = jax.random.split(key, 3)
  params = init(k_init)

  opt = dict(m=jax.tree.map(jnp.zeros_like, params),
             v=jax.tree.map(jnp.zeros_like, params), t=jnp.zeros((), jnp.int32))
  for step in range(args.steps):
    key, k_batch, k_loss = jax.random.split(key, 3)
    x1, label = sample_data(k_batch, args.batch)
    x1 = jax.device_put(x1, jax.P('data'))
    params, opt, l = train_step(params, opt, k_loss, x1, label)
    l = float(l)   # bounds in-flight work; see the note in nanolm.train
    if step % max(1, args.steps // 6) == 0 or step == args.steps - 1:
      print(f'  step {step:5d}  loss {l:.4f}')

  # One vmapped call: unconditional, conditional, and strongly guided.
  ws = jnp.array([0.0, 1.0, 3.0])
  target = jnp.full((args.samples,), args.mode)
  xs = generate_all(params, k_data, args.samples, target, ws)

  print(f'\n{args.samples} samples; conditioning on mode {args.mode}:')
  show(list(xs), [f'w={float(w):g}' + (' (uncond)' if w == 0 else '')
                  for w in ws])
  print(f'\n  w=0: {coverage(xs[0])}/8 modes covered      '
        f'w=1: {100 * hit_rate(xs[1], args.mode):.0f}% on target      '
        f'w=3: {100 * hit_rate(xs[2], args.mode):.0f}% on target')

  if args.check:
    assert coverage(xs[0]) == MODES, 'unconditional samples missed modes'
    assert hit_rate(xs[1], args.mode) > 0.9, 'conditional samples off target'
    assert hit_rate(xs[2], args.mode) > 0.98, 'guided samples off target'
    print('\ncheck: unconditional covers all modes; guidance hits its target')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--devices', type=int, default=util.default_devices(),
                 help='simulated CPU devices; 0 to use real hardware')
  p.add_argument('--steps', type=int, default=3000)
  p.add_argument('--batch', type=int, default=512)
  p.add_argument('--samples', type=int, default=2048)
  p.add_argument('--mode', type=int, default=2, help='mode to condition on')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--check', action='store_true')
  p.add_argument('--offline', action='store_true', help='(never downloads)')
  args = p.parse_args()
  if args.devices:
    jax.config.update('jax_num_cpu_devices', args.devices)
  main(args)
