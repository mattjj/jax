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

"""Training a neural controller through a differentiable simulator.

Demonstrates: `grad` through `scan` (backprop through time), `remat` where it
is measurably load-bearing, `vmap` over goals sharded across devices, `jit`.

A point mass with gravity and quadratic drag must fly to a goal and stop
there. The controller is a small MLP mapping (position, velocity, time, goal)
to thrust; the entire rollout -- hundreds of physics steps, each calling the
network -- is differentiated end to end, and Adam trains the policy weights
directly on "final distance to goal plus fuel". No RL machinery: the
simulator is just a function, so `jax.grad` goes through it.

This is also the file where `jax.remat` is not decorative. Differentiating a
`scan` saves each iteration's intermediates for the backward pass -- here,
the MLP activations, stacked over every timestep. Wrapping the step in
`jax.remat` keeps only the carry (the physics state) and recomputes the rest,
which is the classic backprop-through-time memory trade. The program prints
XLA's measured temporary-memory footprint both ways rather than asserting it
in a comment; at this toy scale it is already an ~18x difference, and it scales
with rollout length times network width.

`--check` asserts the trained controller reaches its goals and that remat
does not change the gradients.

    python examples/diffsim.py            # ~1 min on CPU
    python examples/diffsim.py --check
"""

import argparse
import functools

import numpy as np

import jax
import jax.numpy as jnp

import util

DT = 0.02          # integrator step
STEPS = 500        # rollout length: 10 seconds of simulated time
HIDDEN = 64
GRAVITY = np.array([0.0, -1.0])  # numpy: module level runs before --devices applies
DRAG = 0.1
FUEL = 3e-3        # weight of the fuel (control effort) penalty


def goals(n):
  """Goals arranged on a half-circle above the launch point."""
  th = jnp.pi * (jnp.arange(n) + 0.5) / n
  return 2.0 * jnp.stack([jnp.cos(th), jnp.sin(th)], -1)  # [n, 2]


# -- the controller -----------------------------------------------------------

def init(key):
  sizes = [7, HIDDEN, HIDDEN, 2]  # pos(2) vel(2) t(1) goal(2) -> thrust(2)
  keys = jax.random.split(key, len(sizes))
  return [(jax.random.normal(k, (m, n)) * m ** -0.5, jnp.zeros(n))
          for k, (m, n) in zip(keys, zip(sizes[:-1], sizes[1:]))]


def policy(params, pos, vel, t, goal):
  h = jnp.concatenate([pos, vel, t[None], goal])
  for i, (w, b) in enumerate(params):
    h = h @ w + b
    if i < len(params) - 1:
      h = jnp.tanh(h)
  return 2.0 * jnp.tanh(h)  # thrust, componentwise in [-2, 2]


# -- the simulator ------------------------------------------------------------

def step(carry, t, params, goal):
  """One physics step. All the memory in the backward pass comes from here:
  the MLP activations inside `policy` are what a grad-of-scan saves per
  iteration, and what `jax.remat` chooses to recompute instead.

  The per-step fuel cost is emitted as a scan output and summed afterwards
  rather than accumulated in the carry: a scalar accumulator in the carry of
  a grad-of-vmap-of-scan over a sharded axis currently trips a mesh-tracking
  bug (FINDINGS.md entry 9).
  """
  pos, vel = carry
  u = policy(params, pos, vel, t, goal)
  vel += DT * (u + GRAVITY - DRAG * jnp.linalg.norm(vel) * vel)
  pos += DT * vel
  return (pos, vel), (pos, DT * jnp.sum(u ** 2))


def rollout(params, goal, use_remat):
  """Simulate one trajectory for one goal. `vmap` supplies the batch."""
  body = functools.partial(step, params=params, goal=goal)
  if use_remat:
    body = jax.remat(body)
  ts = jnp.arange(STEPS, dtype=jnp.float32) / STEPS
  (pos, vel), (path, cost) = jax.lax.scan(body, (jnp.zeros(2), jnp.zeros(2)), ts)
  return pos, vel, jnp.sum(cost), path


def loss(params, gs, use_remat):
  # One rollout per goal; the goal axis is sharded over 'data', so each
  # device simulates -- and backpropagates -- only its own trajectories.
  pos, vel, fuel, _ = jax.vmap(rollout, in_axes=(None, 0, None))(
      params, gs, use_remat)
  final_err = jnp.sum((pos - gs) ** 2, -1) + 0.1 * jnp.sum(vel ** 2, -1)
  return jnp.mean(final_err + FUEL * fuel)


@functools.partial(jax.jit, static_argnums=3, donate_argnums=(0, 1))
def train_step(params, opt, gs, use_remat, lr=1e-2, b1=0.9, b2=0.99, eps=1e-8):
  l, g = jax.value_and_grad(loss)(params, gs, use_remat)
  t = opt['t'] + 1
  m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt['m'], g)
  v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, opt['v'], g)
  params = jax.tree.map(
      lambda p, m, v: p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps),
      params, m, v)
  return params, dict(m=m, v=v, t=t), l


def temp_bytes(params, gs, use_remat):
  """XLA's measured scratch memory for one gradient computation."""
  compiled = jax.jit(jax.grad(loss), static_argnums=2).lower(
      params, gs, use_remat).compile()
  return compiled.memory_analysis().temp_size_in_bytes


# -- terminal rendering -------------------------------------------------------

def ascii_paths(paths, gs, w=61, h=21, xr=(-3, 3), yr=(-0.6, 2.8)):
  grid = np.full((h, w), ' ')
  for i, path in enumerate(np.asarray(paths)):
    for x, y in path[::4]:
      c = int((x - xr[0]) / (xr[1] - xr[0]) * w)
      r = int((y - yr[0]) / (yr[1] - yr[0]) * h)
      if 0 <= c < w and 0 <= r < h:
        grid[r, c] = chr(ord('1') + i)
  for i, (x, y) in enumerate(np.asarray(gs)):
    c = int((x - xr[0]) / (xr[1] - xr[0]) * w)
    r = int((y - yr[0]) / (yr[1] - yr[0]) * h)
    grid[r, c] = 'X'
  for row in grid[::-1]:
    print('  ' + ''.join(row))


# -- driver -------------------------------------------------------------------

def main(args):
  jax.set_mesh(jax.make_mesh((jax.device_count(),), ('data',)))
  gs = jax.device_put(goals(args.goals), jax.P('data', None))
  print(f'{args.goals} goals over {jax.device_count()} devices, '
        f'{STEPS}-step rollouts')

  key = jax.random.key(args.seed)
  params = init(key)

  naive, remat = temp_bytes(params, gs, False), temp_bytes(params, gs, True)
  print(f'  backward-pass scratch memory, measured by XLA:\n'
        f'    without remat  {naive / 1e6:8.2f} MB   '
        f'(activations saved for all {STEPS} steps)\n'
        f'    with remat     {remat / 1e6:8.2f} MB   '
        f'({naive / remat:.1f}x less: carry only, rest recomputed)\n')

  opt = dict(m=jax.tree.map(jnp.zeros_like, params),
             v=jax.tree.map(jnp.zeros_like, params), t=jnp.zeros((), jnp.int32))
  for it in range(args.iters):
    params, opt, l = train_step(params, opt, gs, True)
    l = float(l)   # bounds in-flight work; see the note in nanolm.train
    if it % max(1, args.iters // 6) == 0 or it == args.iters - 1:
      print(f'  iter {it:4d}  loss {l:.4f}')

  pos, vel, fuel, paths = jax.jit(
      jax.vmap(rollout, in_axes=(None, 0, None)), static_argnums=2)(
          params, gs, True)
  dist = np.linalg.norm(np.asarray(pos - gs), axis=-1)
  print(f'\n  trajectories (goals marked X), '
        f'mean final distance {dist.mean():.3f}:')
  ascii_paths(paths, gs)

  if args.check:
    assert dist.mean() < 0.1, f'controller misses its goals: {dist}'
    g_naive = jax.jit(jax.grad(loss), static_argnums=2)(params, gs, False)
    g_remat = jax.jit(jax.grad(loss), static_argnums=2)(params, gs, True)
    for (a, _), (b, _) in zip(g_naive, g_remat):
      np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)
    print('\ncheck: goals reached; remat leaves the gradients unchanged')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--devices', type=int, default=util.default_devices(),
                 help='simulated CPU devices; 0 to use real hardware')
  p.add_argument('--goals', type=int, default=8)
  p.add_argument('--iters', type=int, default=500)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--check', action='store_true')
  p.add_argument('--offline', action='store_true', help='(never downloads)')
  args = p.parse_args()
  if args.devices:
    jax.config.update('jax_num_cpu_devices', args.devices)
  main(args)
