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

"""Hamiltonian Monte Carlo in ~50 lines of sampler.

Demonstrates: `grad` (inside the integrator), `vmap` (over chains), `scan`
(over leapfrog steps and over iterations), `jit`, sharding chains over
devices.

HMC is the algorithm JAX's core trio was made for. The leapfrog integrator
needs the gradient of the log-density at every step: that is `jax.grad`,
applied to the same function we'd write anyway. Chains are independent: that
is `jax.vmap`, and because the chain axis is sharded over the 'data' mesh
axis, the chains also run on separate devices without further ceremony. The
integrator and the iteration loop are both `jax.lax.scan`.

The target is a banana-shaped distribution -- a Gaussian pushed through a
polynomial warp -- chosen because it is curved enough to be worth a gradient
sampler and its moments are known exactly:

    x1 = z1,  x2 = z2 + b * (z1**2 - s2)   with z ~ N(0, diag(s2, 1))

so E[x] = 0, Var(x1) = s2, Var(x2) = 1 + 2 b^2 s2^2, Cov(x1, x2) = 0.
`--check` compares the sampler's moments against those closed forms.

    python examples/hmc.py            # a few seconds on CPU
    python examples/hmc.py --check
"""

import argparse

import numpy as np

import jax
import jax.numpy as jnp

import util

B_WARP = 0.5   # banana curvature
S2 = 2.0       # variance of z1
DIM = 2

TRUE_MEAN = np.zeros(DIM)
TRUE_VAR = np.array([S2, 1 + 2 * B_WARP ** 2 * S2 ** 2])


def logp(x):
  """Log-density of the banana, up to a constant."""
  z1 = x[0]
  z2 = x[1] - B_WARP * (x[0] ** 2 - S2)
  return -0.5 * (z1 ** 2 / S2 + z2 ** 2)


# -- the sampler --------------------------------------------------------------

def leapfrog(x, p, eps, steps):
  """Integrate Hamiltonian dynamics: `steps` position/momentum updates."""
  grad_logp = jax.grad(logp)

  def step(carry, _):
    x, p = carry
    p = p + 0.5 * eps * grad_logp(x)
    x = x + eps * p
    p = p + 0.5 * eps * grad_logp(x)
    return (x, p), None

  (x, p), _ = jax.lax.scan(step, (x, p), None, length=steps)
  return x, p


def hmc_step(x, key, eps, steps):
  """One proposal + Metropolis correction, for ONE chain."""
  k_mom, k_acc = jax.random.split(key)
  p = jax.random.normal(k_mom, x.shape)
  x_new, p_new = leapfrog(x, p, eps, steps)
  # Accept with probability exp(H(x, p) - H(x', p')); H = -logp + |p|^2 / 2.
  log_accept = (logp(x_new) - 0.5 * jnp.sum(p_new ** 2)
                - logp(x) + 0.5 * jnp.sum(p ** 2))
  accept = jnp.log(jax.random.uniform(k_acc)) < log_accept
  return jnp.where(accept, x_new, x), accept


def sample(key, chains, iters, eps=0.5, steps=8):
  """All chains, all iterations: `vmap` inside, `scan` outside.

  The step size is tuned for this target: it gives ~0.9 acceptance, in the
  healthy range. (Much closer to 1.0 means the steps are too timid and the
  chain is wasting gradient evaluations -- an earlier eps=0.25 accepted 99%.)
  """
  k_init, k_iter = jax.random.split(key)
  # The chain axis is sharded over 'data': every per-chain computation below
  # -- integrator, gradients, accept -- runs where its chain lives.
  x = jax.random.normal(k_init, (chains, DIM), out_sharding=jax.P('data', None))

  def iteration(x, key):
    # One key per chain, sharded like the chains: `vmap` over a sharded axis
    # requires every mapped input to shard that axis the same way.
    keys = jax.reshard(jax.random.split(key, x.shape[0]), jax.P('data'))
    x, accept = jax.vmap(hmc_step, in_axes=(0, 0, None, None))(
        x, keys, eps, steps)
    return x, (x, accept)

  _, (xs, accepts) = jax.lax.scan(iteration, x,
                                  jax.random.split(k_iter, iters))
  return xs, accepts  # [iters, chains, DIM], [iters, chains]


# -- terminal rendering -------------------------------------------------------

def ascii_scatter(pts, w=57, h=19, xr=(-4.5, 4.5), yr=(-3, 6)):
  grid = np.zeros((h, w), int)
  pts = np.asarray(pts)
  i = ((pts[:, 0] - xr[0]) / (xr[1] - xr[0]) * w).astype(int)
  j = ((pts[:, 1] - yr[0]) / (yr[1] - yr[0]) * h).astype(int)
  ok = (i >= 0) & (i < w) & (j >= 0) & (j < h)
  np.add.at(grid, (j[ok], i[ok]), 1)
  chars = np.array(list(' .:+*#'))
  lvl = np.clip((np.log1p(grid) * 1.7).astype(int), 0, len(chars) - 1)
  for row in chars[lvl][::-1]:
    print('  ' + ''.join(row))


# -- driver -------------------------------------------------------------------

def main(args):
  jax.set_mesh(jax.make_mesh((jax.device_count(),), ('data',)))
  assert args.chains % jax.device_count() == 0, (
      f'{args.chains} chains do not divide over {jax.device_count()} devices')
  print(f'{args.chains} chains over {jax.device_count()} devices, '
        f'{args.iters} iterations each')

  xs, accepts = jax.jit(sample, static_argnums=(1, 2))(
      jax.random.key(args.seed), args.chains, args.iters)
  # Discard warmup, pool chains and iterations.
  keep = np.asarray(xs[args.iters // 4:]).reshape(-1, DIM)
  accept_rate = float(jnp.mean(accepts))

  print(f'  acceptance rate {accept_rate:.2f}, '
        f'{keep.shape[0]} samples after warmup\n')
  ascii_scatter(keep)

  est_mean, est_var = keep.mean(0), keep.var(0)
  print(f'\n  {"":8s} {"mean(x1)":>9s} {"mean(x2)":>9s} '
        f'{"var(x1)":>9s} {"var(x2)":>9s}')
  print(f'  {"exact":8s} {TRUE_MEAN[0]:9.3f} {TRUE_MEAN[1]:9.3f} '
        f'{TRUE_VAR[0]:9.3f} {TRUE_VAR[1]:9.3f}')
  print(f'  {"sampled":8s} {est_mean[0]:9.3f} {est_mean[1]:9.3f} '
        f'{est_var[0]:9.3f} {est_var[1]:9.3f}')

  if args.check:
    assert 0.5 < accept_rate <= 1.0, f'acceptance {accept_rate}'
    np.testing.assert_allclose(est_mean, TRUE_MEAN, atol=0.15)
    np.testing.assert_allclose(est_var, TRUE_VAR, rtol=0.2)
    print('\ncheck: sampled moments match the closed-form moments')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--devices', type=int, default=util.default_devices(),
                 help='simulated CPU devices; 0 to use real hardware')
  p.add_argument('--chains', type=int, default=None,
                 help='default: 8 per device')
  p.add_argument('--iters', type=int, default=1000)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--check', action='store_true')
  p.add_argument('--offline', action='store_true', help='(never downloads)')
  args = p.parse_args()
  if args.devices:
    jax.config.update('jax_num_cpu_devices', args.devices)
  if args.chains is None:
    args.chains = 8 * (args.devices or jax.device_count())
  main(args)
