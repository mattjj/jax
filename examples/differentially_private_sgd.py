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

"""Differentially private SGD: per-example gradients via `vmap`.

Demonstrates: `vmap` of `grad` (per-example gradients), explicit sharding of
the example axis, `jit`.

DP-SGD (Abadi et al., 2016) needs the gradient of *each example* separately:
clip every per-example gradient to l2 norm C, sum, and add Gaussian noise
scaled to C, so no single example can move the model by more than a bounded,
noise-masked amount. In most frameworks per-example gradients mean a loop or
a library extension; in JAX the loop is one word:

    grads = jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))(params, xs, ys)

This is the original JAX DP-SGD example modernized: same clipping norm, noise
multiplier, and batch size as before, but no `jax.example_libraries`, and the
example axis is sharded over 'data', so each device differentiates and clips
exactly the examples it holds -- the summed gradient and the shared noise are
the only cross-device quantities.

Privacy accounting is deliberately not hand-rolled: a wrong epsilon in an
example would be worse than none. If the `dp_accounting` package is
installed, the run reports epsilon; otherwise it says so and carries on.

`--check` asserts every per-example gradient respects the clipping bound and
that the private model still learns.

    python examples/differentially_private_sgd.py            # MNIST, ~2 min
    python examples/differentially_private_sgd.py --offline  # synthetic data
    python examples/differentially_private_sgd.py --check
"""

import argparse
import functools

import numpy as np

import jax
import jax.numpy as jnp

import util

CLIP = 1.0     # per-example gradient l2 bound C
NOISE = 1.1    # noise multiplier sigma: noise stddev is sigma * C
HIDDEN = 256
CLASSES = 10


def init(key, dim):
  sizes = [dim, HIDDEN, CLASSES]
  keys = jax.random.split(key, len(sizes))
  return [(jax.random.normal(k, (m, n)) * m ** -0.5, jnp.zeros(n))
          for k, (m, n) in zip(keys, zip(sizes[:-1], sizes[1:]))]


def logits(params, x):
  for i, (w, b) in enumerate(params):
    x = x @ w + b
    if i < len(params) - 1:
      x = jnp.tanh(x)
  return x


def loss(params, x, y):
  """Loss of ONE example -- `vmap` supplies the batch."""
  return -jax.nn.log_softmax(logits(params, x))[y]


def clip_tree(g, max_norm):
  """Scale a gradient pytree so its global l2 norm is at most `max_norm`."""
  norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(g)))
  return jax.tree.map(lambda x: x * jnp.minimum(1.0, max_norm / norm), g)


@functools.partial(jax.jit, donate_argnums=0)
def private_step(params, key, xs, ys, lr=0.25):
  # THE LINE THIS FILE IS ABOUT: one gradient per example. The example axis
  # of xs/ys is sharded over 'data', so each device differentiates and clips
  # only the examples it holds.
  grads = jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))(params, xs, ys)
  clipped = jax.vmap(clip_tree, in_axes=(0, None))(grads, CLIP)

  # Sum the clipped gradients, add noise calibrated to the clipping bound,
  # and average. The noise is created replicated (`P()`): one shared draw,
  # not one per device.
  n = xs.shape[0]
  leaves, treedef = jax.tree.flatten(clipped)
  keys = jax.random.split(key, len(leaves))
  noisy = [(jnp.sum(g, 0) + NOISE * CLIP * jax.random.normal(
      k, g.shape[1:], out_sharding=jax.P())) / n
           for g, k in zip(leaves, keys)]
  noisy = jax.tree.unflatten(treedef, noisy)
  return jax.tree.map(lambda p, g: p - lr * g, params, noisy), grads


def accuracy(params, xs, ys):
  return float(jnp.mean(jnp.argmax(logits(params, xs), -1) == ys))


def report_epsilon(steps, batch, num_examples, delta=1e-5):
  try:
    import dp_accounting
  except ImportError:
    print('  (install dp_accounting to report the epsilon this run spent)')
    return
  accountant = dp_accounting.rdp.RdpAccountant()
  accountant.compose(dp_accounting.SelfComposedDpEvent(
      dp_accounting.PoissonSampledDpEvent(
          batch / num_examples, dp_accounting.GaussianDpEvent(NOISE)), steps))
  print(f'  epsilon = {accountant.get_epsilon(delta):.2f} at delta = {delta}')


# -- data ---------------------------------------------------------------------

def load(offline, key):
  if not offline:
    try:
      import datasets  # examples/datasets.py, the original MNIST downloader
      train_x, train_y, test_x, test_y = datasets.mnist()
      return (np.asarray(train_x, np.float32), train_y.argmax(-1).astype(np.int32),
              np.asarray(test_x, np.float32), test_y.argmax(-1).astype(np.int32))
    except Exception as e:
      print(f'falling back to synthetic data ({type(e).__name__}: {e})')
  # Offline stand-in: one Gaussian blob per class, 64-dimensional.
  k1, k2, k3 = jax.random.split(key, 3)
  centers = 2.0 * jax.random.normal(k1, (CLASSES, 64))
  y = jax.random.randint(k2, (8192,), 0, CLASSES)
  x = centers[y] + jax.random.normal(k3, (8192, 64))
  x, y = np.asarray(x), np.asarray(y, np.int32)
  return x[:6144], y[:6144], x[6144:], y[6144:]


# -- driver -------------------------------------------------------------------

def main(args):
  jax.set_mesh(jax.make_mesh((jax.device_count(),), ('data',)))
  key = jax.random.key(args.seed)
  key, k_data, k_init = jax.random.split(key, 3)
  train_x, train_y, test_x, test_y = load(args.offline, k_data)
  n = train_x.shape[0]
  print(f'{n} examples of dim {train_x.shape[-1]}, batch {args.batch} '
        f'sharded over {jax.device_count()} devices; '
        f'C={CLIP}, sigma={NOISE}')

  params = init(k_init, train_x.shape[-1])
  rng = np.random.RandomState(args.seed)
  grads = None
  for step in range(args.steps):
    i = rng.randint(0, n, size=args.batch)
    xs = jax.device_put(jnp.asarray(train_x[i]), jax.P('data', None))
    ys = jax.device_put(jnp.asarray(train_y[i]), jax.P('data'))
    key, subkey = jax.random.split(key)
    params, grads = private_step(params, subkey, xs, ys)
    if step % max(1, args.steps // 6) == 0 or step == args.steps - 1:
      acc = accuracy(params, jnp.asarray(test_x), jnp.asarray(test_y))
      print(f'  step {step:4d}  test accuracy {acc:.3f}')

  report_epsilon(args.steps, args.batch, n)

  if args.check:
    acc = accuracy(params, jnp.asarray(test_x), jnp.asarray(test_y))
    assert acc > 0.8, f'private model failed to learn: {acc}'
    clipped = jax.vmap(clip_tree, in_axes=(0, None))(grads, CLIP)
    norms = jax.vmap(lambda g: jnp.sqrt(sum(
        jnp.sum(jnp.square(x)) for x in jax.tree.leaves(g))))(clipped)
    assert float(jnp.max(norms)) <= CLIP * (1 + 1e-5), float(jnp.max(norms))
    print(f'\ncheck: accuracy {acc:.3f} > 0.8, and every per-example '
          f'gradient norm <= {CLIP}')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--devices', type=int, default=util.default_devices(),
                 help='simulated CPU devices; 0 to use real hardware')
  p.add_argument('--batch', type=int, default=256)
  p.add_argument('--steps', type=int, default=300)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--offline', action='store_true', help='synthetic data')
  p.add_argument('--check', action='store_true')
  args = p.parse_args()
  if args.devices:
    jax.config.update('jax_num_cpu_devices', args.devices)
  main(args)
