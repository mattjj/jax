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

"""FSDP with explicit software pipelining, so the collectives can overlap.

Demonstrates: `custom_vjp`, `reduced`/`unreduced` types, `scan` with double
buffering, explicit placement of collectives in a backward pass.

This is the sequel to `nanolm.py`. That file shards the optimizer state but
keeps whole parameters during the forward pass, which is the right default at
small scale. Here we shard the parameters themselves -- FSDP, ZeRO-3 -- so
each device stores only 1/N of every weight and all-gathers a layer's weights
just before using them.

The catch is *overlap*. Writing the all-gather where the weight is used, as
`naive` below does, puts the collective and the matmul that needs it in the
same loop iteration, with a data dependence between them: nothing can hide the
communication. What you want is to issue layer i+1's gather *before* layer i's
matmuls, so the two run concurrently. PyTorch's FSDP does this at runtime by
recording the module order and prefetching from a forward hook; with a `scan`
there are no hooks and no runtime, so we say it in the program instead:

  * the forward carries the *next* layer's gathered weights through the loop,
    which is why the gather appears an iteration early;
  * `unroll=2` gives the double buffer that a one-deep pipeline needs (the
    same thing FSDP2 means when it says a prefetch list of length two or more
    is required for more aggressive overlap, at the cost of memory);
  * a `custom_vjp` does the same for the backward pass, where autodiff would
    otherwise choose the placement for us. `custom_vjp` is JAX's equivalent of
    a backward hook.

Both versions compute the same thing -- `--check` asserts the gradients agree
exactly -- so this file is about *scheduling*, not semantics. It cannot show
you the win: XLA:CPU has no asynchronous collectives, so measuring the overlap
needs a TPU or GPU.

    python examples/fsdp_pipeline.py            # compare the two schedules
    python examples/fsdp_pipeline.py --check    # assert they agree
"""

import argparse
import functools

import numpy as np

import jax
import jax.numpy as jnp

import data
import nanolm
import util
from nanolm import B, D, F, H, L, N, T, V

# Parameters are sharded over 'data' -- this is the FSDP part. Each device
# holds 1/N of every weight and never materializes more than one layer's worth.
# These describe a *single* layer's weights; the stored parameters have a
# leading stacked-over-layers axis, added by `stacked` below.
LAYER_SPECS = dict(
    qkv=jax.P('data', None, None),   # [D, N, 3H]
    proj=jax.P(None, None, 'data'),  # [N, H, D]
    up=jax.P('data', None),          # [D, F]
    down=jax.P(None, 'data'),        # [F, D]
)


def stacked(spec):
  return jax.P(None, *spec)  # the leading axis indexes layers, and is not sharded


FSDP_SPECS = {k: stacked(v) for k, v in LAYER_SPECS.items()}
# The embedding and unembedding stay replicated; the layer stack is where the
# parameters actually are, and where the interesting scheduling lives.
FLAT_SPECS = dict(embed=jax.P(), unemb=jax.P())


def gather(w):
  """Sharded -> replicated: the all-gather.

  The result is typed `reduced={'data'}` rather than plain replicated. Both
  are the same bytes; the difference is that the transpose of this gather is
  then a reduce-scatter we get to place ourselves, instead of an all-reduce
  autodiff inserts on the spot.
  """
  return jax.tree.map(
      lambda x: jax.reshard(x, jax.P(*(None,) * x.ndim, reduced={'data'})), w)


def scatter(w_bar, specs):
  """Unreduced -> sharded: the reduce-scatter, in the backward pass."""
  return jax.tree.map(jax.reshard, w_bar, specs)


def layer(x, w):
  """One transformer layer, exactly nanolm's, over a tuple of weights.

  `ws` is a tuple rather than a dict throughout this file so that the matching
  tuple of `PartitionSpec`s can be a `custom_vjp` `nondiff_argnums` argument,
  which has to be hashable.
  """
  return nanolm.layer(x, dict(zip(nanolm.LAYER_KEYS, w)))[0]


def first(ws, i=0):
  return jax.tree.map(lambda a: a[i], ws)


# -- the two schedules --------------------------------------------------------

def naive(ws, x, specs):
  """Gather each layer's weights exactly where they are used."""
  del specs
  def body(x, w):
    return layer(x, gather(w)), None
  x, _ = jax.lax.scan(body, x, ws)
  return x


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def pipelined(ws, x, specs):
  """Gather layer i+1's weights while layer i's matmuls are running."""
  def body(carry, w_next):
    x, w = carry
    return (layer(x, w), gather(w_next)), None
  # The carry holds the *gathered* weights for the layer we have not run yet:
  # that is the double buffer, and it is why the gather is an iteration early.
  (x, w), _ = jax.lax.scan(body, (x, gather(first(ws))),
                           jax.tree.map(lambda a: a[1:], ws), unroll=2)
  return layer(x, w)


def pipelined_fwd(ws, x, specs):
  # Note what this does *not* do: it never calls `jax.vjp`. It just runs the
  # forward pass and saves each layer's input, and `pipelined_bwd` calls
  # `jax.vjp` fresh from that input. So every layer is recomputed on the
  # backward pass and nothing from the forward is kept alive -- in particular
  # no gathered weight, which is the point, but also no attention
  # intermediates, which for a transformer are the expensive residuals
  # anyway. This is per-layer `jax.remat`, written out.
  #
  # The alternative is to call `jax.vjp` here and keep its residuals, dropping
  # only the gathered weights so they can be re-gathered in the backward pass.
  # That trades memory for recompute and generalizes to layers you would
  # rather not run twice; it needs surgery on the residuals that `jax.vjp`
  # saved. See `tests/pjit_test.py::ShardingInTypesTest::test_fsdp_pipeline_grad`.
  def body(carry, w_next):
    x, w = carry
    return (layer(x, w), gather(w_next)), x
  (x_last, w_last), xs = jax.lax.scan(
      body, (x, gather(first(ws))), jax.tree.map(lambda a: a[1:], ws), unroll=2)
  return layer(x_last, w_last), (ws, xs, x_last)


def pipelined_bwd(specs, res, out_bar):
  ws, xs, x_last = res
  assert L >= 3, 'the pipelined backward pass peels two layers'
  # Last layer, peeled: its weights are gathered and its vjp applied, and the
  # second-to-last layer's gather is issued before the loop starts.
  _, vjp = jax.vjp(layer, x_last, gather(first(ws, -1)))
  x_bar, w_bar = vjp(out_bar)                   # w_bar is unreduced over 'data'

  def body(carry, inputs):
    x_bar, w, w_next_bar = carry
    x_i, w_prev = inputs
    # Gather layer i-1 and reduce-scatter layer i+1's gradient, both while
    # layer i's vjp is running. Two collectives hidden behind one layer.
    _, vjp = jax.vjp(layer, x_i, w)
    x_bar, w_bar = vjp(x_bar)
    return (x_bar, gather(w_prev), w_bar), scatter(w_next_bar, specs)

  (x_bar, w_0, w_1_bar), ws_bar = jax.lax.scan(
      body, (x_bar, gather(first(ws, -2)), w_bar),
      (jax.tree.map(lambda a: a[1:], xs), jax.tree.map(lambda a: a[:-2], ws)),
      reverse=True, unroll=2)

  _, vjp = jax.vjp(layer, first(xs), w_0)
  x_bar, w_0_bar = vjp(x_bar)
  ws_bar = jax.tree.map(
      lambda a, b, c: jnp.concatenate([a[None], b[None], c], axis=0),
      scatter(w_0_bar, specs), scatter(w_1_bar, specs), ws_bar)
  return ws_bar, x_bar


pipelined.defvjp(pipelined_fwd, pipelined_bwd)


# -- model --------------------------------------------------------------------

def init(key):
  specs = {**FSDP_SPECS, **FLAT_SPECS}
  keys = jax.random.split(key, len(nanolm.SHAPES))
  return {k: jax.random.normal(kk, s, out_sharding=specs[k]) * (s[-2] ** -0.5)
          for kk, (k, s) in zip(keys, nanolm.SHAPES.items())}


SPEC_TUPLE = tuple(LAYER_SPECS[k] for k in nanolm.LAYER_KEYS)


def make_loss(stack):
  def loss(params, batch):
    tokens = batch[:, :-1]
    x = params['embed'].at[tokens].get(out_sharding=nanolm.ACTS)
    ws = tuple(params[k] for k in nanolm.LAYER_KEYS)
    x = stack(ws, x, SPEC_TUPLE)
    lg = jnp.einsum('btd,dv->btv', nanolm.rmsnorm(x), params['unemb'],
                    out_sharding=nanolm.ACTS)
    lp = jax.nn.log_softmax(lg)
    return -jnp.mean(jnp.take_along_axis(lp, batch[:, 1:, None], -1))
  return loss


def scan_carry(fn, *args):
  """The types carried across `fn`'s layer loop -- the pipeline, made visible."""
  def find(jaxpr):
    for eqn in jaxpr.eqns:
      if eqn.primitive.name == 'scan':
        return eqn
      for v in eqn.params.values():
        inner = getattr(v, 'jaxpr', v)
        if hasattr(inner, 'eqns') and (found := find(inner)) is not None:
          return found
    return None
  return [str(v.aval) for v in find(jax.jit(fn).trace(*args).jaxpr).outvars]


def main(args):
  jax.set_mesh(jax.make_mesh((args.devices or jax.device_count(),), ('data',)))
  print(f'FSDP over {jax.device_count()} devices, {L} layers')

  key = jax.random.key(args.seed)
  params = init(key)
  print('\n'.join(f'  {k:6s} {jax.typeof(v)}' for k, v in params.items()))

  batch = jax.device_put(
      next(data.batches(data.load(args.offline), B, T, seed=args.seed)
           ).astype(np.int32), jax.P('data', None))

  ws = tuple(params[k] for k in nanolm.LAYER_KEYS)
  x = params['embed'].at[batch[:, :-1]].get(out_sharding=nanolm.ACTS)

  grads = {}
  for name, stack in (('naive', naive), ('pipelined', pipelined)):
    loss = make_loss(stack)
    grads[name] = jax.jit(jax.grad(loss))(params, batch)
    print(f'\n  {name} (loss {jax.jit(loss)(params, batch):.4f}), '
          'values carried across the layer loop:')
    for aval in scan_carry(lambda w, y: stack(w, y, SPEC_TUPLE), ws, x):
      print(f'    {aval}')

  # Exactly equal on more than one device; on a single device the two
  # schedules fuse differently and float32 reassociation shows up.
  for k in grads['naive']:
    a, b = np.asarray(grads['naive'][k]), np.asarray(grads['pipelined'][k])
    err = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
    assert err < 1e-3, f'{k}: relative error {err:.2e}'
  print('\ncheck: the two schedules compute identical gradients')
  print("""
The naive loop carries only the activations: each iteration gathers the
weights it is about to use, and the matmul waits on that gather. The
pipelined loop also carries four `{R:data}` weights -- layer i+1's, gathered
during layer i -- which is the double buffer, and why its gather is off the
critical path.

Note this file does not measure the win, only express it: XLA:CPU has no
asynchronous collectives, so seeing the overlap needs a TPU or GPU.""")


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--devices', type=int, default=util.default_devices(),
                 help='simulated CPU devices; 0 to use real hardware')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--offline', action='store_true')
  p.add_argument('--check', action='store_true',
                 help='(the comparison always runs; this is for the test suite)')
  args = p.parse_args()
  if args.devices:
    jax.config.update('jax_num_cpu_devices', args.devices)
  main(args)
