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

"""Quantization-aware training with a custom array type (hijax).

Demonstrates: hijax types (`HiType`, `VJPHiPrimitive`), tangent types,
straight-through estimators, `jit`, `grad`.

Quantized arrays are the canonical reason to want a *new array type* in JAX:
an int4 tensor and its per-group scales are one logical value, and -- the part
nothing else gets right -- its tangent type is a plain float array. This file
defines that type with `jax.experimental.hijax` and uses it for the thing
that actually needs autodiff: quantization-aware training (QAT).

The scheme is standard, not invented here: symmetric group-wise integer
quantization, one float scale per 64 consecutive weights along the last axis.
MLX's `quantize` uses the same group sizes {32, 64, 128} and bit widths
(affine rather than symmetric); torchao's QAT recipe uses int4 grouped
weights with a straight-through estimator, fine-tuning a pretrained model, as
here. Values stay unpacked int8: QAT *simulates* low precision during
training, and bit-packing belongs to deployment.

The experiment is the standard three-way comparison, swept over bit widths:

  float32    train and evaluate in float32                     (the ceiling)
  PTQ        quantize the trained model, no retraining         (post-training)
  QAT        fine-tune with quantization in the forward pass

and it reproduces the shape of published QAT results: at 4 bits, group-wise
PTQ is nearly free, which is why 4-bit PTQ is what the industry ships; at 3
and especially 2 bits PTQ falls off a cliff, and QAT recovers a large part of
the damage. (`--check` asserts both.) QAT works because of one line on the
type: `to_tangent_aval` says gradients with respect to a quantized value are
`f32`, so with straight-through VJP rules the quantizer differentiates like
the identity and the float master weights keep learning. The end of the file
shows why a pytree cannot express this.

    python examples/quantized.py            # ~30 s on CPU
    python examples/quantized.py --check
"""

import argparse
from dataclasses import dataclass
import functools

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.hijax import (HiType, ShapedArray, VJPHiPrimitive,
                                    register_hitype)

import util

GROUP = 64  # weights per scale, along the last axis


# -- the type -----------------------------------------------------------------

@dataclass
class QArray:
  """Integer values with per-group scales: `bits`-wide ints, stored in int8.

  The components are one logical value -- `qvalue[..., i]` means anything
  only times `scale[..., i // GROUP]`. A hijax type keeps that coupling
  inside the abstraction boundary, and (unlike a pytree) lets us choose the
  type's *tangent* type, which is what makes QAT possible.
  """
  qvalue: jax.Array   # int8[*leading, n], holding values in [-2^(bits-1)+1, ..]
  scale: jax.Array    # f32[*leading, n // GROUP]
  bits: int


@dataclass(frozen=True)
class QArrayTy(HiType):
  shape: tuple[int, ...]
  bits: int
  sharding: object

  # Which array types make up this type, and how values convert -- like the
  # pytree interface, but at the level of types.
  def lo_ty(self):
    s = self.shape[:-1] + (self.shape[-1] // GROUP,)
    return [ShapedArray(self.shape, jnp.dtype('int8'), sharding=self.sharding),
            ShapedArray(s, jnp.dtype('float32'), sharding=self.sharding)]

  def lower_val(self, q):
    return [q.qvalue, q.scale]

  def raise_val(self, qvalue, scale):
    return QArray(qvalue, scale, self.bits)

  # THE LINE THIS FILE IS ABOUT: the tangent of a quantized array is a plain
  # float array. A pytree cannot say this -- a pytree's tangent type is the
  # pytree of its leaves' tangents, and an int8 leaf's tangent is `float0`.
  def to_tangent_aval(self):
    return ShapedArray(self.shape, jnp.dtype('float32'), sharding=self.sharding)

  def str_short(self, short_dtypes=False, mesh_axis_types=False):
    return f'q{self.bits}[{",".join(str(d) for d in self.shape)}]'
  __repr__ = str_short


register_hitype(QArray,
                lambda q: QArrayTy(q.qvalue.shape, q.bits,
                                   jax.typeof(q.qvalue).sharding))


# -- the primitives -----------------------------------------------------------

def _groups(x):
  return x.reshape(*x.shape[:-1], x.shape[-1] // GROUP, GROUP)


class Quantize(VJPHiPrimitive):
  """f32[..., n] -> q{bits}[..., n]: symmetric absmax over groups of 64.

  The VJP is the straight-through estimator: round-and-clip has zero gradient
  almost everywhere, so we differentiate it as if it were the identity.
  """

  def __init__(self, x_aval, bits):
    if x_aval.dtype != jnp.dtype('float32'):
      raise TypeError(f'quantize expects float32, got {x_aval.dtype}')
    if x_aval.shape[-1] % GROUP:
      raise TypeError(f'last axis {x_aval.shape[-1]} not divisible by {GROUP}')
    self.in_avals = (x_aval,)
    self.out_aval = QArrayTy(x_aval.shape, bits, x_aval.sharding)
    self.params = dict(bits=bits)
    super().__init__()

  def expand(self, x):
    qmax = 2 ** (self.params['bits'] - 1) - 1
    g = _groups(x)
    scale = jnp.maximum(jnp.max(jnp.abs(g), axis=-1) / qmax, 1e-8)
    q = jnp.clip(jnp.round(g / scale[..., None]), -qmax, qmax)
    return QArray(q.reshape(x.shape).astype(jnp.int8), scale,
                  self.params['bits'])

  def vjp_fwd(self, nzs_in, x):
    return self(x), None

  def vjp_bwd_retval(self, _res, g):
    return (g,)  # straight through


class Dequantize(VJPHiPrimitive):
  """q{bits}[..., n] -> f32[..., n]."""

  def __init__(self, q_aval):
    self.in_avals = (q_aval,)
    self.out_aval = ShapedArray(q_aval.shape, jnp.dtype('float32'),
                                sharding=q_aval.sharding)
    self.params = {}
    super().__init__()

  def expand(self, qx):
    g = _groups(qx.qvalue.astype('float32')) * qx.scale[..., None]
    return g.reshape(qx.qvalue.shape)

  def vjp_fwd(self, nzs_in, qx):
    return self(qx), None

  def vjp_bwd_retval(self, _res, g):
    return (g,)


def quantize(x, bits):
  return Quantize(jax.typeof(x), bits)(x)


def dequantize(qx):
  return Dequantize(jax.typeof(qx))(qx)


def fake_quant(w, bits):
  """Round-trip through the integer grid: the QAT forward pass, in torchao's
  sense -- simulate quantized numerics in float, differentiate straight
  through."""
  return dequantize(quantize(w, bits))


# -- a small regression model -------------------------------------------------
#
# Two-layer MLP fitting a fixed teacher network. Deliberately small and
# smooth: the subject is the type, not the model. OUT is a multiple of GROUP
# because we quantize along each weight's last axis and w2 is [HIDDEN, OUT].

DIM, HIDDEN, OUT, N_TRAIN = 64, 256, 64, 4096


def init(key):
  k1, k2 = jax.random.split(key)
  return dict(w1=jax.random.normal(k1, (DIM, HIDDEN)) * DIM ** -0.5,
              w2=jax.random.normal(k2, (HIDDEN, OUT)) * HIDDEN ** -0.5)


def predict(params, x, bits=None):
  w1, w2 = params['w1'], params['w2']
  if bits is not None:
    w1, w2 = fake_quant(w1, bits), fake_quant(w2, bits)
  return jnp.tanh(x @ w1) @ w2


def make_data(key, n=N_TRAIN):
  """A fixed teacher labels train and test alike."""
  k1, k2 = jax.random.split(jax.random.key(7))
  teacher = dict(w1=jax.random.normal(k1, (DIM, HIDDEN)) * DIM ** -0.5,
                 w2=jax.random.normal(k2, (HIDDEN, OUT)) * HIDDEN ** -0.5)
  x = jax.random.normal(key, (n, DIM))
  return x, predict(teacher, x)


def mse(params, x, y, bits=None):
  return jnp.mean(jnp.square(predict(params, x, bits) - y))


# The model and optimizer state live in refs (jax.new_ref) and are updated in
# place: no buffers flow out of `train_step`, and there is no donation
# ceremony to get in-place updates. (An earlier version of this file donated
# params instead, and promptly hit the classic footgun: fine-tuning *from*
# the float model donated the very buffers the PTQ baseline still needed.
# With refs, branching a copy is explicit: `jax.new_ref(r[...])`.)

@functools.partial(jax.jit, static_argnums=(4, 5))
def train_step(params, opt, x, y, bits, lr, b1=0.9, b2=0.99, eps=1e-8):
  # With bits set, this is QAT: `grad` differentiates through `fake_quant`
  # straight through, and updates the float32 master weights.
  l, g = jax.value_and_grad(mse)({k: r[...] for k, r in params.items()}, x, y,
                                 bits)
  t = opt['t'][...] + 1
  opt['t'][...] = t
  for k, r in params.items():
    m = opt['m'][k][...] = b1 * opt['m'][k][...] + (1 - b1) * g[k]
    v = opt['v'][k][...] = b2 * opt['v'][k][...] + (1 - b2) * g[k] * g[k]
    r[...] -= lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
  return l


def train(params, x, y, steps, bits=None, lr=3e-3, seed=1):
  """Adam from a copy of `params`. `bits=None` is ordinary training;
  otherwise QAT. Returns plain arrays (`jax.freeze` invalidates the refs)."""
  refs = {k: jax.new_ref(v) for k, v in params.items()}
  opt = dict(m={k: jax.new_ref(jnp.zeros_like(v)) for k, v in params.items()},
             v={k: jax.new_ref(jnp.zeros_like(v)) for k, v in params.items()},
             t=jax.new_ref(jnp.zeros((), jnp.int32)))
  rng = np.random.RandomState(seed)
  for _ in range(steps):
    i = rng.randint(0, x.shape[0], size=256)
    train_step(refs, opt, x[i], y[i], bits, lr)
  return {k: jax.freeze(r) for k, r in refs.items()}


# -- why a pytree can't do this -----------------------------------------------

@jax.tree_util.register_dataclass
@dataclass
class PytreeQArray:
  """The same data as `QArray`, but as a pytree: fine for inference, a dead
  end for training, as `show_pytree_failure` demonstrates."""
  qvalue: jax.Array
  scale: jax.Array


def show_pytree_failure():
  w = jax.random.normal(jax.random.key(0), (4, GROUP))
  g = _groups(w)
  scale = jnp.max(jnp.abs(g), -1) / 7.
  qw = PytreeQArray(
      jnp.round(g / scale[..., None]).reshape(w.shape).astype(jnp.int8), scale)

  def loss(qw):
    deq = _groups(qw.qvalue.astype('float32')) * qw.scale[..., None]
    return jnp.sum(deq.reshape(4, GROUP) ** 2)

  grads = jax.grad(loss, allow_int=True)(qw)
  print('why not a pytree? gradients with respect to one:')
  print(f'  d/d qvalue: {jax.typeof(grads.qvalue)}   <- float0: zero information')
  print(f'  d/d scale:  {jax.typeof(grads.scale)}')
  print("  A pytree's tangent is its leaves' tangents, and an int leaf's")
  print('  tangent is trivial. Only the type itself can declare otherwise.')


# -- driver -------------------------------------------------------------------

def main(args):
  key = jax.random.key(args.seed)
  key, k_data, k_init = jax.random.split(key, 3)
  x, y = make_data(k_data)
  x_test, y_test = make_data(jax.random.key(args.seed + 1), n=1024)

  w = init(k_init)['w1']
  qw = quantize(w, 4)
  grad_ty = jax.jit(jax.grad(lambda w: jnp.sum(fake_quant(w, 4)))).trace(w).out_avals
  print(f'symmetric group-wise quantization, {GROUP} weights per scale:')
  print(f'  weight     {jax.typeof(w)}')
  print(f'  quantized  {jax.typeof(qw)}   '
        f'(qvalue {jax.typeof(qw.qvalue)}, scale {jax.typeof(qw.scale)})')
  print(f'  tangent    {grad_ty}   <- f32, by our to_tangent_aval\n')

  print(f'training float32 for {args.steps} steps...')
  fp = train(init(k_init), x, y, args.steps)
  results = {'float32': (float(mse(fp, x_test, y_test)),) * 2}

  # QAT is *fine-tuning*: it starts from the trained float model, exactly as
  # in torchao's prepare/convert workflow. (Training quantized from scratch
  # with a straight-through estimator is harder, and at these bit widths it
  # loses to PTQ -- try it.)
  for bits in (4, 3, 2):
    ptq = float(mse(fp, x_test, y_test, bits))
    qat_params = train(fp, x, y, args.finetune_steps, bits, lr=3e-4)
    qat = float(mse(qat_params, x_test, y_test, bits))
    results[f'int{bits}'] = (ptq, qat)

  print(f'\ntest MSE (PTQ = quantize after training; '
        f'QAT = fine-tune {args.finetune_steps} steps):')
  print(f'  {"":8s} {"PTQ":>10s} {"QAT":>10s}')
  for name, (ptq, qat) in results.items():
    note = '' if name == 'float32' else f'   QAT recovers {ptq / qat:4.1f}x'
    print(f'  {name:8s} {ptq:10.4f} {qat:10.4f}{note}')
  print('  At 4 bits group-wise PTQ is nearly free -- which is why it is what'
        '\n  everyone ships. QAT earns its keep as the bits get scarce.\n')

  show_pytree_failure()

  if args.check:
    f32 = results['float32'][0]
    assert results['int4'][0] < results['int3'][0] < results['int2'][0], results
    assert results['int2'][1] < 0.7 * results['int2'][0], results
    assert results['int3'][1] < results['int3'][0], results
    assert f32 < results['int4'][0], results
    print('\ncheck: PTQ degrades as bits shrink, and QAT recovers'
          ' (>30% of the int2 damage)')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--devices', type=int, default=util.default_devices(),
                 help='simulated CPU devices; 0 to use real hardware')
  p.add_argument('--steps', type=int, default=3000)
  p.add_argument('--finetune-steps', type=int, default=600)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--check', action='store_true')
  p.add_argument('--offline', action='store_true', help='(never downloads)')
  args = p.parse_args()
  if args.devices:
    jax.config.update('jax_num_cpu_devices', args.devices)
  main(args)
