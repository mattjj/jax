import itertools
import operator

import jax
import jax.numpy as jnp

from functools import partial
from typing import Sequence, Any, Dict, Union, Tuple, List
from jax import core
from jax.util import safe_map, safe_zip, unzip2

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

Val = Any
Scale = Any
IsQDQ = bool

# Rule: track which values have been qdq'd. Whenever we see a dot (maybe: with
# at least one qdq'd input), ensure all its inputs and its output (?) are qdq'd.
# For each qdq we introduce, we need to plumb in and out a scale.


def quantize(fun, args_to_quantize: Sequence[bool], example_args: Sequence[Any]):
  jaxpr = jax.make_jaxpr(fun)(*example_args)
  num_scales = count_scales(jaxpr, args_to_quantize)
  def fun_quantized(scales, *args):
    old_scales = list(scales)
    new_scales = []
    already_qdq = map(operator.not_, args_to_quantize)
    args = map(partial(maybe_qdq, new_scales, old_scales), already_qdq, args)
    _, out_vals = quantize_interpreter(
        jaxpr, args_to_quantize, old_scales, new_scales, *args)
    assert not old_scales and len(scales) == len(new_scales)
    return out_vals, new_scales
  return fun_quantized, num_scales

def count_scales(jaxpr: core.ClosedJaxpr, args_to_quantize: Sequence[bool]
                 ) -> int:
  class Dummy:
    def pop(self, _: int = -1):
      return jnp.float32(1.)
  cell = []
  def count(*args):
    new_scales = []
    quantize_interpreter(jaxpr, args_to_quantize, Dummy(), new_scales, *args)
    cell.append(len(new_scales))

  jax.make_jaxpr(count)(*jaxpr.in_avals)
  num_scales, = cell
  return num_scales + sum(args_to_quantize)

def qiao_quantize(x, quantized_dtype, scale):
  dtype_max = 57344
  scaled_x = jnp.clip(x / scale, -dtype_max, dtype_max)
  return scaled_x.astype(quantized_dtype)

def qiao_dequantize(x, wide_dtype, scale):
  return x.astype(wide_dtype) * scale

def qiao_quantize_dequantize(quantized_dtype, x, scale):
  orig_dtype = x.dtype
  qx = qiao_quantize(x, quantized_dtype, scale)
  return qiao_dequantize(qx, orig_dtype, scale)

def qiao_compute_new_scale(quantized_dtype, x, scale):
  dtype_max = get_fp8_max(quantized_dtype)
  amax = jnp.max(jnp.abs(x)).astype(jnp.result_type(scale))
  # Ensure scale != 0 and avoid divide-by-zero.
  amax = jnp.maximum(amax, 2**-10)
  return 1.1 * amax / dtype_max

def qiao_qdq_and_new_scale(dtype, x, scale):
  qx = qiao_quantize_dequantize(dtype, x, scale)
  new_scale = qiao_compute_new_scale(dtype, x, scale)
  return qx, new_scale

def get_fp8_max(fake_dtype):
  if fake_dtype == FAKE_E4M3:
    return E4M3_MAX
  elif fake_dtype == FAKE_E5M2:
    return E5M2_MAX
  else:
    raise ValueError('Only FAKE_E4M3 and FAKE_E5M2 supported')

FAKE_E4M3 = jnp.float16
FAKE_E5M2 = jnp.bfloat16
E4M3_MAX = 448
E5M2_MAX = 57344

# jax.jit
# def qdq(x, scale):
#   return qiao_qdq_and_new_scale(FAKE_E4M3, x, scale)

def qdq(x, scale):
  return qdq_p.bind(x, scale, dtype=FAKE_E4M3)
qdq_p = jax.core.Primitive('qdq')
qdq_p.multiple_results = True
@qdq_p.def_abstract_eval
def _qdq_abstract_eval(x_aval, scale_aval, *, dtype):
  return x_aval, scale_aval
@qdq_p.def_impl
def _qdq_impl(x, scale, *, dtype):
  return qiao_qdq_and_new_scale(dtype, x, scale)

def maybe_qdq(new_scales, old_scales, already_qdq, x):
  if already_qdq:
    return x
  else:
    x, new_scale = qdq(x, old_scales.pop(0))
    new_scales.append(new_scale)
    return x

def quantize_interpreter(
    jaxpr: core.ClosedJaxpr,
    args_already_quantized: Sequence[bool],
    old_scales: List[Scale],
    new_scales: List[Scale],
    *args: Val
  ) -> Tuple[Sequence[bool], Sequence[Val]]:
  env: Dict[core.Var, Tuple[IsQDQ, Val]] = {}

  def read(x: Union[core.Var, core.Literal]) -> Tuple[IsQDQ, Val]:
    if isinstance(x, core.Literal):
      return (False, x.val)
    else:
      assert isinstance(x, core.Var)
      return env[x]

  def write(is_qdq: bool, v: core.Var, val: Val) -> None:
    env[v] = (is_qdq, val)

  maybe_qdq_ = partial(maybe_qdq, new_scales, old_scales)

  jaxpr, consts = jaxpr.jaxpr, jaxpr.consts
  map(partial(write, False), jaxpr.constvars, consts)
  map(write, args_already_quantized, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_qdq, in_vals = unzip2(map(read, eqn.invars))
    if str(eqn.primitive) == 'dot_general':
      lhs, rhs = in_vals
      lhs_qdq, rhs_qdq = in_qdq
      lhs = maybe_qdq_(lhs_qdq, lhs)
      rhs = maybe_qdq_(rhs_qdq, rhs)
      out = eqn.primitive.bind(lhs, rhs, **eqn.params)  # same dot_general
      out = maybe_qdq_(False, out)
      out_vals = [out]
      out_qdq = [True]
    elif str(eqn.primitive) == 'xla_call':
      # inline inner jits (by just recursively calling quantize_interpreter)
      inner_jaxpr = eqn.params['call_jaxpr']
      out_qdq, out_vals = quantize_interpreter(
          core.ClosedJaxpr(inner_jaxpr, ()), in_qdq, old_scales, new_scales,
          *in_vals)
    else:
      # don't transform any other primitive application (for now...)
      out_vals = eqn.primitive.bind(*in_vals, **eqn.params)
      if not eqn.primitive.multiple_results:
        out_vals = [out_vals]
      out_qdq = [False] * len(out_vals)
    map(write, out_qdq, eqn.outvars, out_vals)
  quantized_outputs, out_vals = unzip2(map(read, jaxpr.outvars))
  return quantized_outputs, out_vals



###

import ipdb, sys, traceback
def info(type, value, tb):
  traceback.print_exception(type, value, tb)
  ipdb.pm()
sys.excepthook = info


def f(weight, net):
  net = jnp.einsum('bm,mn->bn', net, weight)
  return net

weight = jax.random.normal(jax.random.PRNGKey(42), (10, 20))
inputs = jax.random.normal(jax.random.PRNGKey(4), (128, 10))

jaxpr = jax.make_jaxpr(f)(weight, inputs)
print(jaxpr)
print('===')
qf, num_scales = quantize(f, [False, True], example_args=(weight, inputs))
init_scales = [1.] * num_scales
qout, new_scales = qf(init_scales, weight, inputs)

jaxpr = jax.make_jaxpr(qf)(init_scales, weight, inputs)
print(jaxpr)


# TODO
#  * [x] don't hardcode the number of scales, instead do a second pass
#  * [ ] optimization: if a value is consumed more than once, it might get
#        quantized more than once
