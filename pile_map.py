from __future__ import annotations

import pdb, sys, traceback
def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    pdb.pm()
sys.excepthook = info



import collections
import dataclasses
from functools import partial
import itertools as it
import string
from typing import Union, Any, Optional

import jax
import jax.numpy as jnp
from jax import core
from jax.core import Var, Tracer

from jax._src.util import safe_map as map, unzip2

Array = Any

names = (''.join(chars) for n in it.count(1)
         for chars in it.product(string.ascii_lowercase, repeat=n))
var_names = collections.defaultdict(names.__next__)

## Pile data type

# AbstractPile i:n f32[ks.i, ks.i]
# AbstractPile i:n f32[ks.i, ls.i]

@dataclasses.dataclass(frozen=True)
class AbstractPile:
  binder: Var
  length: Union[int, Tracer, Var]
  elt_ty: core.DShapedArray
  def __repr__(self) -> str:
    return f'{var_names[self.binder]}:{self.length} => {self.elt_ty}'
  # def is_indep(self):
  #   return any(

@dataclasses.dataclass(frozen=True)
class Pile:
  aval: AbstractPile
  data: Array  # concatenated!

@dataclasses.dataclass(frozen=True)
class IndexedAxisSize:
  idx: Var
  lengths: Union[Array, Var, Tracer]
  def __repr__(self) -> str:
    return f'{str(self.lengths)}.{var_names[self.idx]}'

def make_pile(aval: AbstractPile, data: List[Array]):
  # aval = AbstractPile i:3 f32[lengths.i]
  # data = [jnp.arange(3), jnp.arange(1), jnp.arange(4)]
  for i, x in enumerate(data):
    expected_aval = subs(aval.elt_ty, aval.binder, i)
    core.typecompat(expected_aval, x)
  axis, = (i for i, d in enumerate(aval.elt_ty.shape)
           if isinstance(d, IndexedAxisSize) and d.idx is aval.binder)
  concat_data = jnp.concatenate(data, axis)
  return Pile(elt_ty, concat_data)

def subs(aval: core.DShapedArray, binder: Var, i: Union[int, Tracer]
         ) -> core.DShapedArray:
  new_shape = []
  for d in aval.shape:
    if isinstance(d, IndexedAxisSize) and d.idx is binder:
      new_shape.append(d.lengths[i])
    elif isinstance(d, int):
      new_shape.append(d)
    else:
      raise NotImplementedError  # TODO(apaszke)
  return aval.update(shape=tuple(new_shape))

##

def idx_var():
  return Var(0, '', core.ShapedArray((), jnp.dtype('int32')))

i = idx_var()
elt_ty = core.DShapedArray((IndexedAxisSize(i, jnp.array([3, 1, 4])),),
                           jnp.dtype('float32'), False)
aval = AbstractPile(i, 3, elt_ty)
print(aval)

pile = make_pile(aval, [jnp.arange(3.), jnp.arange(1.), jnp.arange(4.)])
print(pile)

###

import jax.linear_util as lu

def pile_map(f, x):
  assert isinstance(x, Pile)
  axis_size = x.aval.length
  out, = _pile_map(lu.wrap_init(f), axis_size).call_wrapped(x)
  return out

class PileMapTrace(core.Trace):
  def pure(self, val):
    return PileMapTracer(self, val, False)

  def lift(self, val):
    return PileMapTracer(self, val, False)

  def sublift(self, tracer):
    return PileMapTracer(self, tracer.pile, tracer.batched)

  def process_primitive(self, primitive, tracers, params):
    breakpoint()

class PileMapTracer(core.Tracer):
  val: Union[Pile, Array]
  batched: bool

  def __init__(self, trace, val, batched):
    self._trace = trace
    self.val = val
    self.batched = batched

  @property
  def aval(self):
    aval = core.get_aval(self.val)
    if not self.batched:
      return aval
    binder = aval.binder

    # base_aval = PileTy i:3 f32[lengths.i] where lengths = [3,1,4]
    # return = f32[n] where n is the Tracer result of lengths[i] (and i is a Tracer)
    # i = BatchTracer(iota(3), bdim=0)
    # n = BatchTracer(lengths, bdim=0)
    # TODO substitute

# @lu.transformation
# def _pile_map(f: lu.WrappedFun, axis_size, x):
#   with core.new_main(PileMapTrace, axis_size=axis_size) as main:
#     trace = main.with_cur_sublevel()
#     in_tracers = [PileMapTracer(trace, ...) ...]


###

from jax.interpreters import batching
jax.config.update('jax_dynamic_shapes', True)

# really a 'concat or stack axis', where None means stacked
@dataclasses.dataclass(frozen=True)
class ConcatAxis:
  axis: int
  segment_lengths: Optional[Array]  # array of ints, None means 1s + squeeze


def pile_map(jaxpr: jax.core.ClosedJaxpr, pile: Pile) -> Pile:
  jaxpr, () = jaxpr.jaxpr, jaxpr.consts
  env: Dict[core.Var, tuple[Any, Optional[ConcatAxis]]] = {}

  def read(x):
    return env[x] if type(x) is core.Var else (x.val, None)

  def write(x, val, ax):
    env[x] = (val, ax)

  # TODO remove hardcoding
  assert pile.data.ndim == 1
  lengths = pile.aval.shape[0].lengths
  write(jaxpr.invars[0], lengths, ConcatAxis(0, None))
  write(jaxpr.invars[1], pile.data, ConcatAxis(0, lengths))

  for eqn in jaxpr.eqns:
    # x:f32[m] y:f32[n]
    # can always recover ConcatAxis from types, and env? need also batched
    # indicator
    in_vals, in_axes = unzip2(map(read, eqn.invars))
    if all(c is None for c in in_axes):
      out_vals = eqn.primitive.bind(*in_vals, **eqn.params)
      if not eqn.primitive.multiple_results:
        out_vals = [out_vals]
      out_axes = [None] * len(out_vals)
    elif (rule := dependent_primitive_rules.get(eqn.primitive)):
      out_vals, out_axes = rule(in_vals, in_axes, **eqn.params)
      if not eqn.primitive.multiple_results:
        out_vals, out_axes = [out_vals], [out_axes]
    elif all(c is None or c.segment_lengths is None for c in in_axes):
      in_axes_ = [c and c.axis for c in in_axes]
      rule = batching.primitive_batchers[eqn.primitive]
      out_vals, out_axes_ = rule(in_vals, in_axes_, **eqn.params)
      if not eqn.primitive.multiple_results:
        out_vals, out_axes_ = [out_vals], [out_axes_]
      out_axes = [None if a is None else ConcatAxis(a, None) for a in out_axes_]
    else:
      # rules can assume at least one operand is concat (not stacked), b/c
      # otherwise we would've handled with vmap
      rule = pile_map_rules[eqn.primitive]
      out_vals, out_axes = rule(in_vals, in_axes, **eqn.params)
      if not eqn.primitive.multiple_results:
        out_vals, out_axes = [out_vals], [out_axes]
    map(write, eqn.outvars, out_vals, out_axes)

  (out_val,), (out_axis,) = unzip2(map(read, eqn.outvars))
  if out_axis is None:
    breakpoint()  # TODO broadcast
  elif out_axis.segment_lengths is None:
    return batching.bdim_at_front(out_val, out_axis.axis, len(lengths))
  else:
    breakpoint()  # form a Pile

pile_map_rules = {}
dependent_primitive_rules = {}

from jax._src.lax import lax
from jax.ops import segment_sum

def reduce_sum_rule(in_vals, in_axes: Sequence[ConcatAxis], *,
                    axes: Sequence[int]):
  x, = in_vals
  c, = in_axes
  a, = axes  # TODO
  if c.axis == a:
    segment_lengths = c.segment_lengths
    segment_ids = jnp.repeat(jnp.arange(len(segment_lengths)), segment_lengths)
    out = segment_sum(x, segment_ids, num_segments=len(segment_lengths))
    out_axis = ConcatAxis(c.axis, None)
    return out, out_axis
  else:
    out = lax.reduce_sum_p.bind(x, axes=axes)
    out_axis = ConcatAxis(c.axis - (a <= c.axis), c.segment_lengths)
    return out, out_axis
pile_map_rules[lax.reduce_sum_p] = reduce_sum_rule

def eltwise_rule(prim, in_vals, in_axes, **params):
  # there's at least one with nontrivial segment lengths
  assert not all(c is None or c.segment_lengths is None for c in in_axes)
  # TODO don't compute sum of segment lengths, look up the value!
  # look up the aval of the variable which has nontrivial segment lengths
  segment_lengths = next(c.segment_lengths for c in in_axes
                         if c is not None and c.segment_lengths is not None)

  concat_axis, = {c.axis for c in in_axes if c is not None and
                  c.segment_lengths is not None}
  result_rank = max(x.ndim for x in in_vals)

  # normalize arguments
  in_vals_ = []
  for x, c in zip(in_vals, in_axes):
    if c is None:
      in_vals_.append(batching.broadcast(x, sum(segment_lenghts), 0))
    elif c.segment_lengths is None:
      assert c.axis == 0 and x.ndim == 1  # only happens with stacked scalar operand
      x = jnp.repeat(x, segment_lenghts)
      idx = [None] * result_rank
      idx[concat_axis] = slice(None)
      in_vals_.append(x[tuple(idx)])
    else:
      in_vals_.append(x)  # concat axes are consistent, per above assertion
  del in_axes  # all are now effectively ConcatAxis(0, segment_lengths)

  out_vals = prim.bind(*in_vals, **params)
  out_axes = ConcatAxis(0, segment_lengths)
  return out_vals, out_axes
pile_map_rules[lax.add_p] = partial(eltwise_rule, lax.add_p)
pile_map_rules[lax.cos_p] = partial(eltwise_rule, lax.cos_p)
pile_map_rules[lax.integer_pow_p] = partial(eltwise_rule, lax.integer_pow_p)


def bcast_in_dim_rule(vals, dims, *, shape, broadcast_dimensions):
  operand, *dyn_shape = vals
  operand_dim, *dyn_shape_dims = dims
  if not dyn_shape or all(d is None for d in dyn_shape_dims):
    raise NotImplementedError  # call batching rule
  if len(dyn_shape) > 1:
    raise NotImplementedError
  if broadcast_dimensions:
    raise NotImplementedError  # only broadcast scalars for now
  size, = dyn_shape
  size_dim, = dyn_shape_dims
  assert size_dim.segment_lengths is None  # had to be scalar sizes in jaxpr
  dst_dim, = (i for i, d in enumerate(shape) if d is None)
  data = jnp.repeat(operand, size, axis=operand_dim.axis)
  data = jnp.moveaxis(data, operand_dim.axis, dst_dim) # concat axis pos matters
  return data, ConcatAxis(dst_dim, size)
dependent_primitive_rules[lax.broadcast_in_dim_p] = bcast_in_dim_rule



def f(x):
  y = x + x.shape[0]
  return (jnp.cos(y) ** 2).sum()

jaxpr = jax.make_jaxpr(f, abstracted_axes=('n',))(jnp.arange(3.))
print(jaxpr)

pile = make_pile(aval, [jnp.arange(3.), jnp.arange(1.), jnp.arange(4.)])
y = pile_map(jaxpr, pile)
print(y)


# TODO broadcast_in_dim dynamic shape batching rule upgrade

# TODO toposort f32[m] + m

# TODO if we have let-bound dynamic shapes, could end up with one application of
# pile_map requiring abstracting multiple piles?
# def f(x):
#   n = x.shape[0]
#   m = 2 * n
#   return jnp.zeros((n, m))
# n = [3, 1, 4]
# m = [6, 2, 8]
# i:(Fin 3) => f32[n.i, m.i]
# TODO should allow multiple 'segment lengths' arrays!
#  * would need to generalize ConcatAxis
