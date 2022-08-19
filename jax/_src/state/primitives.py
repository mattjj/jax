# Copyright 2022 Google LLC
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
"""Module for state primitives."""
from functools import partial

from typing import Any, Generic, List, Tuple, TypeVar, Union

from jax import core
from jax import lax
from jax._src import ad_util
from jax._src import pretty_printer as pp
from jax._src.util import safe_map, safe_zip, partition_list, tuple_insert
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp

from jax._src.state.types import ShapedArrayRef, StateEffect

## General utilities

Array = Any
T = TypeVar('T')
class Ref(Generic[T]): pass

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

## get/swap/addupdate implementations

# `get` reads a value from a `Ref` type, a.k.a.:
# a = get_p.bind(x)
# or we can read using indices:
# a = get_p.bind(x, 0, 1)
# Staging out `a = get_p.bind(x)` where the aval of `x` is
# `ShapedArrayRef((3,), np.dtype('float32'))` leads to a jaxpr eqn printed like
#   a:f32[3] <- x[]
get_p = core.Primitive("get")

def _get_impl(ref: Ref, *idx: int, **_):
  del ref, idx
  raise ValueError("Cannot run stateful primitive.")
get_p.def_impl(_get_impl)

Indexer = Tuple[Union[int, slice, jnp.ndarray], ...]

def _unpack_idx(idx: Indexer, ndim: int
               ) -> Tuple[Tuple[int, ...], Tuple[bool, ...]]:
  indexed_dims_ = [type(i) != slice for i in idx]
  _, non_slice_idx = partition_list(indexed_dims_, idx)
  indexed_dims = indexed_dims_ + [False] * (ndim - len(indexed_dims_))
  return (tuple(map(lambda x: jnp.asarray(x, jnp.int32), non_slice_idx)),
          tuple(indexed_dims))

def _get_slice_output_shape(in_shape: Tuple[int, ...],
                            idx_shapes: Tuple[Tuple[int, ...], ...],
                            indexed_dims: Tuple[bool, ...]) -> Tuple[int, ...]:
  shape_suffix = [d for i, d in zip(indexed_dims, in_shape) if not i]
  shape_prefix, = set(idx_shapes) or [()]  # tie fighter
  # Move shape prefix dimensions to the front
  shape = (*shape_prefix, *shape_suffix)
  return shape

def ref_get(ref: Ref, idx: Tuple[Union[int, slice], ...]) -> Array:
  """Reads a value from a `Ref`, a.k.a. value <- ref[idx]."""
  idx, indexed_dims = _unpack_idx(idx, ref.ndim)
  return get_p.bind(ref, *idx, indexed_dims=indexed_dims)

# `swap` mutates a `Ref`, setting its value and returns its previous value.
# b = swap_p.bind(x, a)
# It generalizes the setting operation for a `Ref` as we can ignore the return
# value:
# _ = swap_p.bind(x, a)
# `swap_p` also takes in index arguments following the value, i.e.:
# _ = swap_p.bind(x, a, 0, 1)
# Staging out `b = swap_p.bind(x, a)` where the aval of `x` is
# `ShapedArrayRef((3,), np.dtype('float32'))` and the aval of `a` is
# `ShapedArray((3,), np.dtype('float32'))` leads to a jaxpr eqn printed like
#   b:f32[3], x:Ref{f32[3]} <- x, a
# Staging out `_ = swap_p.bind(x, a, i, j)` where the aval of `x` is
# `ShapedArrayRef((3,), np.dtype('float32'))` , the aval of `a` is
# `ShapedArray((3,), np.dtype('float32'))`, and the avals of both `i` and `j`
# are `ShapedArray((), np.dtype('int32'))` leads to a jaxpr eqn printed like
#   x:Ref{f32[3]}[i, j] <- a
swap_p = core.Primitive("swap")

def _swap_impl(ref: Ref, value: Array, *idx: int, **_):
  del ref, value, idx
  raise ValueError("Cannot run stateful primitive.")
swap_p.def_impl(_swap_impl)

def ref_swap(ref: Ref, idx: Tuple[int, ...], value: Array) -> Array:
  """Sets a `Ref`'s value and returns the original value."""
  idx, indexed_dims = _unpack_idx(idx, ref.ndim)
  return swap_p.bind(ref, value, *idx, indexed_dims=indexed_dims)

def ref_set(ref: Ref, idx: Tuple[int, ...], value: Array) -> None:
  """Sets a `Ref`'s value, a.k.a. ref[idx] <- value."""
  ref_swap(ref, idx, value)

# `addupdate_p` mutates a `Ref`, adding a value to its existing value.
# Semantically,
# ```
# addupdate ref a *idx
# ```
# is equivalent to
# ```
# b = get ref *idx
# c = add b x
# _ = swap ref c *idx
# ```
addupdate_p = core.Primitive('addupdate')
addupdate_p.multiple_results = True

def _addupdate_impl(ref: Ref, value: Array, *idx: int):
  del ref, idx, value
  raise ValueError("Can't evaluate `addupdate` outside a stateful context.")
addupdate_p.def_impl(_addupdate_impl)

def ref_addupdate(ref: Ref, idx: Tuple[int, ...], x: Array) -> None:
  """Mutates a ref with an additive update i.e. `ref[idx] += x`."""
  idx, indexed_dims = _unpack_idx(idx, ref.ndim)
  return addupdate_p.bind(ref, x, *idx, indexed_dims=indexed_dims)

## get/set/addupdate abstract evaluation rules

def _get_abstract_eval(ref_aval: ShapedArrayRef, *idx, indexed_dims):
  if not isinstance(ref_aval, ShapedArrayRef):
    raise ValueError(f"`get` must be called on `Ref` types: {ref_aval}.")
  if len(indexed_dims) != len(ref_aval.shape):
    raise ValueError("`indexed_dims` must be the same length as `Ref` shape.")
  if sum(indexed_dims) != len(idx):
    raise ValueError(f"Invalid `idx` and `indexed_dims`: {idx}, {indexed_dims}")
  idx_shapes = tuple(i.shape for i in idx)
  shape = _get_slice_output_shape(ref_aval.shape, idx_shapes, indexed_dims)
  return (core.ShapedArray(shape, ref_aval.dtype), {StateEffect})
get_p.def_effectful_abstract_eval(_get_abstract_eval)


def _swap_abstract_eval(ref_aval: ShapedArrayRef, val_aval: core.AbstractValue,
                        *idx: core.ShapedArray, indexed_dims: Tuple[bool]):
  if not isinstance(ref_aval, ShapedArrayRef):
    raise ValueError(f"`swap` must be called on `Ref` types: {ref_aval}.")
  if len(indexed_dims) != len(ref_aval.shape):
    raise ValueError("`indexed_dims` must be the same length as `Ref` shape.")
  if sum(indexed_dims) != len(idx):
    raise ValueError(f"Invalid `idx` and `indexed_dims`: {idx}, {indexed_dims}")
  val_aval = core.raise_to_shaped(val_aval)
  assert isinstance(val_aval, core.ShapedArray)
  idx_shapes = tuple(i.shape for i in idx)
  expected_output_shape = _get_slice_output_shape(
      ref_aval.shape, idx_shapes, indexed_dims)
  if expected_output_shape != val_aval.shape:
    raise ValueError("Invalid shape for `swap`. "
                     f"Ref shape: {ref_aval.shape}. "
                     f"Value shape: {val_aval.shape}. "
                     f"Indices: {idx}. ")
  if ref_aval.dtype != val_aval.dtype:
    raise ValueError("Invalid dtype for `swap`. "
                     f"Ref dtype: {ref_aval.dtype}. "
                     f"Value shape: {val_aval.dtype}. ")
  return (core.ShapedArray(expected_output_shape, ref_aval.dtype),
          {StateEffect})
swap_p.def_effectful_abstract_eval(_swap_abstract_eval)


def _addupdate_abstract_eval(ref_aval: ShapedArrayRef,
                             val_aval: core.AbstractValue,
                             *idx: core.ShapedArray, indexed_dims: Tuple[bool]):
  if not isinstance(ref_aval, ShapedArrayRef):
    raise ValueError(f"`addupdate` must be called on `Ref` types: {ref_aval}.")
  if len(indexed_dims) != len(ref_aval.shape):
    raise ValueError("`indexed_dims` must be the same length as `Ref` shape.")
  if sum(indexed_dims) != len(idx):
    raise ValueError(f"Invalid `idx` and `indexed_dims`: {idx}, {indexed_dims}")
  val_aval = core.raise_to_shaped(val_aval)
  assert isinstance(val_aval, core.ShapedArray)
  idx_shapes = tuple(i.shape for i in idx)
  slice_shape = _get_slice_output_shape(
      ref_aval.shape, idx_shapes, indexed_dims)
  if slice_shape != val_aval.shape:
    raise ValueError("Invalid shape for `swap`. "
                     f"Ref shape: {ref_aval.shape}. "
                     f"Value shape: {val_aval.shape}. "
                     f"Indices: {idx}. ")
  if ref_aval.dtype != val_aval.dtype:
    raise ValueError("Invalid dtype for `swap`. "
                     f"Ref dtype: {ref_aval.dtype}. "
                     f"Value shape: {val_aval.dtype}. ")
  return [], {StateEffect}
addupdate_p.def_effectful_abstract_eval(_addupdate_abstract_eval)

## Pretty printing for `get` and `swap` in jaxprs

pp_ref = partial(pp.color, intensity=pp.Intensity.NORMAL,
                 foreground=pp.Color.GREEN)

def _pp_idx(context, non_slice_idx, indexed_dims):
  idx_iter = iter(non_slice_idx)
  idx = ','.join(core.pp_var(next(idx_iter), context) if indexed else ':'
                 for indexed in indexed_dims)
  assert next(idx_iter, None) is None
  return pp.text(idx)

def _get_pp_rule(eqn, context, settings):
  # Pretty prints `a = get x i` as `x[i] <- a`
  y, = eqn.outvars
  x, *idx = eqn.invars
  idx = _pp_idx(context, idx, eqn.params["indexed_dims"])
  lhs = core.pp_vars([y], context, print_shapes=settings.print_shapes)
  # TODO more general get
  return [lhs, pp.text(' <- '), pp_ref(pp.concat([
    pp.text(core.pp_var(x, context)), pp.text('['), idx, pp.text(']')
    ]))]
core.pp_eqn_rules[get_p] = _get_pp_rule

def _swap_pp_rule(eqn, context, settings):
  y, = eqn.outvars
  x, v, *idx = eqn.invars
  idx = _pp_idx(context, idx, eqn.params["indexed_dims"])
  if type(y) is core.DropVar:
    # In the case of a set (ignored return value),
    # pretty print `_ = swap x v i` as `x[i] <- v`
    del y
    return [
      pp_ref(pp.concat([
        pp.text(core.pp_var(x, context)),
        pp.text('['), idx, pp.text(']')
      ])), pp.text(' <- '), pp.text(core.pp_var(v, context))]
  else:
    # pretty-print `y:T = swap x v i` as `y:T, x[i] <- x[i], v`
    x_i = pp.concat([pp.text(core.pp_var(x, context)),
                     pp.text('['), idx, pp.text(']')])
    y = core.pp_vars([y], context, print_shapes=settings.print_shapes)
    return [y, pp.text(', '), pp_ref(x_i), pp.text(' <- '),
            pp_ref(x_i), pp.text(', '), pp.text(core.pp_var(v, context))]
core.pp_eqn_rules[swap_p] = _swap_pp_rule

def _addupdate_pp_rule(eqn, context, settings):
  # pretty-print ` = addupdate x i v` as `x[i] += v`
  () = eqn.outvars
  x, v, *idx = eqn.invars
  idx = _pp_idx(context, idx, eqn.params["indexed_dims"])
  return [
    pp_ref(pp.concat([
        pp.text(core.pp_var(x, context)),
        pp.text('['), idx, pp.text(']')
      ])), pp.text(' += '), pp.text(core.pp_var(v, context))]
core.pp_eqn_rules[addupdate_p] = _addupdate_pp_rule

## get/swap/addupdate JVP rules

def _get_jvp(primals: List[Any], tangents: List[Any], **params: Any):
  ref_primal, *idx = primals
  assert isinstance(ref_primal.aval, ShapedArrayRef)
  ref_tangent, *_ = tangents
  assert isinstance(ref_tangent.aval, ShapedArrayRef)
  return (get_p.bind(ref_primal, *idx, **params),
          get_p.bind(ref_tangent, *idx, **params))  # type: ignore[arg-type]
ad.primitive_jvps[get_p] = _get_jvp

def _swap_jvp(primals: List[Any], tangents: List[Any], **params: Any):
  ref_primal, x_primal, *idx = primals
  assert isinstance(ref_primal.aval, ShapedArrayRef)
  ref_tangent, x_tangent, *_ = tangents
  assert isinstance(ref_tangent.aval, ShapedArrayRef)
  x_tangent = ad_util.instantiate(x_tangent)
  return (swap_p.bind(ref_primal, x_primal, *idx, **params),  # type: ignore[arg-type]
          swap_p.bind(ref_tangent, x_tangent, *idx, **params))  # type: ignore[arg-type]
ad.primitive_jvps[swap_p] = _swap_jvp

def addupdate_jvp_rule(primals: List[Any], tangents: List[Any], **params: Any):
  ref_primal, x_primal, *idx = primals
  ref_tangent, x_tangent, *_ = tangents
  x_tangent = ad_util.instantiate(x_tangent)
  addupdate_p.bind(ref_primal, x_primal, *idx, **params)
  addupdate_p.bind(ref_tangent, x_tangent, *idx, **params)
  return [], []
ad.primitive_jvps[addupdate_p] = addupdate_jvp_rule

##  get/swap/addupdate transpose rules

def _get_transpose(g, ref, *idx, **params):
  # get transpose is addupdate
  if type(g) is not ad_util.Zero:
    addupdate_p.bind(ref, g, *idx, **params)
  return [None] + [None] * len(idx)
ad.primitive_transposes[get_p] = _get_transpose

def _swap_transpose(g, ref, x, *idx, **params):
  # swap transpose is swap
  x_bar = swap_p.bind(ref, ad_util.instantiate(g), *idx, **params)
  return [None, x_bar] + [None] * len(idx)
ad.primitive_transposes[swap_p] = _swap_transpose

def addupdate_transpose(cts_in, ref, x, *idx):
  # addupdate transpose is get
  del cts_in, x
  g = ref_get(ref, idx)
  return [None, g] + [None] * len(idx)
ad.primitive_transposes[addupdate_p] = addupdate_transpose

## get/swap/addupdate partial_eval_custom rules

def _state_partial_eval_custom(prim, saveable, unks_in, inst_in, eqn):
  if any(unks_in):
    res = [v for v, inst in zip(eqn.invars, inst_in) if not inst]
    return None, eqn, [True] * len(eqn.outvars), [True] * len(eqn.outvars), res
  elif saveable(get_p, *[var.aval for var in eqn.invars], **eqn.params):
    return eqn, None, [False] * len(eqn.outvars), [False] * len(eqn.outvars), []
  res = [v for v, inst in zip(eqn.invars, inst_in) if not inst]
  return eqn, eqn, [False] * len(eqn.outvars), [True] * len(eqn.outvars), []

pe.partial_eval_jaxpr_custom_rules[get_p] = partial(_state_partial_eval_custom,
                                                    get_p)
pe.partial_eval_jaxpr_custom_rules[swap_p] = partial(_state_partial_eval_custom,
                                                     swap_p)
pe.partial_eval_jaxpr_custom_rules[addupdate_p] = partial(
    _state_partial_eval_custom, addupdate_p)

##  get/swap/addupdate batching rules

def _compute_output_batch_dimension(indexed_dims: Tuple[bool, ...], ref_dim:
    int, idxs_shape: Tuple[int, ...]):
  num_idxs_to_left = sum(indexed_dims[:ref_dim])
  return ref_dim - num_idxs_to_left + len(idxs_shape)

def _get_vmap(batched_args, batched_dims, *, indexed_dims):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, *idxs = batched_args
  ref_dim, *idx_dims = batched_dims

  ref_is_batched = ref_dim is not batching.not_mapped
  idx_is_batched = any(i_dim is not batching.not_mapped for i_dim in idx_dims)
  bdim_out = 0

  if idx_is_batched:
    # If at least one of the idx is batched, we broadcast them all and move the
    # batch dim to the front.
    idxs = tuple(batching.bdim_at_front(i, d, axis_size) for i, d
                 in zip(idxs, idx_dims))
  idxs_shape, = {i.shape for i in idxs} or [()]
  if ref_is_batched:
    # If ref is batched, we are doing a `get` with an additional axis. If `idxs`
    # are also batched, then we are indexing into the batch axis with an `iota`.
    indexed_dims = tuple_insert(indexed_dims, ref_dim, idx_is_batched)
    if idx_is_batched:
      # If we have batched idx, we need to insert the new iota index
      # The place where we add in the new `iota` index is `ref_dim` so we need
      # to compute what `ref_dim` *would be* if we inserted it into `idxs`
      # instead, because `idxs` doesn't include the non indexed dims.
      idx_place = [i for i, i_dim in enumerate(indexed_dims)
                   if i_dim].index(ref_dim)
      idxs = tuple_insert(idxs, idx_place,
                          lax.broadcasted_iota(jnp.dtype('int32'),
                                               idxs_shape, 0))
    else:
      bdim_out = _compute_output_batch_dimension(indexed_dims, ref_dim,
                                                 idxs_shape)
  return get_p.bind(ref, *idxs, indexed_dims=indexed_dims), bdim_out
batching.primitive_batchers[get_p] = _get_vmap

def _swap_vmap(batched_args, batched_dims, *, indexed_dims):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, val, *idxs = batched_args
  ref_dim, val_dim, *idx_dims = batched_dims
  ref_is_batched = ref_dim is not batching.not_mapped
  val_is_batched = val_dim is not batching.not_mapped
  idx_is_batched = any(i_dim is not batching.not_mapped for i_dim in idx_dims)
  if idx_is_batched:
    # If at least one of the idx is batched, we broadcast them all and move the
    # batch dim to the front.
    idxs = tuple(batching.bdim_at_front(i, d, axis_size) for i, d
                 in zip(idxs, idx_dims))
  idxs_shape, = {i.shape for i in idxs} or [()]
  if ref_is_batched and not idx_is_batched:
    if not val_is_batched:
      val = batching.broadcast(val, axis_size, ref_dim)
    indexed_dims = tuple_insert(indexed_dims, ref_dim, False)
    bdim_out = _compute_output_batch_dimension(indexed_dims, ref_dim, idxs_shape)
  elif idx_is_batched:
    assert ref_is_batched
    assert val_is_batched
    indexed_dims = tuple_insert(indexed_dims, ref_dim, True)
    idx_place = [i for i, i_dim in enumerate(indexed_dims)
                 if i_dim].index(ref_dim)
    idxs = tuple_insert(idxs, idx_place,
                        lax.broadcasted_iota(jnp.dtype('int32'),
                                             idxs_shape, 0))
    val = batching.moveaxis(val, val_dim, 0)
    bdim_out = 0
  return swap_p.bind(ref, val, *idxs, indexed_dims=indexed_dims), bdim_out
batching.primitive_batchers[swap_p] = _swap_vmap

def _addupdate_vmap(batched_args, batched_dims, *, indexed_dims):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, val, *idxs = batched_args
  ref_dim, val_dim, *idx_dims = batched_dims
  ref_is_batched = ref_dim is not batching.not_mapped
  val_is_batched = val_dim is not batching.not_mapped
  idx_is_batched = any(i_dim is not batching.not_mapped for i_dim in idx_dims)
  if idx_is_batched:
    # If at least one of the idx is batched, we broadcast them all and move the
    # batch dim to the front.
    idxs = tuple(batching.bdim_at_front(i, d, axis_size) for i, d
                 in zip(idxs, idx_dims))
  idxs_shape, = {i.shape for i in idxs} or [()]
  if ref_is_batched and not idx_is_batched:
    if not val_is_batched:
      val = batching.broadcast(val, axis_size, ref_dim)
    indexed_dims = tuple_insert(indexed_dims, ref_dim, False)
  elif idx_is_batched:
    assert ref_is_batched
    assert val_is_batched
    indexed_dims = tuple_insert(indexed_dims, ref_dim, True)
    idx_place = [i for i, i_dim in enumerate(indexed_dims)
                 if i_dim].index(ref_dim)
    idxs = tuple_insert(idxs, idx_place,
                        lax.broadcasted_iota(jnp.dtype('int32'),
                                             idxs_shape, 0))
    val = batching.moveaxis(val, val_dim, 0)
  return addupdate_p.bind(ref, val, *idxs, indexed_dims=indexed_dims), []
batching.primitive_batchers[addupdate_p] = _addupdate_vmap

# get [indexed_dims=indexed_dims] x *idxs
#   len(idxs) + [where indexed_dims is false] = x.ndim
# for example
#   x[:, idxs1, :, idxs2, :, :]
# idxs can be int arrays, all have same shape as each other
#
# say x is shape (10, 3, 11, 4, 12, 13)
# say idxs are each of shape (5, 6)
# what is result shape?
# like numpy: (5, 6, 10, 11, 12, 13)
# with numpy, _sometimes_ the bdims stay in place. but we could just always have
# result bdims at front.
#
# for now, don't allow slices like 3:7 (can lower those away)

# TODO
#  [x] get traceable
#  [x] get abstract eval
#  [x] get pprint
#  [-] get vmap
#   [x] batch ref
#   [ ] batch idxs
#  [ ] get lowering
#  [x] other primitives!
#   [x] swap
#   [x] addupdate
#  [ ] transpose
