# Copyright 2022 The JAX Authors.
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
"""Module for the `for_loop` primitive."""
from distutils.errors import UnknownFileError
import functools
import operator

from typing import Any, Callable, Generic, List, Optional, Sequence, Set, Tuple, TypeVar, Union

from jax import core
from jax import lax
from jax import linear_util as lu
from jax.api_util import flatten_fun_nokwargs
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.tree_util import (tree_flatten, tree_structure, tree_unflatten,
                           treedef_tuple, tree_map, tree_leaves, PyTreeDef)
from jax._src import ad_util
from jax._src import dtypes
from jax._src import pretty_printer as pp
from jax._src import source_info_util
from jax._src import state
from jax._src.state import primitives as state_primitives
from jax._src.util import (partition_list, merge_lists, safe_map, safe_zip,
                           split_list, split_dict)
import jax.numpy as jnp

from jax._src.lax.control_flow import loops
from jax._src.lax.control_flow.common import _abstractify, _initial_style_jaxpr

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

## Helpful type aliases
S = TypeVar('S')
T = TypeVar('T')
class Ref(Generic[T]): pass
Array = Any

ReadEffect = state.ReadEffect
WriteEffect = state.WriteEffect
AccumEffect = state.AccumEffect
StateEffect = state.StateEffect
ShapedArrayRef = state.ShapedArrayRef
ref_set = state.ref_set
ref_get = state.ref_get
ref_addupdate = state.ref_addupdate
discharge_state = state.discharge_state


## `for_loop` implementation

for_p = core.Primitive('for')
for_p.multiple_results = True

### Tracing utilities

def _hoist_consts_to_refs(jaxpr: core.Jaxpr) -> core.Jaxpr:
  all_const_avals = [var.aval for var in jaxpr.constvars]
  is_const_ref = [isinstance(var.aval, ShapedArrayRef) for var in
                  jaxpr.constvars]
  const_avals, const_ref_avals = partition_list(is_const_ref, all_const_avals)
  const_avals = [ShapedArrayRef(aval.shape, aval.dtype) for aval in const_avals]  # pytype: disable=attribute-error
  merged_const_avals = merge_lists(is_const_ref, const_avals, const_ref_avals)
  i_aval, *arg_avals = [var.aval for var in jaxpr.invars]
  in_avals = [i_aval, *merged_const_avals, *arg_avals]
  num_consts = len(merged_const_avals)

  def _hoist(i, *consts_args):
    all_consts, args = split_list(consts_args, [num_consts])
    consts, const_refs = partition_list(is_const_ref, all_consts)
    # We immediately read the const values out of the `Ref`s.
    consts = map(lambda x: ref_get(x, ()), consts)
    all_consts = merge_lists(is_const_ref, consts, const_refs)
    return core.eval_jaxpr(jaxpr, all_consts, i, *args)
  hoisted_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(_hoist), in_avals)
  assert not consts, "All consts should have been converted to refs"
  return hoisted_jaxpr

def _trace_to_jaxpr_with_refs(f, state_tree: PyTreeDef,
                              state_avals: Sequence[core.AbstractValue]
                              ) -> Tuple[core.Jaxpr, List[Any], PyTreeDef]:
  f, out_tree_thunk = flatten_fun_nokwargs(
      lu.wrap_init(f), treedef_tuple((tree_structure(0), state_tree)))
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      f, state_avals)
  return jaxpr, consts, out_tree_thunk()

def val_to_ref_aval(x) -> ShapedArrayRef:
  aval = core.raise_to_shaped(core.get_aval(x))
  if type(aval) is not core.ShapedArray:
    raise Exception(f"can't make ref from {x}")
  return ShapedArrayRef(aval.shape, aval.dtype)

def for_loop(nsteps: Union[int, Sequence[int]],
             body: Callable[[Array, Ref[S]], None], init_state: S,
             *, reverse: bool = False, unroll: int = 1) -> S:
  """A for-loop combinator that allows read/write semantics in the loop body.

  `for_loop` is a higher-order function that enables writing loops that can be
  staged out in JIT-ted JAX computations. Unlike `jax.lax.fori_loop`, it allows
  mutation in its body using `Ref`s.

  `for_loop` will initialize `Ref`s with the values in `init_state`. Each
  iteration, `body` will be called with the current `Ref`s, which can be read
  from and written to using `ref_get` and `ref_set`.

  `for_loop` is semantically equivalent to the following Python code:

  ```python
  def for_loop(nsteps, body, init_state):
    refs = tree_map(make_ref, init_state)
    for i in range(nsteps):
      body(i, refs)
    return tree_map(ref_get, refs)
  ```

  Args:
    nsteps: Number of iterations
    body: A callable that takes in the iteration number as its first argument
      and `Ref`s corresponding to `init_state` as its second argument.
      `body` is free to read from and write to its `Ref`s. `body` should
       not return anything.
    init_state: A Pytree of JAX-compatible values used to initialize the `Ref`s
      that will be passed into the for loop body.
    unroll: A positive int specifying, in the underlying operation of the
      `for` primitive, how many iterations to unroll within a single iteration
      of a loop. Higher values may speed up execution time at the cost of longer
      compilation time.
  Returns:
    A Pytree of values representing the output of the for loop.
  """
  if unroll < 1:
    raise ValueError("`unroll` must be a positive integer.")
  if isinstance(nsteps, int):
    nsteps = [nsteps]
  if len(nsteps) > 1:
    outer_step, *rest_steps = nsteps
    def wrapped_body(i, refs):
      vals = tree_map(lambda ref: state.ref_get(ref, ()), refs)
      vals = for_loop(
          rest_steps, functools.partial(body, i), vals, unroll=unroll)
      tree_map(lambda ref, val: state.ref_set(ref, (), val), refs, vals)
    return for_loop(outer_step, wrapped_body, init_state, unroll=unroll)
  nsteps, = nsteps
  flat_state, state_tree = tree_flatten(init_state)
  state_avals = map(val_to_ref_aval, flat_state)
  idx_aval = core.ShapedArray((), jnp.dtype("int32"))
  jaxpr, consts, out_tree = _trace_to_jaxpr_with_refs(
      body, state_tree, [idx_aval, *state_avals])
  if out_tree != tree_structure(None):
    raise Exception("`body` should not return anything.")
  # Remove constvars from jaxpr and turn them into `Ref`s
  jaxpr = _hoist_consts_to_refs(jaxpr)
  which_linear = (False,) * (len(consts) + len(flat_state))
  out_flat = for_p.bind(*consts, *flat_state, jaxpr=jaxpr, nsteps=int(nsteps),
                        reverse=reverse, which_linear=which_linear,
                        unroll=unroll)
  # Consts are `Ref`s so they are both inputs and outputs. We remove them from
  # the outputs.
  out_flat = out_flat[len(consts):]
  return tree_unflatten(state_tree, out_flat)

def run_state_bind(*args, jaxpr: core.Jaxpr, which_linear: Tuple[bool, ...]):
  assert len(args) == len(jaxpr.invars)
  assert not jaxpr.constvars
  for arg, invar in zip(args, jaxpr.invars):
    aval = core.get_aval(arg)
    assert aval.dtype == invar.aval.dtype, (aval, invar.aval)
    assert aval.shape == invar.aval.shape, (aval, invar.aval)
  in_avals = [core.ShapedArray((), jnp.int32), *[v.aval for v in jaxpr.invars]]
  @lu.wrap_init
  def _traceable(_, *args):
    return core.eval_jaxpr(jaxpr, (), *args)
  jaxpr, _, () = pe.trace_to_jaxpr_dynamic(_traceable, in_avals)
  return for_p.bind(*args, jaxpr=jaxpr, nsteps=1, reverse=False,
                    unroll=1, which_linear=which_linear)

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')

def for_bloop(nsteps, body, init_state, *, reverse: bool = False,
              unroll: int = 1):
  def run(refs):
    def wrapped_body(i):
      if reverse:
        i = nsteps - i - 1
      return body(i, refs)
    bloop(wrapped_body, max_iter=nsteps, unroll=unroll)
  return run_state(run, init_state)

def scan(f: Callable[[Carry, X], Tuple[Carry, Y]],
         init: Carry,
         xs: X,
         length: Optional[int] = None,
         reverse: bool = False,
         unroll: int = 1) -> Tuple[Carry, Y]:
  if not callable(f):
    raise TypeError("scan: f argument should be a callable.")
  if unroll < 1:
    raise ValueError("`unroll` must be a positive integer.")
  xs_flat, xs_tree = tree_flatten(xs)

  try:
    lengths = [x.shape[0] for x in xs_flat]
  except AttributeError as err:
    msg = "scan got value with no leading axis to scan over: {}."
    raise ValueError(
      msg.format(', '.join(str(x) for x in xs_flat
                           if not hasattr(x, 'shape')))) from err

  if length is not None:
    length = int(length)
    if not all(length == l for l in lengths):
      msg = ("scan got `length` argument of {} which disagrees with "
             "leading axis sizes {}.")
      raise ValueError(msg.format(length, [x.shape[0] for x in xs_flat]))
  else:
    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
      msg = "scan got values with different leading axis sizes: {}."
      raise ValueError(msg.format(', '.join(str(x.shape[0]) for x in xs_flat)))
    elif len(unique_lengths) == 0:
      msg = "scan got no values to scan over and `length` not provided."
      raise ValueError(msg)
    else:
      length, = unique_lengths

  x_shapes = [x.shape[1:] for x in xs_flat]
  x_dtypes = [dtypes.canonicalize_dtype(x.dtype) for x in xs_flat]
  x_avals = tuple(map(core.ShapedArray, x_shapes, x_dtypes))

  def _create_jaxpr(init):
    init_flat = tree_leaves(init)
    _, in_tree = tree_flatten((init, xs))

    carry_avals = tuple(map(_abstractify, init_flat))
    jaxpr, _, out_tree = _initial_style_jaxpr(
        f, in_tree, carry_avals + x_avals, "scan")
    return jaxpr, out_tree
  jaxpr, out_tree = _create_jaxpr(init)
  _, ys_avals = tree_unflatten(out_tree, jaxpr.out_avals)
  ys = tree_map(lambda aval: jnp.zeros([length, *aval.shape], aval.dtype),
                ys_avals)
  def for_body(i, refs):
    carry_refs, xs_refs, ys_refs = refs
    carry = tree_map(lambda x: x[()], carry_refs)
    x = tree_map(lambda x: x[i], xs_refs)
    carry, y = f(carry, x)
    tree_map(lambda c_ref, c: ref_set(c_ref, (), c), carry_refs, carry)
    tree_map(lambda y_ref, y: ref_set(y_ref, (i,), y), ys_refs, y)
  assert isinstance(length, int)
  init, _, ys = for_loop(length, for_body, (init, xs, ys), reverse=reverse,
                         unroll=unroll)
  return init, ys


def _for_bind(*args, jaxpr, nsteps, reverse, which_linear, unroll):
  assert isinstance(unroll, int)
  assert isinstance(nsteps, int)
  assert isinstance(jaxpr, core.Jaxpr)
  assert isinstance(which_linear, tuple)
  assert isinstance(reverse, bool)
  assert len(args) == len(jaxpr.invars) - 1
  for arg, invar in zip(args, jaxpr.invars[1:]):
    aval = core.get_aval(arg)
    assert aval.dtype == invar.aval.dtype, (aval, invar.aval)
    assert aval.shape == invar.aval.shape, (aval, invar.aval)
  assert len(which_linear) == len(args)
  return core.Primitive.bind(for_p, *args, jaxpr=jaxpr, nsteps=nsteps,
      reverse=reverse, which_linear=which_linear, unroll=unroll)
for_p.def_custom_bind(_for_bind)

@for_p.def_effectful_abstract_eval
def _for_abstract_eval(*avals, jaxpr, **__):
  # Find out for each of the `Ref`s in our jaxpr what effects they have.
  jaxpr_aval_effects = state.get_ref_state_effects(
      [v.aval for v in jaxpr.invars], jaxpr.effects)[1:]
  aval_effects = [set(eff.replace(ref_aval=aval) for eff in effs) for aval, effs
                  in zip(avals, jaxpr_aval_effects)
                  if isinstance(aval, ShapedArrayRef)]
  nonlocal_state_effects = core.join_effects(*aval_effects)
  return list(avals), nonlocal_state_effects

@state.register_discharge_rule(for_p)
def _for_discharge_rule(in_avals, _, *args: Any, jaxpr: core.Jaxpr,
                        reverse: bool, which_linear: Sequence[bool],
                        nsteps: int, unroll: int
                        ) -> Tuple[Sequence[Optional[Any]], Sequence[Any]]:
  out_vals = for_p.bind(*args, jaxpr=jaxpr, reverse=reverse,
                        which_linear=which_linear, nsteps=nsteps,
                        unroll=unroll)
  new_invals = []
  for aval, out_val in zip(in_avals, out_vals):
    new_invals.append(out_val if isinstance(aval, ShapedArrayRef) else None)
  return new_invals, out_vals

def _for_impl(*args, jaxpr, nsteps, reverse, which_linear, unroll):
  del which_linear
  discharged_jaxpr, consts = discharge_state(jaxpr, ())
  def body(i, state):
    i_ = nsteps - i - 1 if reverse else i
    return core.eval_jaxpr(discharged_jaxpr, consts, i_, *state)
  assert all([a.shape == v.aval.shape for a, v in zip(args,
    discharged_jaxpr.invars[1:])])
  return _for_impl_unrolled(body, nsteps, unroll, *args)

def _for_impl_unrolled(body, nsteps, unroll, *args):
  remainder = nsteps % unroll
  i = jnp.int32(0)
  state = list(args)
  if nsteps == 1:
    state = body(0, state)
    return state

  for _ in range(remainder):
    state = body(i, state)
    i = i + 1

  def cond(carry):
    i, _ = carry
    return i < nsteps
  def while_body(carry):
    i, state = carry
    for _ in range(unroll):
      state = body(i, state)
      i = i + 1
    return i, state
  _, state = lax.while_loop(cond, while_body, (i, state))
  return state

mlir.register_lowering(for_p, mlir.lower_fun(_for_impl, multiple_results=True))
for_p.def_impl(functools.partial(xla.apply_primitive, for_p))

def _for_pp_rule(eqn: core.JaxprEqn, context: core.JaxprPpContext,
                 settings: core.JaxprPpSettings) -> List[pp.Doc]:
  nsteps = eqn.params["nsteps"]
  annotation = (source_info_util.summarize(eqn.source_info)
                if settings.source_info else None)
  name_stack_annotation = f'[{eqn.source_info.name_stack}]' if settings.name_stack else None
  lhs = core.pp_vars(eqn.outvars, context, print_shapes=settings.print_shapes)
  if nsteps == 1:
    # Use run_state pretty-printer
    params = dict(which_linear=eqn.params["which_linear"],
                  jaxpr=eqn.params["jaxpr"])
    rhs = [pp.text("run_state", annotation=name_stack_annotation),
           core.pp_kv_pairs(sorted(params.items()), context, settings),
           pp.text(" ") + core.pp_vars(eqn.invars, context)]
  else:
    rhs = [pp.text(eqn.primitive.name, annotation=name_stack_annotation),
           core.pp_kv_pairs(sorted(eqn.params.items()), context, settings),
           pp.text(" ") + core.pp_vars(eqn.invars, context)]
  return [lhs, pp.text(" = ", annotation=annotation), *rhs]
core.pp_eqn_rules[for_p] = _for_pp_rule

def _for_vmap(axis_size, axis_name, main_type, args, dims, *,
              jaxpr, nsteps, reverse, which_linear, unroll):
  init_batched = [d is not batching.not_mapped for d in dims]
  discharged_jaxpr, body_consts = discharge_state(jaxpr, ())
  batched = init_batched
  for _ in range(len(batched)):
    _, out_batched = batching.batch_jaxpr(
        core.ClosedJaxpr(discharged_jaxpr, body_consts),
        axis_size, [False] + batched, instantiate=batched,
        axis_name=axis_name, main_type=main_type)
    if out_batched == batched:
      break
    batched = map(operator.or_, batched, out_batched)
  else:
    raise Exception("Invalid fixpoint")
  args = [batching.broadcast(x, axis_size, 0) if now_bat and not was_bat
          else batching.moveaxis(x, d, 0) if now_bat else x
          for x, d, was_bat, now_bat in zip(args, dims, init_batched, batched)]
  batched_jaxpr_, _ = batching.batch_jaxpr(
      core.ClosedJaxpr(jaxpr, []), axis_size, [False] + batched, [],
      axis_name=axis_name, main_type=main_type)
  batched_jaxpr, () = batched_jaxpr_.jaxpr, batched_jaxpr_.consts  # TODO consts
  out_flat = for_p.bind(*args, jaxpr=batched_jaxpr, nsteps=nsteps,
                        reverse=reverse, which_linear=which_linear,
                        unroll=unroll)
  return out_flat, [0 if b else batching.not_mapped for b in batched]
batching.axis_primitive_batchers[for_p] = _for_vmap

def _for_jvp(primals, tangents, *, jaxpr, nsteps, reverse, which_linear,
             unroll):
  nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  # We need to find out which `Ref`s have nonzero tangents after running the
  # for loop. Ordinarily we do this with a fixed point on the body jaxpr but
  # a `for` body jaxpr is stateful and has no outputs. We therefore discharge
  # the state effect from the jaxpr and we will now have a "symmetric" jaxpr
  # where the inputs line up with the outputs. We use this discharged jaxpr
  # for the fixed point.
  discharged_jaxpr, body_consts = discharge_state(jaxpr, ())
  for _ in range(len(nonzero_tangents)):
    _, out_nonzero_tangents = ad.jvp_jaxpr(
        core.ClosedJaxpr(discharged_jaxpr, body_consts),
        [False] + nonzero_tangents, instantiate=nonzero_tangents)
    if out_nonzero_tangents == nonzero_tangents:
      break
    nonzero_tangents = map(operator.or_, nonzero_tangents, out_nonzero_tangents)
  else:
    raise Exception("Invalid fixpoint")
  tangents = [ad.instantiate_zeros(t) if inst else t
              for t, inst in zip(tangents, nonzero_tangents)]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  closed_jaxpr = core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(closed_jaxpr, [False] + nonzero_tangents, [])
  jvp_jaxpr, () = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts  # TODO consts
  jvp_which_linear = which_linear + (True,) * len(tangents)
  out_flat = for_p.bind(*primals, *tangents, jaxpr=jvp_jaxpr,
                        nsteps=nsteps, reverse=reverse,
                        which_linear=jvp_which_linear, unroll=unroll)
  # `out_flat` includes constant inputs into the `for_loop` which are converted
  # into outputs as well. We don't care about these in AD so we throw them out.
  out_primals, out_tangents = split_list(out_flat, [len(primals)])
  out_tangents_iter = iter(out_tangents)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_value(p)
                  for p, nz in zip(out_primals, nonzero_tangents)]
  return out_primals, out_tangents
ad.primitive_jvps[for_p] = _for_jvp


def _partial_eval_jaxpr_custom(jaxpr, in_unknowns, policy):
  # A simple wrapper around `pe.partial_eval_jaxpr_custom` that assumes all
  # inputs are instantiated and doesn't ensure any outputs are unknown or
  # instantiated.
  return pe.partial_eval_jaxpr_custom(
      jaxpr, in_unknowns, [True] * len(in_unknowns), False, False, policy)

_save_everything = lambda *_, **__: True

def _is_read_only(ref_effects: Set[StateEffect]) -> bool:
  assert len(ref_effects) > 0
  if len(ref_effects) > 1:
    # Means we must have a write or accum effect so not read-only
    return False
  eff, = ref_effects
  return isinstance(eff, ReadEffect)

def _loop_invariant_outputs(jaxpr: core.Jaxpr) -> List[bool]:
  # Get effects for each of the jaxpr inputs and remove the loop index.
  ref_effects = state.get_ref_state_effects(
      [v.aval for v in jaxpr.invars], jaxpr.effects)[1:]
  # We first assume that *read-only `Ref`s* are loop-invariant. We can safely do
  # this because the only way something can be loop-varying is if we write to it
  # at some point. It's *possible* that read-write `Ref`s are loop-invariant but
  # we conservatively assume they aren't.
  loop_invar_refs = [_is_read_only(effs) if effs else True
                     for effs in ref_effects]
  loop_var_refs = map(operator.not_, loop_invar_refs)

  # We'd like to detect if the outputs of the jaxpr are loop-invariant. An
  # output is loop-invariant if it is downstream of only loop-invariant values
  # (seeded by the read-only `Ref`s). If at any point, a loop-varying value
  # interacts with a loop-invariant value, we produce a loop-varying value. We
  # can use `partial_eval` to perform this analysis by treating loop-varying
  # values as "unknown" and loop-invariant values as "known", since when a known
  # and unknown value interact, they produce an unknown value.
  loop_var_inputs = [True, *loop_var_refs]
  _, _, loop_var_outputs, _, _, = _partial_eval_jaxpr_custom(
      jaxpr, loop_var_inputs, _save_everything)
  return map(operator.not_, loop_var_outputs)


def _for_partial_eval(trace: pe.JaxprTrace, *tracers: pe.JaxprTracer,
                      jaxpr: core.Jaxpr, nsteps: int, reverse: bool,
                      which_linear: Tuple[bool, ...],
                      unroll: int) -> List[pe.JaxprTracer]:
  num_inputs = len(tracers)
  assert num_inputs == len(jaxpr.invars) - 1
  in_unknowns = [not t.pval.is_known() for t in tracers]
  # We first need to run a fixpoint to determine which of the `Ref`s are unknown
  # after running the for loop. We want to use the jaxpr to determine which
  # `Ref`s are unknown after executing the for loop body given which `Ref`s are
  # unknown before. However, the jaxpr has no outputs. Instead, we discharge
  # the body and run the fixpoint with the discharged jaxpr. We can do this
  # because the outputs of the jaxpr are one-to-one with the inputs.
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, ())
  discharged_jaxpr = discharged_jaxpr.replace(
      invars=discharged_jaxpr.constvars + discharged_jaxpr.invars,
      constvars=[])
  for _ in range(num_inputs):
    jaxpr_in_unknowns = [False] * len(discharged_consts) + [False, *in_unknowns]
    _, _, out_unknowns, _, _, = pe.partial_eval_jaxpr_custom(
        discharged_jaxpr, jaxpr_in_unknowns, [True] * len(jaxpr_in_unknowns),
          in_unknowns, False, _save_everything)
    out_unknowns = list(out_unknowns)
    if out_unknowns == in_unknowns:
      break
    in_unknowns = map(operator.or_, in_unknowns, out_unknowns)
  else:
    raise Exception("Invalid fixpoint")
  del out_unknowns  # redundant since it's the same as `in_unknowns`
  tracers = tuple(trace.instantiate_const(t) if uk else t  # type: ignore
                  for t, uk in zip(tracers, in_unknowns))

  # We use `partial_eval_jaxpr_custom` here because it won't remove effectful
  # primitives like `get`/`set`.
  jaxpr_known_resout, jaxpr_unknown_resin_, uk_out, inst_out, num_res = \
        _partial_eval_jaxpr_custom(jaxpr, [False, *in_unknowns],
                                   _save_everything)
  core.check_jaxpr(jaxpr_known_resout)
  core.check_jaxpr(jaxpr_unknown_resin_)
  # # `partial_eval_jaxpr_custom` will give us jaxprs that have hybrid `Ref` and
  # regular valued input/outputs. However, we'd like to bind these jaxprs to a
  # `for`, which expects only `Ref` inputs and no output. We need to convert
  # both of these jaxprs into ones that are compatible with `for`.
  # TODO(sharadmv,mattjj): implement "passthrough" optimization.
  # TODO(sharadmv,mattjj): rematerialize loop-dependent values instead of
  # passing the loop index as a residual

  # `jaxpr_known_resout` is a jaxpr that maps from all the input `Refs`
  # to output residual values (none of them should be `Ref`s). We'll need to
  # convert the output residual values into `Ref`s that are initially empty
  # `Ref`s that are written to at the end of the jaxpr.

  # # Loop-invariant residual optimization
  # Here we are interested in finding out which of the residuals are *not*
  # dependent on the loop index. If a residual is not dependent on the loop
  # index, we don't need add an extra loop dimension we're reading from when we
  # convert it from an output into a write.
  if nsteps == 1:
    loop_invar_res = [True] * num_res
  else:
    loop_invar_res = _loop_invariant_outputs(jaxpr_known_resout)

  jaxpr_known, res_avals = _convert_outputs_to_writes(nsteps,
                                                      jaxpr_known_resout,
                                                      loop_invar_res)
  # We now run the known jaxpr to obtain our residual values.
  known_tracers, _ = partition_list(in_unknowns, tracers)
  known_vals = [t.pval.get_known() for t in known_tracers]
  empty_res = map(ad_util.zeros_like_aval, res_avals)
  jaxpr_known_args = [*known_vals, *empty_res]
  # We assume the known inputs are nonlinear which is okay to do for AD but not
  # necessarily okay for general partial eval.
  jaxpr_known_which_linear = (False,) * len(jaxpr_known_args)
  out_flat = for_p.bind(*jaxpr_known_args, jaxpr=jaxpr_known, nsteps=nsteps,
                        reverse=reverse, which_linear=jaxpr_known_which_linear,
                        unroll=unroll)
  known_outputs, residuals = split_list(out_flat, [len(known_tracers)])
  residuals = map(trace.new_instantiated_const, residuals)

  # Now we handle the `jaxpr_unknown` that expects residual values as inputs.
  # This jaxpr is the output of `partial_eval_jaxpr_custom` that marks which
  # inputs are actually used.
  # `partial_eval_jaxpr_custom` doesn't remove extra inputs/outputs for you
  # so we use `dce_jaxpr` here to do that.
  jaxpr_unknown_resin, used_inputs = pe.dce_jaxpr(
        jaxpr_unknown_resin_, [], [True] * num_res + [True, *in_unknowns])
  used_res, (used_i,), used_refs = split_list(used_inputs, [num_res, 1])
  assert all(used_res), "All residuals should be used"
  # To make it compatible with `for`, we need to convert those residual values
  # into `Ref`s.
  jaxpr_unknown = _convert_inputs_to_reads(nsteps, len(res_avals),
                                           jaxpr_unknown_resin,
                                           loop_invar_res)
  # Since not all inputs are used in jaxpr_unknown, we filter the input tracers
  # down using the output of `dce_jaxpr`.
  used_and_known = map(operator.and_, used_refs, map(operator.not_, in_unknowns))
  tracers = [trace.instantiate_const(t) if u_and_k else t for t, u_and_k  # type: ignore
             in zip(tracers, used_and_known)]
  _, known_used = partition_list(used_refs, used_and_known)
  _, used_tracers = partition_list(used_refs, tracers)
  _, used_which_linear = partition_list(used_refs, which_linear)
  which_linear_unknown = (False,) * num_res + tuple(used_which_linear)
  unknown_inputs = [*residuals, *used_tracers]
  # Outputs match inputs so we construct output tracers that look like the input
  # tracers.
  res_ref_unknown_outputs = [
      pe.JaxprTracer(trace, pe.PartialVal.unknown(t.aval), None)
      for t in unknown_inputs]
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack):]
  source = source_info_util.current().replace(name_stack=name_stack)

  assert len(unknown_inputs) == len(res_ref_unknown_outputs)
  assert len(unknown_inputs) == len(jaxpr_unknown.invars) - 1
  eqn = pe.new_eqn_recipe(unknown_inputs, res_ref_unknown_outputs,
                          for_p, dict(jaxpr=jaxpr_unknown, nsteps=nsteps,
                                      reverse=reverse,
                                      which_linear=which_linear_unknown,
                                      unroll=unroll),
                          core.no_effects, source)
  for t in res_ref_unknown_outputs: t.recipe = eqn
  _, unknown_outputs = split_list(res_ref_unknown_outputs, [num_res])
  unknown_outputs, _ = partition_list(known_used, unknown_outputs)
  return merge_lists(in_unknowns, known_outputs, unknown_outputs)
pe.custom_partial_eval_rules[for_p] = _for_partial_eval

def _for_partial_eval_custom(saveable, in_unknowns, in_inst, eqn):
  jaxpr, nsteps, reverse, which_linear, unroll = split_dict(
      eqn.params, ["jaxpr", "nsteps", "reverse", "which_linear", "unroll"])
  num_inputs = len(eqn.invars)
  # We first need to run a fixpoint to determine which of the `Ref`s are unknown
  # after running the for loop. However, the jaxpr has no outputs. Instead, we
  # discharge the body and run the fixpoint with the discharged jaxpr. We can do
  # this because the outputs of the discharged jaxpr are one-to-one with the
  # inputs.
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, ())
  discharged_jaxpr = discharged_jaxpr.replace(
      invars=discharged_jaxpr.constvars + discharged_jaxpr.invars,
      constvars=[])
  in_unknowns, in_inst = list(in_unknowns), list(in_inst)
  out_unknowns, out_inst =  in_unknowns, in_inst
  for _ in range(num_inputs):
    jaxpr_in_unknowns = [False] * len(discharged_consts) + [False, *in_unknowns]
    _, _, out_unknowns, out_inst, _, = pe.partial_eval_jaxpr_custom(
        discharged_jaxpr, jaxpr_in_unknowns, True,
          ensure_out_unknowns=in_unknowns, ensure_out_inst=True,
          saveable=saveable)
    out_unknowns = list(out_unknowns)
    if out_unknowns == in_unknowns:
      break
    in_unknowns = map(operator.or_, in_unknowns, out_unknowns)
  else:
    if num_inputs > 0: raise Exception("Invalid fixpoint")
  del out_unknowns # Redundant since it's the same as `in_unknowns`
  new_inst = [x for x, inst in zip(eqn.invars, in_inst)
              if type(x) is core.Var and not inst]
  in_inst = [True] * len(eqn.invars)

  # We use `partial_eval_jaxpr_custom` here because it won't remove effectful
  # primitives like `get`/`set`.
  jaxpr_known_resout, jaxpr_staged_resin_, _, _, num_res = \
        pe.partial_eval_jaxpr_custom(jaxpr, [False, *in_unknowns],
            [True, *in_inst], [], [], saveable)

  # `partial_eval_jaxpr_custom` will give us jaxprs that have hybrid `Ref` and
  # non-Ref input/outputs. However, we'd like to bind these jaxprs to a
  # `for`, which expects only `Ref` inputs and no output. We need to convert
  # both of these jaxprs into ones that are compatible with `for`.
  # TODO(sharadmv,mattjj): implement "passthrough" optimization.
  # TODO(sharadmv,mattjj): rematerialize loop-dependent values instead of
  # passing the loop index as a residual

  # `jaxpr_known_resout` is a jaxpr that maps from all the input `Refs`
  # to output residual values (none of them should be `Ref`s). We'll need to
  # convert the output residual values into `Ref`s that are initially empty
  # `Ref`s that are written to at the end of the jaxpr.

  # # Loop-invariant residual optimization
  # Here we are interested in finding out which of the residuals are *not*
  # dependent on the loop index. If a residual is not dependent on the loop
  # index, we don't need add an extra loop dimension we're reading from when we
  # convert it from an output into a write.
  if nsteps > 1:
    loop_invar_res = _loop_invariant_outputs(jaxpr_known_resout)
  else:
    loop_invar_res = [True] * len(jaxpr_known_resout.outvars)

  jaxpr_known, res_avals = _convert_outputs_to_writes(nsteps,
                                                      jaxpr_known_resout,
                                                      loop_invar_res)

  known_invars, _ = partition_list(in_unknowns, eqn.invars)
  known_outvars, _ = partition_list(in_unknowns, eqn.outvars)
  newvar = core.gensym()
  resvars = map(newvar, res_avals)

  @lu.wrap_init
  def known(*known_vals):
    empty_res = map(ad_util.zeros_like_aval, res_avals)
    jaxpr_known_args = [*known_vals, *empty_res]
    jaxpr_known_which_linear = (False,) * len(jaxpr_known_args)
    return for_p.bind(*jaxpr_known_args, jaxpr=jaxpr_known, nsteps=nsteps,
                      reverse=reverse, which_linear=jaxpr_known_which_linear,
                      unroll=unroll)
  call_jaxpr_, _, call_jaxpr_consts = pe.trace_to_jaxpr_dynamic(
      known, [v.aval for v in known_invars])
  call_jaxpr = core.ClosedJaxpr(call_jaxpr_, call_jaxpr_consts)
  effects = _inner_to_outer_effects(
      [v.aval for v in call_jaxpr_.invars],
      [v.aval for v in known_invars],
      call_jaxpr_.effects)
  eqn_known = pe.new_jaxpr_eqn(known_invars, [*known_outvars, *resvars],
                               core.closed_call_p, dict(call_jaxpr=call_jaxpr),
                               effects, eqn.source_info)

  jaxpr_staged = _convert_inputs_to_reads(nsteps, len(res_avals),
                                          jaxpr_staged_resin_,
                                          loop_invar_res)
  which_linear_unknown = (False,) * num_res + tuple(which_linear)
  params_staged = dict(eqn.params, jaxpr=jaxpr_staged, reverse=reverse,
                                   nsteps=nsteps,
                                   which_linear=which_linear_unknown, unroll=unroll)
  @lu.wrap_init
  def staged(*res_and_refs):
    out_flat = for_p.bind(*res_and_refs, **params_staged)
    _, ans = split_list(out_flat, [num_res])
    _, ans = partition_list(out_inst, ans)
    return ans
  call_jaxpr_, _, call_jaxpr_consts = pe.trace_to_jaxpr_dynamic(
      staged, [v.aval for v in [*resvars, *eqn.invars]])
  assert len(jaxpr_staged.invars) - 1 == len(call_jaxpr_.invars)
  call_jaxpr = core.ClosedJaxpr(call_jaxpr_, call_jaxpr_consts)
  _, outvars = partition_list(out_inst, eqn.outvars)
  effects = _inner_to_outer_effects(
      [v.aval for v in call_jaxpr_.invars],
      [v.aval for v in [*resvars, *eqn.invars]],
      call_jaxpr_.effects)
  eqn_staged = pe.new_jaxpr_eqn([*resvars, *eqn.invars], outvars,
                               core.closed_call_p, dict(call_jaxpr=call_jaxpr),
                               effects, eqn.source_info)
  new_vars = [*new_inst, *resvars]
  return eqn_known, eqn_staged, in_unknowns, out_inst, new_vars

pe.partial_eval_jaxpr_custom_rules[for_p] = _for_partial_eval_custom

def _convert_outputs_to_writes(
    nsteps: Optional[int], jaxpr: core.Jaxpr, loop_invar_res: Sequence[bool],
    ) -> Tuple[core.Jaxpr, List[core.ShapedArray]]:
  assert not jaxpr.constvars, "Jaxpr shouldn't have constvars."

  in_avals = [v.aval for v in jaxpr.invars]  # [i, *orig_ref_avals]
  @lu.wrap_init
  def eval_jaxpr(i, *refs):
    # We split the refs into the original input refs and the dummy residual
    # refs.
    orig_refs, residual_refs = split_list(refs, [len(in_avals) - 1])
    residual_vals = core.eval_jaxpr(jaxpr, (), i, *orig_refs)
    for res_ref, res_val, loop_invar in zip(residual_refs, residual_vals,
                                            loop_invar_res):
      if loop_invar:
        res_ref[()] = res_val
      else:
        res_ref[i] = res_val
    return []
  # TODO(mattjj, sharadmv): better handling of tokens, which don't have shape/dtype
  res_ref_avals = [ShapedArrayRef(v.aval.shape, v.aval.dtype)  # pytype: disable=attribute-error
                   if loop_invar else
                   ShapedArrayRef((nsteps, *v.aval.shape), v.aval.dtype)  # pytype: disable=attribute-error
                   for v, loop_invar in zip(jaxpr.outvars, loop_invar_res)]
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [*in_avals, *res_ref_avals])
  assert not consts
  return jaxpr, [core.ShapedArray(a.shape, a.dtype) for a in res_ref_avals]

def _convert_inputs_to_reads(
    nsteps: Optional[int], num_res: int, jaxpr: core.Jaxpr,
    loop_invar_res: Sequence[bool]) -> core.Jaxpr:
  assert not jaxpr.constvars, "Jaxpr should not have constvars"

  @lu.wrap_init
  def eval_jaxpr(i, *refs):
    residual_refs, orig_refs = split_list(refs, [num_res])
    residual_vals = [r[()] if loop_invar else r[i] for r, loop_invar
                     in zip(residual_refs, loop_invar_res)]
    () = core.eval_jaxpr(jaxpr, (), *residual_vals, i, *orig_refs)
    return []

  res_val_avals, (i_aval,), orig_ref_avals = \
      split_list([v.aval for v in jaxpr.invars], [num_res, 1])
  res_ref_avals = [ShapedArrayRef(aval.shape, aval.dtype) if loop_invar else
                   ShapedArrayRef((nsteps, *aval.shape), aval.dtype)  # pytype: disable=attribute-error
                   for aval, loop_invar in zip(res_val_avals, loop_invar_res)]

  jaxpr, _, () = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [i_aval, *res_ref_avals, *orig_ref_avals])
  return jaxpr

def transpose_jaxpr(jaxpr: core.Jaxpr, which_linear: Sequence[bool]
                   ) -> core.Jaxpr:
  def trans(i, *args):
    # First we want to run the computation to read all the residual refs. We can
    # do that by using partial evaluation with all linear inputs unknown.
    res_jaxpr, tangent_jaxpr_, *_ = \
        _partial_eval_jaxpr_custom(jaxpr, [False, *which_linear],
                                   _save_everything)
    res_args = [x for x, lin in zip(args, which_linear) if not lin]
    res = core.eval_jaxpr(res_jaxpr, (), i, *res_args)

    # Now that we have residual values, we run the tangent jaxpr. It takes as
    # input the residuals, the loop index, and all the refs (at least, the ones
    # that are used in the body). Luckily, `tangent_jaxpr_` has all known and
    # unknown inputs!
    tangent_jaxpr, used = pe.dce_jaxpr(tangent_jaxpr_, [])
    used_res, (used_i,), used_ct = split_list(used, [len(res), 1])
    primals_args = [*(r for u, r in zip(used_res, res) if u)]
    if used_i:
      primals_args = [*primals_args, i]
    ct_args = [x for x, u in zip(args, used_ct) if u]
    ad.backward_pass(
        tangent_jaxpr, (), False, (), (*primals_args, *ct_args), ())
    return []
  jaxpr_trans, _, _ = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(trans), [v.aval for v in jaxpr.invars])
  return jaxpr_trans

def _for_transpose(in_cts, *args, jaxpr, nsteps, reverse, which_linear, unroll):
  # if any in_ct is nonzero, we definitely want it in args_ (and the
  # corresponding x in args could be an undefined primal, but doesnt have to be)
  # for non-res stuff:
  #   getting and setting => (nonzero ct, UndefinedPrimal arg)
  #   just setting =>        (nonzero ct, not UndefinedPrimal, dummy value)
  #   just getting =>        (zero ct   , UndefinedPrimal arg)
  # for res stuff:
  #                          (zero ct   , not UndefinedPrimal)
  args_ = []
  which_linear_transpose = []
  for x, ct in zip(args, in_cts):
    if   type(ct) is     ad_util.Zero and not ad.is_undefined_primal(x):
      # this is a residual, take x!
      args_.append(x)
      which_linear_transpose.append(False)
    elif type(ct) is     ad_util.Zero and     ad.is_undefined_primal(x):
      # the loop was 'just getting', plug in a zero
      args_.append(ad_util.zeros_like_aval(x.aval))
      which_linear_transpose.append(False)
    elif type(ct) is not ad_util.Zero and not ad.is_undefined_primal(x):
      # the loop was 'just setting', grab that cotangent! x is dummy
      args_.append(ct)
      which_linear_transpose.append(False)
    elif type(ct) is not ad_util.Zero and     ad.is_undefined_primal(x):
      # the loop was 'getting and setting', grab that cotangent!
      args_.append(ct)
      which_linear_transpose.append(True)
  jaxpr_transpose = transpose_jaxpr(jaxpr, which_linear)
  assert len(args_) == len(jaxpr_transpose.invars) - 1
  all_outs = for_p.bind(*args_, jaxpr=jaxpr_transpose, nsteps=nsteps,
                        reverse=not reverse,
                        which_linear=tuple(which_linear_transpose),
                        unroll=unroll)
  ct_outs = [ct if ad.is_undefined_primal(x) else None
             for x, ct in zip(args, all_outs)]
  return ct_outs
ad.primitive_transposes[for_p] = _for_transpose

### Testing utility

def discharged_for_loop(nsteps, body, init_state, *, reverse: bool = False):
  """A `for_loop` implementation that discharges its body right away.

  Potentially useful for testing and benchmarking.
  """
  flat_state, state_tree = tree_flatten(init_state)
  state_avals = map(val_to_ref_aval, flat_state)
  idx_aval = core.ShapedArray((), jnp.dtype("int32"))
  jaxpr, consts, out_tree = _trace_to_jaxpr_with_refs(
      body, state_tree, [idx_aval, *state_avals])
  if out_tree != tree_structure(None):
    raise Exception("`body` should not return anything.")
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, consts)

  def fori_body(i, carry):
    i = jnp.int32(i)
    if reverse:
      i = nsteps - i - 1
    out_flat = core.eval_jaxpr(discharged_jaxpr, discharged_consts,
                               i, *carry)
    return out_flat
  out_flat = loops.fori_loop(0, nsteps, fori_body, flat_state)
  return tree_unflatten(state_tree, out_flat)

def run_state(f, init_state):
  @functools.wraps(f)
  def wrapped_body(_, *args):
    return f(*args)
  return for_loop(1, wrapped_body, init_state)


### Bloop

# bounded_loop :: IntLit -> (() -> Bool) -> (() -> {State h} ()) -> () -> {State h} ()
# bounded_loop max_iters cond_fun body_fun carry =
#   i = 0
#   while cond_fun() and i < max_iters:
#     body_fun()
#     i += 1

class IDHashable:
  val: Any

  def __init__(self, val):
    self.val = val

  def __hash__(self) -> int:
    return id(self.val)

  def __eq__(self, other):
    return type(other) is IDHashable and id(self.val) == id(other.val)

def _unify_consts(cond_consts, body_consts) -> Tuple[List[Any], List[int],
                                                     List[int]]:
  id_map = {}
  cond_indices = [id_map.setdefault(IDHashable(x), len(id_map))
                  for x in cond_consts]
  body_indices = [id_map.setdefault(IDHashable(x), len(id_map))
                  for x in body_consts]
  all_consts = [a.val for a in id_map.keys()]
  return all_consts, cond_indices, body_indices

def _convert_all_inputs_to_refs(jaxpr: core.Jaxpr) -> core.Jaxpr:
  in_avals = [var.aval for var in jaxpr.invars[1:]]
  is_ref = [isinstance(aval, ShapedArrayRef) for aval in in_avals]

  ref_avals = [ShapedArrayRef(aval.shape, aval.dtype)
               if not isinstance(aval, ShapedArrayRef) else aval
               for aval in in_avals]
  def _convert(i, *refs):
    refs_and_vals = [ref if r else ref[()] for ref, r in zip(refs, is_ref)]
    return core.eval_jaxpr(jaxpr, (), i, *refs_and_vals)
  converted_jaxpr_, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(_convert), [core.ShapedArray((), jnp.int32), *ref_avals])
  assert not consts, "All consts should have been converted to refs"
  return converted_jaxpr_

def bloop(body_fun: Callable[[int], None],
          cond_fun: Optional[Callable[[], bool]] = None,
          max_iter: Optional[int] = None,
          unroll: int = 1) -> None:
  has_bound = max_iter is not None
  has_cond = cond_fun is not None
  if not (has_bound or has_cond):
    raise ValueError("Must provide either `max_iter` or `cond_fun` or both.")
  if not isinstance(unroll, int) or unroll < 1:
    raise ValueError(f"`unroll` is not positive integer: {unroll}")
  if unroll != 1 and has_cond:
    raise ValueError("Can only unroll loop when there is no `cond_fun`.")
  if has_bound:
    if max_iter < 0:
      raise ValueError("`max_iter` must be non-negative.")
    elif max_iter == 0:
      # The loop never happens!
      return
  body, _ = flatten_fun_nokwargs(lu.wrap_init(body_fun), tree_structure((0,)))
  body_jaxpr_, _, body_consts = pe.trace_to_jaxpr_dynamic(
      body, [core.ShapedArray((), jnp.int32)])
  if has_cond:
    cond, _ = flatten_fun_nokwargs(lu.wrap_init(cond_fun), treedef_tuple(()))
    cond_jaxpr, _, cond_consts = pe.trace_to_jaxpr_dynamic(cond, ())
    all_consts, cond_idx, body_idx = _unify_consts(cond_consts, body_consts)
    run_first_iteration, = core.eval_jaxpr(cond_jaxpr, cond_consts)
  else:
    cond_jaxpr = None
    cond_idx = []
    body_idx = list(range(len(body_consts)))
    run_first_iteration = True
    all_consts = body_consts

  @lu.wrap_init
  def body_with_cond(iteration, *args):
    body_args = [args[i] for i in body_idx]
    () = core.eval_jaxpr(body_jaxpr_, body_args, iteration)
    if cond_jaxpr is not None:
      cond_args = [args[i] for i in cond_idx]
      pred, = core.eval_jaxpr(cond_jaxpr, cond_args)
    else:
      pred = True
    return [pred]

  all_avals = [
      core.ShapedArray((), jnp.int32),
      *[core.raise_to_shaped(core.get_aval(a)) for a in all_consts]]
  body_jaxpr, _, () = pe.trace_to_jaxpr_dynamic(body_with_cond, all_avals)
  is_ref = [isinstance(aval, ShapedArrayRef) for aval in all_avals[1:]]
  if not all(is_ref):
    # Need to run_state to convert consts into refs (bloop only wants `Ref`
    # inputs)
    non_refs, refs = partition_list(is_ref, all_consts)
    def run_with_const_refs(const_refs):
      all_refs = merge_lists(is_ref, const_refs, refs)
      bloop_p.bind(run_first_iteration, *all_refs, max_iter=max_iter,
                   jaxpr=_convert_all_inputs_to_refs(body_jaxpr),
                   unroll=unroll)
    _ = run_state(run_with_const_refs, non_refs)
  else:
    body_jaxpr = body_jaxpr.replace(
        invars=[body_jaxpr.invars[0], *body_jaxpr.constvars,
                *body_jaxpr.invars[1:]])
    bloop_p.bind(run_first_iteration, *all_consts, max_iter=max_iter,
                 jaxpr=pe.convert_constvars_jaxpr(body_jaxpr),
                 unroll=unroll)
bloop_p = core.Primitive('bloop')

@bloop_p.def_effectful_abstract_eval
def _bloop_abstract_eval(*avals, max_iter, jaxpr, unroll):
  del max_iter, unroll
  pred_aval, *ref_avals = avals
  assert isinstance(pred_aval, core.ShapedArray)
  assert pred_aval.shape == ()
  assert pred_aval.dtype == jnp.bool_
  iter_aval = jaxpr.invars[0].aval
  assert isinstance(iter_aval, core.ShapedArray)
  assert iter_aval.shape == ()
  assert iter_aval.dtype == jnp.int32
  for aval, invar in zip(ref_avals, jaxpr.invars[1:]):
    assert isinstance(aval, ShapedArrayRef)
    assert isinstance(invar.aval, ShapedArrayRef)
    assert aval.shape == invar.aval.shape, (aval. invar.aval)
    assert aval.dtype == invar.aval.dtype, (aval. invar.aval)
  jaxpr_aval_effects = state.get_ref_state_effects(
      [v.aval for v in jaxpr.invars[1:]], jaxpr.effects)
  body_effects = [set(eff.replace(ref_aval=aval) for eff in effs) for aval, effs
                  in zip(ref_avals, jaxpr_aval_effects)
                  if isinstance(aval, ShapedArrayRef)]
  return core.ShapedArray((), jnp.int32), core.join_effects(*body_effects)

def _bloop_lowering_rule(ctx: mlir.LoweringRuleContext, *nodes, max_iter, **_):
  return [mlir.ir_constant(max_iter)]
  # return mlir.lower_fun(
  #     functools.partial(_bloop_discharge, ctx.avals_in, ctx.avals_out))(ctx,
  #         *nodes, **params)
mlir.register_lowering(bloop_p, _bloop_lowering_rule)


@state.register_discharge_rule(bloop_p)
def _bloop_discharge(in_avals, out_avals, *args, max_iter, jaxpr, unroll):
  discharged_jaxpr, body_consts = state.discharge_state(jaxpr, ())

  def cond_fun(carry):
    i, (keep_going, *_) = carry
    return keep_going & (i < max_iter)

  def _body(carry):
    i, (_, *states) = carry
    keep_going, *states = core.eval_jaxpr(discharged_jaxpr, body_consts, i, *states)
    return i + 1, (keep_going, *states)

  def body_fun(carry):
    i, states = carry
    for _ in range(unroll):
      i, states = _body((i, states))
    return i, tuple(states)

  i = jnp.int32(0)
  if max_iter is not None:
    for _ in range(max_iter % unroll):
      (i, args) = _body((i, args))
  num_iters, args = lax.while_loop(cond_fun, body_fun, (i, args))
  return args, num_iters

def _bloop_jvp(primals, tangents, *, max_iter: int, jaxpr: core.Jaxpr,
               unroll: int):
  _, *ref_tangents = tangents
  _, *ref_nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  # We need to find out which `Ref`s have nonzero tangents after running the
  # for loop. Ordinarily we do this with a fixed point on the body jaxpr but
  # a `bloop` body jaxpr is stateful. We use the discharged jaxpr
  # for the fixed point.
  discharged_jaxpr, body_consts = discharge_state(jaxpr, ())
  for _ in range(len(ref_nonzero_tangents)):
    _, out_nonzero_tangents = ad.jvp_jaxpr(
        core.ClosedJaxpr(discharged_jaxpr, body_consts),
        [False, *ref_nonzero_tangents], instantiate=[False, *ref_nonzero_tangents])
    _, *out_ref_nonzero_tangents = out_nonzero_tangents
    if out_ref_nonzero_tangents == ref_nonzero_tangents:
      break
    ref_nonzero_tangents = map(operator.or_, ref_nonzero_tangents,
                               out_ref_nonzero_tangents)
  else:
    raise Exception("Invalid fixpoint")
  ref_tangents = [ad.instantiate_zeros(t) if inst else t
                  for t, inst in zip(ref_tangents, ref_nonzero_tangents)]
  ref_tangents = [t for t in ref_tangents if type(t) is not ad_util.Zero]
  closed_jaxpr = core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(closed_jaxpr, [False, *ref_nonzero_tangents],
                               [False])
  jvp_jaxpr, () = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts  # TODO consts
  num_iters = bloop_p.bind(*primals, *ref_tangents, jaxpr=jvp_jaxpr,
                           max_iter=max_iter, unroll=unroll)
  return num_iters, ad.Zero(core.get_aval(num_iters).at_least_vspace())
ad.primitive_jvps[bloop_p] = _bloop_jvp

def _bloop_partial_eval(trace, *tracers, max_iter, jaxpr, unroll):
  raise NotImplementedError
pe.custom_partial_eval_rules[bloop_p] = _bloop_partial_eval

def _convert_output_to_writes_bloop(
    jaxpr: core.Jaxpr, loop_invar_res: Sequence[bool], max_iter: int
    ) -> Tuple[core.Jaxpr, Sequence[core.ShapedArray]]:
  assert not jaxpr.constvars, "Jaxpr shouldn't have constvars."

  in_avals = [v.aval for v in jaxpr.invars]  # orig_ref_avals
  @lu.wrap_init
  def eval_jaxpr(i, *refs):
    # We split the refs into the original input refs and the dummy residual
    # refs.
    orig_refs, residual_refs = split_list(refs, [len(in_avals) - 1])
    pred, *residual_vals = core.eval_jaxpr(jaxpr, (), i, *orig_refs)
    for res_ref, res_val, loop_invar in zip(residual_refs, residual_vals,
                                            loop_invar_res[1:]):
      if loop_invar:
        res_ref[()] = res_val
      else:
        res_ref[i] = res_val
    return [pred]
  i_aval, *ref_avals = in_avals
  res_ref_avals = [ShapedArrayRef(v.aval.shape, v.aval.dtype)  # pytype: disable=attribute-error
                   if loop_invar else
                   ShapedArrayRef((max_iter, *v.aval.shape), v.aval.dtype)  # pytype: disable=attribute-error
                   for v, loop_invar in zip(jaxpr.outvars[1:],
                                            loop_invar_res[1:])]
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [i_aval, *ref_avals, *res_ref_avals])
  assert not consts
  return jaxpr, [core.ShapedArray(x.shape, x.dtype) for x in res_ref_avals]

def _convert_inputs_linear_loop(
    jaxpr: core.Jaxpr, loop_invar_res: Sequence[bool], max_iter: int,
    ) -> core.Jaxpr:
  assert not jaxpr.constvars, "Jaxpr shouldn't have constvars."
  num_res = len(loop_invar_res)

  in_avals = [v.aval for v in jaxpr.invars]  # orig_ref_avals
  @lu.wrap_init
  def eval_jaxpr(i, *args):
    residuals, refs = split_list(args, [num_res])
    loop_residuals = []
    for res_ref, loop_invar in zip(residuals, loop_invar_res):
      if loop_invar:
        loop_res = res[()]
      else:
        loop_res = res_ref[i]
      loop_residuals.append(loop_res)
    core.eval_jaxpr(jaxpr, (), *loop_residuals, i, *refs)
    return []
  res_avals, (i_aval,), ref_avals = split_list(in_avals, [num_res, 1])
  res_ref_avals = [
      ShapedArrayRef((max_iter, *a.shape), a.dtype)
      if not invar else
      ShapedArrayRef(a.shape, a.dtype) for invar, a
      in zip(loop_invar_res, res_avals)]
  jaxpr_out, _, consts = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [i_aval, *res_ref_avals,
                   *ref_avals])
  assert all(isinstance(v.aval, ShapedArrayRef) for v in jaxpr_out.invars[1:])
  assert not consts
  return jaxpr_out

def _bloop_partial_eval_custom(saveable, in_unknowns, in_inst, eqn):
  pred_in_unknown, *in_unknowns = in_unknowns
  if pred_in_unknown:
    all_true = [True]
    new_inst = [x for x, inst in zip(eqn.invars, in_inst)
                if type(x) is core.Var and not inst]
    return None, eqn, all_true, all_true, new_inst
  pred_in_inst, *in_inst = in_inst
  max_iter, jaxpr, unroll = split_dict(
      eqn.params, ["max_iter", "jaxpr", "unroll"])
  num_inputs = len(eqn.invars)
  # We first need to run a fixpoint to determine which of the `Ref`s are unknown
  # after running the for loop. However, the jaxpr has no outputs. Instead, we
  # discharge the body and run the fixpoint with the discharged jaxpr. We can do
  # this because the outputs of the discharged jaxpr are one-to-one with the
  # inputs.
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, ())
  discharged_jaxpr = pe.convert_constvars_jaxpr(discharged_jaxpr)
  in_unknowns, in_inst = list(in_unknowns), list(in_inst)
  out_unknowns, out_inst =  in_unknowns, in_inst
  pred_out_unknown = pred_in_unknown

  for _ in range(num_inputs):
    jaxpr_in_unknowns = [False] * len(discharged_consts) + in_unknowns
    _, _, out_unknowns, out_inst, _, = pe.partial_eval_jaxpr_custom(
        discharged_jaxpr, [False, *jaxpr_in_unknowns], True,
          ensure_out_unknowns=[False] * num_inputs, ensure_out_inst=True,
          saveable=saveable)
    pred_out_unknown, *out_unknowns = out_unknowns
    if out_unknowns == in_unknowns:
      break
    in_unknowns = map(operator.or_, in_unknowns, out_unknowns)
  else:
    if num_inputs > 0: raise Exception("Invalid fixpoint")
  del out_unknowns # Redundant since it's the same as `in_unknowns`
  if pred_out_unknown:
    all_true = [True]
    new_inst = [x for x, inst in zip(eqn.invars, [pred_in_inst, *in_inst])
                if type(x) is core.Var and not inst]
    return None, eqn, all_true, all_true, new_inst
  new_inst = [x for x, inst in zip(eqn.invars[1:], in_inst)
              if type(x) is core.Var and not inst]
  in_inst = [True] * (len(eqn.invars) - 1)

  # We use `partial_eval_jaxpr_custom` here because it won't remove effectful
  # primitives like `get`/`set`.
  jaxpr_known_resout, jaxpr_staged_resin_, _, _, num_res = \
        pe.partial_eval_jaxpr_custom(jaxpr, [False, *in_unknowns],
                                     [True, *in_inst],
                                     [False], [True], saveable)

  # `partial_eval_jaxpr_custom` will give us jaxprs that have hybrid `Ref` and
  # non-Ref input/outputs. However, we'd like to bind these jaxprs to a
  # `for`, which expects only `Ref` inputs and no output. We need to convert
  # both of these jaxprs into ones that are compatible with `for`.
  # TODO(sharadmv,mattjj): implement "passthrough" optimization.
  # TODO(sharadmv,mattjj): rematerialize loop-dependent values instead of
  # passing the loop index as a residual

  # `jaxpr_known_resout` is a jaxpr that maps from all the input `Refs`
  # to output residual values (none of them should be `Ref`s). We'll need to
  # convert the output residual values into `Ref`s that are initially empty
  # `Ref`s that are written to at the end of the jaxpr.

  # # Loop-invariant residual optimization
  # Here we are interested in finding out which of the residuals are *not*
  # dependent on the loop index. If a residual is not dependent on the loop
  # index, we don't need add an extra loop dimension we're reading from when we
  # convert it from an output into a write.
  loop_invar_res = _loop_invariant_outputs(jaxpr_known_resout)

  jaxpr_known, res_avals = _convert_output_to_writes_bloop(
      jaxpr_known_resout, loop_invar_res, max_iter)
  # jaxpr_known expects an `i` as its first input to determine where to write
  # residuals to.

  known_invars, _ = partition_list([False, *in_unknowns], eqn.invars)
  _, *known_ref_invars = known_invars
  newvar = core.gensym()
  resvars = map(newvar, res_avals)

  @lu.wrap_init
  def known(pred, *known_refs):
    empty_res = map(ad_util.zeros_like_aval, res_avals)
    def run_bloop(all_refs):
      num_iter_ref, res_refs = all_refs
      num_iters = bloop_p.bind(pred, *known_refs, *res_refs,
                               jaxpr=jaxpr_known,
                               max_iter=max_iter,
                               unroll=unroll)
      num_iter_ref[()] = num_iters
      return
    num_iters, res = run_state(run_bloop, (0, empty_res))
    return [num_iters, *res]
  call_jaxpr_, _, call_jaxpr_consts = pe.trace_to_jaxpr_dynamic(
      known, [core.ShapedArray((), jnp.bool_),
              *(v.aval for v in known_ref_invars)])
  call_jaxpr = core.ClosedJaxpr(call_jaxpr_, call_jaxpr_consts)
  # TODO(sharadmv,mattjj): handle when num_iter is a dropvar and prints weird
  num_iter_outvar = eqn.outvars[0]
  effects = _inner_to_outer_effects(
      [v.aval for v in call_jaxpr_.invars[1:]],
      [v.aval for v in known_invars[1:]],
      call_jaxpr_.effects)
  eqn_known = pe.new_jaxpr_eqn(known_invars, [num_iter_outvar, *resvars],
                               core.closed_call_p, dict(call_jaxpr=call_jaxpr),
                               effects, eqn.source_info)

  jaxpr_staged_nodce = _convert_inputs_linear_loop(
      jaxpr_staged_resin_, loop_invar_res[1:], max_iter)
  jaxpr_staged, used_inputs = pe.dce_jaxpr(
      jaxpr_staged_nodce, [],
      instantiate=[True, *[False] * (len(jaxpr_staged_nodce.invars) - 1)])
  assert used_inputs[0]
  # used_inputs = [True, *used_inputs[1:]]  # Keep around loop index!
  _, used_ref_invars = partition_list(used_inputs[1 + num_res:], eqn.invars[1:])
  used_ref_avals = [var.aval for var in used_ref_invars]

  @lu.wrap_init
  def staged(num_iters, *res_and_refs):
    residuals, refs = split_list(res_and_refs, [num_res])
    def run_linear_loop(res_refs):
      _linear_loop(num_iters, res_refs, refs, jaxpr=jaxpr_staged,
                   max_iter=max_iter, unroll=unroll)
    run_state(run_linear_loop, residuals)
    return []
  call_jaxpr_, _, call_jaxpr_consts = pe.trace_to_jaxpr_dynamic(
      staged, [num_iter_outvar.aval, *res_avals, *used_ref_avals])
  run_state_eqn, = call_jaxpr_.eqns
  run_state_eqn_params = run_state_eqn.params
  new_which_linear = (True,) * (len(used_ref_avals) + 1) + (False,) * num_res
  assert len(new_which_linear) == len(run_state_eqn_params['which_linear'])
  call_jaxpr_ = call_jaxpr_.replace(eqns=[
    run_state_eqn.replace(
      params=dict(run_state_eqn_params, which_linear=new_which_linear))
  ])
  call_jaxpr = core.ClosedJaxpr(call_jaxpr_, call_jaxpr_consts)
  assert not call_jaxpr.jaxpr.outvars
  effects = _inner_to_outer_effects(
      [v.aval for v in call_jaxpr_.invars[1 + num_res:]],
      [v.aval for v in used_ref_invars],
      call_jaxpr_.effects)
  eqn_staged = pe.new_jaxpr_eqn([num_iter_outvar, *resvars, *used_ref_invars],
                                [], core.closed_call_p,
                                dict(call_jaxpr=call_jaxpr),
                                effects, eqn.source_info)
  new_vars = [num_iter_outvar, *new_inst, *resvars]
  return eqn_known, eqn_staged, [False], [True], new_vars
pe.partial_eval_jaxpr_custom_rules[bloop_p] = _bloop_partial_eval_custom

linear_loop_p = core.Primitive("linear_loop")
linear_loop_p.multiple_results = True

def _linear_loop(num_iters: Union[core.Tracer, int], residuals: Sequence[Any],
                 carry: Sequence[Any], *, jaxpr: core.Jaxpr, max_iter: int,
                 unroll: int) -> Sequence[Any]:
  # linear_loop :: Int -> (Int -> res -> c --o c) -> res -> c --o c
  # linear_loop num_trips body_fun res carry = 
  #   for i in range(num_trips):
  #     body_fun(i, res, carry)
  which_linear = (False,) * len(residuals) + (True,) * len(carry)
  return linear_loop_p.bind(num_iters, *residuals, *carry, jaxpr=jaxpr,
                            which_linear=which_linear, reverse=False,
                            max_iter=max_iter, unroll=unroll)

def _linear_loop_abstract_eval(*avals, jaxpr: core.Jaxpr,
                               which_linear: Tuple[bool, ...], reverse: bool,
                               max_iter: int, unroll: int):
  del reverse, max_iter, unroll
  assert len(avals) == len(jaxpr.invars)
  assert len(which_linear) == len(avals) - 1
  _, *ref_avals = avals
  assert all(isinstance(ref_aval, ShapedArrayRef) for ref_aval in ref_avals)
  jaxpr_aval_effects = state.get_ref_state_effects(
      [v.aval for v in jaxpr.invars[1:]], jaxpr.effects)
  body_effects = [set(eff.replace(ref_aval=aval) for eff in effs) for aval, effs
                  in zip(ref_avals, jaxpr_aval_effects)
                  if isinstance(aval, ShapedArrayRef)]
  return [], core.join_effects(*body_effects)
linear_loop_p.def_effectful_abstract_eval(_linear_loop_abstract_eval)

def _inner_to_outer_effects(inner_avals, outer_avals, effects):
    jaxpr_aval_effects = state.get_ref_state_effects(
        inner_avals, effects)
    aval_effects = [set(eff.replace(ref_aval=aval) for eff in effs) for aval, effs
                    in zip(outer_avals, jaxpr_aval_effects)
                    if isinstance(aval, ShapedArrayRef)]
    return core.join_effects(*aval_effects)

def _linear_loop_lowering(*args, **kwargs):
  return []
mlir.register_lowering(linear_loop_p, _linear_loop_lowering)

@state.register_discharge_rule(linear_loop_p)
def _linear_loop_discharge(in_avals, out_avals, num_iters, *args,
                           jaxpr: core.Jaxpr, which_linear: Tuple[bool, ...],
                           reverse: bool, max_iter: int, unroll: int):
  del in_avals, out_avals, which_linear
  if unroll != 1:
    raise NotImplementedError
  discharged_jaxpr, body_consts = state.discharge_state(jaxpr, ())

  def body_fun(i, carry):
    if reverse:
      i = num_iters - i - 1
    carry = core.eval_jaxpr(discharged_jaxpr, body_consts, i, *carry)
    return tuple(carry)
  orig_args = args
  args = lax.fori_loop(0, num_iters, body_fun, args)
  for arg, orig in zip(args, orig_args):
    assert arg.shape == orig.shape
  assert [a.shape == v.aval.shape for a, v in zip(args,
    discharged_jaxpr.invars[1:])]
  return [None, *args], []

def _linear_loop_transpose(_, num_iters, *refs, jaxpr: core.Jaxpr,
                           which_linear: Tuple[bool, ...], reverse: bool,
                           max_iter: int, unroll: int):
  # NumIters -> Residuals -> Carry --o Carry
  jaxpr_transpose = transpose_jaxpr(jaxpr, which_linear)
  linear_loop_p.bind(num_iters, *refs, jaxpr=jaxpr_transpose,
                     which_linear=which_linear, reverse=not reverse,
                     max_iter=max_iter, unroll=unroll)
  # linear_loop mutates the `Ref`s.
  return [ad.Zero(core.get_aval(num_iters).at_least_vspace()),
          *[None] * len(refs)]
ad.primitive_transposes[linear_loop_p] = _linear_loop_transpose

def _linear_loop_jvp(primals, tangents, *, jaxpr: core.Jaxpr,
                     which_linear: Tuple[bool, ...], reverse: bool,
                     max_iter: int, unroll: int):
  _, *ref_tangents = tangents
  _, *nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  # We need to find out which `Ref`s have nonzero tangents after running the
  # for loop. Ordinarily we do this with a fixed point on the body jaxpr but
  # a `bloop` body jaxpr is stateful. We use the discharged jaxpr
  # for the fixed point.
  discharged_jaxpr, body_consts = discharge_state(jaxpr, ())
  for _ in range(len(nonzero_tangents)):
    _, out_nonzero_tangents = ad.jvp_jaxpr(
        core.ClosedJaxpr(discharged_jaxpr, body_consts),
        [False, *nonzero_tangents], instantiate=nonzero_tangents)
    if out_nonzero_tangents == nonzero_tangents:
      break
    nonzero_tangents = map(operator.or_, nonzero_tangents,
                           out_nonzero_tangents)
  else:
    raise Exception("Invalid fixpoint")
  ref_tangents = [t for t in ref_tangents if type(t) is not ad_util.Zero]
  closed_jaxpr = core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(closed_jaxpr, [False, *nonzero_tangents], [])
  jvp_jaxpr, () = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts  # TODO consts
  jvp_which_linear = tuple((*which_linear, *(True,) * len(ref_tangents)))
  linear_loop_p.bind(*primals, *ref_tangents, jaxpr=jvp_jaxpr,
                     which_linear=jvp_which_linear, reverse=reverse,
                     max_iter=max_iter, unroll=unroll)
  return [], []
ad.primitive_jvps[linear_loop_p] = _linear_loop_jvp

def _linear_loop_partial_eval_custom(saveable, in_unknowns, in_inst, eqn):
  jaxpr, which_linear, reverse, max_iter, unroll = split_dict(
      eqn.params, ["jaxpr", "which_linear", "reverse", "max_iter", "unroll"])
  # assert all(not linear or (linear and unknown) for linear, unknown
  #            in zip(in_unknowns[1:], which_linear))

  num_inputs = len(eqn.invars)
  # We first need to run a fixpoint to determine which of the `Ref`s are unknown
  # after running the for loop. However, the jaxpr has no outputs. Instead, we
  # discharge the body and run the fixpoint with the discharged jaxpr. We can do
  # this because the outputs of the discharged jaxpr are one-to-one with the
  # inputs.
  orig_in_unknowns = in_unknowns
  num_iter_unknown, *in_unknowns = in_unknowns
  iter_inst, *in_inst = in_inst
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, ())
  discharged_jaxpr = discharged_jaxpr.replace(
      invars=discharged_jaxpr.constvars + discharged_jaxpr.invars,
      constvars=[])
  out_unknowns, out_inst =  in_unknowns, in_inst

  for _ in range(num_inputs):
    jaxpr_in_unknowns = (
        [False] * len(discharged_consts) + [False, *in_unknowns])
    _, _, out_unknowns, out_inst, _, = pe.partial_eval_jaxpr_custom(
        discharged_jaxpr, jaxpr_in_unknowns, True,
          ensure_out_unknowns=in_unknowns, ensure_out_inst=True,
          saveable=saveable)
    out_unknowns = list(out_unknowns)
    if out_unknowns == in_unknowns:
      break
    in_unknowns = map(operator.or_, in_unknowns, out_unknowns)
  else:
    if num_inputs > 0: raise Exception("Invalid fixpoint")
  del out_unknowns # Redundant since it's the same as `in_unknowns`

  def _save_reads(prim, *avals, **params):
    return not (prim is state_primitives.get_p)
  known_invars, _ = partition_list([num_iter_unknown, *in_unknowns], eqn.invars)
  jaxpr_known_resout_, jaxpr_staged_resin_, uk_out, inst_out, num_res = \
        pe.partial_eval_jaxpr_custom(
            jaxpr,
            in_unknowns=[False, *in_unknowns],
            in_inst=True,
            ensure_out_inst=[],
            ensure_out_unknowns=[],
            saveable=_save_reads)
  loop_invar_res = _loop_invariant_outputs(jaxpr_known_resout_)
  jaxpr_known, res_avals = _convert_outputs_to_writes(
      max_iter, jaxpr_known_resout_, loop_invar_res)
  res_ref_avals = [ShapedArrayRef(a.shape, a.dtype) for a in res_avals]
  known_ref_avals = [v.aval for v in known_invars[1:]]
  staged_ref_avals = [v.aval for v in eqn.invars[1:]]
  newvar = core.gensym()
  resvars = map(newvar, res_avals)

  known_which_linear, _ = partition_list(in_unknowns, which_linear)
  ref_avals = [v.aval for v in jaxpr.invars[1:]]
  if num_iter_unknown:
    eqn_known = None
  elif num_res > 0:
    @lu.wrap_init
    def known(num_iter, *known_refs):
      @lu.wrap_init
      def _run_loop(num_iter_ref, *all_refs):
        linear_loop_p.bind(num_iter_ref[()], *all_refs,
                           jaxpr=jaxpr_known, reverse=reverse,
                           max_iter=max_iter,
                           which_linear=(*known_which_linear, *[False] *
                             len(empty_res)),
                           unroll=unroll)

        return []
      empty_res = map(ad_util.zeros_like_aval, res_avals)
      run_state_jaxpr, _, () = pe.trace_to_jaxpr_dynamic(
          _run_loop, [ShapedArrayRef((), jnp.int32), *known_ref_avals,
                      *res_ref_avals])
      linear = (False, *known_which_linear, *[False] * len(empty_res))
      out = run_state_bind(num_iter, *known_refs, *empty_res, 
                     jaxpr=run_state_jaxpr,
                     which_linear=linear)
      _, out_known, out_res = split_list(out, [1, len(known_refs)])
      return out_res
    known_jaxpr_, _, known_consts = pe.trace_to_jaxpr_dynamic(
        known, [core.ShapedArray((), jnp.int32), *known_ref_avals])
    known_jaxpr = core.ClosedJaxpr(known_jaxpr_, known_consts)
    effects = _inner_to_outer_effects(
        [v.aval for v in known_jaxpr_.invars[1:]],
        known_ref_avals,
        known_jaxpr_.effects)
    eqn_known = pe.new_jaxpr_eqn(
        known_invars, resvars, core.closed_call_p,
        dict(call_jaxpr=known_jaxpr),
        effects, eqn.source_info)
    for invar, jaxpr_invar in zip(known_invars, known_jaxpr_.invars):
      assert invar.aval.dtype == jaxpr_invar.aval.dtype
      assert invar.aval.shape == jaxpr_invar.aval.shape
  else:
    effects = _inner_to_outer_effects(
        [v.aval for v in jaxpr_known.invars[1:]],
        known_ref_avals,
        jaxpr_known.effects)
    eqn_known = pe.new_jaxpr_eqn(
        known_invars, [], linear_loop_p,
        dict(jaxpr=jaxpr_known, reverse=reverse,
             which_linear=tuple(known_which_linear), max_iter=max_iter,
             unroll=unroll),
        effects, eqn.source_info)
  jaxpr_staged = _convert_inputs_to_reads(max_iter, num_res,
      jaxpr_staged_resin_, loop_invar_res)
  which_linear_staged = tuple(in_unknowns)
  if num_res > 0:
    @lu.wrap_init
    def staged(num_iter, *res_and_refs):
      res, refs = split_list(res_and_refs, [num_res])
      @lu.wrap_init
      def _run_loop(num_iter_ref, *all_refs):
        linear_loop_p.bind(num_iter_ref[()], *all_refs,
                           jaxpr=jaxpr_staged, reverse=reverse,
                           max_iter=max_iter,
                           which_linear=(*[False] * num_res,
                                         *which_linear_staged),
                           unroll=unroll)

        return []
      run_state_jaxpr, _, () = pe.trace_to_jaxpr_dynamic(
          _run_loop, [ShapedArrayRef((), jnp.int32), *res_ref_avals,
                      *ref_avals])
      linear = (False, *[False] * num_res, *which_linear_staged)
      run_state_bind(num_iter, *res, *refs, 
                     jaxpr=run_state_jaxpr,
                     which_linear=linear)
      return []
    staged_jaxpr_, _, staged_consts = pe.trace_to_jaxpr_dynamic(
        staged, [core.ShapedArray((), jnp.int32), *res_avals, *ref_avals])
    effects = _inner_to_outer_effects(
        [v.aval for v in staged_jaxpr_.invars[1 + num_res:]],
        staged_ref_avals,
        staged_jaxpr_.effects)
    staged_jaxpr = core.ClosedJaxpr(staged_jaxpr_, staged_consts)
    num_iter_var, *ref_vars = eqn.invars
    eqn_staged = pe.new_jaxpr_eqn(
        [num_iter_var, *resvars, *ref_vars],
        [], core.closed_call_p, 
        dict(call_jaxpr=staged_jaxpr),
        effects, eqn.source_info)
    for invar, jaxpr_invar in zip(eqn_staged.invars, staged_jaxpr_.invars):
      assert invar.aval.dtype == jaxpr_invar.aval.dtype
      assert invar.aval.shape == jaxpr_invar.aval.shape
  else:
    effects = _inner_to_outer_effects(
        [v.aval for v in jaxpr_staged.invars[1:]],
        staged_ref_avals,
        jaxpr_staged.effects)
    eqn_staged = pe.new_jaxpr_eqn(
        eqn.invars, [], linear_loop_p,
        dict(jaxpr=jaxpr_staged, reverse=reverse,
             which_linear=tuple(which_linear_staged), max_iter=max_iter,
             unroll=unroll),
        effects, eqn.source_info)
    resvars = []
  new_inst = [x for x, inst in zip(eqn.invars, [iter_inst, *in_inst])
              if type(x) is core.Var and not inst]
  return eqn_known, eqn_staged, [], [], [*new_inst, *resvars]

pe.partial_eval_jaxpr_custom_rules[linear_loop_p] = _linear_loop_partial_eval_custom
