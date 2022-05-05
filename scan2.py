import operator
import ipdb, sys, traceback
def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()
sys.excepthook = info

from functools import partial
from typing import TypeVar, Any, Sequence, List, Tuple, Generic, Callable, Optional

import jax
import jax.numpy as jnp

from jax import core
from jax import linear_util as lu
from jax.config import config
from jax.tree_util import (tree_flatten, tree_unflatten, tree_structure,
                           treedef_tuple)
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.interpreters import mlir
from jax._src import ad_util
from jax._src import source_info_util
from jax._src.api_util import flatten_fun_nokwargs
from jax._src.util import (safe_map, safe_zip, split_list, merge_lists,
                           partition_list)
import jax._src.pretty_printer as pp
import numpy as np


config.update('jax_traceback_filtering', 'off')

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

# State effect
class State: pass
State = State()


def _partial_eval_jaxpr_custom(jaxpr, in_unknowns, policy):
  return pe.partial_eval_jaxpr_custom(
      jaxpr, in_unknowns, [True] * len(in_unknowns), False, False,
      policy)
## primitives for the state effect

def ref_get(ref, idx):
  return get_p.bind(ref, *idx)

get_p = core.Primitive('get')

@get_p.def_effectful_abstract_eval
def get_abstract_eval(ref, *idx):
  return core.ShapedArray(ref.shape[len(idx):], ref.dtype), {State}

def _get_jvp(primals, tangents):
  primal_ref, *idx = primals
  tangent_ref, *_ = tangents
  return ref_get(primal_ref, idx), ref_get(tangent_ref, idx)
ad.primitive_jvps[get_p] = _get_jvp

pp_ref = partial(pp.color, intensity=pp.Intensity.NORMAL,
                 foreground=pp.Color.GREEN)

def _get_pp_rule(eqn, context, settings):
  y, = eqn.outvars
  x, *idx = eqn.invars
  # pretty-print `y = get x i` as `x[i] <- v`
  idx = ','.join(core.pp_var(i, context) for i in idx)
  lhs = core.pp_vars([y], context, print_shapes=settings.print_shapes)
  return pp.concat([lhs, pp.text(' <- '),
                    pp_ref(
                    pp.concat([
                      pp.text(core.pp_var(x, context)),
                      pp.text('['), pp.text(idx), pp.text(']')
                    ]))])
core.pp_eqn_rules[get_p] = _get_pp_rule

def _get_dce_rule(
    used_outs: Sequence[bool],
    eqn: core.JaxprEqn) -> Tuple[List[bool], Optional[core.JaxprEqn]]:
  if any(used_outs):
    return [True] * len(eqn.invars), eqn
  return [False] * len(eqn.invars), None
# pe.dce_rules[get_p] = _get_dce_rule


def ref_set(ref, idx, x):
  _ = ref_swap(ref, idx, x)
def ref_swap(ref, idx, x):
  return swap_p.bind(ref, *idx, x)
swap_p = core.Primitive('swap')

@swap_p.def_effectful_abstract_eval
def swap_abstract_eval(ref, *idx_x):
  *idx, x = idx_x
  return core.raise_to_shaped(x), {State}

def _swap_jvp(primals, tangents):
  primal_ref, *idx, x = primals
  tangent_ref, *_, xdot = tangents
  xdot = ad_util.instantiate(xdot)
  return ref_swap(primal_ref, idx, x), ref_swap(tangent_ref, idx, xdot)
ad.primitive_jvps[swap_p] = _swap_jvp

def _swap_pp_rule(eqn, context, settings):
  y, = eqn.outvars
  x, *idx, v = eqn.invars
  idx = ','.join(core.pp_var(i, context) for i in idx)
  if type(y) is core.DropVar:
    # pretty-print `_ = swap x i v` as `x[i] <- v`
    del y
    return pp.concat([
      pp_ref(pp.concat([
        pp.text(core.pp_var(x, context)),
        pp.text('['), pp.text(idx), pp.text(']')
      ])), pp.text(' <- '), pp.text(core.pp_var(v, context))])
  else:
    # pretty-print `y:T = swap x i v` as `y:T, x[i] <- x[i], v`
    x_i = pp.concat([pp.text(core.pp_var(x, context)),
                     pp.text('['), pp.text(idx), pp.text(']')])
    y = core.pp_vars([y], context, print_shapes=settings.print_shapes)
    return pp.concat([y, pp.text(', '), x_i, pp.text(' <- '),
                      x_i, pp.text(', '), pp.text(core.pp_var(v, context))])
core.pp_eqn_rules[swap_p] = _swap_pp_rule


def ref_addupdate(ref, idx, x):
  addupdate_p.bind(ref, *idx, x)
addupdate_p = core.Primitive('addupdate')
addupdate_p.multiple_results = True

@addupdate_p.def_effectful_abstract_eval
def addupdate_abstract_eval(ref, *idx_x):
  del ref, idx_x  # Unused.
  return [], {State}

def _addupdate_pp_rule(eqn, context, settings):
  () = eqn.outvars
  x, *idx, v = eqn.invars
  # pretty-print ` = addupdate x i v` as `x[i] += v`
  idx = ','.join(core.pp_var(i, context) for i in idx)
  return pp.concat([
    pp_ref(pp.concat([
        pp.text(core.pp_var(x, context)),
        pp.text('['), pp.text(idx), pp.text(']')
      ])), pp.text(' += '), pp.text(core.pp_var(v, context))])
core.pp_eqn_rules[addupdate_p] = _addupdate_pp_rule

def addupdate_jvp_rule(primals, tangents):
  ref_primal, *idx_primal, x_primal = primals
  ref_tangent, *_, x_tangent = tangents
  x_tangent = ad_util.instantiate(x_tangent)
  addupdate_p.bind(ref_primal, *idx_primal, x_primal)
  addupdate_p.bind(ref_tangent, *idx_primal, x_tangent)
  return [], []
ad.primitive_jvps[addupdate_p] = addupdate_jvp_rule


def addupdate_transpose(ctx_in, *ref_idx_x):
  # x += a
  # looks like this:
  # b <- {x}
  # c = a + b
  # {x} <- c
  # transpose should be:
  # cbar <- {x}
  # abar, bbar = cbar, cbar
  # {x} <- bbar
  del ctx_in
  ref, *idx, x = ref_idx_x
  g = ref_get(ref, idx)
  return [None] + [None] * len(idx) + [g]
ad.primitive_transposes[addupdate_p] = addupdate_transpose

## aval for refs

class ShapedArrayRef(core.AbstractValue):
  __slots__ = ['shape', 'dtype']

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  def _getitem(self, tracer, idx):
    if not isinstance(idx, tuple):
      idx = idx,
    return ref_get(tracer, idx)

  def _setitem(self, tracer, idx, val):
    if not isinstance(idx, tuple):
      idx = idx,
    return ref_set(tracer, idx, val)

  def __repr__(self) -> str:
    a = core.ShapedArray(self.shape, self.dtype)
    return f'Ref{{{a.str_short()}}}'

  def at_least_vspace(self):
    return self

core.raise_to_shaped_mappings[ShapedArrayRef] = lambda aval, _: aval

def prnt(jaxpr):
  jaxpr = getattr(jaxpr, 'jaxpr', jaxpr)
  return print(jaxpr.pretty_print(use_color=True))


# @lu.wrap_init
# def f(i, r):
#   x = r[i]
#   r[i] = 2 * x
#   return x + 1,  # flat
# in_avals = [core.ShapedArray((), jnp.dtype('int32')),
#             ShapedArrayRef((4,), jnp.dtype('float32'))]
# jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(f, in_avals)
# prnt(jaxpr)


# AD

def f(r):
  x = r[0]
  r[1] = jnp.cos(x)

in_avals = [ShapedArrayRef((4,), jnp.dtype('float32'))]
jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(lu.wrap_init(lambda r: f(r) or ()), in_avals)
prnt(jaxpr)

print("==> JVP ==>")

@lu.wrap_init
def g(r, rdot):
  jax.jvp(f, (r,), (rdot,))
  return ()

in_avals = [ShapedArrayRef((4,), jnp.dtype('float32')),
            ShapedArrayRef((4,), jnp.dtype('float32'))]
jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(g, in_avals)
prnt(jaxpr)


print("==> PE ==>")
# pe._partial_eval_jaxpr_custom handles effects, at least these ones!
jaxpr_known, jaxpr_staged_, out_unk, out_inst, num_res = \
    _partial_eval_jaxpr_custom(jaxpr, [False, True], lambda *_: True)
prnt(jaxpr_known)
jaxpr_staged, _ = pe.dce_jaxpr(jaxpr_staged_,
                               [True] * len(jaxpr_staged_.outvars))
prnt(jaxpr_staged)
# so just call this sucker in the partial eval rule for the loop


def _get_transpose(g, ref, *idx):
  if type(g) is not ad_util.Zero:
    ref_addupdate(ref, idx, g)
  return [None] + [None] * len(idx)
ad.primitive_transposes[get_p] = _get_transpose

def _swap_transpose(g, ref, *idx_x):
  *idx, x = idx_x
  x_bar = ref_swap(ref, idx, ad_util.instantiate(g))
  return [None] + [None] * len(idx) + [x_bar]
ad.primitive_transposes[swap_p] = _swap_transpose

print("==> TRANSPOSE ==>")
avals = [x.aval for x in jaxpr_staged.outvars]
def trans(res, ref):
  ad.backward_pass(jaxpr_staged, (), (), (), (res, ref), ())
  return []
jaxpr_trans, _, _ = pe.trace_to_jaxpr_dynamic(
    lu.wrap_init(trans), [core.ShapedArray((), jnp.dtype('float32')),
                          ShapedArrayRef((4,), jnp.dtype('float32'))])
prnt(jaxpr_trans)


# discharge!

def discharge_state(jaxpr: core.Jaxpr, consts: Sequence[Any]
                    ) -> Tuple[core.Jaxpr, List[Any]]:
  in_avals = [core.ShapedArray(v.aval.shape, v.aval.dtype)
              if type(v.aval) is ShapedArrayRef
              else v.aval for v in jaxpr.invars]
  assert len(jaxpr.outvars) == 0, jaxpr
  eval_jaxpr = lu.wrap_init(partial(_eval_jaxpr_discharge_state, jaxpr, consts))
  new_jaxpr, _, new_consts = pe.trace_to_jaxpr_dynamic(eval_jaxpr, in_avals)
  return new_jaxpr, new_consts

def _eval_jaxpr_discharge_state(jaxpr: core.Jaxpr, consts: List[Any], *args: Any):
  assert type(jaxpr) == core.Jaxpr
  env: Dict[core.Var, Any] = {}

  def read(x: core.Atom) -> Any:
    if type(x) is core.Literal:
      return x.val
    return env[x]

  def write(v: core.Var, val: Any) -> None:
    env[v] = val

  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    if eqn.primitive is get_p:
      # `y = x[i]` becomes `y = ds x i`
      x, *idx = in_vals
      write(eqn.outvars[0], dynamic_index(x, idx))
    elif eqn.primitive is swap_p:
      # `x_i = swap(x[i], val)` becomes `x_i = ds x i; updated_x = dus x i val`
      x, *idx, val = in_vals
      write(eqn.outvars[0], dynamic_index(x, idx))
      write(eqn.invars[0], dynamic_update_index(x, idx, val))
    elif eqn.primitive is addupdate_p:
      # `x[i] += val` becomes `x = dus x i (val + (ds x i))
      x, *idx, val = in_vals
      ans = dynamic_update_index(x, idx, val + dynamic_index(x, idx))
      write(eqn.invars[0], ans)
    else:
      # standard eval_jaxpr stuff (NOTE assumes no State effects possible here!)
      subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
      ans = eqn.primitive.bind(*subfuns, *map(read, eqn.invars), **bind_params)
      if eqn.primitive.multiple_results:
        map(write, eqn.outvars, ans)
      else:
        write(eqn.outvars[0], ans)
  outvals = map(read, jaxpr.outvars)
  return outvals + [read(v) for v in jaxpr.invars if type(v.aval) is ShapedArrayRef]

def dynamic_index(x, idx):
  if not idx: return x
  ndim = len(x.shape)
  starts = [*idx] + [jax.lax.full_like(idx[0], 0, shape=())] * (ndim - len(idx))
  sizes = (1,) * len(idx) + x.shape[len(idx):]
  out = jax.lax.dynamic_slice(x, starts, sizes)
  return out.reshape(x.shape[len(idx):])

def dynamic_update_index(x, idx, val):
  if not idx: return val
  ndim = len(x.shape)
  starts = [*idx] + [jax.lax.full_like(idx[0], 0, shape=())] * (ndim - len(idx))
  update = val.reshape((1,) * len(idx) + x.shape[len(idx):])
  return jax.lax.dynamic_update_slice(x, update, starts)

# tracing utilities

def _trace_to_jaxpr(f, state_tree, state_avals):
  f, out_tree = flatten_fun_nokwargs(
      lu.wrap_init(f), treedef_tuple((tree_structure(0), state_tree)))
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      f, [core.ShapedArray((), jnp.dtype('int32')), *state_avals])
  jaxpr = _hoist_consts_to_refs(jaxpr)
  # jaxpr has no more consts
  if out_tree() != tree_structure(None): raise Exception
  return jaxpr, consts

def make_ref(x) -> ShapedArrayRef:
  aval = core.raise_to_shaped(core.get_aval(x))
  if type(aval) is not core.ShapedArray:
    raise Exception(f"can't make ref from {x}")
  # if not aval.shape:
  #   raise Exception(f"can't make ref from value with scalar shape {aval.shape}")
  return ShapedArrayRef(aval.shape, aval.dtype)

# Type annotations
S = TypeVar('S')
class Ref(Generic[TypeVar('T')]): pass

def abstractify(x: Any) -> core.AbstractValue:
  return core.raise_to_shaped(core.get_aval(x))

def _hoist_consts_to_refs(jaxpr: core.Jaxpr) -> core.Jaxpr:
  num_consts = len(jaxpr.constvars)

  def _hoist(i, *consts_args):
    const_refs, args = split_list(consts_args, [num_consts])
    consts = [r[()] for r in const_refs]
    return core.eval_jaxpr(jaxpr, consts, i, *args)
  const_avals = [ShapedArrayRef(var.aval.shape, var.aval.dtype) for var in
                 jaxpr.constvars]
  i_aval, *arg_avals = [var.aval for var in jaxpr.invars]
  in_avals = [i_aval, *const_avals, *arg_avals]
  hoisted_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(_hoist), in_avals)
  assert not consts, "All consts should have been converted to refs"
  return hoisted_jaxpr


print('loop!!!!')
# for: Int -> (Int -> Ref s -> {State s} ()) -> s -> s

def for_loop(nsteps: int, body: Callable[[int, Ref[S]], None],
             init_state: S) -> S:
  init_state, state_tree = tree_flatten(init_state)
  jaxpr, consts = _trace_to_jaxpr(body, state_tree, map(make_ref, init_state))
  which_linear = (False,) * (len(consts) + len(init_state))
  out_flat = for_p.bind(*consts, *init_state, jaxpr=jaxpr, nsteps=int(nsteps),
                        reverse=False, which_linear=which_linear)
  out_flat = out_flat[len(consts):]
  return tree_unflatten(state_tree, out_flat)
for_p = core.Primitive('for')
for_p.multiple_results = True

@for_p.def_abstract_eval
def _for_abstract_eval(*_, jaxpr, **__):
  return [core.ShapedArray(v.aval.shape, v.aval.dtype) for v in jaxpr.invars
          if type(v.aval) is ShapedArrayRef]

for_p.def_impl(partial(xla.apply_primitive, for_p))

def _for_impl(*args, jaxpr, nsteps, reverse, which_linear):
  del which_linear
  lowered_jaxpr, consts = discharge_state(jaxpr, ())
  def cond(carry):
    i, _ = carry
    return i < nsteps
  def body(carry):
    i, state = carry
    i_ = nsteps - i - 1 if reverse else i
    new_state = core.eval_jaxpr(lowered_jaxpr, consts, i_, *state)
    return i + 1, new_state
  _, state = jax.lax.while_loop(cond, body, (0, [*args]))
  return state
mlir.register_lowering(for_p, mlir.lower_fun(_for_impl, multiple_results=True))

def _for_jvp(primals, tangents, *, jaxpr, nsteps, reverse, which_linear):
  nonzero_tangents = [type(t) is not ad_util.Zero for t in tangents]
  body_jaxpr, body_consts = discharge_state(jaxpr, ())
  for _ in range(len(nonzero_tangents)):
    _, out_nonzero_tangents = ad.jvp_jaxpr(
        core.ClosedJaxpr(body_jaxpr, body_consts), [False] + nonzero_tangents,
        instantiate=nonzero_tangents)
    if out_nonzero_tangents == nonzero_tangents:
      break
    nonzero_tangents = map(operator.or_, nonzero_tangents, out_nonzero_tangents)
  else:
    raise Exception
  tangents = [ad.instantiate_zeros(t) if inst else t for t, inst in
      zip(tangents, nonzero_tangents)]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  jaxpr_ = core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(jaxpr_, [False] + nonzero_tangents, [])
  jvp_jaxpr, jvp_consts = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts
  jvp_which_linear = ((False,) * len(jvp_consts) + which_linear
                      + (True,) * len(tangents))
  out_flat = for_p.bind(*jvp_consts, *primals, *tangents, jaxpr=jvp_jaxpr,
                        nsteps=nsteps, reverse=reverse,
                        which_linear=jvp_which_linear)
  _, out_primals, out_tangents = split_list(out_flat, [len(jvp_consts), len(primals)])
  out_tangents_iter = iter(out_tangents)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_value(p)
                  for p, nz in zip(out_primals, nonzero_tangents)]
  return out_primals, out_tangents
ad.primitive_jvps[for_p] = _for_jvp

def _for_partial_eval(trace, *tracers, jaxpr, nsteps, reverse, which_linear):
  in_unknowns = [not t.pval.is_known() for t in tracers]
  body_jaxpr, body_consts = discharge_state(jaxpr, ())
  body_jaxpr = body_jaxpr.replace(
      invars=body_jaxpr.constvars + body_jaxpr.invars,
      constvars=[])
  for _ in range(len(in_unknowns)):
    _, _, out_unknowns, _, num_res = \
        _partial_eval_jaxpr_custom(body_jaxpr,
            [False] * len(body_consts) + [False, *in_unknowns], _save_anything)
    out_unknowns = list(out_unknowns)
    if out_unknowns == in_unknowns:
      break
    in_unknowns = map(operator.or_, in_unknowns, out_unknowns)
  else:
    raise Exception
  tracers = [trace.instantiate_const(t) if uk else t
             for t, uk in zip(tracers, out_unknowns)]

  def _remat_state(primitive, *in_avals, **params):
    return primitive not in (get_p, swap_p, addupdate_p)

  if USE_PASSTHROUGH_OPTIMIZATION:
    jaxpr_known_resout, jaxpr_unknown_resin_, _, _, num_res = \
        _partial_eval_jaxpr_custom(jaxpr, [False, *in_unknowns], _remat_state)
    jaxpr_unknown_resin, used_inputs = pe.dce_jaxpr(
        jaxpr_unknown_resin_, [], [True] * num_res + [True, *in_unknowns])
    used_res, (used_i,), used_refs = split_list(used_inputs, [num_res, 1])
  else:
    jaxpr_known_resout, jaxpr_unknown_resin_, _, _, num_res = \
        _partial_eval_jaxpr_custom(jaxpr, [False, *in_unknowns], _save_anything)
    jaxpr_unknown_resin, used_inputs = pe.dce_jaxpr(
        jaxpr_unknown_resin_, [], [True] * num_res + [True, *in_unknowns])
    used_res, (used_i,), used_refs = split_list(used_inputs, [num_res, 1])
  assert all(used_res)
  if USE_LOOP_INVARIANT_OPTIMIZATION:
    (_, _, loop_var_res, _) = pe.partial_eval_jaxpr_nounits(
        core.ClosedJaxpr(jaxpr_known_resout, ()),
        [True] + [False] * sum(map(operator.not_, in_unknowns)), False)
    loop_invar_res = map(operator.not_, loop_var_res)
  else:
    loop_invar_res = [False] * len(jaxpr_known_resout.outvars)

  known_tracers, _ = partition_list(in_unknowns, tracers)
  known_vals = [t.pval.get_known() for t in known_tracers]
  jaxpr_known2, res_avals = convert_outputs_to_writes(
      nsteps, jaxpr_known_resout, loop_invar_res)
  empty_res = [ad_util.zeros_like_aval(a) for a in res_avals]
  out_flat = for_p.bind(*known_vals, *empty_res, jaxpr=jaxpr_known2,
                        nsteps=nsteps, reverse=reverse,
                        which_linear=(False,) * (
                          len(known_vals) + len(empty_res)))
  known_outputs, res = split_list(out_flat, [len(out_flat) - len(empty_res)])
  jaxpr_unknown = convert_inputs_to_reads(nsteps, len(res_avals),
                                          jaxpr_unknown_resin,
                                          loop_invar_res)
  used_and_known = map(operator.and_, used_refs, map(operator.not_, out_unknowns))
  res = map(trace.new_instantiated_const, res)
  tracers = [trace.instantiate_const(t) if u_and_k else t for t, u_and_k
             in zip(tracers, used_and_known)]
  _, used_tracers = partition_list(used_refs, tracers)
  _, used_which_linear = partition_list(used_refs, which_linear)
  which_linear_unknown = (
      (False,) * len(res)
      + tuple(used_which_linear))
  unknown_inputs = [*res, *used_tracers]
  unknown_outputs_ = [pe.JaxprTracer(trace, pe.PartialVal.unknown(t.aval), None)
                      for t in unknown_inputs]
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack):]
  source = source_info_util.current().replace(name_stack=name_stack)
  eqn = new_eqn_recipe(unknown_inputs, unknown_outputs_,
                       for_p, dict(jaxpr=jaxpr_unknown, nsteps=nsteps,
                                   reverse=reverse,
                                   which_linear=which_linear_unknown),
                       jaxpr_unknown.effects, source)
  _, remat_unknown_outputs = split_list(unknown_outputs_, [num_res])
  _, used_out_unknowns = partition_list(used_refs, out_unknowns)
  _, unknown_outputs = partition_list(used_out_unknowns, remat_unknown_outputs)
  for t in unknown_outputs: t.recipe = eqn
  output = merge_lists(out_unknowns, known_outputs, unknown_outputs)
  return output
pe.custom_partial_eval_rules[for_p] = _for_partial_eval

def new_eqn_recipe(in_tracers, out_tracers, prim, params, eff, src):
  assert prim is for_p
  assert len(in_tracers) + 1 == len(params['jaxpr'].invars)
  return pe.new_eqn_recipe(in_tracers, out_tracers, prim, params, eff, src)

def convert_outputs_to_writes(
    nsteps: int, jaxpr: core.Jaxpr, loop_invar_res: Sequence[bool]
  ) -> Tuple[core.Jaxpr, List[core.ShapedArray]]:
  if jaxpr.constvars: raise NotImplementedError  # TODO?

  @lu.wrap_init
  def eval_jaxpr(i, *refs):
    orig_refs, res_refs = split_list(refs, [len(jaxpr.invars) - 1])
    outs = core.eval_jaxpr(jaxpr, (), i, *orig_refs)
    for invar, r, o in zip(loop_invar_res, res_refs, outs):
      if invar:
        r[()] = o
      else:
        r[i] = o
    return []

  in_avals = [v.aval for v in jaxpr.invars]  # [i, *orig_ref_avals]
  jaxpr_known, jaxpr_unknown, out_unknowns, out_avals = (
      pe.partial_eval_jaxpr_nounits(core.ClosedJaxpr(jaxpr, ()),
      [True] + [False] * (len(jaxpr.invars) - 1), False))
  res_ref_avals = [ShapedArrayRef(x.aval.shape, x.aval.dtype) if invar
                   else ShapedArrayRef((nsteps, *x.aval.shape), x.aval.dtype)
                   for invar, x in zip(loop_invar_res, jaxpr.outvars)]
  jaxpr, _, () = pe.trace_to_jaxpr_dynamic(eval_jaxpr, [*in_avals, *res_ref_avals])
  return jaxpr, [core.ShapedArray(a.shape, a.dtype) for a in res_ref_avals]

def convert_inputs_to_reads(
    nsteps: int, num_res: int, jaxpr: core.Jaxpr, loop_invar_res: Sequence[bool]
  ) -> core.Jaxpr:
  if jaxpr.constvars: raise NotImplementedError  # TODO?

  @lu.wrap_init
  def eval_jaxpr(i, *refs):
    res_refs, orig_refs = split_list(refs, [num_res])
    res_vals = [r[()] if invar else r[i] for invar, r in zip(loop_invar_res,
      res_refs)]
    () = core.eval_jaxpr(jaxpr, (), *res_vals, i, *orig_refs)
    return []

  res_val_avals, (i_aval,), orig_ref_avals = \
      split_list([v.aval for v in jaxpr.invars], [num_res, 1])
  res_ref_avals = [ShapedArrayRef(x.shape, x.dtype) if invar
                   else ShapedArrayRef((nsteps, *x.shape), x.dtype)
                   for invar, x in zip(loop_invar_res, res_val_avals)]

  jaxpr, _, () = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [i_aval, *res_ref_avals, *orig_ref_avals])
  return jaxpr

def _save_anything(*_, **__): return True


def _for_transpose(in_cts, *args, jaxpr, nsteps, reverse, which_linear):
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
                        which_linear=tuple(which_linear_transpose))
  ct_outs = [ct if ad.is_undefined_primal(x) else None
             for x, ct in zip(args, all_outs)]
  return ct_outs
ad.primitive_transposes[for_p] = _for_transpose

# TODO better not dce the index
def transpose_jaxpr(jaxpr: core.Jaxpr, which_linear: List[bool]) -> core.Jaxpr:
  which_linear = np.cumsum(which_linear).astype(np.bool_)
  def trans(i, *args):
    # First we want to run the computation to read all the residual refs. We can
    # do that by using partial evaluation with all linear inputs unknown.
    res_jaxpr, tangent_jaxpr_, *_ = \
        _partial_eval_jaxpr_custom(jaxpr, [False, *which_linear],
                                   _save_anything)
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
    ad.backward_pass(tangent_jaxpr, (), False, (), (*primals_args, *ct_args), ())
    return []
  jaxpr_trans, _, _ = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(trans), [v.aval for v in jaxpr.invars])
  return jaxpr_trans

#

def f(x):
  def body(i, refs):
    x_ref, y_ref = refs
    x = x_ref[i]
    y = x
    y_ref[i] = y
    y = y_ref[i]
    y_ref[i] = jnp.sin(y * y)
  n = x.shape[0]
  return for_loop(n, body, (x, jnp.zeros_like(x)))[1]

def f_ref(x):
  return jnp.sin(x ** 2)


USE_LOOP_INVARIANT_OPTIMIZATION = True
USE_PASSTHROUGH_OPTIMIZATION = True

x = jnp.arange(1., 8.)
print("============= F ===========")
prnt(jax.make_jaxpr(f)(x))
print(f(x))
print(f_ref(x))

print("========== F JVP ===========")
print(jax.jvp(f, [x], [x]))
print(jax.jvp(f_ref, [x], [x]))

print("========== F LIN ===========")
print(jax.linearize(f, x)[1](x))
print(jax.linearize(f_ref, x)[1](x))

print("========== F GRAD ===========")
print(jax.grad(lambda x: f(x).sum())(x))
print(jax.grad(lambda x: f_ref(x).sum())(x))

print("========== F JVP-OF-GRAD ===========")
f_grad = jax.grad(lambda x: f(x).sum())
f_ref_grad = jax.grad(lambda x: f_ref(x).sum())
print(jax.jvp(f_grad, (x,), (jnp.ones_like(x),)))
print(jax.jvp(f_ref_grad, (x,), (jnp.ones_like(x),)))

print("========== F LIN-OF-GRAD ===========")
print(jax.linearize(f_grad, x)[1](x))
print(jax.linearize(f_ref_grad, x)[1](x))

print("========== F 2xGRAD ===========")
g     = jax.grad(lambda x: f_grad(x).sum())
g_ref = jax.grad(lambda x: f_ref_grad(x).sum())
print(g(x))
print(g_ref(x))

print("========== F 3xGRAD ===========")
h     = lambda x: jax.grad(lambda x: g(x).sum())(x).sum()
h_ref = lambda x: jax.grad(lambda x: g_ref(x).sum())(x).sum()
print(jax.grad(h)(x))
print(jax.grad(h_ref)(x))

print("========== Loop invariant residuals ==========")
A = jnp.arange(6 * 5.).reshape(6, 5)
xs = jnp.arange(10 * 5.).reshape(10, 5)
ys = jnp.zeros((10, 6))

def body(i, refs):
  x_ref, y_ref = refs
  y_ref[i] = jnp.dot(jnp.cos(A), jnp.sin(x_ref[i]))

def f(xs):
  return for_loop(10, body, (xs, ys))
primals, f_lin = jax.linearize(f, xs)
print(jax.make_jaxpr(f_lin)(xs))

def loss(A):
  def body(i, x_ref):
    x = x_ref[()]
    x_ref[()] = jnp.matmul(A, x)
  init_x = jnp.zeros(A.shape[-1:])
  last_x = for_loop(10, body, init_x)
  return jnp.sum(last_x)

A = jnp.zeros((3, 3))
# The second DUS was unnecessarily replicating A across time.
# We check XLA because _scan_impl is "underneath" the jaxpr language.
print(jax.make_jaxpr(jax.grad(loss))(A))

print("========== Pass through residuals ==========")

def f(x):
  def body(i, refs):
    carry_ref, x_ref, y_ref = refs
    carry_ref[()] = y_ref[i] = x_ref[i] * carry_ref[()]
  carry = 1.
  y = jnp.zeros_like(x)
  return for_loop(x.shape[0], body, (carry, x, y))[2]

print(jax.make_jaxpr(f)(jnp.arange(1., 5.)))
_, f_lin = jax.linearize(lambda xs: f(xs).sum(), jnp.arange(1., 5.))
print(jax.make_jaxpr(f_lin)(jnp.arange(1., 5.)))


# TODO closing over consts, what do? convert to refs?
# TODO don't dce loop counter (maybe add 'instantiate' to dce_jaxpr?)
# TODO partial eval is being wasteful by saving (not rematerializing) pure fns
#      of loop counter
# TODO loop batching
# TODO fixpoints
# TODO nested scans leaving something on the table? how could we nest these
# loops? may need 'heap tags'. could statically give each for an id, and mention
# it in the reference. that doesn't work for standalone functions. maybe can be
# python trace time static, i.e. static in jaxprs. so inner loop can have a
# state effect, like
#  for_loop : Int -> (Int -> Ref h s -> {State h s, effs} ()) -> s
#             -> {effs} s
# whereas if we are okay with nesting being inefficient for now
#   for_loop : Int -> (Int -> Ref h s -> {State h s} ()) -> s -> s
# (may need to require out-of-line functions be pure, at least for now, once we
# have out-of-line functions)
# OR maybe not actually leaving anything on the table...


# TODO pe.partial_eval_jaxpr_custom which does the dce (o/w caller must)
