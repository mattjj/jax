import operator
import ipdb, sys, traceback
def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()
sys.excepthook = info

from functools import partial
from typing import TypeVar, Any, Sequence, List, Tuple, Generic, Callable

import jax
import jax.numpy as jnp

from jax import core
from jax import linear_util as lu
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

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

# State effect
class State: pass
State = State()


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

def _get_pp_rule(eqn, context, settings):
  y, = eqn.outvars
  x, *idx = eqn.invars
  # pretty-print `y = get x i` as `x[i] := v`
  idx = ','.join(core.pp_var(i, context) for i in idx)
  lhs = core.pp_vars([y], context, print_shapes=settings.print_shapes)
  return pp.concat([lhs, pp.text(' = '),
                    pp.text(core.pp_var(x, context)),
                    pp.text('['), pp.text(idx), pp.text(']')])
core.pp_eqn_rules[get_p] = _get_pp_rule



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
  return ref_swap(primal_ref, idx, x), ref_swap(tangent_ref, idx, xdot)
ad.primitive_jvps[swap_p] = _swap_jvp

def _swap_pp_rule(eqn, context, settings):
  y, = eqn.outvars
  x, *idx, v = eqn.invars
  idx = ','.join(core.pp_var(i, context) for i in idx)
  if type(y) is core.DropVar:
    # pretty-print `_ = swap x i v` as `x[i] := v`
    del y
    return pp.concat([pp.text(core.pp_var(x, context)),
                      pp.text('['), pp.text(idx), pp.text('] := '),
                      pp.text(core.pp_var(v, context))])
  else:
    # pretty-print `y:T = swap x i v` as `y:T, x[i] = x[i], v`
    x_i = pp.concat([pp.text(core.pp_var(x, context)),
                     pp.text('['), pp.text(idx), pp.text(']')])
    y = core.pp_vars([y], context, print_shapes=settings.print_shapes)
    return pp.concat([y, pp.text(', '), x_i, pp.text(' := '),
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
  return pp.concat([pp.text(core.pp_var(x, context)),
                    pp.text('['), pp.text(idx), pp.text('] += '),
                    pp.text(core.pp_var(v, context))])
core.pp_eqn_rules[addupdate_p] = _addupdate_pp_rule



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
    pe._partial_eval_jaxpr_custom(jaxpr, [False, True], lambda *_: True)
prnt(jaxpr_known)
jaxpr_staged, _ = pe.dce_jaxpr(jaxpr_staged_,
                               [True] * len(jaxpr_staged_.outvars))
prnt(jaxpr_staged)
# so just call this sucker in the partial eval rule for the loop


def _get_transpose(g, ref, *idx):
  if type(g) is not ad_util.Zero:
    ref_addupdate(ref, idx, g)
  return [None, None]
ad.primitive_transposes[get_p] = _get_transpose

def _swap_transpose(g, ref, *idx_x):
  *idx, x = idx_x
  x_bar = ref_swap(ref, idx, ad_util.instantiate(g))
  return [None] + [None] * len(idx) + [x_bar]
ad.primitive_transposes[swap_p] = _swap_transpose

# print("==> TRANSPOSE ==>")
# avals = [x.aval for x in jaxpr_staged.outvars]
# def trans(res, ref):
#   ad.backward_pass(jaxpr_staged, (), (), (), (res, ref), ())
#   return []
# jaxpr_trans, _, _ = pe.trace_to_jaxpr_dynamic(
#     lu.wrap_init(trans), [core.ShapedArray((), jnp.dtype('float32')),
#                           ShapedArrayRef((4,), jnp.dtype('float32'))])
# prnt(jaxpr_trans)


# discharge!

def discharge_state(jaxpr: core.Jaxpr, consts: Sequence[Any]
                    ) -> Tuple[core.Jaxpr, List[Any]]:
  in_avals = [core.ShapedArray(v.aval.shape, v.aval.dtype)
              if type(v.aval) is ShapedArrayRef
              else v.aval for v in jaxpr.invars]
  eval_jaxpr = lu.wrap_init(partial(_eval_jaxpr_discharge_state, jaxpr, consts))
  new_jaxpr, _, new_consts = pe.trace_to_jaxpr_dynamic(eval_jaxpr, in_avals)
  return new_jaxpr, new_consts

def _eval_jaxpr_discharge_state(jaxpr, consts: List[Any], *args: Any):
  env: Dict[core.Var, Any] = {}

  def read(x: core.Atom) -> Any:
    if type(x) is core.Literal:
      return x.val
    return env[x]

  def write(v: core.Var, val: Any) -> None:
    env[v] = val

  write(core.unitvar, core.unit)
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
  # assert not jaxpr.outvars  # TODO remove this
  return [read(v) for v in jaxpr.invars if type(v.aval) is ShapedArrayRef]

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

# loop

print('loop!!!!')

# Type annotations
S = TypeVar('S')
class Ref(Generic[TypeVar('T')]): pass

def abstractify(x: Any) -> core.AbstractValue:
  return core.raise_to_shaped(core.get_aval(x))

# for: Int -> (Int -> Ref s -> {State s} ()) -> s -> s
def for_loop(nsteps: int, body: Callable[[int, Ref[S]], None],
             init_state: S) -> S:
  init_state, state_tree = tree_flatten(init_state)
  jaxpr, consts = _trace_to_jaxpr(body, state_tree, map(make_ref, init_state))
  out_flat = for_p.bind(*consts, *init_state, jaxpr=jaxpr, nsteps=int(nsteps),
                        reverse=False)
  return tree_unflatten(state_tree, out_flat)
for_p = core.Primitive('for')
for_p.multiple_results = True

def _trace_to_jaxpr(f, state_tree, state_avals):
  f, out_tree = flatten_fun_nokwargs(
      lu.wrap_init(f), treedef_tuple((tree_structure(0), state_tree)))
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      f, [core.ShapedArray((), jnp.dtype('int32')), *state_avals])
  if out_tree() != tree_structure(None): raise Exception
  return pe.convert_constvars_jaxpr(jaxpr), consts

def make_ref(x) -> ShapedArrayRef:
  aval = core.raise_to_shaped(core.get_aval(x))
  if type(aval) is not core.ShapedArray:
    raise Exception(f"can't make ref from {x}")
  if not aval.shape:
    raise Exception(f"can't make ref from value with scalar shape {aval.shape}")
  return ShapedArrayRef(aval.shape, aval.dtype)

@for_p.def_abstract_eval
def _for_abstract_eval(*_, jaxpr, **__):
  return [core.ShapedArray(v.aval.shape, v.aval.dtype) for v in jaxpr.invars
          if type(v.aval) is ShapedArrayRef]

for_p.def_impl(partial(xla.apply_primitive, for_p))

def _for_impl(*args, jaxpr, nsteps, reverse):
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

def _for_jvp(primals, tangents, *, jaxpr, nsteps, reverse):
  tangents = map(ad.instantiate_zeros, tangents)  # TODO handle symbolic zero
  jaxpr_ = core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(jaxpr_, [False] + [True] * len(tangents), True)
  jvp_jaxpr, jvp_consts = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts
  out_flat = for_p.bind(*jvp_consts, *primals, *tangents, jaxpr=jvp_jaxpr,
                        nsteps=nsteps, reverse=reverse)
  return split_list(out_flat, [len(out_flat) // 2])
ad.primitive_jvps[for_p] = _for_jvp

def _for_partial_eval(trace, *tracers, jaxpr, nsteps, reverse):
  in_unknowns = [not t.pval.is_known() for t in tracers]
  body_jaxpr, () = discharge_state(jaxpr, ())
  for _ in range(len(in_unknowns)):
    _, _, out_unknowns, _, num_res = \
        pe._partial_eval_jaxpr_custom(body_jaxpr, [False, *in_unknowns], _save_anything)
    if out_unknowns == in_unknowns:
      break
    in_unknowns = safe_map(operator.or_, in_unknowns, out_unknowns)
  else:
    raise Exception
  tracers = [trace.instantiate_const(t) if uk else t for t, uk in zip(tracers, out_unknowns)]
  jaxpr_known_resout, jaxpr_unknown_resin_, _, _, num_res = \
      pe._partial_eval_jaxpr_custom(jaxpr, [False, *out_unknowns], _save_anything)
  jaxpr_unknown_resin, used_inputs = pe.dce_jaxpr(jaxpr_unknown_resin_, [])
  assert used_inputs[0]  # TODO dont dce i! or maybe just munge input binders
  jaxpr_known, res_avals = convert_outputs_to_writes(nsteps, jaxpr_known_resout)
  empty_res = [ad_util.zeros_like_aval(a) for a in res_avals]
  tracers_known = [t.pval.get_known() for t, uk in zip(tracers, out_unknowns) if
      not uk]
  out_flat = for_p.bind(*tracers_known, *empty_res, jaxpr=jaxpr_known,
                        nsteps=nsteps, reverse=reverse)
  known_outputs, res = split_list(out_flat, [len(out_flat) - len(empty_res)])
  jaxpr_unknown = convert_inputs_to_reads(nsteps, len(res_avals),
                                          jaxpr_unknown_resin)

  res = map(trace.new_instantiated_const, res)
  unknown_inputs = res + [t for t in tracers if not t.pval.is_known()]
  unknown_outputs_ = [pe.JaxprTracer(trace, pe.PartialVal.unknown(t.aval), None)
                      for t in unknown_inputs]
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack):]
  source = source_info_util.current().replace(name_stack=name_stack)
  eqn = pe.new_eqn_recipe(unknown_inputs, unknown_outputs_,
                          for_p, dict(jaxpr=jaxpr_unknown, nsteps=nsteps,
                                      reverse=reverse),
                          jaxpr_unknown.effects, source)
  _, unknown_outputs = split_list(unknown_outputs_, [num_res])
  for t in unknown_outputs: t.recipe = eqn
  return merge_lists(out_unknowns, known_outputs, unknown_outputs)
pe.custom_partial_eval_rules[for_p] = _for_partial_eval
# NOTE we think we'll need 'loop invariant' optimization

def convert_outputs_to_writes(
    nsteps: int, jaxpr: core.Jaxpr
  ) -> Tuple[core.Jaxpr, List[core.ShapedArray]]:
  if jaxpr.constvars: raise NotImplementedError  # TODO?

  @lu.wrap_init
  def eval_jaxpr(i, *refs):
    orig_refs, res_refs = split_list(refs, [len(jaxpr.invars) - 1])
    outs = core.eval_jaxpr(jaxpr, (), i, *orig_refs)
    for r, o in zip(res_refs, outs):
      r[i] = o
    return []

  in_avals = [v.aval for v in jaxpr.invars]  # [i, *orig_ref_avals]
  res_ref_avals = [ShapedArrayRef((nsteps, *x.aval.shape), x.aval.dtype)
                   for x in jaxpr.outvars]
  jaxpr, _, () = pe.trace_to_jaxpr_dynamic(eval_jaxpr, [*in_avals, *res_ref_avals])
  return jaxpr, [core.ShapedArray(a.shape, a.dtype) for a in res_ref_avals]

def convert_inputs_to_reads(
    nsteps: int, num_res: int, jaxpr: core.Jaxpr
  ) -> core.Jaxpr:
  if jaxpr.constvars: raise NotImplementedError  # TODO?

  @lu.wrap_init
  def eval_jaxpr(i, *refs):
    res_refs, orig_refs = split_list(refs, [num_res])
    res_vals = [r[i] for r in res_refs]
    () = core.eval_jaxpr(jaxpr, (), *res_vals, i, *orig_refs)
    return []

  res_val_avals, (i_aval,), orig_ref_avals = \
      split_list([v.aval for v in jaxpr.invars], [num_res, 1])
  res_ref_avals = [ShapedArrayRef((nsteps, *x.shape), x.dtype)
                   for x in res_val_avals]

  jaxpr, _, () = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [i_aval, *res_ref_avals, *orig_ref_avals])
  return jaxpr

def _save_anything(*_, **__): return True


def _for_transpose(in_cts, *args, jaxpr, nsteps, reverse):
  args_ = [ct if ad.is_undefined_primal(x) else x
           for x, ct in zip(args, in_cts)]
  jaxpr_transpose = transpose_jaxpr(jaxpr, map(ad.is_undefined_primal, args))
  all_outs = for_p.bind(*args_, jaxpr=jaxpr_transpose, nsteps=nsteps,
                        reverse=not reverse)
  return [ct if ad.is_undefined_primal(x) else None
          for x, ct in zip(args, all_outs)]
ad.primitive_transposes[for_p] = _for_transpose

# TODO better not dce the index
def transpose_jaxpr(jaxpr: core.Jaxpr, which_linear: List[bool]) -> core.Jaxpr:
  def trans(i, *args):
    primal_jaxpr, tangent_jaxpr_, _, _, _ = \
        pe._partial_eval_jaxpr_custom(jaxpr, [False, *which_linear],
                                      _save_anything)
    tangent_jaxpr, used = pe.dce_jaxpr(tangent_jaxpr_, [])
    primal_args = [x for x, lin in zip(args, which_linear) if not lin]
    res = core.eval_jaxpr(primal_jaxpr, (), i, *primal_args)
    ct_args = [x for x, lin, used in zip(args, which_linear, used[1+len(res):])
               if lin and used]
    ad.backward_pass(tangent_jaxpr, (), False, (), (*res, i, *ct_args), ())
    return []
  jaxpr_trans, _, _ = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(trans), [v.aval for v in jaxpr.invars])
  return jaxpr_trans

#

def f():
  def body(i, ref):
    ref[i] += i
  return for_loop(5, body, jnp.array([3, 1, 4, 1, 5]))

prnt(jax.make_jaxpr(f)())
print(f())


def f(x):
  def body(i, ref):
    x = ref[i]
    # ref[i] = jnp.sin(x)
    ref[i] = x
    ref[i] = (ref[i] + x) / 2.
    # ref[i] = (ref[i] * x) / 2.
  return for_loop(1, body, jnp.array([x]))

prnt(jax.make_jaxpr(f)(3.))
print(f(3.)[0])
print(jax.jvp(f, (3.,), (1.,)))

y, f_lin = jax.linearize(f, 3.)
y_dot = f_lin(1.)
print(y, y_dot)
print(jax.grad(lambda x: f(x)[0])(3.))

# def f(x, y):
#   def body(i, refs):
#     x_ref, y_ref = refs
#     x_ref[0] = jnp.sin(i)
#     y_ref[i] = x_ref[0] * y_ref[i]
#   return for_loop(1, body, (x, y))

# f(jnp.array([1.]), jnp.array([1., 2.]))


def f(x):
  def body(i, refs):
    x_ref, y_ref = refs
    y_ref[i] = x_ref[i] * x_ref[i+1]
  n = x.shape[0]
  _, y = for_loop(n - 1, body, (x, jnp.zeros(n - 1)))
  return y

def f_ref(x):
  return x[:-1] * x[1:]

x = jnp.arange(10.)
print("============= F ===========")
prnt(jax.make_jaxpr(f)(x))
print(f(x))
print(f_ref(x))

print("========== F JVP ===========")
# prnt(jax.make_jaxpr(lambda x, t: jax.jvp(f, (x,), (t,)))(x, x))
print(jax.jvp(f, [x], [x]))
print(jax.jvp(f_ref, [x], [x]))

print("========== F LIN ===========")
# TODO wrong!!
print(jax.linearize(f, x)[1](x))
print(jax.linearize(f_ref, x)[1](x))

print("========== F LIN ===========")
print(jax.grad(lambda x: f(x).sum())(x))
# print(jax.grad(lambda x: f_ref(x).sum())(x))


# TODO loop transpose
# TODO addupdate transpose rule
# TODO loop batching
# TODO fixpoints, need jvp_jaxpr with extra state-is-differentiated input/output
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
