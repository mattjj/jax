from __future__ import annotations

import enum
from functools import partial, lru_cache
import inspect
from typing import (Any, Callable, Optional, Tuple, List, Set, Sequence, Dict,
                    Hashable, Union)
import numpy as np

import jax
from jax import core
from jax.core import Tracer
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util
from jax._src.lax import lax, parallel as lax_parallel
from jax._src.sharding import NamedSharding, PartitionSpec
from jax._src.util import (prod, HashableFunction, unzip2, unzip3,
                           as_hashable_function, memoize)
from jax.api_util import flatten_fun_nokwargs
from jax.experimental import maps
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.interpreters import pxla
from jax.interpreters import ad
from jax.interpreters.pxla import Mesh
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           tree_structure, tree_leaves)
from jax._src.tree_util import (broadcast_prefix, prefix_errors, PyTreeDef,
                                _generate_key_paths)

P = PartitionSpec

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip
traceback_util.register_exclusion(__file__)

# import ipdb, sys, traceback
# def info(type, value, tb):
#     traceback.print_exception(type, value, tb)
#     ipdb.pm()
# sys.excepthook = info


# TODO [-] autodiff w/ sholto@
#        [x] jvp
#        [x] partial eval
#        [ ] transpose
# TODO [x] better eager repr
# TODO [x] fix scalar residual problem
# TODO [x] better errors
#        [x] broadcast_prefix errors
#        [x] if output rank doesn't match out spec for concatenation
#        [x] validate that in_specs / out_specs are indeed pytrees of pspecs
# TODO [ ] remove default rep rule behavior in favor of convenience wrappers,
#          and add good rule coverage
# TODO [ ] actually write thorough tests...
# TODO [ ] try nesting
#        [ ] eager
#        [ ] staged: convert out and then back into manual in lowering rule?
# TODO [ ] try to implement (nested) pmap in terms of shard_map
# TODO [ ] custom_jvp / custom_vjp eager handling
# TODO [ ] name stack

# API

Specs = Any  # PyTree[PartitionSpec]

@traceback_util.api_boundary
def shard_map(f: Callable, mesh: Mesh, in_specs: Specs, out_specs: Specs):
  # TODO improve these error messages
  if not callable(f):
    raise TypeError("shard_map requires a callable for its first argument, "
                    f"but got {f} of type {type(f)}.")
  if not isinstance(mesh, Mesh):
    raise TypeError("shard_map requires a `jax.sharding.Mesh` instance for its "
                    f"second argument, but got {mesh} of type {type(mesh)}.")
  _check_specs(SpecErrorType.input, in_specs)
  _check_specs(SpecErrorType.out, out_specs)

  @traceback_util.api_boundary
  def wrapped(*args):
    fun = lu.wrap_init(f)
    args_flat, in_tree = tree_flatten(args)
    try: in_specs_flat = broadcast_prefix(in_specs, args)
    except ValueError:
      e, *_ = prefix_errors(in_specs, args)
      raise e('shard_map in_specs') from None
    _check_flat_specs(f, in_tree, in_specs, in_specs_flat, args_flat)
    in_names_flat = tuple(map(_canonicalize_spec, in_specs_flat))
    flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)

    @memoize
    def out_names_thunk():
      dummy = tree_unflatten(out_tree(), [object()] * out_tree().num_leaves)
      try: out_specs_flat = broadcast_prefix(out_specs, dummy)
      except ValueError:
        e, *_ = prefix_errors(out_specs, dummy)
        raise e('shard_map out_specs') from None
      return tuple(map(_canonicalize_spec, out_specs_flat))
    try:
      out_flat = shard_map_p.bind(
          flat_fun, *args_flat, mesh=mesh, in_names=in_names_flat,
          out_names_thunk=out_names_thunk)
    except _SpecError as e:
      failures, = e.args
      msg = _spec_error(SpecErrorType.out, f, out_tree(), out_specs, failures)
      if any(fail and not fail.shape for fail in failures):
        msg += (" In particular, for rank 0 outputs which are not constant "
                "over the mesh, add at least one (singleton) axis to them so "
                "that they can be concatenated using out_specs.")
      raise ValueError(msg) from None
    return tree_unflatten(out_tree(), out_flat)
  return wrapped

def _check_specs(error_type: SpecErrorType, specs: Any) -> None:
  maybe_specs = tree_leaves(specs)
  if all(isinstance(p, P) for p in maybe_specs):
    return
  prefix = 'in' if error_type == SpecErrorType.input else 'out'
  specs_aug = _generate_key_paths(specs)
  msgs = [f"  {prefix}_specs{key.pprint()} is {x} of type {type(x).__name__}, "
          for key, x in _generate_key_paths(specs) if not isinstance(x, P)]
  raise TypeError(
      f"shard_map {prefix}_specs argument must be a pytree of "
      f"`jax.sharding.PartitionSpec` instances, but:\n\n"
      + '\n\n'.join(msgs) + '\n\n'
      f"Check the {prefix}_specs values passed to shard_map.")

# Internally use AxisNames = Dict[int, Tuple[AxisName, ...]], not PartitionSpecs
AxisName = Hashable
AxisNames = Dict[int, Tuple[AxisName, ...]]  # TODO make it a hashable dict
def _canonicalize_spec(spec: PartitionSpec) -> AxisNames:
  if isinstance(spec, PartitionSpec):
    return {i: names if isinstance(names, tuple) else (names,)
            for i, names in enumerate(spec) if names is not None}
  else:
    return spec

def _check_flat_specs(f: Callble, in_tree, in_specs: Specs,
                      in_specs_flat: List[P], xs: List) -> None:
  in_avals = [core.raise_to_shaped(core.get_aval(x)) for x in xs]
  fail = [a if not len(p) <= a.ndim else False
          for p, a in zip(in_specs_flat, in_avals)]
  if any(fail):
    msg = _spec_error(SpecErrorType.input, f, in_tree, in_specs, fail)
    raise ValueError(msg)

SpecErrorType = enum.Enum('SpecErrorType', ['input', 'out'])

def _spec_error(error_type: SpecErrorType, f: Callable, tree: PyTreeDef,
                specs: Specs, failures_flat: List[core.ShapedArray]) -> str:
  if error_type == SpecErrorType.input:
    prefix, base = 'in', 'args'
    dummy_args = tree_unflatten(tree, [False] * tree.num_leaves)
    try:
      ba = inspect.signature(f).bind(*dummy_args)
    except (TypeError, ValueError):
      ba = None
  else:
    prefix, base = 'out', f'{f.__name__}(*args)'
  failures = tree_unflatten(tree, failures_flat)
  failures_aug = _generate_key_paths(failures)
  out_specs_ = tree_unflatten(tree_structure(specs), _generate_key_paths(specs))
  leaf = lambda x: type(x) is tuple and len(x) == 2 and type(x[1]) is P
  out_specs_aug = broadcast_prefix(out_specs_, failures, is_leaf=leaf)
  msgs = []
  for (spec_key, spec), (fail_key, fail) in zip(out_specs_aug, failures_aug):
    if fail:
      if error_type == SpecErrorType.input and ba is not None:
        arg_key, *keys = fail_key.keys
        extra = (f", where {base}[{arg_key.key}] is bound to {f.__name__}'s "
                 f"parameter {list(ba.arguments.keys())[arg_key.key]},")
      else:
        extra = ""
      msgs.append(
          f"  {prefix}_specs{spec_key.pprint()} is {spec} which has length "
          f"{len(spec)}, but\n"
          f"  {base}{fail_key.pprint()}{extra} has shape {fail.str_short()}, "
          f"which has rank {fail.ndim} (<{len(spec)})")
  assert msgs
  msg = (f"shard_map applied to the function '{f.__name__}' was given an "
         f"{prefix}_specs entry which is too long to be compatible with the "
         f"corresponding {prefix}put value from the function:\n\n"
         + '\n'.join(msgs) + '\n\n' +
         f"Entries in {prefix}_specs must be of length no greater than the "
         f"number of axes in the corresponding {prefix}put value.\n\n"
         f"Either revise the spec to be shorter, or modify '{f.__name__}' so "
         f"that its {prefix}puts have sufficient rank.")
  return msg

# Primitive

JaxType = Any
MaybeTracer = Union[JaxType, Tracer]

class ShardMapPrimitive(core.Primitive):
  multiple_results = True

  def bind(self, fun: lu.WrappedFun, *args: MaybeTracer, mesh: Mesh,
           in_names: Tuple[AxisNames, ...],
           out_names_thunk: Callable[[], Tuple[AxisNames, ...]]
           ) -> Sequence[MaybeTracer]:
    top_trace = core.find_top_trace(args)
    fun, env_trace_todo = process_env_traces(
        fun, top_trace and top_trace.level, mesh)
    tracers = map(top_trace.full_raise, args)
    outs = top_trace.process_shard_map(  # pytype: disable=attribute-error
        shard_map_p, fun, tracers, mesh=mesh, in_names=in_names,
        out_names_thunk=out_names_thunk)
    return map(core.full_lower, core.apply_todos(env_trace_todo(), outs))

  def get_bind_params(self, params):
    """Goes from jaxpr form to python traceable form."""
    new_params = dict(params)
    jaxpr = new_params.pop('jaxpr')
    subfun = lu.hashable_partial(lu.wrap_init(core.eval_jaxpr), jaxpr, ())
    axes = new_params.pop('out_names')
    new_params['out_names_thunk'] = HashableFunction(lambda: axes, closure=axes)
    return [subfun], new_params

shard_map_p = ShardMapPrimitive('shard_map')

def process_env_traces(fun, top_trace, mesh):
  return fun, lambda: []  # TODO needed for closing over tracers

# Staging

def _shard_map_staging(
    trace: pe.DynamicJaxprTrace, prim: core.Primitive, fun: lu.WrappedFun,
    in_tracers: Sequence[pe.DynamicJaxprTracer], *, mesh: Mesh,
    in_names: Tuple[AxisNames, ...],
    out_names_thunk: Callable[[], Tuple[AxisNames, ...]]
  ) -> Sequence[pe.DynamicJaxprTracer]:
  in_avals = [t.aval for t in in_tracers]
  in_avals_ = map(partial(_shard_aval, mesh), in_names, in_avals)
  with core.new_sublevel(), core.extend_axis_env_nd(mesh.shape.items()):
    jaxpr, out_avals_, consts = pe.trace_to_subjaxpr_dynamic(
        fun, trace.main, in_avals_)
  _check_names(out_names_thunk(), out_avals_)
  out_avals = map(partial(_unshard_aval, mesh), out_names_thunk(), out_avals_)
  source_info = source_info_util.current()
  out_tracers = [pe.DynamicJaxprTracer(trace, a, source_info) for a in out_avals]
  invars = map(trace.getvar, in_tracers)
  constvars = map(trace.getvar, map(trace.instantiate_const, consts))
  outvars = map(trace.makevar, out_tracers)
  in_names= ({},) * len(consts) + tuple(in_names)
  with core.extend_axis_env_nd(mesh.shape.items()):
    jaxpr = pe.convert_constvars_jaxpr(jaxpr)
  params = dict(mesh=mesh, in_names=in_names, out_names=out_names_thunk(),
                jaxpr=jaxpr)
  eqn = pe.new_jaxpr_eqn([*constvars, *invars], outvars, prim, params,
                         jaxpr.effects, source_info)
  trace.frame.add_eqn(eqn)
  return out_tracers
pe.DynamicJaxprTrace.process_shard_map = _shard_map_staging

def _shard_aval(mesh: Mesh, names: AxisNames, aval: core.AbstractValue
                ) -> core.AbstractValue:
  if isinstance(aval, core.ShapedArray):
    return aval.update(tuple(sz // prod(mesh.shape[n] for n in names.get(i, ()))
                             for i, sz in enumerate(aval.shape)))
  else:
    raise NotImplementedError  # TODO add table with handlers

def _unshard_aval(mesh: Mesh, names: AxisNames, aval: core.AbstractValue
                 ) -> core.AbstractValue:
  if isinstance(aval, core.ShapedArray):
    return aval.update(tuple(sz * prod(mesh.shape[n] for n in names.get(i, ()))
                             for i, sz in enumerate(aval.shape)))
  else:
    raise NotImplementedError  # TODO add table with handlers

# Type-checking

def _shard_map_typecheck(*in_atoms, jaxpr, mesh, in_names, out_names):
  for v, x, in_name in zip(jaxpr.invars, in_atoms, in_names):
    if not core.typecompat(v.aval, _shard_aval(mesh, in_name, x.aval)):
      raise core.JaxprTypeError("shard_map argument avals not compatible with "
                                "jaxpr binder avals and in_names")
  with core.extend_axis_env_nd(tuple(mesh.shape.items())):
    core.check_jaxpr(jaxpr)
  in_rep = map(partial(_in_names_to_rep, mesh), in_names)
  out_rep = _output_rep(mesh, jaxpr, in_rep)
  for rep, dst in zip(out_rep, out_names):
    if not _valid_repeats(mesh, rep, dst):
      # TODO add parameter to opt out of check
      raise core.JaxprTypeError("shard_map can't prove output is sufficiently "
                                "replicated")
  out_avals_sharded = [x.aval for x in jaxpr.outvars]
  out_avals = map(partial(_unshard_aval, mesh), out_names, out_avals_sharded)
  return out_avals, jaxpr.effects
core.custom_typechecks[shard_map_p] = _shard_map_typecheck

def _in_names_to_rep(mesh: Mesh, names: AxisNames) -> Set[AxisName]:
  return set(mesh.axis_names) - set(n for ns in names.values() for n in ns)

def _output_rep(mesh: Mesh, jaxpr: core.Jaxpr, in_rep: Sequence[Set[AxisName]],
                ) -> Sequence[Set[AxisName]]:
  env: Dict[core.Var, Set[AxisName]] = {}

  def read(x: core.Atom) -> Set[AxisName]:
    return env[x] if type(x) is core.Var else set(mesh.axis_names)

  def write(v: core.Var, val: Set[AxisName]) -> None:
    env[v] = val

  map(write, jaxpr.constvars, [set(mesh.axis_names)] * len(jaxpr.constvars))
  map(write, jaxpr.invars, in_rep)
  for e in jaxpr.eqns:
    rule = _rep_rules.get(e.primitive, partial(_rep_rule, e.primitive))
    out_rep = rule(*map(read, e.invars), **e.params)
    if e.primitive.multiple_results:
      out_rep = [out_rep] * len(e.outvars) if type(out_rep) is set else out_rep
      map(write, e.outvars, out_rep)
    else:
      write(e.outvars[0], out_rep)
  return map(read, jaxpr.outvars)

def _valid_repeats(mesh: Mesh, rep: Set[AxisName], dst: AxisNames) -> bool:
  unmentioned = set(mesh.axis_names) - {n for ns in dst.values() for n in ns}
  return unmentioned.issubset(rep)

# Lowering

def _shard_map_lowering(ctx, *in_nodes, jaxpr, mesh, in_names, out_names):
  sharded_avals = [v.aval for v in jaxpr.invars]
  in_nodes_ = map(partial(_xla_shard, mesh), in_names, ctx.avals_in,
                  sharded_avals, in_nodes)
  new_axis_context = mlir.SPMDAxisContext(mesh, frozenset(mesh.axis_names))
  sub_ctx = ctx.module_context.replace(axis_context=new_axis_context)
  with core.extend_axis_env_nd(tuple(mesh.shape.items())):
    out_nodes_, _ = mlir.jaxpr_subcomp(sub_ctx, jaxpr, mlir.TokenSet(),
                                       (), *in_nodes_,
                                       dim_var_values=ctx.dim_var_values)
  sharded_avals = [v.aval for v in jaxpr.outvars]
  return map(partial(_xla_unshard, mesh), out_names, sharded_avals,
             ctx.avals_out, out_nodes_)
mlir.register_lowering(shard_map_p, _shard_map_lowering)

def _xla_shard(mesh, names, aval_in, aval_out, x):
  manual_proto = pxla._manual_proto(aval_in, frozenset(mesh.axis_names), mesh)
  result_type, = mlir.aval_to_ir_types(aval_out)
  axes = {name: i for i, ns in names.items() for name in ns}
  sharding_proto = pxla.mesh_sharding_specs(mesh.shape, mesh.axis_names)(
      aval_in, axes).sharding_proto()
  sx = mlir.wrap_with_sharding_op(x, sharding_proto, unspecified_dims=set())
  return [mlir.wrap_with_full_to_shard_op(result_type, sx, manual_proto, set())]

def _xla_unshard(mesh, names, aval_in, aval_out, xs):
  x, = xs
  manual_proto = pxla._manual_proto(aval_in, frozenset(mesh.axis_names), mesh)
  result_type, = mlir.aval_to_ir_types(aval_out)
  sx = mlir.wrap_with_sharding_op(x, manual_proto, unspecified_dims=set())
  axes = {name: i for i, ns in names.items() for name in ns}
  sharding_proto = pxla.mesh_sharding_specs(mesh.shape, mesh.axis_names)(
      aval_out, axes).sharding_proto()
  return mlir.wrap_with_shard_to_full_op(result_type, sx, sharding_proto, set())

# Eager evaluation

def _shard_map_impl(trace, prim, fun, args, *, mesh, in_names, out_names_thunk):
  del prim
  args = map(partial(_device_put_from_names, mesh), in_names, args)
  args = map(partial(_unmatch_spec, mesh), in_names, args)
  in_rep = map(partial(_in_names_to_rep, mesh), in_names)
  with core.new_base_main(ShardMapTrace, mesh=mesh) as main:
    with core.new_sublevel(), core.extend_axis_env_nd(mesh.shape.items()):
      t = main.with_cur_sublevel()
      in_tracers = map(partial(ShardMapTracer, t), in_rep, args)
      ans = fun.call_wrapped(*in_tracers)
      out_tracers = map(t.full_raise, ans)
      outs_, out_rep = unzip2((t.val, t.rep) for t in out_tracers)
      del main, t, in_tracers, ans, out_tracers
  out_avals = [core.mapped_aval(x.shape[0], 0, core.get_aval(x)) for x in outs_]
  _check_names(out_names_thunk(), out_avals)
  return map(partial(_match_spec, mesh), out_rep, out_names_thunk(), outs_)
core.EvalTrace.process_shard_map = _shard_map_impl

def _device_put_from_names(mesh: Mesh, names: AxisNames, x: JaxType) -> JaxType:
  return jax.device_put(x, NamedSharding(mesh, _names_to_pspec(names)))

def _names_to_pspec(names: AxisNames) -> PartitionSpec:
  ndmin = max(names) + 1 if names else 0
  return PartitionSpec(*(names.get(i) for i in range(ndmin)))

def _unmatch_spec(mesh: Mesh, src: AxisNames, x: JaxType) -> JaxType:
  with core.eval_context():
    return jax.jit(_get_unmatcher(mesh, tuple(src.items())))(x)

@lru_cache()
def _get_unmatcher(mesh, src_tup):
  src = _names_to_pspec(dict(src_tup))
  dst = P(mesh.axis_names)
  return shard_map(_add_singleton, mesh, (src,), dst)

def _match_spec(mesh: Mesh, rep: Set[AxisName], dst: AxisNames, x: JaxType
                ) -> JaxType:
  if not _valid_repeats(mesh, rep, dst):
    raise Exception  # TODO add parameter to opt out of check
  return jax.jit(_get_matcher(mesh, tuple(dst.items())))(x)

def _check_names(names: Sequence[AxisNames], avals: core.ShapedArray) -> None:
  # fail = [a if not max(n) < a.ndim - 1 else False for n, a in zip(names, avals)]
  fail = [a if not max(n) < a.ndim else False for n, a in zip(names, avals)]
  if any(fail): raise _SpecError(fail)
class _SpecError(Exception): pass

@lru_cache()
def _get_matcher(mesh, dst_tup):
  src = P(mesh.axis_names)
  dst = _names_to_pspec(dict(dst_tup))
  return shard_map(_rem_singleton, mesh, (src,), dst)

def _rem_singleton(x): return x.reshape(x.shape[1:])
def _add_singleton(x): return x.reshape(1, *x.shape)

class ShardMapTrace(core.Trace):
  mesh: Mesh

  def __init__(self, *args, mesh):
    super().__init__(*args)
    self.mesh = mesh

  def pure(self, val):
    val_ = _unmatch_spec(self.mesh, {}, val)
    return ShardMapTracer(self, set(self.mesh.axis_names), val_)

  def sublift(self, tracer):
    return ShardMapTracer(self, tracer.rep, tracer.val)

  def process_primitive(self, prim, tracers, params):
    in_vals, in_rep = unzip2((t.val, t.rep) for t in tracers)
    f = _primitive_applier(prim, params, self.mesh)
    with core.eval_context(), jax.disable_jit(False):
      out_vals = jax.jit(f)(*in_vals)
    out_rep = _rep_rules.get(prim, partial(_rep_rule, prim))(*in_rep, **params)
    if prim.multiple_results:
      out_rep = [out_rep] * len(out_vals) if type(out_rep) is set else out_rep
      return map(partial(ShardMapTracer, self), out_rep, out_vals)
    return ShardMapTracer(self, out_rep, out_vals)

  def process_call(self, call_primitive, fun, tracers, params):
    if call_primitive is not xla.xla_call_p: raise NotImplementedError
    fun, jaxpr = _something_shady(fun)  # TODO remove when jit is initial style
    bind = HashableFunction(
        lambda *args, **kwargs: call_primitive.bind(fun, *args, **kwargs),
        (call_primitive, fun))
    fake_primitive = pxla._FakePrimitive(multiple_results=True, bind=bind)
    _rep_rules[fake_primitive] = lambda *_, **__: set()
    out_tracers_ = self.process_primitive(fake_primitive, tracers, params)
    out_vals = [t.val for t in out_tracers_]
    out_rep = _output_rep(self.mesh, jaxpr(), [t.rep for t in tracers])
    return map(partial(ShardMapTracer, self), out_rep, out_vals)

@lu.transformation_with_aux
def _something_shady(*args, **kwargs):
  out = yield args, kwargs
  main = core.thread_local_state.trace_state.trace_stack.dynamic  # forgive me
  jaxpr, _ = main.jaxpr_stack[-1].to_jaxpr(out)
  yield out, jaxpr

class ShardMapTracer(core.Tracer):
  rep: Set[AxisName]
  val: JaxType

  def __init__(self, trace, rep, val):
    self._trace = trace
    self.rep = rep
    self.val = val

  @property
  def aval(self):
    aval = core.get_aval(self.val)
    if (isinstance(aval, core.ConcreteArray) and
        self.rep == set(self._trace.mesh.axis_names)):
      with core.eval_context():
        return core.get_aval(self.val[0])
    else:
      aval = core.raise_to_shaped(aval)
      return core.mapped_aval(self._trace.mesh.size, 0, aval)

  def full_lower(self) -> ShardMapTracer:
    return self

  def __str__(self) -> str:
    with core.eval_context():
      blocks = list(self.val)
    mesh = self._trace.mesh
    axis_names = f"({', '.join(map(str, mesh.axis_names))})"
    return '\n'.join(
        f"On {device} at mesh coordinates {axis_names} = {idx}:\n{block}\n"
        for (idx, device), block in zip(np.ndenumerate(mesh.devices), blocks))

def _primitive_applier(prim: core.Primitive, params: core.ParamDict, mesh: Mesh,
                       ) -> Callable:
  return _prim_applier(prim, tuple(params.items()), mesh)

@lru_cache()
def _prim_applier(prim, params_tup, mesh):
  spec = P(mesh.axis_names)

  @partial(shard_map, mesh=mesh, in_specs=spec, out_specs=spec)
  def apply(*args):
    outs = prim.bind(*map(_rem_singleton, args), **dict(params_tup))
    return tree_map(_add_singleton, outs)
  return apply

def _rep_rule(prim, *in_rep, **params):
  raise NotImplementedError(f"no replication rule for {prim}")

_rep_rules = {}
register_rule = lambda prim: lambda rule: _rep_rules.setdefault(prim, rule)
register_standard = lambda prim: _rep_rules.setdefault(prim, _standard_rep_rule)

def _standard_rep_rule(*in_rep, **_):
  return set.intersection(*in_rep)

for o in lax.__dict__.values():
  if isinstance(o, core.Primitive): register_standard(o)

register_standard(lax_parallel.ppermute_p)

@register_rule(lax_parallel.psum_p)
def _psum_rule(*in_rep, axes, axis_index_groups):
  if axis_index_groups is not None: raise NotImplementedError
  axes = (axes,) if not isinstance(axes, tuple) else axes
  return [r | set(axes) for r in in_rep]  # introduces replication

@register_rule(lax_parallel.all_gather_p)
def _all_gather_rule(in_rep, *, all_gather_dimension, axis_name, axis_size,
                     axis_index_groups, tiled):
  if axis_index_groups is not None: raise NotImplementedError
  if not tiled: raise NotImplementedError
  axis_name = (axis_name,) if not isinstance(axis_name, tuple) else axis_name
  return in_rep | set(axis_name)  # introduces replication

@register_rule(lax_parallel.reduce_scatter_p)
def _reduce_scatter_rule(in_rep, *, scatter_dimension, axis_name, axis_size,
                         axis_index_groups, tiled):
  if axis_index_groups is not None: raise NotImplementedError
  if not tiled: raise NotImplementedError
  return in_rep - {axis_name}  # removes replication

@register_rule(lax_parallel.all_to_all_p)
def _reduce_scatter_rule(in_rep, *, split_axis, concat_axis, axis_name,
                         axis_index_groups):
  if axis_index_groups is not None: raise NotImplementedError
  return in_rep - {axis_name}  # removes replication


# Batching

def _shard_map_batch(
    trace: batching.BatchTrace, prim: core.Primitive, fun: lu.WrappedFun,
    in_tracers: Sequence[batching.BatchTracer], mesh: Mesh,
    in_names: Tuple[AxisNames, ...],
    out_names_thunk: Callable[[], Tuple[AxisNames, ...]]
  ) -> Sequence[batching.BatchTracer]:
  in_vals, in_dims = unzip2((t.val, t.batch_dim) for t in in_tracers)
  if all(bdim is batching.not_mapped for bdim in in_dims):
    return prim.bind(fun, *in_vals, mesh=mesh, in_names=in_names,
                     out_names_thunk=out_names_thunk)
  if trace.spmd_axis_name is not None:
    raise NotImplementedError  # TODO add named axis to specs
  fun, out_dims = batching.batch_subtrace(fun, trace.main, tuple(in_dims))
  new_in_names = [{ax + (d is not batching.not_mapped and ax <= d): names[ax]
                   for ax in names} for names, d in zip(in_names, in_dims)]
  @as_hashable_function(closure=out_names_thunk)
  def new_out_names_thunk():
    out_names = out_names_thunk()
    return [{ax + (d is not batching.not_mapped and ax <= d): names[ax]
             for ax in names} for names, d in zip(out_names, out_dims())]
  new_params = dict(mesh=mesh, in_names=new_in_names,
                    out_names_thunk=new_out_names_thunk)
  out_vals = prim.bind(fun, *in_vals, **new_params)
  make_tracer = partial(batching.BatchTracer, trace,
                        source_info=source_info_util.current())
  return map(make_tracer, out_vals, out_dims())
batching.BatchTrace.process_shard_map = _shard_map_batch

# Autodiff

def _shard_map_jvp(self, shard_map_p, f, tracers, mesh, in_names,
                   out_names_thunk):
  primals, tangents = unzip2((t.primal, t.tangent) for t in tracers)
  which_nz = [     type(t) is not ad.Zero           for t in tangents]
  tangents = [t if type(t) is not ad.Zero else None for t in tangents]
  args, in_tree = tree_flatten((primals, tangents))
  f_jvp = ad.jvp_subtrace(f, self.main)
  f_jvp, which_nz_out = ad.nonzero_tangent_outputs(f_jvp)
  tangent_in_names = [ax for ax, nz in zip(in_names, which_nz) if nz]

  @as_hashable_function(closure=out_names_thunk)
  def new_out_names_thunk():
    out_ax = out_names_thunk()
    return (*out_ax, *(ax for ax, nz in zip(out_ax, which_nz_out()) if nz))
  params = dict(mesh=mesh, in_names=(*in_names, *tangent_in_names),
            out_names_thunk=new_out_names_thunk)
  f_jvp, out_tree = ad.traceable(f_jvp, in_tree)
  result = shard_map_p.bind(f_jvp, *args, **params)
  primal_out, tangent_out = tree_unflatten(out_tree(), result)
  tangent_out = [ad.Zero(ad.get_aval(p).at_least_vspace()) if t is None else t
              for p, t in zip(primal_out, tangent_out)]
  return [ad.JVPTracer(self, p, t) for p, t in zip(primal_out, tangent_out)]
ad.JVPTrace.process_shard_map = _shard_map_jvp

def _shard_map_partial_eval(self, shard_map_p, f, tracers, mesh, in_names,
                            out_names_thunk):
  in_pvals = [t.pval for t in tracers]
  in_knowns, unk_in_avals, in_consts = pe.partition_pvals(in_pvals)
  unk_in_names, known_in_names = pe.partition_list(in_knowns, in_names)
  unk_in_avals_sharded = map(partial(_shard_aval, mesh), unk_in_names, unk_in_avals)
  f = pe.trace_to_subjaxpr_nounits(f, self.main, False)
  f = _promote_scalar_residuals(f)
  f_known, aux = pe.partial_eval_wrapper_nounits(f, tuple(in_knowns),
                                                 tuple(unk_in_avals_sharded))
  unk_in_names, known_in_names = pe.partition_list(in_knowns, in_names)

  @as_hashable_function(closure=out_names_thunk)
  def known_out_names():
    out_knowns, _, jaxpr, _ = aux()
    _, out_known_names = pe.partition_list(out_knowns, out_names_thunk())
    assert not any(not v.aval.shape for v in jaxpr.constvars)
    res_names = ({0: (*mesh.axis_names,)},) * len(jaxpr.constvars)
    return (*out_known_names, *res_names)

  known_params = dict(mesh=mesh, in_names=(*known_in_names,),
                      out_names_thunk=known_out_names)
  out = shard_map_p.bind(f_known, *in_consts, **known_params)
  out_knowns, out_avals_sharded, jaxpr, env = aux()
  out_consts, res = pe.split_list(out, [len(out) - len(jaxpr.constvars)])
  with core.extend_axis_env_nd(mesh.shape.items()):
    jaxpr = pe.convert_constvars_jaxpr(jaxpr)
  unknown_out_names, _ = pe.partition_list(out_knowns, out_names_thunk())
  unknown_in_names = (({0: (*mesh.axis_names,)},) * len(res) + ({},) * len(env)
                      + (*unk_in_names,))
  const_tracers = map(self.new_instantiated_const, res)
  env_tracers = map(self.full_raise, env)
  unknown_arg_tracers = [t for t in tracers if not t.is_known()]
  unknown_params = dict(mesh=mesh, in_names=unknown_in_names,
                        out_names=unknown_out_names, jaxpr=jaxpr)
  out_avals = map(partial(_unshard_aval, mesh), unknown_out_names, out_avals_sharded)
  out_tracers = [pe.JaxprTracer(self, pe.PartialVal.unknown(a), None)
                 for a in out_avals]
  eqn = pe.new_eqn_recipe((*const_tracers, *env_tracers, *unknown_arg_tracers),  # type: ignore[arg-type]
                          out_tracers, shard_map_p, unknown_params,
                          jaxpr.effects, source_info_util.current())
  for t in out_tracers: t.recipe = eqn
  return pe.merge_lists(out_knowns, out_tracers, out_consts)
pe.JaxprTrace.process_shard_map = _shard_map_partial_eval

@lu.transformation
def _promote_scalar_residuals(*args, **kwargs):
  jaxpr, (out_pvals, out_consts, env) = yield args, kwargs
  which_scalar = [isinstance(v.aval, core.ShapedArray) and not v.aval.shape
                  for v in jaxpr.constvars]
  out_consts_ = [jax.lax.broadcast(x, (1,)) if scalar else x
                 for x, scalar in zip(out_consts, which_scalar)]
  @lu.wrap_init
  def fun(*args):
    out_consts = [x.reshape(*x.shape[1:]) if scalar else x
                  for x, scalar in zip(out_consts_, which_scalar)]
    return core.eval_jaxpr(jaxpr, out_consts, *args)
  in_avals = [v.aval for v in jaxpr.invars]
  jaxpr, _, out_consts  = pe.trace_to_jaxpr_dynamic(fun, in_avals)
  yield jaxpr, (out_pvals, out_consts, env)

# Crappy in-line tests, to be deleted.

if __name__ == '__main__':
  import os
  import jax
  import jax.numpy as jnp

  jax.config.update('jax_platform_name', 'cpu')
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"

  def f(x):
    # return x
    return 2 * x
    # return jax.lax.sin(x)
    # return x.sum(keepdims=True)

  mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

  sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))
  x = jax.device_put(jnp.arange(8 * 8.).reshape(8, 8), sharding)

  ## eager repr

  # print(x)
  # @partial(shard_map, mesh=mesh, in_specs=P('x', 'y'), out_specs=P('x', 'y'))
  # def f(x):
  #   print(x)
  #   return x
  # f(x)

  ## nesting

  @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
  def f(x):
    @partial(shard_map, mesh=mesh, in_specs=P('x', 'y'), out_specs=P('x', 'y'))
    def g(x):
      return x
    return g(x)
  f(x)

  # # autodiff tests

  # def g(x):
  #   return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))(x)
  # print(jax.jvp(g, [x], [x]))

  # from jax._src.test_util import check_grads

  # check_grads(g, [x], 2, ['fwd'])

  # y, y_dot = jax.jvp(g, [x], [x])

  # y_, g_lin = jax.linearize(g, x)
  # y_dot_ = g_lin(x)

  # print(jnp.allclose(y, y_))
  # print(jnp.allclose(y_dot, y_dot_))

  ## test basics: can we run?

  # @jax.jit
  # def g(x):
  #   return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))(x)
  # print(g(x))

  # def g(x):
  #   return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))(x)
  # print(g(x))


  ## test replication checking against out specs (eager)
  # def f(x):
  #   return 2 * x
  # def g(x):
  #   return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)
  # try:
  #   print(g(x))
  # except:
  #   print('good error!')
  # else:
  #   raise Exception

  # def f(x):
  #   return jax.lax.psum(x, 'x')
  # def g(x):
  #   return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)
  # print(g(x))


  ## test eager conrtrol flow

  # x = jnp.arange(2 * 2.).reshape(2, 2)

  # def f(x):
  #   y = jax.lax.psum(x, ('x', 'y'))
  #   if y < 0:
  #     return x
  #   else:
  #     return -x

  # def g(x):
  #   return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))(x)
  # print(g(x))

  # ## test outer jit detects shard_map's mesh
  # x = jnp.array(2.0)
  # f = shard_map(lambda x: x.reshape(1, *x.shape), mesh, P(), P('x'))
  # y = jax.jit(f)(x)  # doesnt work
  # print(y)


  # ## vmap

  # x = jax.device_put(jnp.arange(8 * 8.).reshape(8, 8), sharding)

  # def f(x):
  #   return jax.lax.mul(2., x)

  # def g(x):
  #   return shard_map(f, mesh, in_specs=(P('y'),), out_specs=P('y'))(x)
  # y = jax.vmap(g, axis_name='x')(x)
  # print(y)

  # # y = jax.vmap(g, spmd_axis_name='x')(x)
  # # print(y)


  ## test tree prefix error

#   @partial(shard_map, mesh=mesh, in_specs=([P('x', 'y')],), out_specs=P('x', 'y'))
#   def f(x):
#     print(x)
#     return x
#   try: f([x, x])
#   except: print('good error')
#   else: raise Exception

#   @partial(shard_map, mesh=mesh, in_specs=([P('x', 'y')],), out_specs=P('x', 'y'))
#   def f(x):
#     return x
#   print(f([x]))

  ## rank errors

  # def foo(): return {'hi': [3.]}
  # try:
  #   shard_map(foo, mesh=mesh, in_specs=(), out_specs={'hi': P('x')})()
  # except ValueError:
  #   print('good error')
  # else:
  #   raise Exception("uh oh")

  # try:
  #   jax.jit(lambda: shard_map(foo, mesh=mesh, in_specs=(), out_specs={'hi': P('x')})())()
  # except ValueError:
  #   print('good error')
  # else:
  #   raise Exception("uh oh")

  # def foo(x): pass
  # try:
  #   shard_map(foo, mesh=mesh, in_specs=({'hi': P('x')},), out_specs=())({'hi': [jnp.array(3.)]})
  # except ValueError:
  #   print('good error')
  # else:
  #   raise Exception('uh oh')
