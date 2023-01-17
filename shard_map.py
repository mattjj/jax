from functools import partial, lru_cache

from typing import (Any, Callable, Optional, Tuple, List, Set, Sequence, Dict,
                    Hashable, Union)

import jax
from jax import core
from jax.core import Tracer
from jax import linear_util as lu
from jax import util
from jax._src import dispatch
from jax._src import pjit
from jax._src import source_info_util
from jax._src.util import prod, HashableFunction, unzip2, unzip3
from jax._src.sharding import NamedSharding, PartitionSpec
from jax._src.lax import parallel as lax_parallel
from jax.api_util import flatten_fun_nokwargs
from jax.experimental import maps
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.interpreters import pxla
from jax.interpreters.pxla import Mesh
from jax.tree_util import tree_flatten
from jax.tree_util import tree_map
from jax.tree_util import tree_unflatten
from jax._src.tree_util import broadcast_prefix
import numpy as np

P = PartitionSpec

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

# TODO [x] tree prefix handling
# TODO [x] ShardMapTrace.process_call
# TODO [x] adapt eager mode to use subset of mesh names (to check replication)
# TODO [ ] vmap rule
# TODO [ ] static checker for replication checks (in typecheck rule)
# TODO [ ] custom_jvp / custom_vjp handling
# TODO [ ] try to implement pmap in terms of shard_map
# TODO [ ] better error if output rank doesn't match out spec for concatenation
# TODO [ ] name stack
# TODO [ ] refine eager-shmap-of-jit checking when (p)jit is initial-style


# API

Specs = Any  # PyTree[PartitionSpec]

def shard_map(f: Callable, mesh: Mesh, in_specs: Specs, out_specs: Specs):
  def wrapped(*args):
    fun = lu.wrap_init(f)
    args_flat, in_tree = tree_flatten(args)
    in_specs_flat = broadcast_prefix(in_specs, args)
    in_names_flat = tuple(map(_canonicalize_spec, in_specs_flat))
    flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
    def out_names_thunk():
      dummy = tree_unflatten(out_tree(), [object()] * out_tree().num_leaves)
      out_specs_flat = broadcast_prefix(out_specs, dummy)
      return tuple(map(_canonicalize_spec, out_specs_flat))
    out_flat = shard_map_p.bind(
        flat_fun, *args_flat, mesh=mesh, in_names=in_names_flat,
        out_names_thunk=out_names_thunk)
    return tree_unflatten(out_tree(), out_flat)
  return wrapped

# Internally we use AxisNames instead of PartitionSpecs
AxisName = Hashable
AxisNames = Dict[int, Tuple[AxisName, ...]]  # TODO make it a hashable dict
def _canonicalize_spec(spec: PartitionSpec) -> AxisNames:
  if isinstance(spec, PartitionSpec):
    return {i: names if isinstance(names, tuple) else (names,)
            for i, names in enumerate(spec) if names is not None}
  else:
    return spec


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
    raise NotImplementedError  # TODO needed for round-tripping through a jaxpr
shard_map_p = ShardMapPrimitive('shard_map')

def process_env_traces(fun, top_trace, mesh):
  return fun, lambda: []  # TODO needed for closing over tracers


# Staging

def _shard_map_staging(
    trace: pe.DynamicJaxprTrace, prim: core.Primitive, fun: lu.WrappedFun,
    in_tracers: Sequence[pe.DynamicJaxprTracer], *, mesh: Mesh,
    in_names: Tuple[AxisNames],
    out_names_thunk: Callable[[], Tuple[AxisNames, ...]]
  ) -> Sequence[pe.DynamicJaxprTracer]:
  in_avals = [t.aval for t in in_tracers]
  in_avals_ = map(partial(_shard_aval, mesh), in_names, in_avals)
  with core.new_sublevel(), core.extend_axis_env_nd(mesh.shape.items()):
    jaxpr, out_avals_, consts = pe.trace_to_subjaxpr_dynamic(
        fun, trace.main, in_avals_)
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
  out_avals_ = [x.aval for x in jaxpr.outvars]
  out_avals = map(partial(_unshard_aval, mesh), out_names, out_avals_)
  return out_avals, jaxpr.effects
core.custom_typechecks[shard_map_p] = _shard_map_typecheck


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
  in_repl = [set(mesh.axis_names) - set(n for ns in names.values() for n in ns)
             for names in in_names]
  with core.new_base_main(ShardMapTrace, mesh=mesh) as main:
    with core.new_sublevel(), core.extend_axis_env_nd(mesh.shape.items()):
      t = main.with_cur_sublevel()
      in_tracers = map(partial(ShardMapTracer, t), in_repl, args)
      ans = fun.call_wrapped(*in_tracers)
      out_tracers = map(t.full_raise, ans)
      outs_, out_repl = unzip2((t.val, t.repl) for t in out_tracers)
      del main, t, in_tracers, ans, out_tracers
  outs = map(partial(_match_spec, mesh), out_repl, out_names_thunk(), outs_)
  return outs
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

def _match_spec(mesh: Mesh, repl: Set[AxisName], dst: AxisNames, x: JaxType
                ) -> JaxType:
  unmentioned = set(mesh.axis_names) - {n for ns in dst.values() for n in ns}
  if not unmentioned.issubset(repl):
    raise Exception  # TODO add parameter to opt out of check
  return jax.jit(_get_matcher(mesh, tuple(dst.items())))(x)

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
    return ShardMapTracer(self, tracer.repl, tracer.val)

  def process_primitive(self, prim, tracers, params):
    in_vals, in_rep = unzip2((t.val, t.repl) for t in tracers)
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
    bind = HashableFunction(
        lambda *args, **kwargs: call_primitive.bind(fun, *args, **kwargs),
        (call_primitive, fun))
    fake_primitive = pxla._FakePrimitive(multiple_results=True, bind=bind)
    # TODO until initial-style jit, assume no replication on jit output
    _rep_rules[fake_primitive] = lambda *_, **__: set()
    return self.process_primitive(fake_primitive, tracers, params)

class ShardMapTracer(core.Tracer):
  repl: Set[AxisName]
  val: JaxType

  def __init__(self, trace, repl, val):
    self._trace = trace
    self.repl = repl
    self.val = val

  @property
  def aval(self):
    aval = core.get_aval(self.val)
    if (isinstance(aval, core.ConcreteArray) and
        self.repl == set(self._trace.mesh.axis_names)):
      with core.eval_context():
        return core.get_aval(self.val[0])
    else:
      aval = core.raise_to_shaped(aval)
      return core.mapped_aval(self._trace.mesh.size, 0, aval)

  def full_lower(self):
    return self

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

# Repl rules are for tracking when values are guaranteed to be equal across
# mapped instances, ultimately used for checking when it's safe to omit an axis
# name in an out_spec. Without the safety check, we wouldn't need names tracked.
# Some collectives have outputs equal across all instances along some mesh axes
# (psum and other reducers, all_gather) while others are sure to introduce
# divergences (axis_index). The default is to intersect the arguments'
# sets of names along which inputs are repeated (i.e. output only repeated along
# an axis if all inputs are repeated along that axis). We need rules for
# collectives and HOPs.

def _rep_rule(prim, *in_repl, **params):
  return set.intersection(*in_repl)

_rep_rules = {}
register_rule = lambda prim: lambda rule: _rep_rules.setdefault(prim, rule)

@register_rule(lax_parallel.psum_p)
def _psum_rule(*in_repl, axes, axis_index_groups):
  if axis_index_groups is not None: raise NotImplementedError
  return [r | set(axes) for r in in_repl]


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
    # return x.sum(keepdims=True)

  mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
  pspec = P('x', 'y')

  sharding = jax.sharding.NamedSharding(mesh, pspec)
  x = jax.device_put(jnp.arange(8 * 8.).reshape(8, 8), sharding)

  ## test basics: can we run?

  @jax.make_jaxpr
  def g(x):
    return shard_map(f, mesh, in_specs=(pspec,), out_specs=pspec)(x)
  print(g(x))

  @jax.jit
  def g(x):
    return shard_map(f, mesh, in_specs=(pspec,), out_specs=pspec)(x)
  print(g(x))

  def g(x):
    return shard_map(f, mesh, in_specs=(pspec,), out_specs=pspec)(x)
  print(g(x))


  ## test replication checking against out specs (eager)
  def f(x):
    return jax.lax.psum(x, ('x',))
  def g(x):
    return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)
  try:
    print(g(x))
  except:
    print('good error!')


  ## test eager conrtrol flow

  x = jnp.arange(2 * 2.).reshape(2, 2)

  def f(x):
    y = jax.lax.psum(x, ('x', 'y'))
    # if y > 0:
    if jax.lax.gt(y, 0.):
      return x
    else:
      return -x

  def g(x):
    return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))(x)

  print(g(x))


  ## test outer jit detects shard_map's mesh
  x = jnp.array(2.0)
  f = shard_map(lambda x: x.reshape(1, *x.shape), mesh, P(), P('x'))
  y = jax.jit(f)(x)  # doesnt work
  print(y)

