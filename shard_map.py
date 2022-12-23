from functools import partial

from typing import (Any, Callable, Optional, Tuple, List, Sequence, Dict,
                    Hashable, Union)

from jax import core
from jax.core import Tracer
from jax import linear_util as lu
from jax import util
from jax._src import source_info_util
from jax._src.util import prod
from jax.api_util import flatten_fun_nokwargs
from jax.experimental import maps
from jax.experimental. maps import Mesh
from jax.experimental import pjit
from jax.experimental.pjit import PartitionSpec
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import pxla
from jax.tree_util import tree_flatten
from jax.tree_util import tree_map
from jax.tree_util import tree_unflatten
import numpy as np

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


# API

AxisName = Hashable
AxisNames = Dict[int, Tuple[AxisName, ...]]
Specs = Any  # PyTree[Union[PartitionSpec, AxisNames]]

def shard_map(f: Callable, mesh: Mesh, in_specs: Specs, out_specs: Specs):
  def wrapped(*args):
    fun = lu.wrap_init(f)
    args_flat, in_tree = tree_flatten(args)
    in_specs_flat, in_tree_ = tree_flatten(in_specs)
    assert in_tree == in_tree_  # TODO tree prefix
    in_names_flat = tuple(map(_canonicalize_spec, in_specs_flat))
    flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
    def out_names_thunk():
      out_specs_flat, out_tree_ = tree_flatten(out_specs)
      assert out_tree() == out_tree_  # TODO tree prefix
      return tuple(map(_canonicalize_spec, out_specs_flat))
    out_flat = shard_map_p.bind(
        flat_fun, *args_flat, mesh=mesh, in_names=in_names_flat,
        out_names_thunk=out_names_thunk)
    return tree_unflatten(out_tree(), out_flat)
  return wrapped

def _canonicalize_spec(spec: Union[PartitionSpec, AxisNames]) -> AxisNames:
  if isinstance(spec, PartitionSpec):
    return {i: names if isinstance(names, tuple) else (names,)
            for i, names in enumerate(spec)}
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
        fun, tracers, mesh=mesh, in_names=in_names,
        out_names_thunk=out_names_thunk)
    return map(core.full_lower, core.apply_todos(env_trace_todo(), outs))

  def get_bind_params(self, params):
    raise NotImplementedError  # TODO
shard_map_p = ShardMapPrimitive('shard_map')

def process_env_traces(fun, top_trace, mesh):
  return fun, lambda: []  # TODO


# Staging

def _shard_map_staging(
    trace: pe.DynamicJaxprTrace, fun: lu.WrappedFun,
    in_tracers: Sequence[pe.DynamicJaxprTracer], *, mesh: Mesh,
    in_names: Tuple[AxisNames],
    out_names_thunk: Callable[[], Tuple[AxisNames, ...]]
  ) -> Sequence[pe.DynamicJaxprTracer]:
  in_avals = [_shard_aval(mesh, ns, t.aval)
              for t, ns in zip(in_tracers, in_names)]
  with core.new_sublevel(), core.extend_axis_env_nd(mesh.shape.items()):
    jaxpr, out_avals_, consts = pe.trace_to_subjaxpr_dynamic(fun, trace.main, in_avals)
  out_avals = [_unshard_aval(mesh, ns, a)
               for a, ns in zip(out_avals_, out_names_thunk())]
  source_info = source_info_util.current()
  out_tracers = [pe.DynamicJaxprTracer(trace, a, source_info) for a in out_avals]
  invars = map(trace.getvar, in_tracers)
  constvars = map(trace.getvar, map(trace.instantiate_const, consts))
  outvars = map(trace.makevar, out_tracers)
  in_names= ({},) * len(consts) + tuple(in_names)
  params = dict(mesh=mesh, in_names=in_names, out_names=out_names_thunk(),
                jaxpr=pe.convert_constvars_jaxpr(jaxpr))
  eqn = pe.new_jaxpr_eqn([*constvars, *invars], outvars, shard_map_p,
                         params, jaxpr.effects, source_info)
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


# Lowering

def _shard_map_lowering(ctx, *in_nodes, jaxpr, mesh, in_names, out_names):
  sharded_avals = [v.aval for v in jaxpr.invars]
  in_nodes_ = map(partial(_xla_shard, mesh), in_names, ctx.avals_in,
                  sharded_avals, in_nodes)
  new_axis_context = mlir.SPMDAxisContext(mesh, frozenset(mesh.axis_names))
  # TODO(sharadmv): name stack
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

# def _shard_map_impl(trace, fun, args, *, mesh, in_pspecs, out_pspecs_thunk):
#   breakpoint()
#   # TODO check pspecs are consistent with args shardings, by comparing
#   # OpShardings
#   with core.new_base_main(ShardMapTrace, mesh=mesh) as main:
#     with core.new_sublevel(), core.extend_axis_env_nd(mesh.shape.items()):
#       t = main.with_cur_sublevel()
#       tracers = [ShardMapTracer(t, x, p) for x, p in zip(args, pspecs)]
#       ans = fun.call_wrapped(*tracers)
#       out_tracers = map(t.full_raise, ans)
#       # TODO shouldn't be ignoring computed out_pspecs, gotta make things match
#       out_vals, _ = unzip2((t.val, t.pspec) for t in out_tracers)
#       del main, t, tracers, ans, out_tracers
#   return out_vals
# core.EvalTrace.process_shard_map = _shard_map_impl

# class ShardMapTrace(core.Trace):
#   def __init__(self, *args, mesh):
#     super().__init__(*args)
#     self.mesh = mesh

#   def pure(self, val):
#     return ShardMapTracer(self, val, (None,) * np.ndim(val))

#   def sublift(self, tracer):
#     return ShardMapTracer(self, tracer.val, tracer.pspec)

#   def process_primitive(self, primitive, tracers, params):
#     # TODO execute_sharded_on_local_devices(...)
#     raise NotImplementedError("Eager shard map not supported.")

# class ShardMapTracer(core.Tracer):
#   def __init__(self, trace, val, pspec):
#     self._trace = trace
#     self.val = val
#     self.pspec = pspec

#   @property
#   def aval(self):
#     aval = core.raise_to_shaped(core.get_aval(self.val))
#     new_shape = [sz // prod(m.shape[n] for n in names)
#                  for sz, names in zip(aval.shape, self.pspec)]
#     new_shape = [sz // prod(self._trace.mesh[n] for n in self.names[i])
#                  for i, sz in enumerate(aval.shape)]
#     return aval.update(shape=tuple(new_shape))

#   def full_lower(self):
#     return self


##

import os, re
import jax
from jax.experimental import maps
from jax.experimental import pjit
import jax.numpy as jnp

n = 8
xla_flags = os.getenv("XLA_FLAGS", "")
xla_flags = re.sub(
    r"--xla_force_host_platform_device_count=\S+", "", xla_flags
).split()
os.environ["XLA_FLAGS"] = " ".join(
    ["--xla_force_host_platform_device_count={}".format(n)] + xla_flags
)

P = pjit.PartitionSpec


def f(x):
  return jax.lax.mul(2., x)

mesh = Mesh(np.array(jax.devices()[:8]).reshape(4, 2), ('x', 'y'))
pspec = P('x', 'y')
sharding = jax.sharding.NamedSharding(mesh, pspec)
x = jax.device_put(jnp.arange(8 * 8.).reshape(8, 8), sharding)


@jax.make_jaxpr
def g(x):
  return shard_map(f, mesh, in_specs=(pspec,), out_specs=pspec)(x)
print(g(x))

@jax.jit
def g(x):
  return shard_map(f, mesh, in_specs=(pspec,), out_specs=pspec)(x)
print(g(x))
