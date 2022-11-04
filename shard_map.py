import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)


from typing import Tuple, Optional

import numpy as np

from jax import core
from jax.tree_util import tree_flatten, tree_unflatten
from jax.api_util import flatten_fun_nokwargs, shaped_abstractify
from jax import linear_util as lu
from jax.util import safe_map, safe_zip, unzip2
from jax.interpreters import partial_eval as pe
from jax.interpreters import mlir
from jax.interpreters import pxla
from jax.experimental import PartitionSpec
from jax._src import source_info_util
map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

import jax
import jax.numpy as jnp

jax.config.update('jax_array', True)


# Notes:
#  1. it'd be nice if we could avoid annotating the input shardings, and instead
#     just use argument sharding. that could work eagerly, but when staged out
#     we have a phase ordering problem: xla spmd partitioner doesn't decide
#     shardings until we've generated all the HLO for it, but we'd need to
#     change what (static-shape) HLO we generate based on input shardings
#     (specifically the axis sizes). So for now we just have explicit
#     annotations. When we lower, we can generate with_sharding_constraint. That
#     is, we'd like
#       shard_map(f, x)
#     but today we can only do
#       shard_map(f, x, sharding)
#  2. May be nice to do shard_map(f, x, sharding) or shard_map(f)(x, sharding)
#     rather than shard_map(f, sharding)(x) just to underscore that we should
#     think of 'sharding' as bound at the same time as 'x' (rather than 'x'
#     being bound later), per #1 above. On the other hand, it looks unusual...
#  3. We need names. Shardings don't have names. Maybe we want to pass in one
#     Mesh, and a PSpec for each argument? At least for staged version.
#  4. That API looks similar to old pjit, except: (a) mesh is explicit, (b) we
#     don't need output sharding annotations (trivially inferrable b/c map).
#  5. Need either annotations for outputs, or compute it ourselves by some
#     policy (per primitive) and return to user.  # TODO justify why
#  6. TODO Handle partial manual control. How to express "don't care, spmd
#     partitioner can handle this axis" vs "do not handle this axis, but I want
#     it replicated"?
#  7. TODO actually we do need transfer rules for each primitive, for how names
#     transfer through, so that we can support eager. Analogy to vmap.
#  8. Maybe we can have: staged out jaxpr form has explicit output annotations
#     (eg for transposition), but trace-time form can infer output (needed for
#     eager).

class ShardMapPrimitive(core.Primitive):
  multiple_results = True

  def bind(self, fun, *args, mesh, in_pspecs, out_pspecs_thunk):
    top_trace = core.find_top_trace(args)
    fun, env_trace_todo = process_env_traces(
        fun, top_trace and top_trace.level, mesh)
    tracers = map(top_trace.full_raise, args)
    outs = top_trace.process_shard_map(
        fun, tracers, mesh=mesh, in_pspecs=in_pspecs,
        out_pspecs_thunk=out_pspecs_thunk)
    return map(core.full_lower, core.apply_todos(env_trace_todo(), outs))

  def get_bind_params(self, params):
    breakpoint()


shard_map_p = ShardMapPrimitive('shard_map')

def process_env_traces(fun, top_trace, mesh):
  return fun, lambda: []  # TODO

def _shard_map_impl(trace, fun, args, *, mesh, in_pspecs, out_pspecs_thunk):
  # TODO check pspecs are consistent with args shardings, by comparing
  # OpShardings
  with core.new_base_main(ShardMapTrace, mesh=mesh) as main:
    with core.new_sublevel(), core.extend_axis_env_nd(mesh.shape.items()):
      t = main.with_cur_sublevel()
      tracers = [ShardMapTracer(t, x, p) for x, p in zip(args, pspecs)]
      ans = fun.call_wrapped(*tracers)
      out_tracers = map(t.full_raise, ans)
      out_vals, _ = unzip2((t.val, t.pspec) for t in out_tracers)
      del main, t, tracers, ans, out_tracers
  return out_vals
core.EvalTrace.process_shard_map = _shard_map_impl

def _shard_map_staging(trace, fun, in_tracers, *, mesh, in_pspecs, out_pspecs_thunk):
  in_avals = [x.aval.update(shape=tuple(d if n is None else d // mesh.shape[n]
                                        for d, n in zip(x.shape, p)))
              for x, p in zip(in_tracers, in_pspecs)]
  with core.new_sublevel():
    jaxpr, out_avals_, consts = pe.trace_to_subjaxpr_dynamic(
        fun, trace.main, in_avals)
  out_pspecs = [canonicalize_pspec(x.aval.shape, p)
                for x, p in zip(jaxpr.outvars, out_pspecs_thunk())]
  out_avals = [x.aval.update(shape=tuple(d if n is None else d * mesh.shape[n]
                                         for d, n in zip(x.aval.shape, p)))
               for x, p in zip(jaxpr.outvars, out_pspecs)]
  source_info = source_info_util.current()
  out_tracers = [pe.DynamicJaxprTracer(trace, a, source_info)
                 for a in out_avals]
  invars = map(trace.getvar, in_tracers)
  constvars = map(trace.getvar, map(trace.instantiate_const, consts))
  outvars = map(trace.makevar, out_tracers)
  params = dict(mesh=mesh, in_pspecs=in_pspecs, out_pspecs=out_pspecs,
                jaxpr=pe.convert_constvars_jaxpr(jaxpr))
  eqn = pe.new_jaxpr_eqn([*constvars, *invars], outvars, shard_map_p,
                         params, jaxpr.effects, source_info)
  trace.frame.add_eqn(eqn)
  return out_tracers
pe.DynamicJaxprTrace.process_shard_map = _shard_map_staging


def _shard_map_lowering(ctx, *in_vals, jaxpr, mesh, in_pspecs, out_pspecs):
  in_axes_maps = [{name: i for i, name in enumerate(p) if name is not None}
                  for p in in_pspecs]
  in_vals_ = [pxla._full_to_shard_lowering(ctx, x, axes=a, mesh=mesh,
                                           manual_axes=frozenset())[0]
              for x, a in zip(in_vals, in_axes_maps)]
  sub_ctx = ctx.module_context
  outs, _ = mlir.jaxpr_subcomp(sub_ctx, jaxpr, mlir.TokenSet(), (), *in_vals_)
  breakpoint()
mlir.register_lowering(shard_map_p, _shard_map_lowering)


class ShardMapTrace(core.Trace):
  def __init__(self, *args, mesh):
    super().__init__(*args)
    self.mesh = mesh

  def pure(self, val):
    return ShardMapTracer(self, val, (None,) * np.ndim(val))

  def sublift(self, tracer):
    return ShardMapTracer(self, tracer.val, tracer.pspec)

  def process_primitive(self, primitive, tracers, params):
    in_arrs, in_pspecs = unzip2((t.val, t.pspec) for t in tracers)
    breakpoint()
    # TODO execute_sharded_on_local_devices(...)

class ShardMapTracer(core.Tracer):
  def __init__(self, trace, val, pspec):
    self._trace = trace
    self.val = val
    self.pspec = pspec

  @property
  def aval(self):
    aval = core.raise_to_shaped(core.get_aval(self.val))
    mesh_shape = self._trace.mesh.shape
    new_shape = [d if axis_name is None else d // mesh_shape[axis_name]
                 for d, axis_name in zip(aval.shape, self.pspec)]
    return aval.update(shape=tuple(new_shape))

  def full_lower(self):
    return self

def shard_map(f, mesh, args, in_pspecs, out_pspecs):
  f = lu.wrap_init(f)
  args_flat, in_tree = tree_flatten(args)
  in_pspecs_flat, in_tree_ = tree_flatten(in_pspecs)
  in_pspecs_flat = [canonicalize_pspec(x.shape, p)
                    for x, p in zip(args_flat, in_pspecs_flat)]
  assert in_tree == in_tree_
  f, out_tree = flatten_fun_nokwargs(f, in_tree)
  def out_pspecs_thunk():
    out_pspecs_flat, out_tree_ = tree_flatten(out_pspecs)
    assert out_tree() == out_tree_
    return out_pspecs_flat
  out_flat = shard_map_p.bind(
      f, *args_flat, mesh=mesh, in_pspecs=in_pspecs_flat,
      out_pspecs_thunk=out_pspecs_thunk)
  return tree_unflatten(out_tree(), out_flat)

def canonicalize_pspec(shape: Tuple[int, ...], p: PartitionSpec
                       ) -> Tuple[Optional[core.AxisName], ...]:
  assert len(p) == len(shape)
  return tuple(p)

##

from jax.experimental import maps
P = PartitionSpec


def f(x):
  return jax.lax.mul(2., x)

mesh = maps.Mesh(np.array(jax.devices()[:8]).reshape(4, 2), ('x', 'y'))
pspec = P('x', 'y')
sharding = jax.sharding.NamedSharding(mesh, pspec)
x = jax.device_put(jnp.arange(8 * 8.).reshape(8, 8), sharding)


@jax.make_jaxpr
def g(x):
  return shard_map(f, mesh, (x,), in_pspecs=(pspec,), out_pspecs=pspec)
print(g(x))

@jax.jit
def g(x):
  return shard_map(f, mesh, (x,), in_pspecs=(pspec,), out_pspecs=pspec)
print(g(x))
