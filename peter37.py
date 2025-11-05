from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax._src import core
from jax._src.util import safe_map, safe_zip, toposort
from jax._src.core import typeof
from jax._src.pjit import jit_p

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

@dataclass
class Recipe:
  in_tracers: list[VozTracer]
  out_tracers: None | list[VozTracer]  # TODO weakrefs
  params: dict

class ConstRecipe: pass

class VozTracer(core.Tracer):
  def __init__(self, trace, aval, maybe_val, recipe):
    self._trace = trace
    self._aval = aval
    self.maybe_val = maybe_val
    self.recipe = recipe

  @property
  def aval(self):
    return self._aval

  def __repr__(self):
    if not self.maybe_val:
      force(self)
    return repr(self.maybe_val[0])

  @property
  def parents(self):
    if self.maybe_val:
      return []
    elif isinstance(self.recipe, ConstRecipe):
      return []
    elif isinstance(self.recipe, Recipe):
      return self.recipe.in_tracers
    else:
      assert False

class VozTrace(core.Trace):
  def process_primitive(self, primitive, args, params):
    assert primitive.name == 'jit'
    in_tracers = map(self.to_voz_tracer, args)
    node = Recipe(in_tracers, None, params)
    out_avals = params['jaxpr'].out_avals
    out_tracers = [VozTracer(self, a, [], node) for a in out_avals]
    node.out_tracers = out_tracers
    return out_tracers

  def to_voz_tracer(self, x):
    if isinstance(x, VozTracer):
      return x
    else:
      return VozTracer(self, typeof(x), [x], ConstRecipe())

@dataclass(frozen=True)
class Input:
  pass

@dataclass(frozen=True)
class PjitNode:
  inputs: tuple[int, ...]
  params_tuple: tuple

def force(x: VozTracer) -> None:
  assert not x.maybe_val
  tracers: list[VozTracer] = toposort([x])
  graph, consts = tracers_to_graph(tracers)
  with core.set_current_trace(core.eval_trace):
    val = _force(graph, consts)
  x.maybe_val.append(val)

def tracers_to_graph(tracers):
  consts = []
  graph = []
  input_idxs = {id(t): i for i, t in enumerate(tracers)}
  for i, t in enumerate(tracers):
    if t.maybe_val:
      consts.append(t.maybe_val[0])
      graph.append(Input())
    else:
      assert isinstance(t.recipe, Recipe)
      inputs = [input_idxs[id(t)] for t in t.parents]
      graph.append(PjitNode(tuple(inputs), tuple(t.recipe.params.items())))
  return tuple(graph), consts

@jax.jit(static_argnums=0)
def _force(graph, consts):
  print('compiling!')
  consts_ = iter(consts)
  vals = []
  for g in graph:
    if isinstance(g, Input):
      vals.append(next(consts_))
    else:
      assert isinstance(g, PjitNode)
      in_vals = [vals[i] for i in g.inputs]
      outs = jit_p.bind(*in_vals, **dict(g.params_tuple))
      vals.extend(outs)
  return vals[-1]

voz_trace = VozTrace()
core.trace_ctx.set_trace(voz_trace)

x = jnp.sin(1.0)
y = jnp.cos(x)
print(y)

x = jnp.sin(2.0)
y = jnp.cos(x)
print(y)
