from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import typeof

from jax._src import core
from jax._src.tree_util import FlatTree as ft
from jax._src.interpreters.ad import GradAccum, ValAccum, NullAccum
from jax._src.util import safe_map as map, safe_zip as zip, unzip2

def unft_fun(ft_fun):
  return lambda *args: ft_fun(ft.flatten(args)).unflatten()

def ft_fun(pytree_fun):
  return lambda args_ft: ft.flatten(pytree_fun(*args_ft.unflatten()))

def vjp(f, *primals):
  primals_out_ft, f_vjp_ft = _vjp(ft_fun(f), ft.flatten(primals))
  return primals_out_ft.unflatten(), unft_fun(f_vjp_ft)

def _vjp(f, primals):
  tape = []
  with core.take_current_trace() as parent_trace:
    tag = core.TraceTag()
    trace = VJPTrace(parent_trace, tag, tape)
    tracers_in = primals.map(trace.new_arg)
    left_accums = tracers_in.map(lambda x: x.accum)
    with core.set_current_trace(trace):
      tracers_out = f(tracers_in)
    primals_out, right_accs = tracers_out.map(trace.primal_accum_pair).unzip2()
    del trace, tracers_in, tracers_out

  def bwd(right_cts):
    right_accs.map2(lambda acc, ct: acc.accum(ct), right_cts)
    while tape: tape.pop()()
    return left_accums.map(lambda x: x.freeze())

  return primals_out, bwd

class VJPTracer(core.Tracer):
  _trace: VJPTrace
  primal: Any
  accum: GradAccum

  @property
  def aval(self):
    return typeof(self.primal)

  def __init__(self, trace, primal, accum):
    self.accum = accum
    self.primal = primal
    self._trace = trace

class VJPTrace(core.Trace):
  parent_trace: core.Trace
  tape: list
  tag: core.TraceTag

  def new_arg(self, x):
    return VJPTracer(self, x, ValAccum(typeof(x).to_ct_aval()))

  def primal_accum_pair(self, x):
    if isinstance(x, VJPTracer) and x._trace.tag is self.tag:
      return x.primal, x.accum
    else:
      return x, NullAccum(typeof(x).to_ct_aval())

  def process_primitive(self, prim, tracers, params):
    primals, left_accums = unzip2(map(self.primal_accum_pair, tracers))
    primal_out, res, bwd = rules[prim](*primals)
    right_accum = ValAccum(typeof(primal_out).to_ct_aval())
    def thunk():
      bwd(res, right_accum.freeze(), *left_accums)
    self.tape.append(thunk)
    return VJPTracer(self, primal_out, right_accum)

  def __init__(self, parent_trace, tag, tape):
    super().__init__()
    self.tape = tape
    self.tag = tag
    self.parent_trace = parent_trace


from jax._src.lax import lax
rules = {}

def sin_vjp(x):
  bwd = lambda x, g, acc: acc.accum(lax.mul(lax.cos(x), g))
  return lax.sin(x), x, bwd
rules[lax.sin_p] = sin_vjp


###

x = 3.
y, sin_vjp = vjp(jax.lax.sin, x)
x_bar, = sin_vjp(1.)
print(y, x_bar)

# TODO next: get to scan, by figuring out HOP. beware: nonlocal res, nonlocal
# accums

# options:
#  1. one-shot VJPs. you can't even zero-grads to reset things
#  2. like #1 but put a jit on it by default
#  3. build a dag / jaxpr. backward_pass of today

