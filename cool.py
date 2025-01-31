from __future__ import annotations
from dataclasses import dataclass

import jax
from jax._src import core
from jax._src.interpreters import partial_eval as pe
from jax._src import linear_util as lu
from jax._src.util import safe_map, safe_zip

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

class CoolFun:
  in_avals: tuple[core.AbstractValue, ...]

  def __hash__(self):
    assert False

  def __eq__(self, other):
    assert False

  def call(self, *args):
    assert False

  def jvp(self, primals, tangents):
    # return jax.jvp(self.call, primals, tangents)
    jaxpr = module.cool_module[self]
    jvp_prim = JVPOf(self)
    if jvp_prim not in module.cool_module:
      jvp_jaxpr, _ = ad.jvp_jaxpr(jaxpr, [True] * len(primals), True)
      module.cool_module[jvp_prim] = jvp_jaxpr
    out_primals_tangents = cool_p.bind(*primals, *tangents, prim=jvp_prim)
    return _split(out_primals_tangents)

def _split(lst):
  n = len(lst)
  return lst[:n//2], lst[n//2:]

cool_p = core.Primitive('cool')
cool_p.multiple_results = True

@cool_p.def_abstract_eval
def _cool_abstract_eval(*in_avals, prim):
  assert [*in_avals] == [*prim.in_avals]
  return prim.out_avals

@cool_p.def_impl
def _cool_impl(*args, prim):
  return core.jaxpr_as_fun(module.cool_module[prim])(*args)

def bind_cool_call(prim, *args):
  if prim not in module.cool_module:
    module.cool_module[prim] = None
    jaxpr, out_avals, consts, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(prim.call), prim.in_avals)
    module.cool_module[prim] = core.ClosedJaxpr(jaxpr, consts)
  return cool_p.bind(*args, prim=prim)

from jax._src.interpreters import ad

def _cool_jvp(primals, tangents, *, prim):
  return prim.jvp(primals, tangents)
ad.primitive_jvps[cool_p] = _cool_jvp

@dataclass(frozen=True)
class JVPOf(CoolFun):
  prim: CoolFun

#

import jax.numpy as jnp

def einsum(einstr, x, y):
  in_avals = core.get_aval(x), core.get_aval(y)
  out, = bind_cool_call(CoolEinsum(in_avals, einstr), x, y)
  return out

@dataclass(frozen=True)
class CoolEinsum(CoolFun):
  in_avals: tuple[core.AbstractValue, ...]
  einstr: str

  def __str__(self):
    return f'einsum {self.einstr}'

  def call(self, *args):
    return jnp.einsum(self.einstr, *args),


def add(x, y):
  in_avals = tuple(map(core.get_aval, (x, y)))
  out, = bind_cool_call(CoolAdd(in_avals), x, y)
  return out

@dataclass(frozen=True)
class CoolAdd(CoolFun):
  in_avals: tuple[core.AbstractValue, ...]
  def __str__(self): return 'add'
  def call(self, x, y):
    return jnp.add(x, y),

def mul(x, y):
  in_avals = tuple(map(core.get_aval, (x, y)))
  out, = bind_cool_call(CoolMul(in_avals), x, y)
  return out

@dataclass(frozen=True)
class CoolMul(CoolFun):
  in_avals: tuple[core.AbstractValue, ...]
  def __str__(self): return 'mul'
  def call(self, x, y):
    return jnp.multiply(x, y),
  def jvp(self, primals, tangents):
    x, y = primals
    xdot, ydot = tangents
    z = mul(x, y)
    zdot =  add(mul(x, ydot), mul(xdot, y))
    return [z], [zdot]

def fact(n):
  in_avals = out_avals = core.get_aval(n),
  out, = bind_cool_call(CoolFact(in_avals), n)
  return out

@dataclass(frozen=True)
class CoolFact(CoolFun):
  in_avals: tuple[core.AbstractValue, ...]
  def __str__(self): return 'Fact!'
  def call(self, n):
    return jax.lax.cond(n == 0, lambda: 1, lambda: n * fact(n - 1)),
  @property
  def out_avals(self):
    return core.get_aval(1),

#


x = y = jnp.ones(3)

# def f(x, y):
#   a = einsum('i,j->ij', x, y)
#   b = einsum('i,j->ij', x, y)
#   return mul(add(a, b), a)
# jaxpr = jax.make_jaxpr(f)(x, y)
# print(jaxpr)
# f(x, y)  # it runs!


# def g(x):
#   x = add(x, x)
#   x = add(x, x)
#   return mul(x, x)

# jaxpr = jax.make_jaxpr(g)(x)
# print(jaxpr)

# _, d = jax.jvp(g, (x,), (x,))
# print(d)

# jaxpr = jax.make_jaxpr(lambda x, y: jax.jvp(g, (x,), (x,)))(x, x)
# print(jaxpr)

@dataclass(frozen=True)
class Module:
  cool_module: dict[CoolFun, core.ClosedJaxpr]
  def __str__(self):
    items = []
    for prim, body in self.cool_module.items():
      items.append(f'{prim} =\n{body}\n')
    return '\n'.join(items)

module = Module({})


jaxpr = jax.make_jaxpr(fact)(5)
print()
print(module)

from jax._src.lax.control_flow import conditionals
@conditionals.cond_p.def_impl
def cond_impl(idx, *args, branches):
  return core.jaxpr_as_fun(branches[idx])(*args)

print()
print(fact(5))

# print()
# for k, v in cool_module.items():
#   print(k)
#   print(v)


# why?
# * preserve higher-level structure (eg dont inline jax.numpy)
#   for later overloading
# * faster trace times, fewer bugs
# * nicer pretty-prints than existing out-of-line
#
# * hijax
