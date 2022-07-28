import jax
import jax.numpy as jnp

from jax import core
from jax._src import prng

import ipdb, sys, traceback
def info(type, value, tb):
  traceback.print_exception(type, value, tb)
  ipdb.pm()
sys.excepthook = info

#aval = core.ShapedArray((), core.AbstractKey('fry'))
aval = core.ShapedArray((), core.AbstractKey(prng.threefry_prng_impl))
print(aval)

from jax.interpreters import partial_eval as pe
from jax._src.api_util import flatten_fun_nokwargs
import jax.linear_util as lu

def make_jaxpr(f, *in_avals):
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(lambda *xs: [f(*xs)]), in_avals)
  return core.ClosedJaxpr(jaxpr, consts)

def f(key):
  k1, k2 = jax.random.split(key, 2)
  x = jax.random.normal(k1, (1234, 5000))
  y = jax.random.uniform(k2, (1234, 5000))
  return x + y

jaxpr = make_jaxpr(f, aval)
print(jaxpr)

with jax.disable_jit():
  x = f(jax.random.PRNGKey(jnp.array(27)))
  print(x[:3, :4])
  print(x.mean())

print()

def f(seed):
  key = jax.random.PRNGKey(seed)
  return jax.random.uniform(key, (10000, 20000))

jaxpr = make_jaxpr(f, core.ShapedArray((), jnp.dtype('uint32')))
print(jaxpr)

with jax.disable_jit():
  x = f(jnp.array(27))
  print(x[:3, :4])
  print(x.mean())

print()

# fv = jax.vmap(f)

# jaxpr = make_jaxpr(fv, core.ShapedArray((3,), jnp.dtype('uint32')))
# print(jaxpr)

# with jax.disable_jit():
#   x = fv(jnp.array([27, 12, 9]))
#   print(x[:, :3, :4])
#   print(x.mean(axis=[1, 2]))

# print()

x = f(jnp.array(27))
print(x[:3, :4])
print(x.mean())

print()

x = jax.jit(f)(jnp.array(27))
print(x[:3, :4])
print(x.mean())

print()




# do we need a function i32[N] -> fry[N] in the jaxpr language
# or can it always come in as an argument
# k:fry[] = make_key s:i32[]
