import jax
import jax.numpy as jnp

from jax import core

aval = core.ShapedArray((), core.AbstractKey('fry'))
print(aval)

from jax.interpreters import partial_eval as pe
from jax._src.api_util import flatten_fun_nokwargs
import jax.linear_util as lu

def make_jaxpr(f, *in_avals):
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(lambda *xs: [f(*xs)]), in_avals)
  return core.ClosedJaxpr(jaxpr, consts)

def f(key):
  x = jax.random.normal(key, (3,))
  return x

jaxpr = make_jaxpr(f, aval)
print(jaxpr)

# { lambda k:key[]

# do we need a function i32[N] -> fry[N] in the jaxpr language
# or can it always come in as an argument
# k:fry[] = make_key s:i32[]


