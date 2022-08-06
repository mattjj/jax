import jax
import jax.numpy as jnp
from jax import (
    make_jaxpr,
)

from jax import core
from jax._src import prng

import ipdb, sys, traceback
def info(type, value, tb):
  traceback.print_exception(type, value, tb)
  ipdb.pm()
sys.excepthook = info

seed = jnp.array(27, dtype=jnp.dtype('uint32'))

def f(key):
  k1, k2 = jax.random.split(key, 2)
  x = jax.random.normal(k1, (1234, 5000))
  y = jax.random.uniform(k2, (1234, 5000))
  return x + y

key = jax.random.PRNGKey(jnp.array(0, dtype=int))
print(key)
jaxpr = make_jaxpr(f)(key)
print(jaxpr)

with jax.disable_jit():
  x = f(jax.random.PRNGKey(seed))
  print(x[:3, :4])
  print(x.mean())

print()

def f(seed):
  key = jax.random.PRNGKey(seed)
  return jax.random.uniform(key, (10000, 20000))

jaxpr = make_jaxpr(f)(key)
print(jaxpr)

with jax.disable_jit():
  x = f(seed)
  print(x[:3, :4])
  print(x.mean())

print()

seeds = jnp.array([27, 12, 9], dtype=jnp.dtype('uint32'))
fv = jax.vmap(f)
jaxpr = make_jaxpr(fv)(seeds)
print(jaxpr)

with jax.disable_jit():
  x = fv(seeds)
  print(x[:, :3, :4])
  print(x.mean(axis=[1, 2]))

print()

x = f(seed)
print(x[:3, :4])
print(x.mean())

print()

x = jax.jit(f)(seed)
print(x[:3, :4])
print(x.mean())

print()




# do we need a function i32[N] -> fry[N] in the jaxpr language
# or can it always come in as an argument
# k:fry[] = make_key s:i32[]
