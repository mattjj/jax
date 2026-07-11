import jax
from jax import numpy as jnp

#jax.config.update("jax_dynamic_shapes", True)

@jax.jit
def foo(N: int):
    arr = jnp.arange(N)
    raveled = arr.ravel()
    return raveled

print(foo(5))
