# Adapted: jax._src.core.mutable_array (removed) -> jax.new_ref (current API).
import jax
import jax.numpy as jnp

a = jnp.float32(0)
a_ref = jax.new_ref(a)

@jax.jit
@jax.grad
def f(x):
    a_ref[()] = x  # writing function input to a global mutable array
    return 2 * x

x = jnp.float32(3.)
print(f(x))
print("a_ref now:", a_ref[()])
