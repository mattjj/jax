import jax
import jax.numpy as jnp

@jax.custom_jvp
def multiply_no_nan(x, y):
  return jax.lax.select(jnp.equal(y, 0.), jnp.zeros_like(x), x * y)

@multiply_no_nan.defjvp
def multiply_no_nan_jvp(primals, tangents):
  x, y = primals
  dx, dy = tangents
  return (multiply_no_nan(x, y), multiply_no_nan(dx, y) + multiply_no_nan(x, dy))

x = jnp.array(jnp.inf)
y = jnp.array(0.0)

print(jax.grad(multiply_no_nan, (0, 1))(x, y))  # want: (0.0, inf)
