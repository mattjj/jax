from functools import partial
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')


@partial(jax.jit, device=jax.devices('cpu')[0])
def f1(x):
  return jnp.sin(jnp.sin(x))


@partial(jax.jit, device=jax.devices('cpu')[1])
def f2(x):
  return jnp.cos(jnp.cos(x))

def f(x):
  y = f1(x)
  z = f2(y)
  return z


jaxpr = jax.make_jaxpr(f)(3.)
print(jaxpr)
print('===')
print(jax.jit(f).lower(3.).compiler_ir(dialect="mhlo"))
