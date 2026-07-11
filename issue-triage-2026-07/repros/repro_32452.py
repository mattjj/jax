# Issue 32452: side effects in custom_vjp forward under jit -> UnexpectedTracerError
import jax
import jax.numpy as jnp

def f(x):
  x_stats = {'absmax': jnp.zeros(())}

  def fwd(x):
    x_stats['absmax'] = jnp.max(jnp.abs(x))
    return x, ()

  def bwd(_, g):
    return g,

  fwd_bwd = jax.custom_vjp(lambda x: fwd(x)[0])
  fwd_bwd.defvjp(fwd, bwd)
  return fwd_bwd(x), x_stats['absmax']

x = jnp.full((), 42.)
print(jax.jit(f)(x))
