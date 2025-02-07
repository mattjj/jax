from functools import partial

import jax
import jax.numpy as jnp

def hermit(f):
  f = jax.jit(f)
  def f_hermit(*args):
    out = jax.tree.map(jnp.zeros_like, jax.eval_shape(f, *args))
    def cond(carry):
      i, _ = carry
      return i < jax.lax.optimization_barrier(1)
    def body(carry):
      i, out = carry
      return i + 1, f(*args)
    _, out = jax.lax.while_loop(cond, body, (0, out))
    return out
  return f_hermit

##

@jax.custom_vjp
def f(x):
  return jnp.sin(jnp.sin(x))

def f_fwd(x):
  y, _, _ = h(x)
  return y, x

def f_bwd(x, y_bar):
  _, cos_x, cos_sin_x = h(x)
  return cos_x * (cos_sin_x * y_bar),

f.defvjp(f_fwd, f_bwd)

@hermit
def h(x):
  return jnp.sin(jnp.sin(x)), jnp.cos(x), jnp.cos(jnp.sin(x))

f(3.)
jax.grad(lambda x: f(f(x)))(3.)

##

def ad_hermit(f):
  f_ = jax.custom_vjp(f)
  h = hermit(partial(jax.vjp, f))

  def f_fwd(*args):
    out, _ = h(*args)
    return out, args

  def f_bwd(args, out_bar):
    _, f_vjp = h(*args)
    return f_vjp(out_bar)

  f_.defvjp(f_fwd, f_bwd)
  return f_

##

@ad_hermit
def f(x):
  return jnp.sin(jnp.sin(x))


jax.grad(f)(3.)

# TODO sharding annotations
