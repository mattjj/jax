import jax
import jax.numpy as jnp
from jax._src.lax.control_flow.for_loop import bloop, run_state

def f(refs):
  i_ref, x_ref, y_ref = refs
  def cond():
    return i_ref[()] < x_ref.shape[0]

  def body():
    i = i_ref[()]
    y_ref[i] = jnp.sin(x_ref[i])
    i_ref[()] += 1

  bloop(10, cond, body)

def sin(x):
  return run_state(f, (0, x, jnp.zeros_like(x)))[2]

print(sin(jnp.arange(4.)))
print(jax.make_jaxpr(sin)(jnp.arange(4.)))

out = jax.jvp(sin, (jnp.arange(4.),), (jnp.ones(4),))
print(out)

_, f_lin = jax.linearize(sin, jnp.arange(4.))
print(jax.make_jaxpr(f_lin)(1.))
