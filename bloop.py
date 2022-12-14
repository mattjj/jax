import jax
import jax.numpy as jnp
from jax._src.lax.control_flow.for_loop import bloop, run_state

def f(x_ref):
  def cond():
    return x_ref[()] < 5

  def body():
    x_ref[()] += 1

  bloop(10, cond, body)


jaxpr = jax.make_jaxpr(lambda: run_state(f, 0))()
print(jaxpr)

run_state(f, 0)
