import jax
import jax.numpy as jnp
from jax._src.lax.control_flow.for_loop import bloop, run_state

def f(x_ref):
  def cond():
    return True

  def body():
    x_ref[()] += 1.

  bloop(2, cond, body)


jaxpr = jax.make_jaxpr(lambda: run_state(f, 1.))()

print(run_state(f, 1.))
# jax.jvp(lambda x: run_state(f, x), (1.,), (1.,))
