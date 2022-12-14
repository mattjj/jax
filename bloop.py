import jax
import jax.numpy as jnp
from jax._src.lax.control_flow.for_loop import bloop, run_state

def f(x_ref):
  def cond():
    return x_ref[()] < 5

  def body():
    x_ref[()] += 1.

  bloop(2, cond, body)


jaxpr = jax.make_jaxpr(lambda: run_state(f, 1.))()

print(jaxpr)
print(run_state(f, 1.))
out = jax.jvp(lambda x: run_state(f, x), (1.,), (1.,))
print(out)

print(jax.make_jaxpr(lambda x, t: jax.jvp(lambda x: run_state(f, x), (x,),
  (t,)))(1., 1.))
