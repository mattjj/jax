from functools import partial
import jax

def test():
  @partial(jax.remat, static_argnums=(0,))
  def g(x):
    with jax.ensure_compile_time_eval():
      x_pos = float(x) > 0
    if x_pos:
      return jax.lax.sin(x), 3.
    else:
      return jax.lax.cos(x), 4.
  def f(x):
    x, _ = g(x)
    return x
  print(jax.grad(f)(2.))

test()
# check trace state is not corrupted afterwards
print("after:", jax.numpy.sin(1.0))
print("OK")
