from functools import partial
import jax

# exactly the pattern promised by the jax.checkpoint docstring
@partial(jax.checkpoint, static_argnums=(1,))
def foo(x, y):
  with jax.ensure_compile_time_eval():
    y_pos = y > 0
  if y_pos:
    return jax.lax.sin(x)
  else:
    return jax.lax.cos(x)

print("concrete static arg:", jax.grad(foo)(2., 1.))

# issue variant: the static arg is itself a tracer from grad
def f(x):
  return foo(x, x)
print("tracer static arg:", jax.grad(f)(2.))
print("OK")
