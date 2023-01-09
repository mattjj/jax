import jax
import jax.numpy as jnp
from jax._src.lax.control_flow.for_loop import bloop, run_state
import numpy as np

from jax.config import config
config.update('jax_traceback_filtering', 'off')

# def f(refs):
#   i_ref, x_ref, y_ref = refs
#   def cond():
#     return i_ref[()] < x_ref.shape[0]

#   def body():
#     y_ref[()] = jnp.sin(x_ref[()])
#     i_ref[()] += 1

#   bloop(10, cond, body)

# def foo(x):
#   return run_state(f, (0, x, jnp.zeros_like(x)))[2]

# print(jax.make_jaxpr(foo)(jnp.arange(2.)))
# print(foo(jnp.arange(2.)))

# def f(refs):
#   i_ref, x_ref, y_ref = refs
#   def cond():
#     return i_ref[()] < x_ref.shape[0]

#   def body():
#     i = i_ref[()]
#     y_ref[i] = jnp.sin(x_ref[i])
#     i_ref[()] += 1

#   bloop(10, cond, body)

# def sin(x):
#   return run_state(f, (0, x, jnp.zeros_like(x)))[2]

# out = jax.jvp(sin, (jnp.arange(4.),), (jnp.ones(4),))
# print(out)

# _, f_lin = jax.linearize(sin, jnp.arange(4.))
# print(jax.make_jaxpr(f_lin)(jnp.ones(4)))
# print(f_lin(jnp.ones(4)))

# from jax._src import test_util as jtu

# jtu.check_grads(sin, (jnp.arange(4.),), order=3)
# print("Done!")

# def make_vjp(f):
#   def f_vjp(*args):
#     out_primal_py, vjp_py = jax.vjp(f, *args)
#     return vjp_py(out_primal_py)[0]
#   return f_vjp
# sin_vjp = make_vjp(sin)
# sin_vjp2 = make_vjp(jnp.sin)

# print(sin_vjp(jnp.arange(4.)))
# print(sin_vjp2(jnp.arange(4.)))

# print(jax.jvp(sin_vjp, (jnp.arange(4.),), (jnp.ones(4),)))
# print(jax.jvp(sin_vjp2, (jnp.arange(4.),), (jnp.ones(4),)))

# x, sin_vjp_lin = jax.linearize(sin_vjp, jnp.arange(4.))
# y, sin_vjp2_lin = jax.linearize(sin_vjp2, jnp.arange(4.))
# x = jnp.arange(4.)
# print(jax.linearize(sin_vjp, x)[0])
# print(jax.linearize(sin_vjp2, x)[0])
# print(x, y)

# print(jax.make_jaxpr(cos)(jnp.arange(4.)))
# print(sin(jnp.arange(4.)))
# print(cos(jnp.arange(4.)))
# print(sin(jnp.arange(4.)))
# print(cos(jnp.arange(4.)))
# print(jax.make_jaxpr(sin)(jnp.arange(4.)))
# print(jax.make_jaxpr(cos)(jnp.arange(4.)))
# neg_sin = jax.grad(lambda x: cos(x).sum())
# print(neg_sin(jnp.arange(4.)))
# print(jax.make_jaxpr(neg_sin)(jnp.arange(4.)))
# neg_cos = jax.grad(lambda x: neg_sin(x).sum())
# print(jax.make_jaxpr(neg_cos)(jnp.arange(4.)))
# print(jax.grad(lambda x: cos(x).sum())(jnp.arange(4.)))
# print(jax.grad(lambda x: cos(x).sum())(jnp.arange(4.)))

def for_loop(nsteps, body, init_state, *, reverse: bool = False,
             unroll: int = 1):
  def run(refs):
    def wrapped_body(i):
      if reverse:
        i = nsteps - i - 1
      return body(i, refs)
    bloop(wrapped_body, max_iter=nsteps, unroll=unroll)
  return run_state(run, init_state)

def f(x_ref):
  def body(i):
    x_ref[()] += 1
  bloop(body, max_iter=5)

# print(run_state(f, 0))

def body(i, x_ref):
  x_ref[()]
  jax.debug.print("i={i}", i=i)
print(for_loop(10, body, 0., reverse=True, unroll=3))
print(jax.jit(lambda x: for_loop(10, body, x, reverse=True,
  unroll=1)).lower(0.).compiler_ir())
