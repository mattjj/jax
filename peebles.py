# import pdb, sys, traceback
# def info(type, value, tb):
#     traceback.print_exception(type, value, tb)
#     pdb.pm()
# sys.excepthook = info

import numpy as np
import jax
import jax.numpy as jnp

@jax.jit
def f(x, y):
  z = x + y
  w = jnp.divide(x, y)
  return z + w

@jax.jit
def g(x, y):
  z = x + y
  w = jax.lax.div(z, y)
  return z + w

@jax.jit
def h(x, y):
  return x / y

# i've been commenting-in lines from here to see what happens:

with jax.debug_nans():
  jnp.divide(1., 1.); jnp.divide(0., 0.)  # second call hits different path!

  # jax.jvp(jnp.log, (0.,), (0.,))
  # first, in _python_pjit_helper, we hit the `not run_impl` case, with
  #   jaxpr = { lambda ; a:f32[]. let b:f32[] = log a in (b,) }
  # then we do these calls:
  #   pjit_p.bind jaxpr = { lambda ; a:f32[]. let b:f32[] = log a in (b,) }
  #     pjit_jvp_rule
  #       pjit_p.bind jaxpr = { lambda ; a:f32[] b:f32[]. let c:f32[] = log a; d:f32[] = div b a in (c, d) }
  #         _pjit_call_impl_python
  #           dispatch.check_special -> raise InternalFloatingPointError
  #
  # this is why we have an InternalFloatingPointError: to re-run original user
  # callable, we need to bubble control back up to _python_pjit_helper, because
  # pjit_p.bind and callees don't have the user callable

  # we need __is_primitive__ because otherwise we get a recursion, since the
  # impl rule for a primitive is jit-this-one-primitive


  # do we want special behavior for numpy functions, where
  #  1. we dont recurse into them if we get a nan, except
  #  2. if we're transforming them, then we do recurse
  # example: bottom out on
  #   b = (x + y) / (x - y)
  #       ~~~~~~~~^~~~~~~~~
  # rather than on div_p.bind
  #
  # maybe let's punt on this


  # jnp.divide(0., 0.)
  # jnp.log(-1)
  # jax.jit(jax.lax.div)(1., 0.)  # shouldnt nan error
  # jax.jit(jax.lax.div)(0., 0.)
  # f(1., 0.)  # shouldnt error
  # f(0., 0.)
  # g(1., 0.)  # shouldnt error
  # g(0., 0.)  # no inner jit
  # h(0., 0.)
  # jax.jit(lambda x: x)(jnp.nan)

  # jax.grad(f)(0., 0.)  # nan on fwd pass
  # jax.jvp(jnp.log, (0.,), (0.,))
  # jax.jvp(jax.lax.log, (0.,), (0.,))

  # nan on bwd pass
  # we have a pjit_p.bind that is not underneath python_pjit_helper
  # the calls are like:
  #   vjp
  #     ad.backward_pass
  #       pjit._pjit_transpose
  #         pjit_p.bind
  #           _pjit_call_impl_python
  jax.vjp(jnp.log, 0.)[1](0.)  # TODO



@jax.jit
def e(x, y):
    a = x * y
    b = (x + y) / (x - y)
    c = a + 2
    return a + b * c

x = jnp.array([2., 5.])
y = jnp.array([3., 5.])
e(x, y).block_until_ready()

x = jnp.array([2., 0.])
y = jnp.array([3., 0.])
with jax.debug_nans():
  e(x, y)



# TODO
# [ ] good error messages
# [ ] fix apply_primitive error message not to talk about dispatch.py
# [ ] apply in_layouts / in_shardings when we re-run
# [ ] think about multi-host, comms, effects
# [x] make ad error message talk about user code... maybe backward_pass should
#     have built in try/except around rules?
# [x] (unrelated) remove reducing_transposes

