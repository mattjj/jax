from contextlib import contextmanager
import jax
import jax.numpy as jnp
from jax._src import config

# TODO on JAX
#  [ ] boxes
#  [ ] ad
#  [ ] refs

RematLevel = int | None  # data RematLevel = NoAD | ADWithRematLevel Int

# N: int | None = 2  # hardcoded
# n: int | None = 0

@contextmanager
def increment_remat_level():
  n = config.remat_level.value
  try:
    config.update('jax_remat_level', n + 1)
    yield
  finally:
    config.update('jax_remat_level', n)

def remat_level():
  return config.remat_level.value

def get_remat_box():
  return remat_box
remat_box = jax.new_ref(0.)

def remat(f):
  f_ = jax.custom_vjp(f)
  def fwd(x):
    return f(x), x  # fwd
  def bwd(x, g):
    with increment_remat_level():
      _, f_vjp = jax.vjp(f, x)  # rematted fwd
      return f_vjp(g)           # bwd
  f_.defvjp(fwd, bwd)
  return f_


@jax.jit
@remat
@remat
@jax.jit
def f(x):
  x1 = jnp.sin(x)
  if remat_level() == 0:
    jax.debug.print('n=0')
  elif remat_level() == 1:
    jax.debug.print('n=1')
    get_remat_box()[...] = jax.lax.stop_gradient(x1)
  elif remat_level() == 2:
    jax.debug.print('n=2')
    x1 = merge_primal1_and_tangent2(get_remat_box()[...], x1)
  else:
    assert False, breakpoint()
  x2 = jnp.sin(x1)
  x3 = jnp.sin(x2)
  return x3

@jax.custom_jvp
def merge_primal1_and_tangent2(x1, x2):
  jax.debug.print('BAR')
  return x1
@merge_primal1_and_tangent2.defjvp
def _(primals, tangents):
  (x, _), (_, t) = primals, tangents
  return x, t

y2, g = jax.value_and_grad(f)(3.)
print(y2, jnp.sin(jnp.sin(jnp.sin(3.))))
print(g, jnp.cos(3.) * jnp.cos(jnp.sin(3.)) * jnp.cos(jnp.sin(jnp.sin(3.))) )


# @jax.jit
# @remat
# def f(x):
#   if n == 0:
#     return x ** 2
#   elif n == 1:
#     # could do a custom_vjp/custom_lin here if you want
#     return x ** 3 / x
#   else:
#     raise Exception
# y2, g = jax.value_and_grad(f)(3.)
# print(y2, 3. ** 2)
# print(g, 2. * 3)



# @jax.jit
# @remat
# def f(x):
#   x1 = jnp.sin(x)
#   if n == 0:
#     get_remat_box()[...] = x1
#   else:
#     x1 = merge_primal1_and_tangent2(get_remat_box()[...], x1)
#   x2 = jnp.sin(x1)
#   x3 = jnp.sin(x2)
#   return x3

# @jax.custom_jvp
# def merge_primal1_and_tangent2(x1, x2):
#   assert False
# @merge_primal1_and_tangent2.defjvp
# def _(primals, tangents):
#   (x, _), (_, t) = primals, tangents
#   return x, t

# y2, g = jax.value_and_grad(f)(3.)
# print(y2, jnp.sin(jnp.sin(jnp.sin(3.))))
# print(g, jnp.cos(3.) * jnp.cos(jnp.sin(3.)) * jnp.cos(jnp.sin(jnp.sin(3.))) )





from functools import partial
import itertools as it


import jax
import jax.numpy as jnp
import jax.ad_checkpoint
from jax.tree_util import tree_flatten, tree_unflatten

jax.config.update('jax_platform_name', 'cpu')  # suppress warning

from rich.console import Console
from rich.table import Table
import rich.text


def print_fwd_bwd(f, *args, **kwargs) -> None:
  args, in_tree = tree_flatten((args, kwargs))

  def f_(*args):
    args, kwargs = tree_unflatten(in_tree, args)
    return f(*args, **kwargs)

  fwd = jax.make_jaxpr(lambda *args: jax.vjp(f_, *args))(*args).jaxpr

  y, f_vjp = jax.vjp(f_, *args)
  res, in_tree = tree_flatten(f_vjp)

  def g_(*args):
    *res, y = args
    f_vjp = tree_unflatten(in_tree, res)
    return f_vjp(y)

  bwd = jax.make_jaxpr(g_)(*res, y).jaxpr

  table = Table(show_header=False, show_lines=True, padding=(1, 2, 0, 2), box=None)
  table.add_row("[bold green]forward computation:",
                "[bold green]backward computation:")
  table.add_row(rich.text.Text.from_ansi(str(fwd)),
                rich.text.Text.from_ansi(str(bwd)))
  console = Console(width=240)
  console.print(table)

print_fwd_bwd(f, 3.)
