from typing import Any

from jax import core
import jax.numpy as jnp

Array = Any

import pdb, sys, traceback
def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    pdb.pm()
sys.excepthook = info


class PRNGKeyArray:
  data: Array
  shape = property(lambda self: self.data.shape[:-1])

  def __init__(self, data):
    self.data = data

  def reshape(self, newshape, order=None):
    del order  # Ignored.
    return PRNGKeyArray(jnp.reshape(self.data, (*newshape, -1)))

  def concatenate(self, key_arrs, axis):
    axis = axis % len(self.shape)
    arrs = [self.data, *[k.data for k in key_arrs]]
    return PRNGKeyArray(jnp.concatenate(arrs, axis))

jnp.register_stackable(PRNGKeyArray)


def print_type(x):
  print(_print_type(x))

def _print_type(x) -> str:
  if isinstance(x, PRNGKeyArray):
    return f'PRNGKeyArray({_print_type(x.data)})'
  else:
    aval = core.raise_to_shaped(core.get_aval(x))
    return aval.str_short(short_dtypes=True)

data = jnp.arange(12, dtype='uint32').reshape(6, 2)
print_type(data)

x = PRNGKeyArray(data)
y = jnp.reshape(x, (2, 3))
print_type(y.data)

# So given that current jnp.reshape reflects on object method, we don't need any
# separate mechanism for overloading reshape.

# Next:
#  * concatenate
#  * broadcast (optional)
#  * check_arraylike
#
# Test:
#  * jnp.array
#  * jnp.stack

z = jnp.stack([y, y, y, y])
print_type(z)

# So reflecting works well! That is, if we make concatenate (and broadcast) ask
# their arguments how to perform the operation, then overloading is done.
# Any reason not to?
# It might tempt mutation, whereas a handler would suggest against that (?).
