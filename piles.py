import jax
import jax.numpy as jnp

jax.config.update('jax_dynamic_shapes', True)
jax.config.update('jax_disable_jit', True)

import numpy as np
from jax._src import core
from jax._src.interpreters.batching import pile_axis, Pile, PileTy, IndexedAxisSize

def pile_stack(xs: list[jax.Array]) -> Pile:
  suffix_shape, = {x.shape[1:] for x in xs}
  max_length = max(len(x) for x in xs)
  lengths = jnp.array([len(x) for x in xs])
  lengths = jax.lax.convert_element_type(lengths, core.bint(max_length))
  xs_padded = jnp.stack([jnp.zeros((max_length, *suffix_shape), dtype=x.dtype
                                   ).at[:x.shape[0]].set(x) for x in xs])
  # jax.vmap(lambda l, xp: xp[:l, :], out_axes=pile_axis)(lengths, xs_padded)

  binder = core.Var(0, '', core.ShapedArray((), np.dtype('int32')))
  elt_ty = core.DShapedArray((IndexedAxisSize(binder, lengths), *suffix_shape),
                             xs_padded.dtype)
  aval = PileTy(binder, len(xs), elt_ty)
  xs_pile = Pile(aval, xs_padded)
  return xs_pile

###

def f(x):
  return x.sum()

xs = [jnp.ones(3), jnp.ones(1), jnp.ones(4)]

x_pile = pile_stack(xs)
print(x_pile)


f_mapped = jax.vmap(f, in_axes=pile_axis, out_axes=0, axis_size=3)
y_arr = f_mapped(x_pile)
print(y_arr)

