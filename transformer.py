import jax
import jax.numpy as jnp

jax.config.update('jax_disable_jit', True)
jax.config.update('jax_dynamic_shapes', True)

# def fprop_layer(params, x, t0, i):
def fprop_layer(params, x):
  ((xnorm_scale, xnorm_bias), (wqkv, wqkv_bias), (wo, wo_bias),
   (ynorm_scale, ynorm_bias), (w_i, w_i_bias), (w_o, w_o_bias)) = params
  xnorm = jax.nn.normalize(x) * xnorm_scale + xnorm_bias
  qkv = jnp.einsum('bte,ihqe->ibthq', xnorm, wqkv) + wqkv_bias[:, None, None]
  q, k, v = qkv
  outer = jnp.einsum('bthq,bshq->btsh', q, k) / jnp.asarray(
      jnp.sqrt(v.shape[-1]), dtype=x.dtype)

  # # s refers to timestep attended to; t refers to timestep attending
  # s = jnp.arange(outer.shape[2])[None, None, :]
  # t = (0 if t0 is None else t0[:, None, None]
  #      ) + jnp.arange(outer.shape[1])[None, :, None]
  # if i is not None or t0 is not None:
  #   invalid = t < s
  #   outer = outer - jnp.asarray(
  #       jnp.inf, dtype=x.dtype) * invalid[:, :, :, None]

  alpha = jax.nn.softmax(outer, 2)
  inner = jnp.einsum('btsh,bshq->bthq', alpha, v)
  y = jnp.einsum('bthq,hqe->bte', inner, wo) + wo_bias + x
  ynorm = jax.nn.normalize(y) * ynorm_scale + ynorm_bias
  act = jax.nn.gelu(jnp.einsum('bte,ef->btf', ynorm, w_i) + w_i_bias)
  z = jnp.einsum('btf,fe->bte', act, w_o) + w_o_bias + y
  return z

params = [
    (jnp.ones(1024), jnp.zeros(1024)),  # xnorm_scale, xnorm_bias
    (jnp.ones((3, 16, 64, 1024)), jnp.zeros((3, 16, 64))),  # wqkv, wqkv_bias
    (jnp.ones((16, 64, 1024)), jnp.zeros(1024)),  # wo, wo_bias
    (jnp.ones(1024), jnp.zeros(1024)),  # ynorm_scale, ynorm_bias
    (jnp.ones((1024, 4096)), jnp.zeros(4096)),  # w_i, w_i_bias
    (jnp.ones((4096, 1024)), jnp.zeros(1024)),  # w_o, w_o_bias
]

# x = jnp.zeros((8, 512, 1024))
# fprop_layer(params, x)

xs = [
    jnp.zeros((512, 1024)),
    jnp.zeros((386, 1024)),
    jnp.zeros((420, 1024)),
]

import numpy as np
from jax._src import core
from jax._src.interpreters.batching import pile_axis, Pile, PileTy, IndexedAxisSize

def pile_stack(xs: list[jax.Array]) -> Pile:
  max_length = max(len(x) for x in xs)
  lengths = jnp.array([len(x) for x in xs])
  lengths = jax.lax.convert_element_type(lengths, core.bint(max_length))
  xs_padded = jnp.stack([jnp.zeros((max_length, 1024), dtype=x.dtype
                                   ).at[:x.shape[0]].set(x) for x in xs])
  # jax.vmap(lambda l, xp: xp[:l, :], out_axes=pile_axis)(lengths, xs_padded)

  # binder = i
  binder = core.Var(0, '', core.ShapedArray((), np.dtype('int32')))
  # elt_ty = f32[[3, 1, 4].i, 1024]
  elt_ty = core.DShapedArray((IndexedAxisSize(binder, lengths), 1024),
                             xs_padded.dtype)
  # aval = i:(Fin 3) => f32[[3, 1, 4].i, 1024]
  aval = PileTy(binder, len(xs), elt_ty)
  xs_pile = Pile(aval, xs_padded)
  return xs_pile

xs_pile = pile_stack(xs)


fprop_batched = jax.vmap(fprop_layer,
                         in_axes=(None, pile_axis),
                         out_axes=pile_axis,
                         axis_size=3)
fprop_batched(params, xs_pile)


# TODO
#  * make .astype work
#  * make max(lengths) work with core.bint ?
#  * terrible error message with .at[...].set(...) when ranks are wrong...
#  * make slicing work
