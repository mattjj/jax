import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
from jax._src.lax.control_flow.for_loop import for_loop

def id(xs):
  _, xs = jax.lax.scan(lambda _, x: (None, x), None, xs, length=xs.shape[0])
  return xs

jaxpr_transpose = jax.make_jaxpr(jax.vjp(id, jnp.zeros(3))[1])(jnp.zeros(3))
print(jaxpr_transpose)

# jaxpr_transpose = jax.xla_computation(jax.vjp(id, jnp.zeros(3))[1])(jnp.zeros(3))
# print(jaxpr_transpose.as_hlo_text())


# region_0.6 {
#   arg_tuple.7 = (s32[], f32[3]{0}, f32[3]{0}) parameter(0)
#   get-tuple-element.8 = s32[] get-tuple-element(arg_tuple.7), index=0
#   constant.11 = s32[] constant(1)
#   add.17 = s32[] add(get-tuple-element.8, constant.11)
#   get-tuple-element.9 = f32[3]{0} get-tuple-element(arg_tuple.7), index=1
#   get-tuple-element.10 = f32[3]{0} get-tuple-element(arg_tuple.7), index=2
#   constant.12 = s32[] constant(3)
#   subtract.13 = s32[] subtract(constant.12, get-tuple-element.8)
#   subtract.14 = s32[] subtract(subtract.13, constant.11)
#   dynamic-slice.15 = f32[1]{0} dynamic-slice(get-tuple-element.10, subtract.14), dynamic_slice_sizes={1}
#   dynamic-update-slice.16 = f32[3]{0} dynamic-update-slice(get-tuple-element.9, dynamic-slice.15, subtract.14)
#   ROOT tuple.18 = (s32[], f32[3]{0}, f32[3]{0}) tuple(add.17, dynamic-update-slice.16, get-tuple-element.10)
# }

# print('======')

def id(xs):
  def body_fun(carry, _):
    i, xs = carry
    x = jax.lax.dynamic_index_in_dim(xs, i, keepdims=False)
    xs = jax.lax.dynamic_update_index_in_dim(xs, x, i, 0)
    return (i + 1, xs), None

  (_, xs), _ = jax.lax.scan(body_fun, (0, xs), None, length=xs.shape[0])
  return xs

jaxpr_transpose = jax.make_jaxpr(jax.vjp(id, jnp.zeros(3))[1])(jnp.zeros(3))
print(jaxpr_transpose)

# jaxpr_transpose = jax.xla_computation(jax.vjp(id, jnp.zeros(3))[1])(jnp.zeros(3))
# print(jaxpr_transpose.as_hlo_text())

# # region_0.8 {
# #   arg_tuple.9 = (s32[], f32[3]{0}) parameter(0)
# #   get-tuple-element.10 = s32[] get-tuple-element(arg_tuple.9), index=0
# #   constant.17 = s32[] constant(1)
# #   add.29 = s32[] add(get-tuple-element.10, constant.17)
# #   get-tuple-element.11 = f32[3]{0} get-tuple-element(arg_tuple.9), index=1
# #   constant.15 = f32[1]{0} constant({0})
# #   constant.16 = s32[3]{0} constant({0, 1, 2})
# #   constant.18 = s32[] constant(3)
# #   subtract.19 = s32[] subtract(constant.18, get-tuple-element.10)
# #   subtract.20 = s32[] subtract(subtract.19, constant.17)
# #   dynamic-slice.21 = s32[1]{0} dynamic-slice(constant.16, subtract.20), dynamic_slice_sizes={1}
# #   reshape.22 = s32[] reshape(dynamic-slice.21)
# #   dynamic-update-slice.23 = f32[3]{0} dynamic-update-slice(get-tuple-element.11, constant.15, reshape.22)
# #   constant.12 = f32[] constant(0)
# #   broadcast.13 = f32[3]{0} broadcast(constant.12), dimensions={}
# #   dynamic-slice.24 = f32[1]{0} dynamic-slice(get-tuple-element.11, reshape.22), dynamic_slice_sizes={1}
# #   constant.14 = f32[] constant(0)
# #   reduce.25 = f32[] reduce(dynamic-slice.24, constant.14), dimensions={0}, to_apply=region_1.4
# #   reshape.26 = f32[1]{0} reshape(reduce.25)
# #   dynamic-update-slice.27 = f32[3]{0} dynamic-update-slice(broadcast.13, reshape.26, reshape.22)
# #   add.28 = f32[3]{0} add(dynamic-update-slice.23, dynamic-update-slice.27)
# #   ROOT tuple.30 = (s32[], f32[3]{0}) tuple(add.29, add.28)
# # }

def body(i, refs):
  x_ref, y_ref = refs
  y_ref[i] = x_ref[i]
def id(x):
  return for_loop(x.shape[0], body, (x, jnp.zeros_like(x)))[1]
print(jax.xla_computation(jax.vjp(id, jnp.zeros(3))[1])(jnp.zeros(3)).as_hlo_text())
