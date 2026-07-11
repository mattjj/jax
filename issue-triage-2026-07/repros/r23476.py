# Adapted: TPU 2x4 mesh -> 8 CPU devices via XLA_FLAGS;
# jax.experimental.shard_map -> jax.shard_map; dropped check_rep kwarg (renamed at head).
import jax
from jax import numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax import shard_map

device_mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape([2, 4]), ('x', 'y'))

x_sharding = jax.sharding.NamedSharding(mesh=device_mesh, spec=P('x'))
xy_sharding = jax.sharding.NamedSharding(mesh=device_mesh, spec=P('x', 'y'))

def foo_with_float_arg_no_cond(to_add_slice, global_state):
    def foo_capturing_something(state_slice):
        return to_add_slice + state_slice

    vmap_capture = jax.vmap(foo_capturing_something)
    shmap_vmap_capture = shard_map(vmap_capture, mesh=device_mesh,
                                   in_specs=P('y'), out_specs=P('y'))
    result = shmap_vmap_capture(global_state)
    return result

global_state = jax.device_put(jnp.ones(shape=[2, 4]), xy_sharding)
float_vector = jax.device_put(jnp.array([0.0, 1.0]), x_sharding)

r1 = jax.vmap(foo_with_float_arg_no_cond)(float_vector, global_state)
print("no spmd_axis_name OK:", r1.shape)
r2 = jax.vmap(foo_with_float_arg_no_cond, spmd_axis_name='x')(float_vector, global_state)
print("spmd_axis_name OK:", r2.shape)
print(r2)
