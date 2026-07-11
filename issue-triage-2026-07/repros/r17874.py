# Adapted: chex.set_n_cpu_devices(10) -> XLA_FLAGS host platform device count 8;
# jax.experimental.shard_map -> jax.shard_map (keyword args); sizes 10 -> 8.
import functools
import jax
import jax.numpy as jnp
from jax import shard_map as shmap

P = jax.sharding.PartitionSpec
mesh = jax.sharding.Mesh(jax.devices(), axis_names=['x'])
assert len(jax.devices()) == 8, jax.devices()


@functools.partial(jax.vmap, spmd_axis_name='x')
def shmap_explicit_arg(x):
    return shmap(lambda x: x, mesh=mesh, in_specs=(P(),), out_specs=P())(x)

shape = shmap_explicit_arg(jnp.zeros((8,))).shape
assert shape == (8,), f"explicit-arg shape: {shape}"
print("explicit arg OK:", shape)


@functools.partial(jax.vmap, spmd_axis_name='x')
def shmap_closed_arg(x):
    return shmap(lambda: x, mesh=mesh, in_specs=(), out_specs=P())()

shape = shmap_closed_arg(jnp.zeros((8,))).shape
assert shape == (8,), f"closed-over-arg shape: {shape}"
print("closed-over arg OK:", shape)
