import jax
import jax.numpy as jnp
import functools

mesh = jax.sharding.Mesh(jax.devices()[:2], axis_names=('x',))
spec = jax.sharding.PartitionSpec('x')
sharding = jax.sharding.NamedSharding(mesh, spec)

@functools.partial(
    jax.shard_map, mesh=mesh, in_specs=(spec, spec), out_specs=spec,
)
def f(x, y):
  return x + y

x = jax.device_put(jnp.arange(8), sharding)
f(x)  # Missing second argument
