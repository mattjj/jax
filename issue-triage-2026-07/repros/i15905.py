import jax
import jax.numpy as jnp
import numpy as np

P = jax.sharding.PartitionSpec

mesh = jax.sharding.Mesh(np.reshape(np.array(jax.devices()[:4]), (2, 2)),
                         axis_names=['x', 'y'])
x = jax.device_put(jnp.ones((2, 2)))

y = jax.vmap(
  jax.shard_map(
      lambda x: x ** 2,
      mesh=mesh,
      in_specs=(P('y'),),
      out_specs=P('y')
  ),
  axis_name='x',
  spmd_axis_name='x',
)(x)
print(y)
print("OK")
