import numpy as np
import jax.numpy as jnp
print("numpy:", np.floor_divide(2, 0))
print("jax  :", jnp.floor_divide(2, 0))
for a, b in [(2,0), (-2,0), (0,0), (7,0)]:
    print(f"floor_divide({a},{b}): numpy={np.floor_divide(a,b)} jax={jnp.floor_divide(a,b)}")
