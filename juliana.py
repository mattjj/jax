import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, Mesh

def f(x):
  x = jnp.sin(x)
  x = jax.lax.with_sharding_constraint(x, P())
  return x


mlir.special_paralle_rules[...] = ...

with Mesh(jax.devices(), ('x',)):
  jax.jit(f).lower(3., partir=True)
  # jax.jit(f).lower(3., special_rules=...)


# Option 1: pass rules dict/table directly into lower()
# Option 2: pass some key into lower, and register rules in some module-level
# (global) dictionary, then in lowering rules check that key against that
# module-level dict

