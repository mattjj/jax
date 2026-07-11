# Issue 36703: pmap(vmap(fn)) with top_k[0]-in-where + separate top_k[1]
# Adapted: run on CPU with forced host device count (original used 2 GPUs).
import jax
import jax.numpy as jnp

N = 50
K = 5
N_ITEMS = 20
N_DEV = len(jax.devices())
print("devices:", N_DEV, jax.devices())
assert N_DEV >= 2

padded = ((N_ITEMS + N_DEV - 1) // N_DEV) * N_DEV
shard_indices = (jnp.arange(padded) % N_ITEMS).reshape(N_DEV, -1)

def fn(i):
    x = jnp.arange(N, dtype=jnp.float32) * (1.0 + i * 0.01)
    top_vals = jax.lax.top_k(x, K)[0]
    hard_sum = jnp.sum(top_vals)
    soft_sum = hard_sum * 0.99
    result = jnp.where(True, hard_sum, soft_sum)
    idx = jax.lax.top_k(x, K)[1]
    return result + jnp.float32(jnp.sum(idx))

out = jax.pmap(jax.vmap(fn))(shard_indices)
print("OK, output shape:", out.shape)
print(out)
