import jax
from jax import numpy as np
print("num devices:", jax.device_count())

for batches in [
    [np.ones((2, 2)), np.ones((14, 1))],  # works
    [np.ones((3, 2))],                    # works
    [np.ones((13, 1))],                   # works
    [np.ones((13, 1)), np.ones((3, 2))],  # pmap fails
]:
    def compute_batch(batch):
        return np.ones((7))

    @jax.jit
    def pmap_over_batches(batches):
        ret = [
            jax.pmap(
                compute_batch,
                devices=jax.devices()[:len(batch)],
            )(batch).flatten()
            for batch in batches
        ]
        return ret

    try:
        pmap_ret = pmap_over_batches(batches)
        print("OK:", [b.shape for b in batches], "->", [r.shape for r in pmap_ret])
    except Exception as e:
        print("FAIL:", [b.shape for b in batches], "->", type(e).__name__, str(e)[:300])
