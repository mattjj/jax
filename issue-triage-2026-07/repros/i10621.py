import time
import jax
import jax.numpy as jnp
from jax import lax

def signbit(x):
    return lax.shift_right_logical(lax.bitcast_convert_type(x, jnp.int32), 31)

def eigvalsrs_tridiagonal_bisection(
    a: jnp.ndarray, # (n,) diag
    b: jnp.ndarray, # (n - 1,) sub-diag
):
    b2 = b ** 2
    def count(x):
        def scan_f(carry, data):
            q, c = carry
            ai, b2i = data
            q = (ai - x) - (b2i * lax.reciprocal(q))
            return (q, lax.add(c, signbit(q))), None

        return lax.scan(scan_f, (1.0, jnp.int32(0)), (a, jnp.pad(b2, (1, 0))), unroll=48)[0][1]

    b_abs = lax.abs(b)
    r = jnp.pad(b_abs, (1, 0)) + jnp.pad(b_abs, (0, 1))
    emax = jnp.max(a + r)
    emin = jnp.min(a - r)
    norm = lax.max(lax.abs(emax), lax.abs(emin))
    n = a.size
    upper0 = emax + norm * 3e-7 * n
    lower0 = emin - norm * 3e-7 * n

    @jax.vmap
    def bisection(cnt):
        def step(carry, _):
            lower, upper = carry
            mid = (lower + upper) / 2
            pred = count(mid) <= cnt
            lower = lax.select(pred, mid, lower)
            upper = lax.select(pred, upper, mid)
            return (lower, upper), None

        # only works for fp32 now
        lower, upper = lax.scan(step, (lower0, upper0), None, length=24, unroll=3)[0]
        return (lower + upper) / 2

    return bisection(jnp.arange(n))

key1, key2 = jax.random.split(jax.random.PRNGKey(0))
n = 2
m = 95
a = jax.random.normal(key1, (n, m))
b2 = jax.random.gamma(key2, jnp.arange(m - 1, 0, -1), (n, m - 1))
b = jnp.sqrt(b2)
# adapted: skipped printing the (huge) jaxpr from the original repro
t0 = time.time()
out = jax.vmap(eigvalsrs_tridiagonal_bisection)(a, b)
out.block_until_ready()
print("completed in %.1f s" % (time.time() - t0))
print(out)
