# Issue 34139: jnp.sinc gradient has large errors near (but not at) zero.
# Compare jax.grad(jnp.sinc) against an accurate reference derivative.
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# Reference: sinc(x) = sin(pi x)/(pi x); d/dx = (pi x cos(pi x) - sin(pi x)) / (pi x^2)
# Near zero use Maclaurin series (in float64 / longdouble, accurate).
def ref_dsinc(x):
    x = np.asarray(x, dtype=np.longdouble)
    pix = np.pi * x
    # series: -pi^2 x/3 + pi^4 x^3/30 - pi^6 x^5/840 + pi^8 x^7/45360 - ...
    small = np.abs(pix) < 0.5
    xs = x
    series = (-np.pi**2 * xs / 3 + np.pi**4 * xs**3 / 30
              - np.pi**6 * xs**5 / 840 + np.pi**8 * xs**7 / 45360
              - np.pi**10 * xs**9 / 3991680)
    with np.errstate(all='ignore'):
        direct = (pix * np.cos(pix) - np.sin(pix)) / (np.pi * xs * xs)
    return np.where(small, series, direct)

for dtype, name in [(np.float64, "float64"), (np.float32, "float32")]:
    xs = np.logspace(-12 if dtype == np.float64 else -6, 0, 200).astype(dtype)
    g = jax.vmap(jax.grad(jnp.sinc))(jnp.asarray(xs))
    g = np.asarray(g, dtype=np.longdouble)
    ref = ref_dsinc(xs.astype(np.float64))
    rel = np.abs(g - ref) / np.abs(ref)
    i = int(np.argmax(rel))
    eps = np.finfo(dtype).eps
    print(f"{name}: max rel err = {float(rel.max()):.3e} at x = {xs[i]:.3e} "
          f"(grad={float(g[i]):.6e}, ref={float(ref[i]):.6e}); eps={eps:.2e}")
    # also value exactly at 0 for sanity
    g0 = jax.grad(jnp.sinc)(dtype(0.0))
    print(f"{name}: grad at exactly 0 = {g0} (exact: 0.0)")
