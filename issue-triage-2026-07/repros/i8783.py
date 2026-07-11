from jax.experimental.ode import odeint
from jax import jit, value_and_grad, vmap
import jax.numpy as jnp
import jax
# adapted: `from jax.config import config` removed long ago
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

T = 1.
X = -1j * jnp.array([[0., 1.], [1., 0.]], dtype=complex)
Y = -1j * jnp.array([[0., -1j], [1j, 0.]], dtype=complex)

def err_obj(a, b_vals):
    def err(b):
        def rhs(y, t):
            return (b * X + a * (t**2) * Y) @ y

        # adapted: odeint now requires float t (was complex in original repro)
        res = odeint(rhs, y0=jnp.eye(2, dtype=complex), t=jnp.array([0., T]), rtol=1e-6, atol=1e-6)
        return jnp.abs((X * res[-1]).sum())**2 / 4

    all_err = vmap(err)(b_vals)
    return all_err.sum()

b_vals = jnp.array([1., 2., 3., 4., 5.])
print(jit(value_and_grad(lambda a: err_obj(a, b_vals)))(1.))
