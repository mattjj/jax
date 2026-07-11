from jax.experimental.ode import odeint
from jax import value_and_grad, vmap, grad
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

T = 1.
X = -1j * jnp.array([[0., 1.], [1., 0.]], dtype=complex)
Y = -1j * jnp.array([[0., -1j], [1j, 0.]], dtype=complex)

def err(a, b):
    def rhs(y, t):
        return (b * X + a * (t**2) * Y) @ y
    res = odeint(rhs, y0=jnp.eye(2, dtype=complex), t=jnp.array([0., T]), rtol=1e-6, atol=1e-6)
    return jnp.abs((X * res[-1]).sum())**2 / 4

b_vals = jnp.array([1., 2., 3., 4., 5.])

print("forward scalar:", err(1., 2.))
print("forward vmap:", vmap(lambda b: err(1., b))(b_vals))
print("grad no vmap:", grad(err)(1., 2.))
print("grad + vmap:", grad(lambda a: vmap(lambda b: err(a, b))(b_vals).sum())(1.))
