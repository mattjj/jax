# Minimal reconstruction of #5552: grad of lax.custom_root with cg in tangent_solve
# (original repro was only in a Colab; this follows the issue's described pattern)
import jax
import jax.numpy as jnp
from jax import lax, jvp
import jax.scipy.sparse.linalg as jsla

jax.config.update("jax_enable_x64", True)

N = 20
key = jax.random.PRNGKey(0)
M = jax.random.normal(key, (N * 2, N * 2))
A = M @ M.T + jnp.eye(N * 2) * (N * 2)  # SPD

def solve(x):  # x shape (N, 2)
    def f(y):
        return (A @ y.reshape(-1)).reshape(N, 2) - x

    def forward_solve(f, y0):
        return jnp.linalg.solve(A, x.reshape(-1)).reshape(N, 2)

    def tangent_solve_cg(g, y):
        j = lambda v: jvp(g, (y,), (v,))[1]
        x_sol, _ = jsla.cg(j, y)
        return x_sol

    return lax.custom_root(f, jnp.zeros_like(x), forward_solve, tangent_solve_cg)

x = jax.random.normal(jax.random.PRNGKey(1), (N, 2))
val = solve(x)
print("forward ok, residual:", jnp.abs((A @ val.reshape(-1)).reshape(N, 2) - x).max())
g = jax.grad(lambda x: solve(x).sum())(x)
print("grad ok:", g.shape, jnp.isfinite(g).all())
