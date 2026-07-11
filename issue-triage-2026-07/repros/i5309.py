from jax import numpy as np
import jax

def f(x, p):
    return x - p

key = jax.random.PRNGKey(0)
p_test = jax.random.normal(key, (10,))

@jax.custom_jvp
def solve_gmres(p):
    return p

@solve_gmres.defjvp
def solve_gmres_jvp(primals, tangents):
    p, = primals
    dp, = tangents
    x = solve_gmres(p)
    f_x, f_p = jax.jacfwd(f, argnums=(0, 1))(x, p)
    dx, _ = jax.scipy.sparse.linalg.gmres(f_x, -f_p @ dp)
    return x, dx

print("jacfwd:", jax.jacfwd(solve_gmres)(p_test)[0, :3])  # works
print("jacrev:", jax.jacrev(solve_gmres)(p_test)[0, :3])  # fails per issue
