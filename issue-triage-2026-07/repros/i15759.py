import numpy as np
import jax.numpy as jnp
from jax import vmap, jit, grad
from jax.lax import switch
from jax.experimental.ode import odeint

def A0(t):
    return 2.

def A1(a, t):
    return a**2

y0 = np.random.rand(2)
T = np.pi * 1.232

def test_func(a):
    eval_list = [A0, lambda t: A1(a, t)]

    def single_eval(idx, t):
        return switch(idx, eval_list, t)
    multiple_eval = vmap(single_eval, in_axes=(0, None))
    idx_list = jnp.array([0, 1])
    rhs = lambda y, t: multiple_eval(idx_list, t) * y

    out = odeint(
        rhs,
        y0=y0,
        t=jnp.array([0, T], dtype=float),
        atol=1e-13,
        rtol=1e-13
    )
    return out

print(jit(grad(lambda a: test_func(a)[-1][1].real))(1.))
print("OK")
