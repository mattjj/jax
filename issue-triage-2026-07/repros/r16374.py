import jax
import jax.numpy as jnp


@jax.custom_jvp
def f(x):
    y = jnp.zeros((1,) + x.shape)
    return y.at[0].set(x)


@f.defjvp
def f_jvp(primals, tangents):
    (x,) = primals
    (t_x,) = tangents
    out = jnp.broadcast_to(x, (1,) + x.shape)
    t_out = jnp.broadcast_to(t_x, (1,) + t_x.shape)
    return out, t_out

x = tx = jnp.zeros((1, 0))
out = jax.jvp(jax.jit(jax.vmap(f)), (x,), (tx,))
print("OK", jax.tree_util.tree_map(lambda a: a.shape, out))
