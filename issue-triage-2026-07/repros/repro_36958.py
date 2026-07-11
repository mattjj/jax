# Issue 36958: scan+vmap wrong results (reported GPU-only; issue says not present on CPU).
import jax
import jax.numpy as jnp

def forward(weight, bias, x):
    return weight @ x + bias

def rollout_bug(weight, bias, key, points: int, steps: int):
    dt = 1 / steps
    def rollout_step(x, _):
        vf = jax.vmap(forward, (None, None, 0))(weight, bias, x)
        return x + dt * vf, x
    res, _ = jax.lax.scan(rollout_step, jax.random.normal(key, (points, 2)), length=steps)
    return res

def rollout_ok(weight, bias, key, points: int, steps: int):
    dt = 1 / steps
    def rollout_step(x, _):
        vf = jax.vmap(forward, (None, None, 0))(weight, bias, x)
        return x + dt * vf, (x, vf)
    res, _ = jax.lax.scan(rollout_step, jax.random.normal(key, (points, 2)), length=steps)
    return res

def rollout_manual(weight, bias, key, points: int, steps: int):
    dt = 1 / steps
    x = jax.random.normal(key, (points, 2))
    for _ in range(steps):
        x = x + dt * jax.vmap(forward, (None, None, 0))(weight, bias, x)
    return x

print("backend:", jax.default_backend())
wkey, bkey = jax.random.split(jax.random.PRNGKey(0))
lim = 2**-0.5
weight = jax.random.uniform(wkey, (2, 2), minval=-lim, maxval=lim)
bias = jax.random.uniform(bkey, (2,), minval=-lim, maxval=lim)

res_bug = rollout_bug(weight, bias, jax.random.PRNGKey(0), 64, 10)
res_ok = rollout_ok(weight, bias, jax.random.PRNGKey(0), 64, 10)
res_manual = rollout_manual(weight, bias, jax.random.PRNGKey(0), 64, 10)

print("max |bug - manual|:", float(jnp.max(jnp.abs(res_bug - res_manual))))
print("max |ok  - manual|:", float(jnp.max(jnp.abs(res_ok - res_manual))))
