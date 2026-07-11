import jax
jax.config.update("jax_error_checking_behavior_divide", "raise")
import jax.numpy as jnp
try:
    print(jnp.floor_divide(2, 0))
except Exception as e:
    print(type(e).__name__, ":", str(e)[:200])
