import jax
import jax.lax as lax
import functools as ft

@ft.partial(jax.custom_jvp, nondiff_argnums=(0,))
def f(x, y):
    print("f")
    return y

@f.defjvp
def f_jvp(x, y, tang_y):
    print("f_jvp")
    x + 1  # Crashes on this line per the issue
    (y,) = y
    (tang_y,) = tang_y
    return y, tang_y

def g(y, x):
    return lax.cond(x < y, f, lambda _x, _y: _y, x, y)

print(jax.grad(g)(1.0, 1.0))
