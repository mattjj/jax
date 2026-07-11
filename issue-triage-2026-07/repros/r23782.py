import sys
import jax
from jax import lax
from jax.experimental.attrs import jax_getattr


class C:
    def __init__(self):
        self.vals = dict(x=0)


state = C()


def f(y):
    v = jax_getattr(state, "vals")
    return v["x"] + y


def f_iter(i, v):
    return f(v)


def do_loop():
    return lax.fori_loop(0, 5, f_iter, 0)


use_jit = "--jit" in sys.argv
if use_jit:
    do_loop = jax.jit(do_loop)

state.vals = dict(x=1, z=2)  # --bug case
print(f"running with {state.vals=}, jit={use_jit}")
print(f"{do_loop()=}")
