import jax
from functools import partial

def f(N):
  return N

jit_f = jax.jit(f, static_argnums=0)

@partial(jax.jit, static_argnums=0)
def g1(N):
  fN = f(N)
  return float(fN)

print("g1(1) =", g1(1))

@partial(jax.jit, static_argnums=0)
def g2(N):
  fN = jit_f(N)
  return float(fN)

try:
  print("g2(1) =", g2(1))
except Exception as e:
  print("g2 FAIL:", type(e).__name__, str(e)[:250])
