import jax
import jax.numpy as jnp
from jax._src.hijax import HiPrimitive

def accum_grad_in_ref(x):
  return accum_grad_in_ref_p.bind(x)

class AccumGradInRef(HiPrimitive):
  ref_primitive = True

  def abstract_eval(self, x):
    return x, set()

  def to_lojax(self, x):
    return x

  def jvp(self, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return accum_grad_in_ref(x), accum_grad_in_ref(x_dot)

accum_grad_in_ref_p = AccumGradInRef('grad_ref')

##

def f(x):
  x = accum_grad_in_ref(x)
  return jnp.sin(x) + jnp.cos(x)

g = jax.grad(f)(1.)
print(jax.make_jaxpr(lambda: jax.grad(f)(1.))())
print(g)

print(jax.grad(lambda x: jnp.sin(x) + jnp.cos(x))(1.))
