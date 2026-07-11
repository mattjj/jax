# Minimal stand-in for mpi4jax's token-threading allreduce primitive
# (mpi4jax itself needs MPI; this reproduces the same AD/token structure).
import jax
import jax.numpy as jnp
from jax.interpreters import ad, mlir
from jax._src import core
try:
    from jax.extend.core import Primitive
except ImportError:
    from jax.core import Primitive

allreduce_p = Primitive('dummy_allreduce')
allreduce_p.multiple_results = True

def allreduce(x, token=None):
    if token is None:
        token = jax.lax.create_token()
    return allreduce_p.bind(x, token)

@allreduce_p.def_abstract_eval
def _abstract(x, token):
    return core.ShapedArray(x.shape, x.dtype), core.abstract_token

def _transpose(cts, x, token):
    ct_res, ct_token = cts
    # mpi4jax's transpose rule threads the (cotangent) token back through
    # another allreduce bind; ct_token is Zero(AbstractToken()).
    res, new_token = allreduce_p.bind(ct_res, ct_token)
    return res, new_token

ad.primitive_transposes[allreduce_p] = _transpose
mlir.register_lowering(allreduce_p, lambda ctx, x, tok: (x, tok))

arr = jnp.ones((3, 2))

def f(x):
    (res,) = jax.linear_transpose(lambda x: allreduce(x)[0], arr)(x)
    return res

res = jax.jit(f)(arr)
print(res)
print("OK")
