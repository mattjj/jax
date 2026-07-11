import jax

@jax.custom_jvp
def f(x, y):
    return x + y

@f.defjvp
def f_jvp(primals, tangents):
    _, ty = tangents
    print("tangent for y:", ty, "dtype:", getattr(ty, 'dtype', None))
    assert ty.dtype != jax.numpy.int32, "BUG: integer tangent passed for integer arg"
    return f(*primals), tangents[0]

print(jax.jvp(lambda x: f(x, 1), (1.,), (1.,)))
print("OK")
