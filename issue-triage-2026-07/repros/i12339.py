import jax

def f(i):
    return jax.numpy.array([.8, .9])[i].sum()

fp = jax.grad(f, allow_int=True)
i = jax.numpy.array([0, 0, 1])
fp_i = fp(i)
print("dtype:", fp_i.dtype)
for label, thunk in [("fp_i * .3", lambda: fp_i * .3),
                     ("fp_i + fp_i", lambda: fp_i + fp_i),
                     ("i + fp_i", lambda: i + fp_i)]:
    try:
        print(label, "=", thunk())
    except Exception as e:
        print(label, "-> ERROR:", type(e).__name__, str(e)[:150])
