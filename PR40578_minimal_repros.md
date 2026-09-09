# Minimal repros: jax-ml/jax#40578's `_balanced_cmp` vs. the follow-up on this branch

Four toy cases that fail (or change behavior) with the `_balanced_cmp` merged in
https://github.com/jax-ml/jax/pull/40578 and pass with the version in this
branch's `jax/_src/lax/lax.py`. In every case the follow-up's output is
identical to the pre-PR rule's. The outputs below were produced by running the
script at the bottom with each rule installed (jax 0.10.2 venv, rule
monkeypatched into `max_p` / `min_p`; see `PR40578_review.md` for why).

## 1. Forward mode with Python-scalar primals

`jax.vjp` canonicalizes its inputs before tracing; `jax.jvp` and `jax.jacfwd` do
not, so `JVPTracer.primal` is a raw Python `float` and the PR's `x.dtype`
raises. `jnp.maximum` is unaffected because dtype promotion re-binds the tracer
before it reaches `max_p`.

```python
jax.jvp(lax.max, (1.0, 2.0), (1.0, 1.0))
# PR:        AttributeError: 'float' object has no attribute 'dtype'
# follow-up: (Array(2., dtype=float32, weak_type=True), Array(1., dtype=float32, weak_type=True))

jax.jacfwd(lambda y: lax.max(jnp.array([1., 2., 3.]), y))(2.0)
# PR:        AttributeError: 'float' object has no attribute 'dtype'
# follow-up: [1.  0.5 0. ]
```

## 2. Complex operands

`lax.max` / `lax.min` order complex values lexicographically on `(real, imag)`
and differentiated before the PR (the old rule only used `eq`). The PR's `gt`
rejects complex dtypes.

```python
z = jnp.array([1+1j, 2+0j, 2+1j], dtype=jnp.complex64)   # elementwise: z <, ==, > w
w = jnp.array([2+0j, 2+0j, 1+5j], dtype=jnp.complex64)
jax.grad(lambda z: jnp.sum(jnp.real(lax.max(z, w))))(z)
# PR:        TypeError: gt does not accept dtype complex64 at position 0.
# follow-up: [0. +0.j 0.5+0.j 1. +0.j]
```

## 3. Weak types

The PR's mask is always strongly typed, so gradients and tangents of weakly
typed inputs come back strongly typed. Every other lax JVP rule (abs, clamp,
sin, mul, where, relu) preserves weak types, and so did the pre-PR max/min
rule. The second line shows the user-visible consequence through type
promotion.

```python
jax.grad(lambda x: lax.max(x, 0.))(1.).aval.weak_type
# PR:        False
# follow-up: True

(jax.grad(lambda x: jnp.maximum(x, 0.))(1.) * jnp.ones(2, jnp.bfloat16)).dtype
# PR:        float32
# follow-up: bfloat16
```

## 4. Eager reverse mode under an explicit mesh

One device is enough. Under `jax.grad`, the JVP rule runs inside a
`JaxprTrace`; the PR's float fill `0.5` goes through `stage` and becomes a
known tracer, and `lax.full` then drops the concrete sharding for a tracer fill
value, so `half` is replicated while `ones` / `zeros` (integer fills, which are
evaluated eagerly) are sharded. Under `jax.jit` both rules work.

```python
mesh = jax.make_mesh((1,), ("x",), axis_types=(AxisType.Explicit,))
x = jax.device_put(jnp.arange(4.), NamedSharding(mesh, P("x")))
y = jax.device_put(jnp.full((4,), 2.), NamedSharding(mesh, P()))
with jax.set_mesh(mesh):
  jax.grad(lambda a, b: jnp.sum(lax.max(a, b)), argnums=(0, 1))(x, y)
# PR:        ShardingTypeError: select cases must have the same shardings, got [... P('x') ..., ... P(None) ...]
# follow-up: (Array([0. , 0. , 0.5, 1. ], dtype=float32), Array([1. , 1. , 0.5, 0. ], dtype=float32))
```

## Corresponding tests

Each case has a test in `tests/lax_autodiff_test.py` on this branch,
parameterized over `lax.max` and `lax.min` where applicable:

| case | test |
|---|---|
| 1 | `test_max_min_jvp_python_scalar_primals` |
| 2 | `test_max_min_complex_grad` |
| 3 | `test_max_min_grad_weak_type` |
| 4 | `test_max_min_grad_eager_explicit_sharding` |

Running `pytest tests/lax_autodiff_test.py -k test_max_min` with the PR's rule
installed fails exactly these seven tests (four cases, three of them
parameterized over two ops) and passes the PR's own two NaN tests. With the
follow-up rule or the pre-PR rule, all nine pass.

## Script

Public API only. Run it once on the PR's commit and once on this branch to
reproduce the two columns above.

```python
import jax, jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P, AxisType

def run(label, fn):
  try:
    print(f"{label}: {fn()}")
  except Exception as e:
    print(f"{label}: {type(e).__name__}: {str(e).splitlines()[0][:90]}")

# 1. forward mode with Python-scalar primals
run("1a jvp(lax.max, (1.0, 2.0), (1.0, 1.0))",
    lambda: jax.jvp(lax.max, (1.0, 2.0), (1.0, 1.0)))
run("1b jacfwd(lambda y: lax.max(jnp.array([1., 2., 3.]), y))(2.0)",
    lambda: jax.jacfwd(lambda y: lax.max(jnp.array([1., 2., 3.]), y))(2.0))

# 2. complex operands
z = jnp.array([1+1j, 2+0j, 2+1j], dtype=jnp.complex64)
w = jnp.array([2+0j, 2+0j, 1+5j], dtype=jnp.complex64)
run("2  grad(lambda z: sum(real(lax.max(z, w))))(z)",
    lambda: jax.grad(lambda z: jnp.sum(jnp.real(lax.max(z, w))))(z))

# 3. weak types
run("3a grad(lambda x: lax.max(x, 0.))(1.) weak_type",
    lambda: jax.grad(lambda x: lax.max(x, 0.))(1.).aval.weak_type)
run("3b grad(lambda x: jnp.maximum(x, 0.))(1.) * ones(bf16) dtype",
    lambda: (jax.grad(lambda x: jnp.maximum(x, 0.))(1.)
             * jnp.ones(2, jnp.bfloat16)).dtype)

# 4. eager reverse mode under an explicit mesh
mesh = jax.make_mesh((1,), ("x",), axis_types=(AxisType.Explicit,))
x = jax.device_put(jnp.arange(4.), NamedSharding(mesh, P("x")))
y = jax.device_put(jnp.full((4,), 2.), NamedSharding(mesh, P()))
def eager_explicit_grad():
  with jax.set_mesh(mesh):
    return jax.grad(lambda a, b: jnp.sum(lax.max(a, b)), argnums=(0, 1))(x, y)
run("4  eager grad under explicit mesh", eager_explicit_grad)
```

Output with the PR's rule:

```
1a jvp(lax.max, (1.0, 2.0), (1.0, 1.0)): AttributeError: 'float' object has no attribute 'dtype'
1b jacfwd(lambda y: lax.max(jnp.array([1., 2., 3.]), y))(2.0): AttributeError: 'float' object has no attribute 'dtype'
2  grad(lambda z: sum(real(lax.max(z, w))))(z): TypeError: gt does not accept dtype complex64 at position 0. Accepted dtypes at position 0 are subtyp
3a grad(lambda x: lax.max(x, 0.))(1.) weak_type: False
3b grad(lambda x: jnp.maximum(x, 0.))(1.) * ones(bf16) dtype: float32
4  eager grad under explicit mesh: ShardingTypeError: select cases must have the same shardings, got [NamedSharding(mesh=AbstractMesh('x': 1, ax
```

Output with the follow-up rule (and with the pre-PR rule):

```
1a jvp(lax.max, (1.0, 2.0), (1.0, 1.0)): (Array(2., dtype=float32, weak_type=True), Array(1., dtype=float32, weak_type=True))
1b jacfwd(lambda y: lax.max(jnp.array([1., 2., 3.]), y))(2.0): [1.  0.5 0. ]
2  grad(lambda z: sum(real(lax.max(z, w))))(z): [0. +0.j 0.5+0.j 1. +0.j]
3a grad(lambda x: lax.max(x, 0.))(1.) weak_type: True
3b grad(lambda x: jnp.maximum(x, 0.))(1.) * ones(bf16) dtype: bfloat16
4  eager grad under explicit mesh: (Array([0. , 0. , 0.5, 1. ], dtype=float32), Array([1. , 1. , 0.5, 0. ], dtype=float32))
```
