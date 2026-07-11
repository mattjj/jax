import jax
def f(x):
    return x * 1.0
jaxpr = jax.make_jaxpr(f)(1.0)
print("jaxpr:      ", jaxpr)
print("eqns[0]:    ", jaxpr.eqns[0])
print("invars:     ", jaxpr.jaxpr.invars, "eqn invars:", jaxpr.eqns[0].invars, "eqn outvars:", jaxpr.eqns[0].outvars)
