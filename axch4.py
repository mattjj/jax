import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

# test1: dumb log1p, when x is close to 0. sin is just a decoy.
def log1p(x):
  y = jnp.sin(x)
  return jnp.log(1. + y)

# test2: dumb log beta, when a is large and b is small
from jax.scipy.special import gammaln
def betaln(a, b):
  return gammaln(a) + gammaln(b) - gammaln(a + b)


###

import operator as op
from typing import Any, Dict, List, Callable
from jax import core
from jax._src.util import safe_map, safe_zip
from jax._src import ad_util
from jax._src import dtypes
from jax._src import source_info_util
map = safe_map
zip = safe_zip


Val = Any
Val64 = Any
Val32 = Any

def foo(jaxpr: core.ClosedJaxpr, *args):
  env: Dict[core.Var, Val64] = {}
  perturbation_env: Dict[core.Var, Val64] = {}

  def read(x: core.Atom) -> Val64:
    if isinstance(x, core.Literal):
      return x.val
    return env[x]

  def write(v: core.Var, val: Val64) -> None:
    env[v] = val

  def write_perturbation(v: core.Var, val: Val64) -> None:
    perturbation_env[v] = val

  map(write, jaxpr.jaxpr.constvars, jaxpr.consts)
  map(write, jaxpr.jaxpr.invars, args)
  for eqn in jaxpr.jaxpr.eqns:
    inputs: List[Val64] = map(read, eqn.invars)
    outputs: Union[Val64, List[Val64]] = eqn.primitive.bind(*inputs, **eqn.params)

    # TODO assumes 'params' doesn't have dtype parameters
    inputs_32: List[Val32] = map(demote_to_32, inputs)
    outputs_32: Union[Val32, List[Val32]] = eqn.primitive.bind(*inputs_32, **eqn.params)

    if eqn.primitive.multiple_results:
      outputs_32_ = map(promote_to_64, outputs_32)
      perturbations = [x - y for x, y in zip(outputs_32_, outputs)]
    else:
      outputs_32_ = promote_to_64(outputs_32)
      perturbations = outputs_32_ - outputs

    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, outputs)
      map(write_perturbation, eqn.outvars, perturbations)
    else:
      write(eqn.outvars[0], outputs)
      write_perturbation(eqn.outvars[0], perturbations)

  zero_perturbations = {v: ad_util.zeros_like_aval(v.aval) for v in perturbation_env}
  sensitivity_env = jax.grad(make_perturbation_fn(jaxpr))(zero_perturbations, *args)

  scores = {v: jnp.vdot(perturbation_env[v] / env[v], sensitivity_env[v])
            for v in perturbation_env}
  worst_offender = max((v for eqn in jaxpr.jaxpr.eqns for v in eqn.outvars),
                       key=lambda v: jnp.abs(scores[v]))
  for worst_offender in sorted(scores, key=lambda v: jnp.abs(scores[v])):
    eqn, = (eqn for eqn in jaxpr.jaxpr.eqns if worst_offender in eqn.outvars)
    src = source_info_util.summarize(eqn.source_info)
    print(f"at {src} we applied {eqn.primitive.name} with inputs:\n" +
          '\n'.join(f'  val={read(x)}' for x in eqn.invars) + '\n' +
          f"but the output(s) had value / absolute / relative error:\n" +
          '\n'.join(f'  {env[v]} / {perturbation_env[v]} / {perturbation_env[v] / env[v]}'
                    for v in eqn.outvars) + '\n' +
          f"and this resulted in an elasticity score of {scores[worst_offender]}\n"
          )

x64_to_x32 = {
    jnp.dtype('float64'): jnp.dtype('float32')
}
x32_to_x64 = {v:k for k, v in x64_to_x32.items()}

def demote_to_32(x):
  new_dtype = x64_to_x32[dtypes.dtype(x)]
  return jax.lax.convert_element_type(x, new_dtype)

def promote_to_64(x):
  new_dtype = x32_to_x64[dtypes.dtype(x)]
  return jax.lax.convert_element_type(x, new_dtype)


def make_perturbation_fn(jaxpr: core.ClosedJaxpr) -> Callable:
  def fn(perturbation_env, *args):
    env = {}

    def read(x: core.Atom) -> Val:
      return env[x] if isinstance(x, core.Var) else x.val

    def read_perturbation(v: core.Var) -> Val:
      return perturbation_env[v]

    def write(v: core.Var, val: Val) -> None:
      env[v] = val

    map(write, jaxpr.jaxpr.constvars, jaxpr.consts)
    map(write, jaxpr.jaxpr.invars, args)
    for eqn in jaxpr.jaxpr.eqns:
      inputs = map(read, eqn.invars)
      outputs = eqn.primitive.bind(*inputs, **eqn.params)
      perturbations = map(read_perturbation, eqn.outvars)
      if eqn.primitive.multiple_results:
        outputs = map(op.add, outputs, perturbations)
        map(write, eqn.outvars, outputs)
      else:
        outputs = outputs + read_perturbation(eqn.outvars[0])
        write(eqn.outvars[0], outputs)

    out, = map(read, jaxpr.jaxpr.outvars)
    return out
  return fn


# x = 1e-4
# jaxpr = jax.make_jaxpr(log1p)(x)
# foo(jaxpr, x)

def exp_gamma_log_prob(concentration, log_rate, x):
  y = jnp.exp(x + log_rate)
  log_unnormalized_prob = concentration * x - y
  # log_unnormalized_prob = (1e1 + log_unnormalized_prob_) - 1e1
  log_normalization = jax.lax.lgamma(concentration) - concentration * log_rate
  return log_unnormalized_prob - log_normalization

jaxpr = jax.make_jaxpr(exp_gamma_log_prob)(117.67729, 159.94534, -155.34862)
foo(jaxpr, 117.67729, 159.94534, -155.34862)





# TODO recurse into higher-order primitives
