# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare custom_root with its implementation before the HiJAX conversion."""

import collections
from functools import partial
import itertools
import operator

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
import jax.numpy as jnp
from jax._src import ad_util
from jax._src import api
from jax._src import api_util
from jax._src import core
from jax._src import custom_derivatives
from jax._src import flattree as ft
from jax._src import test_util as jtu
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow.common import _check_tree
from jax._src.lib import hlo
from jax._src.util import split_list

jax.config.parse_flags_with_absl()


# The implementation from 0896c305, with documentation and type annotations
# omitted. Keep the eager tracing and custom_jvp rule here: comparing only with
# the mathematical derivative would miss changes in staging and AD codegen.
_RootTuple = collections.namedtuple('_RootTuple', 'f, solve, l_and_s')


def _split_root_args(args, const_lengths):
  params_list = split_list(args, list(const_lengths))
  return _RootTuple(*params_list[:-1]), params_list[-1]


def _legacy_custom_root(f, initial_guess, solve, tangent_solve, has_aux=False):
  guess_flat = ft.flatten(initial_guess)
  guess_avals = guess_flat.map(core.typeof)
  f_debug = api_util.debug_info("custom_root", f, (initial_guess,), {})
  args_avals = ft.pack(((guess_avals,), {}))
  f_jaxpr, out_avals = pe.trace_to_jaxpr(f, args_avals, f_debug)
  f_jaxpr, f_consts = pe.separate_consts(f_jaxpr)
  _check_tree("f", "initial_guess", out_avals.tree, guess_avals.tree, False)

  solve_debug = api_util.debug_info("custom_root solve", solve,
                                    (f, initial_guess), {}, static_argnums=(0,))
  solve_jaxpr, solution_avals = pe.trace_to_jaxpr(
      partial(solve, f), args_avals, solve_debug)
  solve_jaxpr, solve_consts = pe.separate_consts(solve_jaxpr)
  _check_tree("solve", "initial_guess", solution_avals.tree, guess_flat.tree, has_aux)

  def linearize_and_solve(x, b):
    _, f_jvp = api.linearize(f, x)
    return tangent_solve(f_jvp, b)

  tangent_debug = api_util.debug_info("custom_root tangent_solve", tangent_solve,
                                      (initial_guess, initial_guess), {})
  tangent_avals = ft.pack(((guess_avals, guess_avals), {}))
  tangent_jaxpr, out_avals = pe.trace_to_jaxpr(
      linearize_and_solve, tangent_avals, tangent_debug)
  tangent_jaxpr, tangent_consts = pe.separate_consts(tangent_jaxpr)
  _check_tree("tangent_solve", "x", out_avals.tree, guess_flat.tree, False)

  all_consts = [f_consts, solve_consts, tangent_consts]
  const_lengths = _RootTuple(*map(len, all_consts))
  jaxprs = _RootTuple(f_jaxpr, solve_jaxpr, tangent_jaxpr)
  solution_flat = _legacy_root_bind(
      const_lengths, jaxprs, *itertools.chain.from_iterable(all_consts), *guess_flat)
  return solution_avals.update(solution_flat).unflatten()


@partial(custom_derivatives.custom_jvp, nondiff_argnums=(0, 1))
def _legacy_root_bind(const_lengths, jaxprs, *args):
  params, initial_guess = _split_root_args(args, const_lengths)
  return core.jaxpr_as_fun(jaxprs.solve)(*(params.solve + initial_guess))


@_legacy_root_bind.defjvp
def _legacy_root_jvp(const_lengths, jaxprs, primals, tangents):
  params, _ = _split_root_args(primals, const_lengths)
  sol = _legacy_root_bind(const_lengths, jaxprs, *primals)
  solution, aux = split_list(sol, [len(jaxprs.f.out_avals)])
  params_dot, _ = _split_root_args(tangents, const_lengths)
  f = core.jaxpr_as_fun(jaxprs.f)
  linearize_and_solve = partial(
      core.jaxpr_as_fun(jaxprs.l_and_s), *params.l_and_s)
  f_at_solution = lambda *params: f(*params, *solution)
  _, f_at_solution_lin = api.linearize(f_at_solution, *params.f)
  rhs = f_at_solution_lin(*params_dot.f)
  solution_dot = list(map(operator.neg, linearize_and_solve(*solution, *rhs)))
  return solution + aux, solution_dot + list(map(ad_util.zero_from_primal, aux))


def _ones(tree):
  return jax.tree.map(jnp.ones_like, tree)


def _sum(tree):
  return sum(map(jnp.sum, jax.tree.leaves(tree)))


_TRANSFORMS = (
    'primal', 'jvp', 'vjp', 'linearize', 'linear_transpose', 'grad',
    'value_and_grad', 'jacfwd', 'jacrev', 'jacfwd_jacfwd', 'jacfwd_jacrev',
    'jacrev_jacfwd', 'jacrev_jacrev', 'vmap_grad', 'grad_vmap')


def _transform(f, transform):
  match transform:
    case 'primal':
      return f
    case 'jvp':
      return lambda x: jax.jvp(f, (x,), (_ones(x),))
    case 'vjp':
      def vjp(x):
        y, pullback = jax.vjp(f, x)
        return pullback(_ones(y))
      return vjp
    case 'linearize':
      return lambda x: jax.linearize(f, x)[1](_ones(x))
    case 'linear_transpose':
      def transpose(x):
        y, pushforward = jax.linearize(f, x)
        return jax.linear_transpose(pushforward, x)(_ones(y))
      return transpose
    case 'grad' | 'value_and_grad':
      return getattr(jax, transform)(lambda x: _sum(f(x)))
    case 'vmap_grad':
      batched_grad = jax.vmap(jax.grad(lambda x: _sum(f(x))))
      return lambda x: batched_grad(jax.tree.map(lambda x: jnp.stack([x, x + 1]), x))
    case 'grad_vmap':
      return jax.grad(lambda x: _sum(jax.vmap(f)(
          jax.tree.map(lambda x: jnp.stack([x, x + 1]), x))))
    case _:
      for mode in reversed(transform.split('_')):
        f = getattr(jax, mode)(f)
      return f


def _elementwise_root(impl, has_aux):
  def root(a):
    def f(x):
      return jax.tree.map(lambda x, a: x * x - a, x, a)

    def solve(f, x):
      def step(_, x):
        return jax.tree.map(lambda x, y: x - y / (2 * x), x, f(x))
      x = lax.fori_loop(0, 5, step, x)
      return (x, jnp.array(5.)) if has_aux else x

    def tangent_solve(g, b):
      return jax.tree.map(operator.truediv, b, g(_ones(b)))

    return impl(f, _ones(a), solve, tangent_solve, has_aux=has_aux)
  return root


def _canonical_hlo(module):
  options = hlo.HloPrintOptions.canonical()
  options.print_metadata = False
  options.print_ids = False
  options.print_backend_config = True
  options.print_large_constants = True
  options.print_control_dependencies = True
  options.canonicalize_computations = True
  options.canonicalize_instruction_names = True
  # Only omit the module header (which names the Python function). Shapes,
  # constants, operand order, layouts, and backend configuration remain visible.
  return '\n'.join(module.to_string(options).splitlines()[1:])


class CustomRootHloTest(jtu.JaxTestCase):

  @parameterized.product(transform=_TRANSFORMS, shape=['scalar', 'vector', 'tree'],
                         has_aux=[False, True])
  def test_elementwise_hlo_matches_legacy(self, transform, shape, has_aux):
    a = jnp.array(2.) if shape == 'scalar' else jnp.array([2., 3.])
    if shape == 'tree':
      a = {'vector': a, 'scalar': jnp.array(4.)}
    lowered = [jax.jit(_transform(_elementwise_root(impl, has_aux), transform)).lower(a)
               for impl in (_legacy_custom_root, lax.custom_root)]
    self.assertEqual(*[_canonical_hlo(f.compiler_ir('hlo').as_hlo_module())
                       for f in lowered])
    compiled = [f.compile() for f in lowered]
    self.assertAllClose(compiled[0](a), compiled[1](a), rtol=1e-5, atol=1e-6)
    if jtu.test_device_matches(['cpu']):
      self.assertEqual(*[_canonical_hlo(f.runtime_executable().hlo_modules()[0])
                         for f in compiled])

  @parameterized.product(
      modes=list(itertools.product(('jacfwd', 'jacrev'), repeat=3)))
  def test_third_derivatives_match_legacy(self, modes):
    # Higher-order AD can regroup cotangent additions, so do not require text
    # equality here. Exercise all eight forward/reverse compositions.
    a = jnp.array([2., 3.])
    functions = [jax.jit(_transform(_elementwise_root(impl, False), '_'.join(modes)))
                 for impl in (_legacy_custom_root, lax.custom_root)]
    self.assertAllClose(functions[0](a), functions[1](a), rtol=1e-5, atol=1e-6)

  @parameterized.product(transform=_TRANSFORMS, case=['coefficients', 'jacfwd', 'jacrev'])
  def test_nonlinear_and_dense_derivatives_match_legacy(self, transform, case):
    # Dense Jacobians and parameter-dependent coefficients need not generate
    # identical HLO: symbolic-zero pruning and cotangent grouping can differ.
    def root(impl, args):
      a, p = args
      if case == 'coefficients':
        f = lambda x: p * x * x + jnp.sin(x) - a
        step = lambda x: x - f(x) / (2 * p * x + jnp.cos(x))
        tangent_solve = lambda g, b: b / g(_ones(b))
      else:
        matrix = jnp.array([[2., .2], [.3, 1.]])
        f = lambda x: matrix @ (x * x) - a
        step = lambda x: x - jnp.linalg.solve(matrix * (2 * x)[None, :], f(x))
        tangent_solve = lambda g, b: jnp.linalg.solve(getattr(jax, case)(g)(_ones(b)), b)

      def solve(f, x):
        return lax.fori_loop(0, 5, lambda _, x: step(x), x)

      return impl(f, _ones(a), solve, tangent_solve)

    args = (jnp.array([2., 3.]), jnp.array([1.1, 1.2]))
    lowered = [jax.jit(_transform(partial(root, impl), transform)).lower(args)
               for impl in (_legacy_custom_root, lax.custom_root)]
    compiled = [f.compile() for f in lowered]
    self.assertAllClose(compiled[0](args), compiled[1](args), rtol=1e-5, atol=1e-6)
    # value_and_grad can also change where the primal reduction is scheduled.
    if transform in ('primal', 'vjp', 'grad', 'jacrev'):
      self.assertEqual(*[_canonical_hlo(f.compiler_ir('hlo').as_hlo_module())
                         for f in lowered])
      if jtu.test_device_matches(['cpu']):
        self.assertEqual(*[_canonical_hlo(f.runtime_executable().hlo_modules()[0])
                           for f in compiled])


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
