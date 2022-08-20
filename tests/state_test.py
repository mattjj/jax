# Copyright 2022 Google LLC
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
from functools import partial
import itertools as it
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import jax
from jax import core
from jax import lax
from jax import linear_util as lu
from jax.config import config
from jax.interpreters import partial_eval as pe
from jax._src import test_util as jtu
from jax._src.util import tuple_insert
import jax.numpy as jnp

from jax._src import state

config.parse_flags_with_absl()

class StatePrimitivesTest(jtu.JaxTestCase):

  def test_cant_eval_get_primitive(self):
    with self.assertRaises(ValueError):
      state.get_p.bind(jnp.ones(5))

  def test_cant_eval_swap_primitive(self):
    with self.assertRaises(ValueError):
      state.swap_p.bind(jnp.ones(5), jnp.zeros(5))

  def test_cant_eval_addupdate_primitive(self):
    with self.assertRaises(ValueError):
      state.addupdate_p.bind(jnp.ones(5), jnp.zeros(5))

  def test_get_abstract_aval_must_take_in_refs(self):
    ref_aval = core.ShapedArray((), jnp.float32)
    def f(x_ref):
      return [state.ref_get(x_ref, ())]
    with self.assertRaises(ValueError):
      pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), [ref_aval])

  @parameterized.named_parameters(
      dict(testcase_name="trivial_get", ref_shape=(1, 2),
           ref_dtype=jnp.float32,
           idx=(), out_shape=(1, 2), out_dtype=jnp.float32),
      dict(testcase_name="get_with_index", ref_shape=(1, 2),
           ref_dtype=jnp.float32,
           idx=(0,), out_shape=(2,), out_dtype=jnp.float32),
      dict(testcase_name="get_with_nonleading_index", ref_shape=(1, 2),
           ref_dtype=jnp.float32,
           idx=(slice(None), 0), out_shape=(1,), out_dtype=jnp.float32),
      dict(testcase_name="get_with_array_index", ref_shape=(1, 2, 3, 4),
           ref_dtype=jnp.float32,
           idx=(np.array([0, 1]),), out_shape=(2, 2, 3, 4),
           out_dtype=jnp.float32),
      dict(testcase_name="get_with_multiple_array_index",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(np.array([0, 1]), np.array([0, 1])),
           out_shape=(2, 2, 4), out_dtype=jnp.float32),
      dict(testcase_name="get_with_nonleading_multiple_array_index",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           idx=(slice(None), np.array([0, 1]), slice(None), np.array([0, 1])),
           out_shape=(2, 1, 2), out_dtype=jnp.float32),
  )
  def test_get_abstract_eval(self, ref_shape, ref_dtype, idx, out_shape=None,
                             out_dtype=None, should_error=False):
    ref_aval = state.ShapedArrayRef(ref_shape, ref_dtype)
    def f(x_ref):
      out = state.ref_get(x_ref, idx)
      return [out]
    if should_error:
      with self.assertRaises(Exception):
        pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), [ref_aval])
    else:
      jaxpr, out_avals, _ = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(f), [ref_aval])
      self.assertSetEqual(jaxpr.effects, {state.StateEffect})
      self.assertLen(out_avals, 1)
      out_aval, = out_avals
      self.assertIsInstance(out_aval, core.ShapedArray)
      self.assertEqual(out_aval.shape, out_shape)
      self.assertEqual(out_aval.dtype, out_dtype)

  def test_swap_abstract_eval_must_take_in_refs(self):
    ref_aval = core.ShapedArray((), jnp.float32)
    val_aval = core.ShapedArray((), jnp.float32)
    def f(x_ref, val):
      return [state.ref_swap(x_ref, (), val)]
    with self.assertRaises(ValueError):
      pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), [ref_aval, val_aval])

  @parameterized.named_parameters(
      dict(testcase_name="invalid_val_shape", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(), should_error=True),
      dict(testcase_name="invalid_val_shape_slice", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(slice(None),), should_error=True),
      dict(testcase_name="trivial_swap", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(1, 2), val_dtype=jnp.float32,
           idx=(), out_shape=(1, 2), out_dtype=jnp.float32),
      dict(testcase_name="bad_dtype", ref_shape=(1, 2),
           ref_dtype=jnp.int32, val_shape=(1, 2), val_dtype=jnp.float32,
           idx=(), should_error=True),
      dict(testcase_name="swap_with_index", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(0,), out_shape=(2,), out_dtype=jnp.float32),
      dict(testcase_name="swap_with_nonleading_index", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(1,), val_dtype=jnp.float32,
           idx=(slice(None), 0), out_shape=(1,), out_dtype=jnp.float32),
      dict(testcase_name="swap_with_nonleading_index_bad_val", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(slice(None), 0), should_error=True),
      dict(testcase_name="swap_with_array_index", ref_shape=(1, 2, 3, 4),
           ref_dtype=jnp.float32, val_shape=(2, 2, 3, 4), val_dtype=jnp.float32,
           idx=(np.array([0, 1]),), out_shape=(2, 2, 3, 4),
           out_dtype=jnp.float32),
      dict(testcase_name="swap_with_multiple_array_index",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           val_shape=(2, 2, 4), val_dtype=jnp.float32,
           idx=(np.array([0, 1]), np.array([0, 1])),
           out_shape=(2, 2, 4), out_dtype=jnp.float32),
      dict(testcase_name="swap_with_nonleading_multiple_array_index",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           val_shape=(2, 1, 2), val_dtype=jnp.float32,
           idx=(slice(None), np.array([0, 1]), slice(None), np.array([0, 1])),
           out_shape=(2, 1, 2), out_dtype=jnp.float32),
  )
  def test_swap_abstract_eval(self, ref_shape, ref_dtype,
      val_shape, val_dtype, idx, out_shape=None, out_dtype=None,
      should_error=False):
    ref_aval = state.ShapedArrayRef(ref_shape, ref_dtype)
    val_aval = core.ShapedArray(val_shape, val_dtype)
    def f(x_ref, val):
      out = state.ref_swap(x_ref, idx, val)
      return [out]
    if should_error:
      with self.assertRaises(Exception):
        pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), [ref_aval, val_aval])
    else:
      jaxpr, out_avals, _ = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(f), [ref_aval, val_aval])
      self.assertSetEqual(jaxpr.effects, {state.StateEffect})
      self.assertLen(out_avals, 1)
      out_aval, = out_avals
      self.assertIsInstance(out_aval, core.ShapedArray)
      self.assertEqual(out_aval.shape, out_shape)
      self.assertEqual(out_aval.dtype, out_dtype)

  @parameterized.named_parameters(
      dict(testcase_name="invalid_val_shape", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(), should_error=True),
      dict(testcase_name="invalid_val_shape_slice", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(slice(None),), should_error=True),
      dict(testcase_name="trivial_addupdate", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(1, 2), val_dtype=jnp.float32,
           idx=(), out_shape=(1, 2), out_dtype=jnp.float32),
      dict(testcase_name="bad_dtype", ref_shape=(1, 2),
           ref_dtype=jnp.int32, val_shape=(1, 2), val_dtype=jnp.float32,
           idx=(), should_error=True),
      dict(testcase_name="addupdate_with_index", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(0,), out_shape=(2,), out_dtype=jnp.float32),
      dict(testcase_name="addupdate_with_nonleading_index", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(1,), val_dtype=jnp.float32,
           idx=(slice(None), 0)),
      dict(testcase_name="addupdate_with_nonleading_index_bad_val", ref_shape=(1, 2),
           ref_dtype=jnp.float32, val_shape=(2,), val_dtype=jnp.float32,
           idx=(slice(None), 0), should_error=True),
      dict(testcase_name="addupdate_with_array_index", ref_shape=(1, 2, 3, 4),
           ref_dtype=jnp.float32, val_shape=(2, 2, 3, 4), val_dtype=jnp.float32,
           idx=(np.array([0, 1]),)),
      dict(testcase_name="addupdate_with_multiple_array_index",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           val_shape=(2, 2, 4), val_dtype=jnp.float32,
           idx=(np.array([0, 1]), np.array([0, 1]))),
      dict(testcase_name="addupdate_with_nonleading_multiple_array_index",
           ref_shape=(1, 3, 2, 4), ref_dtype=jnp.float32,
           val_shape=(2, 1, 2), val_dtype=jnp.float32,
           idx=(slice(None), np.array([0, 1]), slice(None), np.array([0, 1]))),
  )
  def test_addupdate_abstract_eval(self, ref_shape, ref_dtype,
      val_shape, val_dtype, idx, out_shape=None, out_dtype=None,
      should_error=False):
    ref_aval = state.ShapedArrayRef(ref_shape, ref_dtype)
    val_aval = core.ShapedArray(val_shape, val_dtype)
    def f(x_ref, val):
      state.ref_addupdate(x_ref, idx, val)
      return []
    if should_error:
      with self.assertRaises(Exception):
        pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), [ref_aval, val_aval])
    else:
      jaxpr, out_avals, _ = pe.trace_to_jaxpr_dynamic(
          lu.wrap_init(f), [ref_aval, val_aval])
      self.assertSetEqual(jaxpr.effects, {state.StateEffect})
      self.assertLen(out_avals, 0)

  def test_addupdate_abstract_eval_must_take_in_refs(self):
    ref_aval = core.ShapedArray((), jnp.float32)
    val_aval = core.ShapedArray((), jnp.float32)
    def f(x_ref, val):
      return [state.ref_addupdate(x_ref, (), val)]
    with self.assertRaises(ValueError):
      pe.trace_to_jaxpr_dynamic(lu.wrap_init(f), [ref_aval, val_aval])

  def test_can_represent_get_and_swap_in_jaxprs(self):

    def body(x):
      x[()] = jnp.int32(1)
      x[()] = jnp.int32(2)
      return (x[()],)
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((), jnp.int32)])
    self.assertLen(consts, 0)
    self.assertListEqual(out_avals, [core.ShapedArray((), jnp.int32)])
    self.assertEqual(jaxpr.eqns[0].primitive, state.swap_p)
    self.assertEqual(jaxpr.eqns[1].primitive, state.swap_p)
    self.assertEqual(jaxpr.eqns[2].primitive, state.get_p)

  def test_can_represent_addupdate_in_jaxprs(self):

    def body(x):
      state.ref_addupdate(x, (), jnp.int32(1))
      return (x[()],)
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((), jnp.int32)])
    self.assertLen(consts, 0)
    self.assertListEqual(out_avals, [core.ShapedArray((), jnp.int32)])
    self.assertEqual(jaxpr.eqns[0].primitive, state.addupdate_p)

  def test_get_custom_pretty_printing_rule(self):
    def body(x_ref):
      x = x_ref[()]
      return [x]
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((), jnp.int32)])
    self.assertIn("b:i32[] <- a[]", jaxpr.pretty_print(use_color=False))

    def body(x_ref):
      x = x_ref[:, 0]
      return [x]
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((1, 2), jnp.int32)])
    self.assertIn("b:i32[1] <- a[:,0]", jaxpr.pretty_print(use_color=False))

  def test_set_custom_pretty_printing_rule(self):
    def body(x_ref):
      x_ref[()] = jnp.int32(2)
      return []
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((), jnp.int32)])
    self.assertIn("a[] <- 2", jaxpr.pretty_print(use_color=False))

    def body(x_ref, val):
      x_ref[:, 0] = val
      return []
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((1, 2), jnp.int32),
                             core.ShapedArray((1,), jnp.int32)])
    self.assertIn("a[:,0] <- b", jaxpr.pretty_print(use_color=False))

  def test_swap_custom_pretty_printing_rule(self):
    def body(x_ref):
      x = state.ref_swap(x_ref, (), jnp.int32(2))
      return [x]
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((), jnp.int32)])
    self.assertIn("b:i32[], a[] <- a[], 2", jaxpr.pretty_print(use_color=False))

    def body(x_ref, val):
      x = state.ref_swap(x_ref, (slice(None), 0), val)
      return [x]
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((1, 2), jnp.int32),
                             core.ShapedArray((1,), jnp.int32)])
    self.assertIn("c:i32[1], a[:,0] <- a[:,0], b",
                  jaxpr.pretty_print(use_color=False))

  def test_addupdate_custom_pretty_printing_rule(self):
    def body(x_ref):
      state.ref_addupdate(x_ref, (), jnp.int32(2))
      return []
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((), jnp.int32)])

    self.assertIn("a[] += 2", jaxpr.pretty_print(use_color=False))

    def body(x_ref, val):
      state.ref_addupdate(x_ref, (slice(None), 0), val)
      return []
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.ShapedArrayRef((1, 2), jnp.int32),
                             core.ShapedArray((1,), jnp.int32)])
    self.assertIn("a[:,0] += b", jaxpr.pretty_print(use_color=False))


  def test_get_jvp(self):

    def f(r):
      x = r[()]
      return jnp.cos(x)

    def g(r, rdot):
      return jax.jvp(f, (r,), (rdot,))

    in_avals = [state.ShapedArrayRef((), jnp.dtype('float32')),
                state.ShapedArrayRef((), jnp.dtype('float32'))]
    jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(lu.wrap_init(g), in_avals)
    self.assertEqual(jaxpr.eqns[0].primitive, state.get_p)
    self.assertEqual(jaxpr.eqns[1].primitive, state.get_p)

  def test_swap_jvp(self):

    def f(a):
      x = a[()]
      a[()] = jnp.sin(x)
      return a[()]

    def g(r, rdot):
      return jax.jvp(f, (r,), (rdot,))

    in_avals = [state.ShapedArrayRef((), jnp.dtype('float32')),
                state.ShapedArrayRef((), jnp.dtype('float32'))]
    jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(lu.wrap_init(g), in_avals)
    self.assertEqual(jaxpr.eqns[0].primitive, state.get_p)
    self.assertEqual(jaxpr.eqns[1].primitive, state.get_p)
    self.assertEqual(jaxpr.eqns[2].primitive, lax.sin_p)
    self.assertEqual(jaxpr.eqns[3].primitive, lax.cos_p)
    self.assertEqual(jaxpr.eqns[4].primitive, lax.mul_p)
    self.assertEqual(jaxpr.eqns[5].primitive, state.swap_p)
    self.assertEqual(jaxpr.eqns[6].primitive, state.swap_p)

  def test_addupdate_jvp(self):

    def f(a):
      state.ref_addupdate(a, (), 1.)
      return a[()]

    def g(r, rdot):
      return jax.jvp(f, (r,), (rdot,))

    in_avals = [state.ShapedArrayRef((), jnp.dtype('float32')),
                state.ShapedArrayRef((), jnp.dtype('float32'))]
    jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(lu.wrap_init(g), in_avals)
    self.assertEqual(jaxpr.eqns[0].primitive, state.addupdate_p)
    self.assertEqual(jaxpr.eqns[1].primitive, state.addupdate_p)
    self.assertEqual(jaxpr.eqns[2].primitive, state.get_p)
    self.assertEqual(jaxpr.eqns[3].primitive, state.get_p)

  @parameterized.parameters(
      dict(ref_shape=ref_shape, ref_bdim=ref_bdim, idx_shape=idx_shape,
           indexed_dims=indexed_dims, idx_bdims=idx_bdims, out_bdim=out_bdim,
           op=op)
      for ref_shape in [(2, 3), (4, 2, 3), ()]
      for ref_bdim in range(1 + len(ref_shape))
      for idx_shape in [(), (1,), (5,), (5, 6)]
      for indexed_dims in it.product([True, False], repeat=len(ref_shape))
      for idx_bdims in it.product([None, *range(1 + len(idx_shape))],
                                  repeat=sum(indexed_dims))
      for out_bdim in range(1 + len(ref_shape) - sum(indexed_dims)
                            + len(idx_shape) * any(indexed_dims))
      for op in [
          lambda x_ref, indexer: [x_ref[indexer]],
          # TODO fix these other guys bro
          # lambda x_ref, indexer: [
          #     state.ref_swap(x_ref, indexer,
          #                    jnp.ones(x_ref.shape, x_ref.dtype)[indexer])],
          # lambda x_ref, indexer: (
          #     state.ref_addupdate(x_ref, indexer,
          #                         jnp.ones(x_ref.shape, x_ref.dtype)[indexer])
          #     or [jnp.ones(x_ref.shape, x_ref.dtype)[indexer]])
      ]
  )
  def test_get_vmap(self, ref_shape, ref_bdim, idx_shape, indexed_dims,
                    idx_bdims, out_bdim, op):
    axis_size = 7
    out_shape = tuple([d for d, b in zip(ref_shape, indexed_dims) if not b])
    if any(indexed_dims):
      out_shape = (*idx_shape, *out_shape)

    def maybe_insert(shape, idx):
      if idx is None:
        return shape
      return tuple_insert(shape, idx, axis_size)

    batched_ref_shape = maybe_insert(ref_shape, ref_bdim)
    ref_aval = state.ShapedArrayRef(ref_shape, jnp.dtype('float32'))
    bat_ref_aval = state.ShapedArrayRef(batched_ref_shape, jnp.dtype('float32'))

    idx_avals = [core.ShapedArray(idx_shape, jnp.dtype('int32'))
                 for _ in idx_bdims]
    bat_idx_avals = [
        core.ShapedArray(maybe_insert(idx_shape, idx_bdim), jnp.dtype('int32'))
        for idx_bdim in idx_bdims]

    def f(x_ref, *idxs):
      idxs_ = iter(idxs)
      indexer = tuple([next(idxs_) if b else slice(None) for b in indexed_dims])
      # return op(x_ref, indexer)
      foo_ = jnp.ones(x_ref.shape, x_ref.dtype)
      foo = foo_[None][(0, *indexer)]
      # breakpoint()
      x_ref[indexer] = foo
      return [foo]

    rng = self.rng()
    a = rng.randn(*bat_ref_aval.shape)
    his = [d for d, b in zip(ref_aval.shape, indexed_dims) if b]
    idxs = [rng.randint(low=0, high=hi, size=i.shape)
            for i, hi in zip(bat_idx_avals, his)]

    # discharge-of-vmap
    f_batched = jax.vmap(f, in_axes=(ref_bdim, *idx_bdims), out_axes=[out_bdim])
    stateful_jaxpr, _, stateful_consts = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(f_batched), [bat_ref_aval, *bat_idx_avals])
    jaxpr, consts = state.discharge_state(stateful_jaxpr, stateful_consts)
    discharge_of_vmap_ans = core.eval_jaxpr(jaxpr, consts, a, *idxs)
    # vmap-of-discharge
    stateful_jaxpr, _, stateful_consts = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(f), [ref_aval, *idx_avals])
    jaxpr_, consts_ = state.discharge_state(stateful_jaxpr, stateful_consts)
    f_batched = jax.vmap(partial(core.eval_jaxpr, jaxpr_, consts_),
                         in_axes=(ref_bdim, *idx_bdims),
                         out_axes=[out_bdim, ref_bdim])
    vmap_of_discharge_ans = f_batched(a, *idxs)

    self.assertAllClose(discharge_of_vmap_ans, vmap_of_discharge_ans,
                        check_dtypes=False)


class StateDischargeTest(jtu.JaxTestCase):

  def test_discharge_get(self):
    def f(a_ref):
      a = state.ref_get(a_ref, ())
      return [a + 1]
    in_avals = [state.ShapedArrayRef((), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that just adds 1.
    discharged_jaxpr, _ = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 2)
    self.assertEqual(discharged_jaxpr.eqns[0].primitive, lax.add_p)
    # Should be able to evaluate this jaxpr
    self.assertListEqual(core.eval_jaxpr(discharged_jaxpr, (),
                                         jnp.float32(1.)), [2., 1.])

  def test_discharge_get_with_slice(self):
    def f(a_ref):
      a = state.ref_get(a_ref, (0, 1))
      return [a + 1]
    in_avals = [state.ShapedArrayRef((4, 3, 2), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that just adds 1.
    discharged_jaxpr, () = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 2)
    self.assertIn(lax.dynamic_slice_p,
                  set(eqn.primitive for eqn in discharged_jaxpr.eqns))
    # Should be able to evaluate this jaxpr
    inval = jnp.arange(24., dtype=jnp.float32).reshape((4, 3, 2))
    outval, refval = core.eval_jaxpr(discharged_jaxpr, (), inval)
    self.assertTrue((outval == inval[0, 1] + 1).all())
    self.assertTrue((refval == inval).all())

  def test_discharge_get_with_gather(self):
    def f(a_ref):
      a = a_ref[jnp.array([0, 1])]
      return [a + 1]
    in_avals = [state.ShapedArrayRef((4, 3), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(f), in_avals)
    discharged_jaxpr, discharged_consts = state.discharge_state(
        stateful_jaxpr, consts)
    inval = jnp.arange(4 * 3, dtype=jnp.float32).reshape((4, 3))
    outval, refval = core.eval_jaxpr(discharged_jaxpr, discharged_consts, inval)
    self.assertTrue((outval == inval[jnp.array([0, 1])] + 1).all())
    self.assertTrue((refval == inval).all())

  def test_discharge_set(self):
    def f(a_ref, b):
      state.ref_set(a_ref, (), b + 1)
      return []
    in_avals = [state.ShapedArrayRef((), jnp.dtype('float32')),
                core.ShapedArray((), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that ignores the first
    # value and returns second value plus 1.
    discharged_jaxpr, _ = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 2)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertEqual(core.eval_jaxpr(discharged_jaxpr, (), jnp.float32(0.),
                                     jnp.float32(1.))[0], 2.)
    self.assertEqual(core.eval_jaxpr(discharged_jaxpr, (), jnp.float32(2.),
                                     jnp.float32(1.))[0], 2.)

  def test_discharge_set_with_slice(self):
    def f(a_ref):
      state.ref_set(a_ref, (0, 1), jnp.ones(2, dtype=jnp.dtype('float32')))
      return []
    in_avals = [state.ShapedArrayRef((4, 3, 2), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that just adds 1.
    discharged_jaxpr, () = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertIn(lax.dynamic_update_slice_p,
                  set(eqn.primitive for eqn in discharged_jaxpr.eqns))
    self.assertIn(lax.dynamic_slice_p,
                  set(eqn.primitive for eqn in discharged_jaxpr.eqns))
    # Should be able to evaluate this jaxpr
    inval = jnp.arange(24., dtype=jnp.float32).reshape((4, 3, 2))
    refval, = core.eval_jaxpr(discharged_jaxpr, (), inval)
    self.assertTrue((refval == inval.at[0, 1].set(1.)).all())

  def test_discharge_set_with_gather(self):
    def f(a_ref):
      a_ref[jnp.array([0, 1])] = jnp.ones((2, 3), 'float32')
      return []
    in_avals = [state.ShapedArrayRef((4, 3), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    discharged_jaxpr, discharged_consts = state.discharge_state(
        stateful_jaxpr, consts)
    inval = jnp.arange(4 * 3, dtype=jnp.float32).reshape((4, 3))
    refval, = core.eval_jaxpr(discharged_jaxpr, discharged_consts, inval)
    self.assertTrue((refval == inval.at[jnp.array([0, 1])].set(1.)).all())

  def test_discharge_addupdate(self):
    def f(a_ref, b):
      state.ref_addupdate(a_ref, (), b + 1)
      return []
    in_avals = [state.ShapedArrayRef((), jnp.dtype('float32')),
                core.ShapedArray((), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    # Discharging should just turn this into a jaxpr that adds the first value,
    # second value, and 1.
    discharged_jaxpr, _ = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 2)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertEqual(core.eval_jaxpr(discharged_jaxpr, (), jnp.float32(0.),
                                     jnp.float32(1.))[0], 2.)
    self.assertEqual(core.eval_jaxpr(discharged_jaxpr, (), jnp.float32(2.),
                                     jnp.float32(1.))[0], 4.)

  def test_discharge_addupdate_with_slice(self):
    def f(a_ref):
      state.ref_addupdate(a_ref, (0, 1),
                             jnp.ones(2, dtype=jnp.dtype('float32')))
      return []
    in_avals = [state.ShapedArrayRef((4, 3, 2), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    discharged_jaxpr, _ = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 1)
    self.assertIn(lax.dynamic_update_slice_p,
                  set(eqn.primitive for eqn in discharged_jaxpr.eqns))
    self.assertIn(lax.add_p,
                  set(eqn.primitive for eqn in discharged_jaxpr.eqns))
    self.assertIn(lax.dynamic_slice_p,
                  set(eqn.primitive for eqn in discharged_jaxpr.eqns))
    inval = jnp.arange(24., dtype=jnp.float32).reshape((4, 3, 2))
    refval, = core.eval_jaxpr(discharged_jaxpr, (), inval)
    self.assertTrue((refval == inval.at[0, 1].add(1.)).all())

  def test_discharge_addupdate_with_gather(self):
    def f(a_ref):
      state.ref_addupdate(a_ref, (jnp.array([0, 1]),),
                          jnp.ones((2, 3), 'float32'))
      return []
    in_avals = [state.ShapedArrayRef((4, 3), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    discharged_jaxpr, discharged_consts = state.discharge_state(
        stateful_jaxpr, consts)
    inval = jnp.arange(4 * 3, dtype=jnp.float32).reshape((4, 3))
    refval, = core.eval_jaxpr(discharged_jaxpr, discharged_consts, inval)
    self.assertTrue((refval == inval.at[jnp.array([0, 1])].add(1.)).all())

  def test_discharge_jaxpr_with_multiple_outputs(self):
    def f(a_ref):
      a = state.ref_get(a_ref, ())
      b = a + 1
      return [a, b]
    in_avals = [state.ShapedArrayRef((4,), jnp.dtype('float32'))]
    stateful_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(f),
                                                          in_avals)
    discharged_jaxpr, _ = state.discharge_state(stateful_jaxpr, consts)
    self.assertLen(discharged_jaxpr.invars, 1)
    self.assertLen(discharged_jaxpr.outvars, 3)
    inval = jnp.arange(4., dtype=jnp.float32)
    a, b, refval = core.eval_jaxpr(discharged_jaxpr, (), inval)
    self.assertTrue((a == inval).all())
    self.assertTrue((b == inval + 1).all())
    self.assertTrue((refval == inval).all())


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
