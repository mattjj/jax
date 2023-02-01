# Copyright 2023 The JAX Authors.
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
import os
import unittest

from absl.testing import absltest
import numpy as np

import jax
from jax import lax
from jax.config import config
from jax.experimental.maps import Mesh
from jax.experimental.pjit import PartitionSpec as P
from jax.interpreters import pxla
from jax._src import sharding
from jax._src import ad_checkpoint
from jax._src import debugging
from jax._src import dispatch
from jax._src import test_util as jtu
from jax._src.lib import xla_bridge
import jax.numpy as jnp

from shard_map import shard_map  # TODO(mattjj): move me

config.parse_flags_with_absl()

# Helper for some tests.
def create_inputs(a_sharding, b_sharding):
  x, y, z = 2, 2, 2  # pylint: disable=invalid-name
  devices = np.array(jax.devices()[:x * y * z]).reshape((x, y, z))
  mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
  b, e, f = 8, 8, 8  # pylint: disable=invalid-name
  m1 = jax.device_put(
      jnp.arange(b * e).reshape((b, e)),
      jax.sharding.NamedSharding(mesh, a_sharding))
  m2 = jax.device_put(
      jnp.arange(e * f).reshape((e, f)),
      jax.sharding.NamedSharding(mesh, b_sharding))
  return mesh, m1, m2

# Run all tests with 8 CPU devices.
def setUpModule():
  global prev_xla_flags
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (flags_str +
                               " --xla_force_host_platform_device_count=8")
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


class ShardMapTest(jtu.JaxTestCase):

  def test_identity(self):
    mesh, a, _ = create_inputs(P('z', ('x', 'y')), P(None, None))
    assert a.device_buffers[0].shape == (4, 2)

    def identity(x):
      return x

    @jax.jit
    def fwd(a):
      c = shard_map(
          lambda x: x,
          mesh,
          in_specs=(P('z', ('x', 'y')),),
          out_specs=P('z', ('x', 'y')))(a)
      return c

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (4, 2))

  def test_all_gather(self):
    mesh, a, _ = create_inputs(P('z', ('x', 'y')), P(None, None))
    assert a.device_buffers[0].shape == (4, 2)

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('z', ('x', 'y')),), out_specs=P(None, ('x', 'y')))
    def fwd(a):
      return lax.all_gather(a, 'z', axis=0, tiled=True)

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (8, 2))

  def test_matmul_partial(self):
    raise unittest.SkipTest("invalid replication asserted by out_spec?")

    mesh, a, b = create_inputs(P('z', 'y'), P('y', None))
    assert a.device_buffers[0].shape == (4, 4)

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('z', 'y'), P('y', None)), out_specs=P('z', None))
    def fwd(a):
      c = jnp.matmul(a, b)  # [B.z, F] {y.unreduced}
      return c

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (4, 8))

  def test_matmul_reduce_scatter(self):
    mesh, a, b = create_inputs(P('z', 'y'), P('y', None))
    assert a.device_buffers[0].shape == (4, 4)

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('z', 'y'), P('y', None)),
             out_specs=P(('z', 'y'), None))
    def fwd(a, b):
      c = jnp.matmul(a, b)  # [B.z, F] {y.unreduced}
      return lax.psum_scatter(c, 'y', scatter_dimension=0, tiled=True)

    c = fwd(a, b)
    self.assertEqual(c.device_buffers[0].shape, (2, 8))

  def test_collective_permute(self):
    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=('x'))
    a = jax.device_put(
        jnp.arange(8 * 8).reshape((8, 8)),
        jax.sharding.NamedSharding(mesh, P('x', None)))

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=(P('x', None),),
             out_specs=P('x', None))
    def fwd(a):
      axis_size = lax.psum(1, 'x')
      perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
      return lax.ppermute(a, 'x', perm=perm)

    c = fwd(a)
    self.assertAllClose(c[1, :], a[0, :])

  def test_all_to_all(self):
    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=('x'))
    a = jax.device_put(
        jnp.arange(8 * 8).reshape((8, 8)),
        jax.sharding.NamedSharding(mesh, P('x', None)))

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x', None),), out_specs=P(None, 'x'))
    def fwd(a):
      return lax.all_to_all(a, 'x', split_axis=1, concat_axis=1, tiled=True)

    c = fwd(a)
    assert (c == jnp.reshape(a.T, (1, 64))).all()

  def test_eager_repr(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    s = None

    @partial(shard_map, mesh=mesh, in_specs=P('x', 'y'), out_specs=P('x', 'y'))
    def f(x):
      nonlocal s
      s = str(x)
      return x
    _ = f(np.arange(8 * 8.).reshape(8, 8))

    self.assertIsInstance(s, str)
    self.assertIn('at mesh coordinates', s)

  def test_jvp_basic(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jnp.sin(jnp.cos(x)), mesh,
                  in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))
    args = np.arange(4 * 4.).reshape(4, 4),
    jtu.check_grads(g, args, 2, ['fwd'])
    jtu.check_grads(jax.jit(g), args, 2, ['fwd'])

  def test_linearize_basic(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jnp.sin(jnp.cos(x)), mesh,
                  in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))
    x = np.arange(4 * 4.).reshape(4, 4)

    y, y_dot = jax.jvp(g, [x], [x])

    y_, g_lin = jax.linearize(g, x)
    y_dot_ = g_lin(x)

    self.assertAllClose(y, y_, check_dtypes=False)
    self.assertAllClose(y_dot, y_dot_, check_dtypes=False)

  def test_linearize_basic_repres(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jax.lax.sin(jax.lax.cos(x)), mesh,
                  in_specs=(P('x',),), out_specs=P('x',))
    x = np.arange(4.)

    y, y_dot = jax.jvp(g, [x], [x])

    y_, g_lin = jax.linearize(g, x)
    y_dot_ = g_lin(x)

    self.assertAllClose(y, y_, check_dtypes=False)
    self.assertAllClose(y_dot, y_dot_, check_dtypes=False)

  def test_linearize_basic_repres_jit(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jnp.sin(jnp.cos(x)), mesh,
                  in_specs=(P('x',),), out_specs=P('x',))
    x = np.arange(4.)

    y, y_dot = jax.jvp(g, [x], [x])

    y_, g_lin = jax.linearize(g, x)
    y_dot_ = g_lin(x)

    self.assertAllClose(y, y_, check_dtypes=False)
    self.assertAllClose(y_dot, y_dot_, check_dtypes=False)

  def test_replication_checker_eager(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = np.arange(8 * 8.).reshape(8, 8)

    def f(x):
      return 2 * x
    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)

    with self.assertRaisesRegex(ValueError, 'statically inferred'):
      g(x)

    def f(x):
      return jax.lax.psum(x, 'x')
    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)
    _ = g(x)  # doesn't crash

  def test_replication_checker_jit(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = np.arange(8 * 8.).reshape(8, 8)

    def f(x):
      return 2 * x
    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)

    with self.assertRaisesRegex(ValueError, 'statically inferred'):
      jax.jit(g)(x)

    def f(x):
      return jax.lax.psum(x, 'x')
    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)
    _ = jax.jit(g)(x)  # doesn't crash

  def test_process_env_traces(self):
    mesh = Mesh(np.array(jax.devices()[:4]), ('x',))
    x = np.arange(8.)

    def g(x):
      y = (3. * x).sum()
      z = shard_map(lambda x: 2 * x * y, mesh,
                    in_specs=(P('x'),), out_specs=P('x'))(np.arange(8.))
      return z

    jtu.check_grads(g, (x,), modes=['fwd'], order=2)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
