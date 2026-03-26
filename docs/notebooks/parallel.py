# ---
# Copyright 2021 The JAX Authors.
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
#
# jupyter:
#   jupytext:
#     formats: ipynb,md:myst,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# [![Open in
# Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax/blob/main/docs/notebooks/parallel.ipynb)

# # Distributed arrays and automatic parallelization
#
# <!--* freshness: { reviewed: '2025-12-02' } *-->
#
# JAX has three styles of multi-device distributed parallelism, which can be
# mixed and composed. They differ in how much the compiler automatically decides
# versus how much is controlled explicitly in the program:
#
#  * **Compiler-based automatic sharding** is where you program as if using a single
#  "global view" machine, and the compiler chooses how to shard data (with some
#  user-provided constraints via `with_sharding_constraint`) and how to
#  partition computation into per-device programs with collectives.
#  * **Explicit sharding and automatic partitioning** is where you still have a
#  global view but data shardings are explicit in JAX types, inspectable using
#  `jax.typeof`. The compiler still partitions the computation.
#  * **Manual per-device programming** is where you have a per-device view of
#  data and computation, and write explicit communication collectives like
#  `jax.lax.psum`.

# | Mode | View? | Explicit sharding? | Explicit Collectives? |
# |---|---|---|---|
# | Auto | Global | ❌ | ❌ |
# | Explicit | Global | ✅ | ❌ |
# | Manual | Per-device | ✅ | ✅ |

# Before getting into details, here's a quick example using explicit mode.
# First, we create a `jax.Array` sharded across multiple devices:

from __future__ import annotations
import enum

import jax
import jax.numpy as jnp
jax.config.update('jax_num_cpu_devices', 8)

# +
jax.set_mesh(jax.make_mesh((4, 2), ('X', 'Y')))  # explicit mode by default

x = jnp.arange(8 * 2.).reshape(8, 2)
x = jax.device_put(x, jax.P('X', 'Y'))
print(jax.typeof(x))  # f32[8@X, 2@Y]

# +
jax.debug.visualize_array_sharding(x)

# Next, we'll apply a computation to it and observe that the result values are
# stored across multiple devices too:

y = jnp.sin(x).T
print(jax.typeof(y))  # f32[8@Y, 2@X]

# The `jnp.sin` and transpose computations were automatically parallelized
# across the devices on which the input values (and output values) are stored.

# To understand these modes and how to switch among them, we first need to
# understand meshes.

# ## A `Mesh` is a grid of devices with named axes

# To describe how data and computation are distributed across devices, we first
# organize our devices into a multi-dimensional grid called a `Mesh`.
# Because communication happens along mesh axes, the mesh shape and device order
# can determine communication performance. The mesh should reflect the
# physical connection topology among the devices.

# We distinguish between _concrete_ and _abstract_ meshes. An abstract mesh
# comprises only a shape, axis names, and axis types reflecting the mode of each
# axis:

class AbstractMesh:
  axis_sizes: tuple[int, ...]
  axis_names: tuple[str, ...]
  axis_types: tuple[AxisType, ...]

class AxisType(enum.Enum):
  Auto = enum.auto()
  Explicit = enum.auto()
  Manual = enum.auto()

# A concrete mesh additionally includes physical device objects with e.g.
# precise coordinates:

import numpy as np

class Mesh:
  devices: np.ndarray[jax.Device]
  axis_names: tuple[str, ...]
  axis_types: tuple[AxisType, ...]

  @property
  def axis_sizes(self) -> tuple[int, ...]:
    return self.devices.shape

# At the top level of a program (i.e. not under a `jit`) we can create a
# concrete `Mesh` directly [using
# the class constructor](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Mesh),
# which lets us specify the exact device order, or using the `jax.make_mesh`
# helper, which automatically chooses a device order by taking the underlying
# hardware topology into account:

mesh = jax.make_mesh((4, 2), ('X', 'Y'))
print(mesh)

# By default, all mesh axis types are `AxisType.Explicit`.

# To avoid threading `mesh` throughout your program, use `jax.set_mesh` to set
# a concrete mesh globally:

jax.set_mesh(mesh)

# You can also use `with jax.set_mesh(mesh): ...` as a context manager. At the
# top level only, the concrete mesh can be queried using `jax.get_mesh() ->
# jax.sharding.Mesh`.

# Under a jit, only the abstract mesh can be queried and changed. Use
# `jax.sharding.get_abstract_mesh() -> jax.sharding.AbstractMesh` to query the
# current abstract mesh, and use `with jax.sharding.use_abstract_mesh(m:
# AbstractMesh): ...` to change the abstract mesh within a context. The axis
# sizes, axis names, and axis types can be changed, but the total size of the
# mesh (i.e. the product of the axis sizes) must not change.

# We haven't explained shardings yet, but here's a toy example of changing
# abstract meshes inside a `jax.jit`:

@jax.jit
def f(x):
  abstract_mesh = jax.sharding.AbstractMesh((8,), ('A',), (jax.sharding.AxisType.Explicit,))
  with jax.sharding.use_abstract_mesh(abstract_mesh):
    y = jax.reshard(x, jax.P('A', None))
    return y * 2

z = f(x)
print(jax.typeof(z))  # f32[8@A, 2]

# ## A `Sharding` describes how array values are laid out over a `Mesh`

# A `jax.sharding.Sharding` describes distributed memory layout. That is, it
# describes how an array's entries are stored in the physical memories of
# different devices, i.e. how it's _sharded_ over devices.

# At the top level, every `jax.Array` has an associated `Sharding`, which
# consists of a concrete `Mesh` along with a `jax.sharding.PartitionSpec`
# (aliased to `jax.P`):

print(x.sharding)
jax.debug.visualize_array_sharding(x)

# Here, `PartitionSpec('X', 'Y')` expresses that the first and second axes of
# the array `x` are sharded over the mesh axes 'X' and 'Y', respectively.
# We can see how that translates to physical storage using `addressable_shards`:

for s in x.addressable_shards:
  print(s.device)
  print(s.data)
  print()
import sys; sys.exit(0)

# We can use `jax.device_put` (or `jax.reshard`) to produce a new array that is
# sharded over the same mesh of devices but with a different layout specified by
# a `jax.P`.
# (`jax.device_put` is a runtime-level API with more features than
# `jax.reshard`.)
# Since we have a mesh in context, via the `jax.set_mesh` above, we can pass
# `jax.P` instances directly to `jax.device_put`:

y = jax.device_put(x, jax.P('Y', 'X'))
print(y.sharding)
jax.debug.visualize_array_sharding(y)

# +
y = jax.device_put(x, jax.P('X', None))
print(y.sharding)
jax.debug.visualize_array_sharding(y)

# Here, because the mesh axis name 'Y' is not mentioned in `jax.P('X', None)`,
# the array is replicated over the mesh axis 'Y'. (As a shorthand, trailing
# `None` placeholders can be omitted, so that P('X', None) here means the same
# thing as P('X'). But it doesn’t hurt to be explicit!)

# By using tuples of axis names inside a `PartitionSpec`, we can shard one array
# axis over multiple mesh axes:

y = jax.device_put(x, jax.P(('X', 'Y')))
print(y.sharding)
jax.debug.visualize_array_sharding(y)

# So an array's data can be replicated over a mesh axis, or one of its array
# axes can be sharded over that mesh axis, but there's another possibility too:
# an array can be _unreduced_ over a mesh axis:

y = jax.device_put(x, jax.P('X', None, unreduced={'Y'}))
print(y.sharding)

# Note that because every array has its own `Sharding` instance, and every
# `Sharding` instance has its own `Mesh` instance, arrays in scope can be
# associated with different meshes. To illustrate, we can use `jax.device_put`
# with a full `jax.NamedSharding` instance argument rather than using the
# in-context mesh:

mesh2 = jax.make_mesh((8,), ('A',))
z = jax.device_put(x, jax.NamedSharding(mesh2, jax.P('A', None)))
print(z.sharding)
print(y.sharding)

# Now that we understand mesh shapes, axis names, and shardings at the top
# level, we can dive into mesh axis types and how Explicit and Auto modes
# differ.

# ## Explicit sharding mode puts sharding in the trace-time types

# In explicit sharding mode, shardings are always queryable via `jax.typeof`,
# even under a `jax.jit`:

print(jax.typeof(x).sharding)

# +
jax.jit(lambda x: print(jax.typeof(x).sharding))(x)

# In terms of the printed representation, the type language is roughly:
#
#  <array_type> ::= <dtype>[<size_and_sharding>, ...]
#  <size_and_sharding> ::= <size> | <size>@<MeshAxisName>
#
# Where
#  * The MeshAxisNames in scope are those from `jax.typeof(x).sharding.mesh`
#  * Each MeshAxisName must be of Explicit axis type
#  * Each MeshAxisName can be mentioned at most once

# These shardings associated with JAX-level types propagate through operations.
# For example:

arg0 = jax.device_put(np.arange(4).reshape(4, 1), jax.P("X", None))
arg1 = jax.device_put(np.arange(8).reshape(1, 8), jax.P(None, "Y"))

result = arg0 + arg1

print(f"{jax.typeof(arg0)=!s}")
print(f"{jax.typeof(arg1)=!s}")
print(f"{jax.typeof(result)=!s}")

# We can do the same type querying under a `jit`:

@jax.jit
def add_arrays(x, y):
  ans = x + y
  print(f"{jax.typeof(arg0)=!s}")
  print(f"{jax.typeof(arg1)=!s}")
  print(f"{jax.typeof(result)=!s}")
  return ans

add_arrays(arg0, arg1)

# Given the input and output shardings, the computation itself is automatically
# partitioned over devices. The compiler inserts communication operations as
# needed. For example:

x = jax.random.normal(jax.random.key(0), (8192, 8192),
                      out_sharding=jax.P('X', 'Y'))
print(jax.typeof(x))

# +
y = x.sum(0)
print(jax.typeof(y))

# Here, when partitioning the computation, the compiler automatically inserts
# communication collectives to perform the reduction:

compile_txt = jax.jit(lambda x: x.sum(0)).lower(x).compile().as_text()
print('all-reduce(' in compile_txt)

### Result shardings follow simple rules, or error and require annotation

# Each primitive operation has a sharding propagation rule to determine the
# sharding of the result a function of input shardings. If there is not an
# obvious output sharding, an error is rasied. The goal is to get important
# parallelism decisions in your face, rather than hide them so you might
# accidentally miss them. Put another way, sharding propagation rules prefer to
# error and require annotation rather than falling back to arbitrarily chosen
# defaults.

# Each op is able to implement its own sharding propagation rule, but the usual
# pattern is:
#  1. For each output array axis, identify it with zero or more corresponding
#  input array axes.
#  2. If all those input axes are sharded the same as each other, shard the
#  output axis the same way; otherwise, error (and require an explicit
#  `out_sharding` argument).
#  3. After all output array axes are decided that way, if an output array
#  sharding  mentions the same mesh axis more than once, error (and require an
#  explicit `out_sharding`).

# Here are some example rules:
# * nullary ops like `jnp.zeros`, `jnp.arange`: These ops create arrays out of whole
# cloth so they don’t have input shardings to propagate. Their output is
# unsharded by default unless overridden by the `out_sharding` kwarg.
# * unary elementwise ops like `sin`, `exp`: The output is sharded the same as
# the input.
# * binary ops (`+`, `-`, `*` etc.): Axis shardings of “zipped” dimensions must
# match (or be None). “Outer product” dimensions (dimensions that appear in only
# one argument) are sharded as they are in the input. If the result ends up
# mentioning a mesh axis more than once it’s an error.

# The contraction ops like `jnp.dot` and `jnp.einsum` also have some interesting
# cases. For example, the result of `jnp.dot(x: f32[8,4@x], y:f32[4@x,16])`,
# where the shared contracting axis is sharded the same way, could reasonably be:
# * `f32[8,16]` (doing an all-reduce)
# * `f32[8@x,16]` (a reduce-scatter on the first axis)
# * `f32[8,16@x]` (a reduce-scatter on the second axis)
# * `f32[8,16]{U:x}` (no communication)
# Instead of automatically choosing one, JAX errors in this case and requires an
# `out_sharding` be provided, e.g. `jnp.dot(x, y, out_sharding=jax.P('x',
# None))`. But there are other cases that induce communication
# that JAX does perform automatically, like
# `jnp.dot(x:f32[8,4], y:f32[4@x,16])` results in an `f32[8,16]`, likely by
# doing an all-gather on `y` as in FSDP.

