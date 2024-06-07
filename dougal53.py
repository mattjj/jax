import pdb, sys, traceback
def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    pdb.pm()
sys.excepthook = info

from functools import partial
from dataclasses import dataclass
import jax
from jax._src import core
from jax._src.util import safe_map as map, safe_zip as zip, unflatten

@dataclass
class Douple:
  elts: tuple

def pack(*elts):
  return pack_p.bind(*elts)

def splat(tup):
  return splat_p.bind(tup)

pack_p = core.Primitive('pack')
splat_p = core.Primitive('splat')
splat_p.multiple_results = True

### impls

@pack_p.def_impl
def pack_impl(*vals):
  return Douple(vals)

@splat_p.def_impl
def splat_impl(tup):
  return tup.elts

### test impls

import jax.numpy as jnp

tup = pack(1, jnp.arange(2), 3.)
elts = splat(tup)
print(elts)  # [1, Array([0, 1], dtype=int32), 3.0]


tup = pack(jnp.arange(2.), tup)
elts = splat(tup)
print(elts)  # [Array([0., 1.], dtype=float32), Douple(...)]

### staging

@dataclass(frozen=True)
class AbstractDouple:
  types: tuple
  def str_short(self, short_dtypes=False) -> str:
    return f"Douple({', '.join(t.str_short(short_dtypes) for t in self.types)})"
  strip_named_shape = strip_weak_type = lambda self: self  # ugh

def typeof(x: Douple):
  types = [core.raise_to_shaped(core.get_aval(e)) for e in x.elts]
  return AbstractDouple(tuple(types))
core.pytype_aval_mappings[Douple] = typeof
core.raise_to_shaped_mappings[AbstractDouple] = lambda x, _: x

@pack_p.def_abstract_eval
def pack_abstract_eval(*elts):
  elts = tuple(map(core.raise_to_shaped, elts))
  return AbstractDouple(elts)

@splat_p.def_abstract_eval
def splat_abstract_eval(tup):
  assert type(tup) is AbstractDouple
  return tup.types

### test staging

@jax.make_jaxpr
def f(x, y, z):
  tup = pack(x, y, z)
  x, y, z = splat(tup)
  return x, y, z

jaxpr = f(1, jnp.arange(2), 3.)
print(jaxpr)
# { lambda ; a:i32[] b:i32[2] c:f32[]. let
#     d:Douple(i32[], i32[2], f32[]) = pack a b c
#     e:i32[] f:i32[2] g:f32[] = splat d
#   in (e, f, g) }

### lowering and dispatch

from jax._src.interpreters import mlir, pxla, xla
mlir.register_representation(AbstractDouple, lambda tup: tup.types)
mlir.register_lowering(pack_p, lambda ctx, *xs: [xs])
mlir.register_lowering(splat_p, lambda ctx, xs: xs)

def shard_douple(dup, shardings):
  return map(pxla.shard_arg, dup.elts, shardings)
pxla.shard_arg_handlers[Douple] = shard_douple

def douple_result_handler(aval, shardings, committed):
  handlers = [pxla.global_result_handlers[type(t)](t, s, committed)
              for t, s in zip(aval.types, shardings)]
  lens = map(len, map(mlir.representation, aval.types))
  def handler(bufs):
    bufss = unflatten(bufs, lens)
    return Douple([h(bs) for h, bs in zip(handlers, bufss)])
  return handler
pxla.global_result_handlers[AbstractDouple] = douple_result_handler

def canonicalize_douple(dup):
  return Douple(map(xla.canonicalize_dtype, dup.elts))
xla.canonicalize_dtype_handlers[Douple] = canonicalize_douple

# TODO support sharding annotations...?

### test lowering and dispatch

dup = jax.jit(pack)(1, 2, 3)
print(dup)
elts = jax.jit(splat)(dup)
print(elts)

cons = lambda x, dup: (x, *splat(dup))
out = jax.jit(cons)(1, Douple((2, 3)))
print(out)

### autodiff

from jax._src import ad_util
from jax._src.interpreters import ad

def tangent_type_of(dup):
  vspaces = [x.at_least_vspace() for x in dup.types]
  return AbstractDouple((*vspaces,))
AbstractDouple.at_least_vspace = tangent_type_of

def add_dups(dup1, dup2):
  return pack(*map(ad_util.add_jaxvals, splat(dup1), splat(dup2)))
ad_util.adders[AbstractDouple] = add_dups

def zeros(dup):
  return pack(*map(ad_util.zeros_like_aval, dup.types))
ad_util.aval_zeros_likers[AbstractDouple] = zeros

ad.deflinear(pack_p, splat)
ad.deflinear(splat_p, lambda cts: [pack(*map(ad.instantiate_zeros, cts))])

### test autodiff

def f(dup):
  x, *_ = splat(dup)
  y, *_ = splat(dup)
  return jnp.sin(x + y)
print(jax.grad(f)(pack(1., 2.)))

def g(dup):
  return 3.
print(jax.grad(g)(pack(1., 2.)))

### vmap

from jax._src.interpreters import batching

def axis_size_handler(dup, i):
  return batching.axis_size(splat(dup)[0], i)
batching.axis_size_handlers[AbstractDouple] = axis_size_handler

def splat_batcher(vals, dims):
  (dup,), (d,) = vals, dims
  elts = splat(dup)
  return elts, [d] * len(elts)
batching.primitive_batchers[splat_p] = splat_batcher

def pack_batcher(vals, dims):
  size, = {batching.axis_size(x, d) for x, d in zip(vals, dims)
           if d is not batching.not_mapped}
  vals = map(partial(batching.bdim_at_front, size=size), vals, dims)
  return pack(*vals), 0
batching.primitive_batchers[pack_p] = pack_batcher

### test vmap

def f(dup):
  x, y = splat(dup)
  return pack(x + y)
tup = pack(jnp.arange(3), jnp.arange(3.))
jax.vmap(f)(tup)

### scan

def map_dup(size, axis, aval):
  types = map(partial(core.mapped_aval, size, axis), aval.types)
  return AbstractDouple(tuple(types))
def unmap_dup(size, _, axis, aval):
  types = map(partial(core.unmapped_aval, size, _, axis), aval.types)
  return AbstractDouple(tuple(types))
core.aval_mapping_handlers[AbstractDouple] = map_dup, unmap_dup

### test scan

def body(_, x):
  x, = splat(x)
  return None, pack(jnp.sin(x))
_, y = jax.lax.scan(body, None, pack(jnp.arange(3.)))
print(y)
