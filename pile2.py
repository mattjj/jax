from __future__ import annotations

import pdb, sys, traceback
def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    pdb.pm()
sys.excepthook = info


import collections
import itertools as it
import string
from typing import Tuple, NamedTuple, Any

import numpy as np

from jax import core
from jax.core import Var, Tracer, AxisSize, DShapedArray, AbstractBInt

Array = Any

# Examples:
#   * [3, 1, 4].i
#   * n.i
#   * n.i.j
class IndexedAxisSize:
  idx: Tuple[Var, ...]
  sizes: Union[Array, Var, Tracer]  # int dtype, rank == len(idx)

  def __init__(self, idx: Tuple[Var, ...], sizes: Union[Array, Var, Tracer]):
    aval = sizes.aval if type(sizes) is Var else core.get_aval(sizes)
    assert len(idx) == len(aval.shape)
    self.idx = idx
    self.sizes = sizes

  def __repr__(self) -> str:
    idx_str = '.'.join(var_names[i] for i in self.idx)
    # idx_str = '.'.join(map(str, self.idx))
    data_str = str(self.sizes)
    return f'{data_str}.{idx_str}'

###

def Var(aval=core.ShapedArray((), np.dtype('int32'))):
  return core.Var(0, '', aval)

names = (''.join(chars) for n in it.count(1)
         for chars in it.product(string.ascii_lowercase, repeat=n))
var_names = collections.defaultdict(names.__next__)

print(IndexedAxisSize((Var(),), np.array([3, 1, 4])))          # [3,1,4].a
print(IndexedAxisSize((Var(), Var()), np.array([[3, 1, 4]])))  # [[3,1,4]].b.c

###

# Examples:
#   * i:3 => f32[[3, 1, 4].i]
#   * i:3 => f32[3]  -- could be flattened to 3x3 array
#   * i:2 => j:3 => f32[[[3, 1, 4], [1, 5, 9]].i.j]
# TODO could we just have .*.* and always apply in fixed order?
class PileTy(NamedTuple):
  binder: Var
  size: Union[int, Var, Tracer, IndexedAxisSize]
  ty: Union[PileTy, DShapedArray]

  def __repr__(self) -> str:
    binder_str = var_names[self.binder]
    # binder_str = str(self.binder)
    return f'{binder_str}:{self.size} => {self.ty}'

###

# d:3 => f32[3]
d = Var()
pt = PileTy(d, 3, DShapedArray((3,), np.dtype('float32'), False))
print(pt)

# d:3 => f32[[3, 1, 4].d]
d = Var()
pt = PileTy(d, 3, DShapedArray((IndexedAxisSize((d,), np.array([3, 1, 4])),),
                               np.dtype('float32'), False))
print(pt)

e = Var()
rect = DShapedArray((IndexedAxisSize((d, e), np.array([[3, 1, 4], [1, 5, 9]])),),
                    np.dtype('float32'), False)
pt = PileTy(d, 2, PileTy(e, 3, rect))
print(pt)

###

class Pile:
  ty: PileTy
  data: Array  # padded
  def __init__(self, ty: PileTy, data: Array):
    self.ty = ty
    self.data = data

  def __repr__(self) -> str:
    ty, sizes, binders = self.ty, [], []
    while isinstance(ty, PileTy):
      sizes.append(ty.size)
      binders.append(ty.binder)
      ty = ty.ty
    rect_shape = ty.shape
    del ty

    outs = []
    for idx in it.product(*map(range, sizes)):
      env = dict(zip(binders, idx))
      slices = [slice(d.sizes[tuple(env[i] for i in d.idx)])
                if isinstance(d, IndexedAxisSize) else slice(None)
                for d in rect_shape]
      full_index = (*idx, *slices)
      data = self.data[full_index]
      outs.append(f'{idx}:\n{data}')
    return f'{self.ty} with values:\n' + '\n\n'.join(outs)

###

import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')

d = Var()
pt = PileTy(d, 3, DShapedArray((IndexedAxisSize((d,), np.array([3, 1, 4])),),
                               np.dtype('float32'), False))
pile = Pile(pt, jnp.arange(4*3).reshape(3, 4))
print(pile)

e = Var()
rect = DShapedArray((IndexedAxisSize((d, e), np.array([[3, 1, 4], [1, 5, 9]])),),
                    np.dtype('float32'), False)
pt = PileTy(d, 2, PileTy(e, 3, rect))
pile = Pile(pt, jnp.arange(2*3*9).reshape(2, 3, 9))
print(pile)


# would have a BatchTracer wrapping a Pile, mapped along axis 0
# that Pile would look like `PileTy i:3 f32[[3, 1, 4].i]`
#                           `PileTy i:BIntTy(3) f32[[3, 1, 4].i]`




# p : i:(Fin 3) => f32[[3, 1, 4].i]
# p = vmap(jnp.arange)(jnp.array([3, 1, 4]))
# maybe no!

# maybe can only pile_map a pile_iota!
# i.e. don't overload, and don't turn an array output into a pile output
# (and if we want to eventually, we can do it 'at the api level', i.e. in a
# wrapper


# let's have pile_vmap and pile_scan (or pile_for)

# TODO pile_vmap


##


# tmap : (i:n -> a(i) -> b(i)) -> (j:n=>a) -> (k:n=>b)

# tmap(jnp.sin)(pile)


# pile introduction forms?
# option 1:
#  * pile_iota : n:int -> PileTy i:n i32[]
#  * pile_map : 
#  * (pile zip?)

# option 2:
#  * pile_for like pile_map but w/o indexing built in, i.e. call pile_get in body


# elimination forms
#  * pile_get : 



# let's start with the primitive version, b/c we need it to express the vmap
# version anyway

import operator as op

import jax.linear_util as lu
from jax.interpreters import partial_eval as pe
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.api_util import flatten_fun_nokwargs

jax.config.update('jax_dynamic_shapes', True)


def tmap(fun: Callable) -> Callable:
  def fun_tmap(x):
    breakpoint()
    pile_ty = x.ty
    # TODO substitute 
    arg_ty = type_substitute(pile_ty.binder, pe.DBIdx(0), pile_ty.ty)
    in_type = [(pile_ty.binder.aval, False), (arg_ty, True)]
    fun_ = lu.annotate(lu.wrap_init(fun), tuple(in_type))
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic2(fun_, None)
    out_flat = tmap_p.bind(*consts, *args_flat,
                           jaxpr=pe.convert_constvars_jaxpr(jaxpr))
    return out_flat
  return fun_tmap

tmap_p = core.Primitive('tmap')
tmap_p.multiple_results = True

@tmap_p.def_impl
def tmap_impl(*args, jaxpr):
  breakpoint()

d = Var()
pt = PileTy(d, 3, DShapedArray((IndexedAxisSize((d,), np.array([3, 1, 4])),),
                               np.dtype('float32'), False))
pile = Pile(pt, jnp.arange(4*3).reshape(3, 4))


core.pytype_aval_mappings[Pile] = op.attrgetter('ty')

# TODO abstracting the arrays in indexing expressions
core.raise_to_shaped_mappings[PileTy] = lambda x, weak_type: x

tmap(lambda x: [jnp.sin(x)])(pile)

