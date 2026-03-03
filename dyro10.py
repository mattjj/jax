from dataclasses import dataclass
import jax
import jax.numpy as jnp

from jax._src.tree_util import tree_structure, tree_flatten, PyTreeDef
from jax._src import core
from jax._src.core import typeof
from jax._src.state import discharge
from jax._src.state import primitives
from jax._src.state.types import TransformedRef, AbstractRef, Transform

x = jnp.ones(10)
x_ref = jax.new_ref(x)
xt_ref = x_ref.at[2:7]

get_p = core.Primitive('get2')
@get_p.def_abstract_eval
def _get_abstract_eval(x):
  return x.aval.inner_aval

@discharge.register_discharge_rule(get_p)
def _get_discharge(in_avals, out_avals, x):
  a, = in_avals
  idx, tree = a.transforms
  return discharge._get_discharge_rule([a.base_aval], out_avals, x, *idx, tree=tree)

class TRefTy(AbstractRef):
  aval: AbstractRef
  base_aval: AbstractRef
  transforms: tuple[tuple[int, ...], PyTreeDef]

  def __init__(self, aval, base_aval, transforms):
    self.aval = aval
    self.base_aval = base_aval
    self.transforms = transforms

  def __eq__(self, other):
    return (isinstance(other, TRefTy) and 
            self.base_aval == other.base_aval and
            self.transforms == other.transforms)

  def __hash__(self):
    return hash((self.base_aval, self.transforms))

  def _getitem(self, tracer, idx):
    if idx != Ellipsis: raise NotImplementedError
    return get_p.bind(tracer)

  ndim = property(lambda self: self.aval.ndim)
  dtype = property(lambda self: self.aval.dtype)
  shape = property(lambda self: self.aval.shape)
  inner_aval = property(lambda self: self.base_aval.inner_aval)

  def str_short(self, short_dtypes):
    return f'TRefTy{{{self.aval.str_short(short_dtypes=True)}}}'

def _typeof(x):
  leaves_, tree = tree_flatten(x.transforms)
  leaves = tuple(map(int, leaves_))
  return TRefTy(x.type, typeof(x.ref), (leaves, tree))
core.pytype_aval_mappings[TransformedRef] = _typeof

@jax.jit
def f(x_ref):
  return x_ref[...]

f(xt_ref)


# @jax.jit
# def f():
#   x_ref = jax.new_ref(x)
#   return x_ref.at[1][...]
