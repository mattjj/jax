from dataclasses import dataclass
from functools import partial
from typing import Callable
import jax
import jax.numpy as jnp
from jax._src import core
from jax._src.tree_util import FlatTree
from jax._src.hijax import MutableHiType, QDD, register_hitype, HiPrimitive, box_effect

log_effect = box_effect

@dataclass(frozen=True)
class Monoid:
  op: Callable
  id: Callable

@dataclass(frozen=True)
class CatMonoid(Monoid):
  pass
cat = CatMonoid(None, None)

@dataclass(frozen=True)
class LogQDD(QDD):
  ft: FlatTree  # FlatTree[AbstractValue]
  mons: tuple[Monoid, ...]

  @classmethod
  def fresh(self):
    return LogQDD(FlatTree.flatten({}), ())

  def to_tangent_qdd(self):
    return LogQDD(self.ft.map(lambda a: a.to_tangent_aval()),
                  tuple(m.to_tangent_monoid() for m in self.mons))

class Log:
  _dct: dict
  _mons: dict

  def __init__(self):
    self._dct = {}
    self._mons = {}

  def cur_qdd(self):
    return LogQDD(FlatTree.flatten(self._dct).map(core.typeof),
                  tuple(self._mons[k] for k in sorted(self._mons)))

  def append(self, key, val, mon=cat):
    log_append(self, key, val, mon)

  def __repr__(self) -> str:
    return f'Log({self._dct})'

def log_append(log, key, val, mon=cat):
  log_append_p.bind(log, val, key=key, mon=mon)

class LogTy(MutableHiType):
  has_qdd = True
  append = core.aval_method(log_append)

  def __hash__(self): return hash(Log)
  def __eq__(self, other): return isinstance(other, Log)
  def str_short(self, short_dtypes=False, **_) -> str: return 'Log'

  def lo_ty_qdd(self, state: LogQDD, /) -> list[core.AbstractValue]:
    return list(state.ft)

  def new_from_loval(self, fresh_logqdd) -> Log:
    assert fresh_logqdd == LogQDD.fresh()
    return Log()  # will be mutated

  def read_loval(self, state: LogQDD, log: Log) -> list:
    assert FlatTree.flatten(log._dct).map(core.typeof) == state.ft
    return list(FlatTree.flatten(log._dct))

  def update_from_loval(self, state: LogQDD, log: Log, *lo_vals) -> None:
    new_stuff = state.ft.update(lo_vals).unflatten()
    new_mons = state.ft.update(state.mons).unflatten()
    log._dct = dict(log._dct, **new_stuff)
    log._mons = dict(log._mons, **new_mons)

  def to_tangent_aval(self):
    return LogTy()

register_hitype(Log, lambda _: LogTy())

class LogAppend(HiPrimitive):
  multiple_results = True  # no results

  def abstract_eval(self, log_ty, val_ty, *, key, mon):
    log_qdd = log_ty.mutable_qdd.cur_val
    new_ft = FlatTree.flatten({**log_qdd.ft.unflatten(), key: val_ty})
    new_mons = FlatTree.flatten({**log_qdd.ft.update(log_qdd.mons).unflatten(), key: mon})
    log_ty.mutable_qdd.update(LogQDD(new_ft, tuple(new_mons)))
    return [], {log_effect}

  def to_lojax(_, log, val, *, key, mon):
    log._dct = {**log._dct, key: val}
    log._mons = {**log._mons, key: mon}
    return []
log_append_p = LogAppend('log_append')

###

@jax.jit
def f(l, x):
  l.append('x', x * 2)

l = Log()
f(l, 3.)
print(l._dct)


l = Log()
def body(c, x):
  l.append('x', c + x)
  return c, ()
_, () = jax.lax.scan(body, 0., jnp.arange(3.))
print(l._dct)


l = Log()

@jax.custom_vjp
def foo(l, x):
  return x
def foo_fwd(l, x):
  return x, l
def foo_bwd(l, g):
  l.append('g', g)
  return None, g
foo.defvjp(foo_fwd, foo_bwd)


def f(l, x):
  def body(x, _):
    x = 2 * x
    x = foo(l, x)
    return x, ()
  x, () = jax.lax.scan(body, x, (), length=3)
  return x

jax.grad(partial(f, l))(1.0)
print(l._dct)

# TODO: this is always a stacking monoid, but what if we want some reduction?
# gotta teach scan about it, since it's got to either put it in the carry or the
# extensive output, and if it's in the carry it's got to call the reduction.
# so how about some new methods on hitypes that scan can call?
# currently, in _scan_to_lojax we move non-concat qddful hi-binders from consts
# to carry, then expand them. but if some components are concat-y and some
# aren't... the bookkeeping gets annoying.
# what about the scan-index idea? maybe the type's components aren't tagged one
# way or the other, it's just the uses. as in, the use of the whole log is
# stacking or not.
