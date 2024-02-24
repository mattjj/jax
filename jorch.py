from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import attrs

### convenience/syntax tools for defining layers (optional!)

class Param:
  name: str
  trainable: bool

  def __init__(self, trainable=True):
    self.trainable = trainable

  # https://docs.python.org/3/reference/datamodel.html#object.__set_name__
  def __set_name__(self, owner_cls, name: str):
    self.name = '_' + name

  # https://docs.python.org/3/reference/datamodel.html#object.__get__
  def __get__(self, instance, owner=None) -> jax.Array:
    if instance is None: return self
    return attrs.jax_getattr(instance, self.name)

  # https://docs.python.org/3/reference/datamodel.html#object.__set__
  def __set__(self, instance, val: jax.Array):
    attrs.jax_setattr(instance, self.name, val)

class Model:
  def init_params(self, key):
    assert False  # override

  def __call__(self, x):
    assert False  # override

  @property
  def trainable_params(self):
    params = [(self, param.name) for param in type(self).__dict__.values()
              if type(param) is Param and param.trainable]
    params.extend([p for mod in self.__dict__.values() if isinstance(mod, Model)
                   for p in mod.trainable_params])
    return params

### layers definitions

@dataclass(eq=False)
class Dense(Model):
  n_in: int
  n_out: int

  w = Param()
  b = Param()

  def init_params(self, key):
    k1, k2 = random.split(key)
    self.w = random.normal(k1, (self.n_in, self.n_out))
    self.b = random.normal(k1, (self.n_out,))

  def __call__(self, x):
    return x @ self.w + self.b

@dataclass(eq=False)
class FFResBlock(Model):
  dense_in: Dense
  dense_out: Dense

  def __init__(self, model_dims: int, hidden_dims: int):
    self.dense_in = Dense(model_dims, hidden_dims)
    self.dense_out = Dense(hidden_dims, model_dims)

  def init_params(self, key):
    k1, k2 = random.split(key)
    self.dense_in.init_params(k1)
    self.dense_out.init_params(k2)

  def __call__(self, x):
    return x + self.dense_out(jax.nn.relu(self.dense_in(x)))

### losses

@dataclass
class Loss:
  model: Model
  grads: None | dict[tuple[Any, str], jax.Array] = None

  def __call__(self, inputs, targets) -> float:
    return self.loss_fn(self.model(inputs), targets)

  def grad(self, inputs, targets):
    gradfun = attrs.grad(self, attrs=self.model.trainable_params)
    self.grads = gradfun(inputs, targets)

  def loss_fn(self, inputs, targets):
    assert False  # override

@dataclass
class SquaredError(Loss):
  def loss_fn(self, predictions, targets):
    return jnp.mean((predictions - targets) ** 2)

### optimizer library

@dataclass
class Optimizer:
  loss: Loss
  def step(self):
    for (o, a), g in loss.grads.items():
      x = attrs.jax_getattr(o, a)
      new_x = self.step_fn(x, g)
      attrs.jax_setattr(o, a, new_x)

@dataclass
class SGD(Optimizer):
  loss: Loss
  lr: float
  def step_fn(self, x, g):
    return x - self.lr * g

### example

model = FFResBlock(3, 4)

model.init_params(random.key(0))

x = jnp.ones(3)
y = jnp.ones(3)

loss = SquaredError(model)
optimizer = SGD(loss, 1e-2)

@jax.jit
def step(inputs, targets):
  loss.grad(inputs, targets)
  optimizer.step()

for _ in range(10):
  step(x, y)
  print(loss(x, y))

