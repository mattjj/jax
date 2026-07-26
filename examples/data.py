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

"""Byte-level text data for the language modeling examples.

There is no tokenizer here on purpose: bytes are a perfectly good vocabulary
for a small example, and it keeps the examples dependency-free. The vocabulary
size is therefore always 256.
"""

import os
import urllib.request

import numpy as np

_DATA = os.environ.get("JAX_EXAMPLE_DATA", "/tmp/jax_example_data")
_TINY_SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

VOCAB_SIZE = 256


def tiny_shakespeare(url: str = _TINY_SHAKESPEARE) -> np.ndarray:
  """Downloads ~1MB of Shakespeare and returns it as a uint8 array."""
  os.makedirs(_DATA, exist_ok=True)
  path = os.path.join(_DATA, "tiny_shakespeare.txt")
  if not os.path.isfile(path):
    print(f"downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
  with open(path, "rb") as f:
    return np.frombuffer(f.read(), dtype=np.uint8)


def synthetic(num_bytes: int = 1 << 20, seed: int = 0) -> np.ndarray:
  """Offline stand-in for `tiny_shakespeare`: a learnable byte sequence.

  A random walk over the alphabet, so the next byte is largely predictable
  from the previous one and the loss visibly goes down with no network access.
  Spaces and capitals are sprinkled in deterministically so that byte-level
  transforms like upper-casing are not no-ops here (see `lora.py`).
  """
  rng = np.random.RandomState(seed)
  steps = rng.randint(-2, 3, size=num_bytes).cumsum() % 26
  out = (steps + ord("a")).astype(np.uint8)
  out[::16] = ord(" ")     # word separators
  out[1::16] -= 32         # capitalize the letter after each space
  return out


def load(offline: bool = False) -> np.ndarray:
  if offline:
    return synthetic()
  try:
    return tiny_shakespeare()
  except Exception as e:  # no network: still want the example to run
    print(f"falling back to synthetic data ({type(e).__name__}: {e})")
    return synthetic()


def batches(data: np.ndarray, batch_size: int, seq_len: int, seed: int = 0):
  """Yields uint8 arrays of shape `(batch_size, seq_len + 1)` forever.

  The extra element is so that a batch contains both the inputs `x[:, :-1]`
  and the targets `x[:, 1:]`.
  """
  rng = np.random.RandomState(seed)
  while True:
    i = rng.randint(0, len(data) - seq_len - 1, size=batch_size)
    yield np.stack([data[j:j + seq_len + 1] for j in i])


def decode(tokens) -> str:
  return bytes(np.asarray(tokens, dtype=np.uint8)).decode("utf-8", "replace")


def encode(text: str) -> np.ndarray:
  return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
