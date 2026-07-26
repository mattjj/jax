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

"""Autoregressive sampling with a sharded, in-place KV cache.

Demonstrates: refs (mutable arrays), explicit sharding, `jit`, dynamic slices.

The cache is a pair of `jax.new_ref` mutable arrays that live on device across
the whole decode loop. Each step writes one position in place -- no donation
ceremony, no cache flowing through the function signature -- and the cache
stays sharded over the 'data' and 'model' mesh axes the whole time, so
sampling parallelizes the same way training does.

Prefill and decode are the *same* function, `forward`, called with a different
number of query positions: the prompt length for prefill, one for each decode
step. That falls out of passing query positions explicitly rather than
assuming they start at zero.

For a production-grade treatment of these ideas on real models, see
https://github.com/jax-ml/jax-llm-examples.

    python examples/sample.py                     # train, then sample
    python examples/sample.py --params run.npz    # sample from saved params
    python examples/sample.py --check             # verify against no cache

Without `--params` this trains a model first, which is most of the runtime.
Expect Shakespeare-shaped English rather than Shakespeare: the model here is
850k parameters over bytes, and the point of the file is the cache.
"""

import argparse

import numpy as np

import jax
import jax.numpy as jnp

import data
import util
import nanolm
from nanolm import B, H, L, N, T

CACHE = jax.P(None, 'data', None, 'model', None)  # [L, B, T, N, H]
ACTS = nanolm.ACTS


def new_cache(batch, length):
  shape = (L, batch, length, N, H)
  return (jax.new_ref(jnp.zeros(shape, out_sharding=CACHE)),
          jax.new_ref(jnp.zeros(shape, out_sharding=CACHE)))


def forward(params, k_cache, v_cache, tokens, start):
  """Runs `tokens` at positions `start ...`, appending to the cache in place.

  `tokens` has a static length, so this compiles once for the prompt and once
  more for the (length-1) decode steps.
  """
  seq = tokens.shape[1]
  q_pos = start + jnp.arange(seq)
  # Attend to every cache position at or before the query position. Positions
  # after `start + seq` are still zeros, and are masked out by the same rule.
  mask = jnp.arange(k_cache.shape[2])[None, :] <= q_pos[:, None]

  x = params['embed'].at[tokens].get(out_sharding=ACTS)
  for i in range(L):
    q, k, v = jnp.split(
        jnp.einsum('btd,dnh->btnh', nanolm.rmsnorm(x), params['qkv'][i]), 3, -1)
    # The only mutation in the program: one dynamic slice of the cache per
    # layer. `jax.ds(start, seq)` is a dynamically-offset, statically-sized
    # slice, which is what makes this a fixed-shape write.
    k_cache[i:i+1, :, jax.ds(start, seq)] = k[None]
    v_cache[i:i+1, :, jax.ds(start, seq)] = v[None]
    a = jax.nn.dot_product_attention(
        q, k_cache[i:i+1, ...][0], v_cache[i:i+1, ...][0], mask=mask)
    x += jnp.einsum('btnh,nhd->btd', a, params['proj'][i], out_sharding=ACTS)
    h = jax.nn.gelu(jnp.einsum('btd,df->btf', nanolm.rmsnorm(x), params['up'][i]))
    x += jnp.einsum('btf,fd->btd', h, params['down'][i], out_sharding=ACTS)
  return jnp.einsum('btd,dv->btv', nanolm.rmsnorm(x), params['unemb'],
                    out_sharding=ACTS)


@jax.jit
def step(params, k_cache, v_cache, tokens, start, key, temperature):
  logits = forward(params, k_cache, v_cache, tokens, start)[:, -1]
  greedy = jnp.argmax(logits, -1)
  sampled = jax.random.categorical(key, logits / temperature, axis=-1)
  return jnp.where(temperature > 0, sampled, greedy).astype(jnp.int32)


def generate(params, prompt, num_tokens, key, temperature, length=T):
  """Prefills `prompt`, then decodes `num_tokens` tokens one at a time."""
  batch = prompt.shape[0]
  k_cache, v_cache = new_cache(batch, length)
  prompt = jax.device_put(prompt, jax.P('data', None))

  key, subkey = jax.random.split(key)
  token = step(params, k_cache, v_cache, prompt, 0, subkey, temperature)
  # Pulling each token back to the host is what a streaming sampler does
  # anyway, and it keeps only one step's work in flight at a time.
  out, pos = [np.asarray(token)], prompt.shape[1]

  for _ in range(num_tokens - 1):
    key, subkey = jax.random.split(key)
    token = step(params, k_cache, v_cache, token[:, None], pos, subkey,
                 temperature)
    out.append(np.asarray(token))
    pos += 1
  # The refs were created here, so this is where they can be freed.
  jax.freeze(k_cache)
  jax.freeze(v_cache)
  return np.stack(out, axis=1)


def check(params, key):
  """Cached decoding must match running the whole prefix through the model."""
  rows = jax.sharding.get_mesh().shape['data']  # batch must divide the 'data' axis
  prompt = jnp.asarray(np.tile(data.encode('First Citizen:\n')[None], (rows, 1)),
                       jnp.int32)
  tokens = generate(params, prompt, 24, key, temperature=0.0)
  full = np.concatenate([np.asarray(prompt), tokens], axis=1)
  # Uncached reference: nanolm's own forward pass over the full sequence.
  ref = np.asarray(jax.jit(nanolm.logits)(
      params, jax.device_put(full[:, :-1], jax.P('data', None))))
  np.testing.assert_array_equal(ref.argmax(-1)[:, -tokens.shape[1]:], tokens)
  print('check: cached greedy decoding matches the uncached model')


def main(args):
  jax.set_mesh(jax.make_mesh(tuple(int(x) for x in args.mesh.split(',')),
                             ('data', 'model')))
  key = jax.random.key(args.seed)

  if args.params:
    # Inference needs no gradients, so the parameters go back to their plain
    # tensor-parallel layout -- no `reduced` cast.
    params = {k: jax.device_put(jnp.asarray(v), nanolm.PARAM_SPECS[k])
              for k, v in np.load(args.params).items()}
  else:
    print(f'no --params given; training for {args.train_steps} steps first')
    key, subkey = jax.random.split(key)
    batch_iter = data.batches(data.load(args.offline), B, T, seed=args.seed)
    params = nanolm.train(nanolm.init(subkey), batch_iter, args.train_steps)

  if args.check:
    return check(params, key)

  prompt = jnp.asarray(np.tile(data.encode(args.prompt)[None], (args.batch, 1)),
                       jnp.int32)
  print(f'prompt {jax.typeof(jax.device_put(prompt, jax.P("data", None)))}, '
        f'cache {jax.typeof(new_cache(args.batch, T)[0])}')
  tokens = generate(params, prompt, args.tokens, key, args.temperature)
  for row in tokens:
    print(f'  {args.prompt!r} -> {data.decode(row)!r}')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--devices', type=int, default=util.default_devices(),
                 help='simulated CPU devices; 0 to use real hardware')
  p.add_argument('--mesh', default=None, help='"data,model" mesh shape')
  p.add_argument('--params', default=None, help='.npz written by nanolm.py --save')
  p.add_argument('--train-steps', type=int, default=1200)
  p.add_argument('--prompt', default='First Citizen:\n')
  p.add_argument('--tokens', type=int, default=64)
  p.add_argument('--batch', type=int, default=8)
  p.add_argument('--temperature', type=float, default=0.8)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--offline', action='store_true')
  p.add_argument('--check', action='store_true')
  args = p.parse_args()
  if args.devices:
    jax.config.update('jax_num_cpu_devices', args.devices)
  if args.mesh is None:
    args.mesh = util.default_mesh(args.devices or jax.device_count())
  main(args)
