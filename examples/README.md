# JAX examples

Small, self-contained programs that each demonstrate a few JAX features doing
real work. They depend on nothing but `jax` and `numpy` — no neural network
library — because they are here to teach JAX, not a framework.

## The examples

| File | Demonstrates | Runtime |
|---|---|---|
| [`nanolm.py`](nanolm.py) | explicit sharding, tensor parallelism, ZeRO-2, `reduced`/`unreduced`, `jit`, `grad`, `scan`, `remat` | ~1 min |
| [`fsdp_pipeline.py`](fsdp_pipeline.py) | FSDP, `custom_vjp`, software-pipelined collectives | ~1 min |
| [`lora.py`](lora.py) | `vmap` over a sharded axis, multi-adapter training and serving | ~3 min |
| [`quantized.py`](quantized.py) | hijax custom array types, tangent types, quantization-aware training | ~30 s |
| [`flow_matching.py`](flow_matching.py) | a generative model with classifier-free guidance; `grad`, `vmap`, `scan` | ~1 min |
| [`hmc.py`](hmc.py) | Hamiltonian Monte Carlo: `grad` in the integrator, `vmap`ed sharded chains | ~10 s |
| [`diffsim.py`](diffsim.py) | a neural controller trained through a differentiable simulator; `remat`, measured | ~1 min |
| [`sample.py`](sample.py) | refs (mutable arrays), sharded KV cache, dynamic slices | ~6 min |
| [`moe.py`](moe.py) | `shard_map`, `all_to_all`, expert parallelism | ~1 min |
| [`differentially_private_sgd.py`](differentially_private_sgd.py) | `vmap` of `grad`: per-example gradients over a sharded batch | ~2 min |
| [`ffi/`](ffi) | calling custom C++ and CUDA kernels from JAX | |
| [`jax_cpp/`](jax_cpp) | ahead-of-time lowering and running from C++ | |
| [`k8s/`](k8s) | multi-host JAX on Kubernetes | |

See [`MODERNIZATION.md`](MODERNIZATION.md) for the plan and its history, and
[`FINDINGS.md`](FINDINGS.md) for a running log of JAX rough edges these
examples have turned up.

## Running them

Every example runs on simulated CPU devices by default, so the sharding is
real and inspectable without an accelerator:

```
python examples/nanolm.py
python examples/moe.py
python examples/sample.py
```

`sample.py` trains a model before sampling from it, so if you want to sample
more than once, train once and reuse the parameters:

```
python examples/nanolm.py --steps 1200 --save /tmp/nanolm.npz
python examples/sample.py --params /tmp/nanolm.npz
```

Each takes a `--mesh` argument, so the same file demonstrates different
parallelization strategies:

```
python examples/nanolm.py --mesh 8,1   # pure FSDP
python examples/nanolm.py --mesh 1,8   # pure tensor parallelism
python examples/nanolm.py --mesh 4,2   # both
```

and each has a `--check` mode that verifies the parallel computation against
an unsharded reference:

```
python examples/nanolm.py --check
python examples/moe.py --check
python examples/sample.py --check
python examples/lora.py --check
python examples/quantized.py --check
python examples/flow_matching.py --check
python examples/hmc.py --check
python examples/diffsim.py --check
python examples/differentially_private_sgd.py --check --offline
```

To run on real hardware instead of simulated devices, pass `--devices 0` and a
`--mesh` matching your machine.

Two things worth knowing about the simulated-device mode. Every array axis a
mesh axis shards must divide evenly, which is why these examples pick sizes
like 8 attention heads — it lets every factorization of an 8-device mesh work.
And every simulated device is backed by the same CPU thread pool, so a program
that runs far ahead of the device — JAX dispatches asynchronously — can queue
more concurrent collectives than the pool has threads and stall. The training
loops here wait on each step's loss to keep that bounded; if you write your own
loop over simulated devices and it hangs in an all-gather, that is why.

## Reading order

`nanolm.py` first: it is the smallest complete thing, and the two spec tables
at the top of it are the only place its parallelism is expressed. Then
`sample.py`, which reuses that model for inference; `lora.py`, which fine-tunes
it four different ways at once; `fsdp_pipeline.py`, which shards the parameters
too and shows what it costs to keep the resulting collectives off the critical
path; and `moe.py`, which is where automatic partitioning stops being enough and
`shard_map` takes over. The remaining three stand alone: `quantized.py` defines
a new array type and trains through it, and `flow_matching.py` and `hmc.py` are
the "JAX is not only for transformers" corner of the directory.

The documentation these are meant to accompany is
[Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
and [`shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html).
For full-scale, performance-tuned implementations of real open-weights models,
see [jax-llm-examples](https://github.com/jax-ml/jax-llm-examples).
