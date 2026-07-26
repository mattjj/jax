# How PyTorch overlaps communication with compute

Notes for the JAX team, written while deciding how `examples/nanolm.py` should
handle FSDP. The question that prompted them: *PyTorch clearly gets
compute/communication overlap in FSDP — how is that actually expressed, given
there's no compiler doing it?*

Everything below was run against torch 2.13.0 on CPU with a single gloo
process. The companion script is
[`pytorch_overlap_demo.py`](pytorch_overlap_demo.py); every output block here is
its real output. You don't need a GPU or a multi-process launcher to see the
structure, because the structure is all in program order.

## The one-sentence version

PyTorch has no compiler in the loop, so overlap is expressed as **"start the
collective earlier in Python program order, don't wait for it, and wait
later."** Two ingredients make that possible: `async_op=True`, which returns a
handle instead of blocking, and *hooks*, which are how you inject "do this now"
into the backward pass, where you have no program order of your own to write
into.

## The primitive: `async_op=True`

```python
work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
...       # anything here runs concurrently with the collective
work.wait()
```

`all_reduce` normally blocks. With `async_op=True` it returns a `Work` handle;
on GPU it enqueues the NCCL kernel on a **separate CUDA stream**, so it runs
concurrently with compute on the main stream, and `work.wait()` inserts a
cross-stream dependency rather than blocking the host thread. Everything
between launch and wait is the overlap window.

That's the whole mechanism. Everything else is policy: *which* collective to
launch, *when*, and how far ahead.

## What a hook is

An autograd hook is a callback attached to a node in the autograd graph,
invoked by the engine when it processes that node. The forward pass is ordinary
Python you can edit; the backward pass is generated, so hooks are the only way
to say "run this at this point in backward."

Two flavors matter.

**`tensor.register_hook(fn)`** fires with that tensor's gradient, the moment
it's computed:

```python
out = layer(out)
out.register_hook(lambda g, i=i: h.append(f"activation grad for layer {i}"))
```

```
forward: layer 0
forward: layer 1
forward: layer 2
hooks fired in this order: ['activation grad for layer 2',
                            'activation grad for layer 1',
                            'activation grad for layer 0']
```

**`param.register_post_accumulate_grad_hook(fn)`** fires once a *parameter's*
`.grad` is final. This is DDP's hook point:

```
  layer 2.weight grad ready     <- while layers 1 and 0 are still computing
  layer 1.weight grad ready
  layer 0.weight grad ready
```

That reverse ordering is the entire opportunity. Layer 2's gradient is finished
and could be in flight across the network while layers 1 and 0 are still doing
matmuls.

## DDP = hooks + bucketing

Hand-rolled, and this runs:

```python
class Bucket:
    def add(self, p):
        self.pending.append(p); self.size += p.numel()
        if self.size >= self.cap: self.flush()

    def flush(self):
        flat = torch._utils._flatten_dense_tensors([p.grad for p in self.pending])
        work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
        self.works.append((work, flat, list(self.pending)))
        self.pending, self.size = [], 0

    def wait(self):
        for work, flat, params in self.works:
            work.wait()
            ...  # unflatten back into p.grad

for p in reversed(list(m.parameters())):
    p.register_post_accumulate_grad_hook(lambda p, b=bucket: b.add(p))
```

```
  launched all-reduce for 4 params (144 elems)
  launched all-reduce for 4 params (144 elems)
  waited for all buckets
```

Real DDP is this plus engineering:

- Gradients are coalesced into **buckets** (default ~25 MB) so you issue a few
  large all-reduces instead of thousands of tiny ones. Bucketing is a pure
  latency/efficiency play, unrelated to correctness.
- Bucket order starts as reverse `model.parameters()` order, which merely
  *approximates* backward order — so DDP **rebuilds the buckets after the first
  iteration** using the order gradients actually arrived in.
- `no_sync()` suppresses the hooks for gradient accumulation.

Note what's absent: with DDP, parameters are replicated, so there are no
forward-pass collectives at all. Only gradients move. This is all nanoGPT uses.

## FSDP2: the same idea, but it has to run *ahead*

Gradient all-reduce is the easy case: the gradient already exists when the hook
fires, so you overlap it with work that comes *after*. A parameter all-gather is
the hard case — you need the weight *before* you can compute with it, so
overlapping means running **ahead** of the compute. That's prefetch:

```python
layers[0].unshard()                      # launch gather for layer 0
for i, layer in enumerate(layers):
    if i + 1 < len(layers):
        layers[i + 1].unshard()          # <-- launched BEFORE we wait on i
    w = layer.wait()
    h = h @ w
    layer.reshard()
```

```
  gather(0)
  gather(1)
  wait(0) -> matmul(0)
  reshard(0)
  gather(2)
  wait(1) -> matmul(1)
  reshard(1)
  gather(3)
  wait(2) -> matmul(2)
  reshard(2)
  wait(3) -> matmul(3)
```

`matmul(i)` sits between `gather(i+1)` and `wait(i+1)`. That gap is the
overlap, and there is exactly one gather in flight at a time — a software
pipeline of depth two.

FSDP2 automates this. `fully_shard(block)` makes each transformer block its own
parameter group; the runtime records the module execution order on iteration 1
and, in the `pre_forward` of block *i*, issues block *i+1*'s all-gather on a
dedicated stream. Backward is always `BACKWARD_PRE` — the docs say that's the
only ordering that overlaps correctly, since `BACKWARD_POST` prefetches wrong
under nesting.

The user-facing knobs are `set_modules_to_forward_prefetch` and
`set_modules_to_backward_prefetch`, and the doc sentence worth internalizing is:

> Passing a singleton list containing the previous FSDP module gives the same
> all-gather overlap behavior as the default overlap behavior, while passing a
> list with at least length two is required for more aggressive overlap and
> will use more reserved memory.

**That is pipeline depth, exposed as a user knob**, with the memory cost stated
plainly.

## What the reference repos actually do

Worth knowing, because it's easy to assume everyone runs FSDP:

- **nanoGPT** — plain DDP (`train.py:212`). Params replicated, gradient
  all-reduce overlapped by bucketed hooks. No forward collectives.
- **modded-nanogpt** — **ZeRO-2 with sharded weight update, not FSDP**. Params
  are full during fwd/bwd; the only collectives are in the optimizer step,
  structured as: launch `reduce_scatter_tensor(async_op=True)` for every param
  in `scatter_order`; then walk `work_order`, `future.wait()` on each param's
  reduce, compute its update on the shard, and immediately launch
  `all_gather_into_tensor(async_op=True)`; then wait on the gathers. Overlap is
  hand-scheduled by two hard-coded parameter orderings, with comments saying so:
  *"Process smaller/faster params first while large reduces complete"*,
  *"Large, polar express - process last to maximize overlap"*, and *"Explicit
  scatter_order and work_order for communication scheduling (no backward
  hooks)"*. Its parameters are *banks* — `qk_bank`, `vo_bank`, `mlp_bank`,
  stacked across layers — the same layout a JAX `scan` wants.
- **torchtitan** — the actual FSDP2 reference, per the above.
- **MLX** — not a useful reference here: data parallel via
  `all_sum`/`average_gradients`, whose optimization is batching many small
  comms into one large one. No parameter sharding in the forward pass.

## Why this matters for JAX

The structural difference is not that PyTorch is cleverer. It's that **PyTorch's
layer loop is an unrolled Python `for`**, so "prefetch layer *i+1*" costs
nothing — it's just moving a call earlier in program order. Hooks are needed
*only* for the backward pass, because that's the one part of the program the
user didn't write.

JAX inverts both halves of that:

| PyTorch | JAX |
|---|---|
| backward pass is generated; hooks inject into it | `custom_vjp` — **you write the backward pass** |
| layer loop is unrolled; program order is free | `scan` is a real loop; the body boundary is a barrier |
| prefetch depth ≥ 2 as a runtime list | `unroll=2` on the `scan` |
| separate CUDA stream + `work.wait()` | XLA's async decomposition and scheduler |

So the JAX translation of FSDP2's prefetch is: carry the *next* layer's gathered
weights through the `scan` (so the gather is issued an iteration early), use
`unroll=2` for the double buffer, and use `custom_vjp` to do the same in the
backward pass. That is exactly what `tests/pjit_test.py::test_fsdp_pipeline_grad`
does, and what `examples/fsdp_pipeline.py` now demonstrates.

Two slogans that seem worth keeping:

- **`custom_vjp` is JAX's backward hook.**
- **`unroll=2` is FSDP2's "list of at least length two."**

## Loose ends

- `torch.compile`/Inductor has a pass that reorders collectives against compute
  — the compiler analogue, and the closest thing to XLA's version of this
  problem. I did not verify its behavior; treat as a pointer, not a claim.
- Everything here is single-rank on CPU, so it demonstrates *program structure*,
  not achieved overlap. Measuring the win needs multiple GPUs.

## Sources

- [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)
- [KellerJordan/modded-nanogpt](https://github.com/kellerjordan/modded-nanogpt)
- [torchtitan FSDP docs](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- [FSDP2 API reference](https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html)
- [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [MLX distributed](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
