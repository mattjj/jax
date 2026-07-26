"""How PyTorch expresses compute/communication overlap, in four runnable pieces.

Companion to notes/pytorch-overlap.md. Runs on CPU with a single gloo process
-- no GPU, no multi-process launcher -- because the point is the *scheduling
structure*, which is visible from one rank. Nothing here needs a real cluster.

    python notes/pytorch_overlap_demo.py

Verified against torch 2.13.0.
"""

import os, torch, torch.nn as nn, torch.distributed as dist
torch.set_num_threads(1)
torch.manual_seed(0)

print("=" * 70)
print("1. A tensor hook fires DURING backward, the moment that grad exists")
print("=" * 70)

def mlp(n):
    return nn.Sequential(*[nn.Linear(8, 8) for _ in range(n)])

m = mlp(3)
x = torch.randn(4, 8)

# register_hook on an intermediate activation: called with that tensor's grad
h = []
out = x
for i, layer in enumerate(m):
    out = layer(out)
    out.register_hook(lambda g, i=i: h.append(f"activation grad for layer {i}"))
    print(f"forward: layer {i}")
out.sum().backward()
print("hooks fired in this order:", h)

print()
print("=" * 70)
print("2. A post-accumulate-grad hook fires when a PARAM's .grad is ready")
print("=" * 70)
print("   This is the DDP hook point: 'this parameter's gradient is final,")
print("   launch its all-reduce now' -- while earlier layers are still running.")

m = mlp(3)
order = []
for i, layer in enumerate(m):
    layer.weight.register_post_accumulate_grad_hook(
        lambda p, i=i: order.append(f"layer {i}.weight grad ready"))
m(x).sum().backward()
print("\n".join("  " + s for s in order))
print("  ^ reverse layer order: layer 2's grad is ready while 1 and 0 are")
print("    still being computed. That is the whole overlap opportunity.")

print()
print("=" * 70)
print("3. Hand-rolled DDP: bucket grads, launch async all-reduce from the hook")
print("=" * 70)

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29511")
dist.init_process_group("gloo", rank=0, world_size=1)

m = mlp(4)
log = []

class Bucket:
    """Collect params until `cap` elements, then all-reduce the whole bucket."""
    def __init__(self, cap):
        self.cap, self.pending, self.size, self.works = cap, [], 0, []

    def add(self, p):
        self.pending.append(p); self.size += p.numel()
        if self.size >= self.cap:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        flat = torch._utils._flatten_dense_tensors([p.grad for p in self.pending])
        # async_op=True returns immediately; the collective runs in the
        # background while the rest of the backward pass keeps computing.
        work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
        log.append(f"launched all-reduce for {len(self.pending)} params "
                   f"({self.size} elems)")
        self.works.append((work, flat, list(self.pending)))
        self.pending, self.size = [], 0

    def wait(self):
        for work, flat, params in self.works:
            work.wait()
            for p, g in zip(params, torch._utils._unflatten_dense_tensors(
                    flat, [p.grad for p in params])):
                p.grad.copy_(g)
        log.append("waited for all buckets")

bucket = Bucket(cap=100)
# Register in REVERSE parameter order, which approximates backward order --
# this is exactly what DDP does, and why it rebuilds buckets after iteration 1
# using the order gradients actually arrived in.
for p in reversed(list(m.parameters())):
    p.register_post_accumulate_grad_hook(lambda p, b=bucket: b.add(p))

m(x).sum().backward()
bucket.flush()
bucket.wait()
print("\n".join("  " + s for s in log))

print()
print("=" * 70)
print("4. FSDP-shaped prefetch: gather layer i+1 before running layer i")
print("=" * 70)

class Sharded(nn.Module):
    """Toy: holds a shard of a weight, gathers it on demand."""
    def __init__(self, w, rank, world):
        super().__init__()
        self.shard = w.chunk(world, dim=0)[rank].clone()
        self.full = None

    def unshard(self):                       # launch the all-gather, don't wait
        buf = torch.empty(self.shard.shape[0] * dist.get_world_size(),
                          self.shard.shape[1])
        self.work = dist.all_gather_single(buf, self.shard, async_op=True)
        self.full = buf
        return self.work

    def wait(self):
        self.work.wait(); return self.full

    def reshard(self):
        self.full = None

layers = [Sharded(torch.randn(8, 8), 0, 1) for _ in range(4)]
trace = []

# The prefetch pattern, written out. Note the shape: one gather is always
# in flight *ahead* of the compute that needs it.
layers[0].unshard()
trace.append("gather(0)")
h = torch.randn(4, 8)
for i, layer in enumerate(layers):
    if i + 1 < len(layers):
        layers[i + 1].unshard()               # <-- launched BEFORE we wait
        trace.append(f"gather({i+1})")
    w = layer.wait()
    trace.append(f"wait({i}) -> matmul({i})")
    h = h @ w
    layer.reshard()
    trace.append(f"reshard({i})")
print("  " + "\n  ".join(trace))
print()
print("  Between 'gather(i+1)' and 'wait(i+1)' sits 'matmul(i)'. That gap is")
print("  the overlap. FSDP2 does this for you by recording the module order")
print("  on iteration 1 and replaying it as prefetch.")

dist.destroy_process_group()
