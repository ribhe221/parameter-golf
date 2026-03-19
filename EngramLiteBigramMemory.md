# Engram-Lite Bigram Memory for `parameter-golf`

## Summary

This module is a very small Engram-inspired memory path for the `parameter-golf` baseline.
It is intentionally **not** full Engram.

What it keeps:

- hashed local memory
- context-dependent gating
- early residual injection

What it drops:

- multi-order memory
- multi-head hash retrieval
- multi-layer memory insertions
- tokenizer canonicalization
- post-memory convolution
- large sparse-memory systems machinery

The implementation lives in `train_gpt.py` and is controlled by:

- `USE_BIGRAM_MEMORY`
- `BIGRAM_HASH_ROWS`
- `BIGRAM_MEM_DIM`
- `BIGRAM_MEM_LR`

Default reviewed setting:

- `BIGRAM_HASH_ROWS = 2048`
- `BIGRAM_MEM_DIM = 32`

## Mechanism

Let:

- `u_t` be the token id at position `t`
- `z_t in R^d` be the token embedding after RMSNorm
- `H` be the number of hash rows
- `r` be the low-rank memory width

Define the hashed bigram id

```text
h_t = 0,                                 if t = 0
h_t = 1 + (((a * u_t) XOR (b * u_{t-1})) mod H),   if t > 0
```

with fixed integer multipliers `a` and `b`.

The module parameters are:

- memory table `E in R^((H+1) x r)`
- projection `P in R^(r x d)`
- gate vector `w in R^d`
- gate bias `beta in R`
- scalar gain `alpha in R`

The memory path is:

```text
m_t = E[h_t] P
g_t = sigma(w^T z_t + beta)
z_t' = z_t + alpha g_t m_t
```

The baseline model then uses `z_t'` as both the main residual stream and the persistent `x0` stream:

```text
x_t^(0) = x0_t^(0) = z_t'
```

Because each transformer block mixes the current residual stream with `x0`, the memory signal is available throughout depth without adding per-layer memory parameters.

## Why This Can Help

The scientific bet is simple:

1. Small language models are weak at reconstructing frequent local transition structure.
2. A hashed bigram memory gives a cheap learned prior for common token transitions.
3. The gate suppresses the memory when the current token state does not support it.
4. The low-rank factorization keeps the memory cheap enough for the challenge.

This follows the core lesson from the NanoGPT bigram experiment:

- local hashed memory can help
- early injection matters
- the useful part is the residual prior, not the giant sparse table or sparse communication stack

It also follows the scale-down lesson from Engram theory:

- keep the memory idea
- remove the large-model extras first

## Parameter Count

The module parameter count is

```text
N_mem = (H + 1) r + r d + d + 2
```

For the reviewed default `H = 2048`, `r = 32`, `d = 512`:

```text
N_mem = (2049)(32) + (32)(512) + 512 + 2 = 82,466
```

This is still a small addition relative to the baseline model.

## Challenge-Fit Estimate

The baseline published record reports:

- total submission size: `15,863,489` bytes
- remaining headroom to `16,000,000`: `136,511` bytes

This implementation increased `train_gpt.py` by about `3,669` bytes.

Under this repo's serializer, the reviewed default is attractive because `2048 x 32` is just above the small-tensor fp16 passthrough threshold, so the table is stored in per-row int8 form instead of fp16.

Approximate added model bytes before metadata and zlib effects:

```text
table:      65,568 int8 bytes + 4,098 fp16 scale bytes
proj:       16,384 fp16 params = 32,768 bytes
gate:       512 fp16 params + bias
alpha:      scalar
total ~= 103,462 bytes
```

So the rough size budget is:

```text
15,863,489 + 3,669 + 103,462 ~= 15,970,620 bytes
```

That suggests the design is **likely still under the 16 MB cap**, but with limited margin, so the final trained export must still be checked directly.

## Training-Time Estimate

The extra compute per token is dominated by:

- one `r -> d` projection
- one `d -> 1` gate

For `r = 32`, `d = 512`, this is tiny relative to the full transformer.
On the challenge baseline, the expected training-time overhead is roughly:

- likely low single-digit percent
- plausibly around `1%` to `3%`
- unlikely to be the dominant bottleneck

The memory-table lookup itself is cheap; the main risk is not FLOPs, but whether the extra kernels and memory traffic reduce throughput more than expected.

## What This Is Not

This implementation is **not proof of improvement**.
It is a strong first architectural test.

Most likely outcomes:

- modest improvement if the model benefits from better local transition priors
- no change if the baseline already models these transitions well enough
- regression if collisions dominate or if the injected signal is too noisy

The highest-value next experiment is a clean ablation:

1. baseline
2. bigram memory on
3. compare final roundtrip `val_bpb`, artifact size, and step time

