# DfNet4 Performance Audit: Mamba Backbone Dominance

**Date**: 2026-04-05 (updated 2026-04-06)
**Component**: `df_mlx` training pathway
**Hardware**: Apple M3 Pro (18 GPU cores, 36GB)

## Summary

An adversarial performance audit of the `df_mlx` train_dynamic pathway
reveals that the **Mamba backbone accounts for 80-96% of forward pass
time**. Python-level training loop optimizations provide negligible speedup
because `mx.compile` CSE already optimizes the compiled graph.

**Key result**: `mx.checkpoint` on `_selective_scan` eliminates a
catastrophic memory cliff at batch≥32, delivering **2× speedup** for
forward+backward and **7.4× speedup** for the selective scan backward pass
at that batch size, with only ~5% overhead at smaller batches.

## Component-Level Benchmark Results

Run with: `python -m df_mlx.benchmark_model_components --batch-sizes 1,4`

### Batch=1, SeqLen=40, f32

| Component | Mean | % of Forward |
|-----------|------|-------------|
| **full_forward** | **5.43ms** | **100%** |
| backbone_mamba | 4.37ms | 80.5% |
| └ selective_scan | 1.05ms | 19.4% |
| └ linear projections | ~3.3ms | ~61% |
| encoder | 0.67ms | 12.4% |
| decoders + DfOp + VAD | ~0.5ms | 9.1% |
| **forward_backward** | **14.34ms** | **264%** |

### Batch=4, SeqLen=40, f32

| Component | Mean | % of Forward |
|-----------|------|-------------|
| **full_forward** | **23.15ms** | **100%** |
| backbone_mamba | 21.66ms | 93.6% |
| └ selective_scan | 5.32ms | 23.0% |
| └ linear projections | ~16.3ms | ~71% |
| encoder | 0.80ms | 3.5% |
| decoders + DfOp + VAD | ~0.7ms | 3.0% |
| **forward_backward** | **48.81ms** | **211%** |

### BF16 Comparison

BF16 provides **no meaningful speedup** on M3 Pro — the chip lacks
dedicated bf16 compute units (unlike NVIDIA tensor cores). Memory
bandwidth savings exist but are offset by conversion overhead.

## Key Insight: mx.compile CSE

Operations inside `@mx.compile` boundaries are traced once into a
computation graph. The compiler's CSE pass deduplicates identical
operations with the same inputs. This means:

- **Python-level caching** (e.g., precomputing z-scored energy) is
  redundant — the compiler already does this
- **Lazy operations** (`.astype()`, `mx.array(float)`) are graph nodes,
  not immediate copies — "caching" them saves nothing
- **Small-tree traversals** (gradient accumulation) differ by microseconds

## Optimization Paths

### ✅ IMPLEMENTED: Gradient Checkpoint on Selective Scan

Wrapping `_selective_scan` with `mx.checkpoint` eliminates ~4GB of
intermediate array storage during backprop by recomputing the scan
during the backward pass.

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| forward_backward batch=32 | 765ms (σ=1129ms) | 383ms (σ=11ms) | **2.0× faster, 100× less variance** |
| selective_scan batch=32 | 346ms (P95=1218ms) | 47ms (P95=50ms) | **7.4× faster** |
| forward_backward batch≤16 | baseline | +5% | Negligible overhead |

**Root cause**: The iterative doubling scan creates ~60 intermediate arrays
of shape `(batch, 64, d_inner=512, d_state=16)`. At batch≥32, these total
~4GB and cause catastrophic memory pressure with sporadic GPU stalls.
`mx.checkpoint` tells MLX to recompute these intermediates during backprop
instead of storing them.

**Commit**: `0b5ff63`

### A. Metal Kernel for Selective Scan (~20% forward pass)

The parallel prefix scan in `mamba.py:_selective_scan` creates ~60 graph
nodes via an iterative doubling Python loop. A custom Metal kernel could
replace these with a single GPU dispatch, saving ~20% of forward pass
time (~10% of a full train step including backward).

**Complexity**: High. Requires Metal shader development, MLX custom kernel
API integration, and careful numerical validation.

### B. Architecture Tuning (up to 50% savings)

The Mamba backbone's linear projections (in_proj: d_model→2*d_inner,
out_proj: d_inner→d_model) are the dominant cost. Reducing `expand_factor`
from 2→1 would halve `d_inner` and these projection costs. Reducing
`nb_layers` from 2→1 would halve backbone passes. Both are quality
trade-offs requiring evaluation runs.

### C. No Further Code Optimization

The training loop is well-optimized:
- `mx.compile` wraps forward+backward+optimizer ✓
- Data pipeline uses async prefetch ✓
- Gradient ops use `tree_map` ✓
- Sync boundaries are controlled by `eval_frequency` ✓
- Python overhead between steps is <0.1ms ✓

## Completed Optimizations

| Change | Commit | Real Impact |
|--------|--------|-------------|
| Gate sync-metric loss recomp | `6400f0a` | Minimal (sync boundaries only) |
| tree_map for gradient ops | `cb7b55c` | ~microseconds |
| z-score precomputation | `66d4acf` | Zero (CSE redundancy) |
| Dead FP32 code removal | `66d4acf` | Cleanup |
| Component benchmark tool | `5bdd5f5` | New diagnostic |
| **Gradient checkpoint scan** | **`0b5ff63`** | **2× batch=32 fwd+bwd, 7.4× scan** |

## Recommendations

1. Use `sync_mode: fast` to skip ALL per-component loss recomputation at
   sync boundaries (the one optimization with real impact for users who
   don't need detailed per-step metrics)

2. Run `benchmark_model_components` to profile your specific hardware
   before pursuing optimizations

3. For significant speedup, the path is either a custom Metal kernel for
   the selective scan (~20%) or architectural changes (~50%)
