# GPU Inference Optimization for GLM-ASR

A portfolio case study on profiling and optimizing **GLM-ASR** inference with **PyTorch, Triton, and GPU profiling**.

> This repository is a portfolio reconstruction of the optimization work. It intentionally does **not** publish the original coursework template, assignment instructions, model weights, or the full submitted implementation.

## Highlights

- Reduced end-to-end inference latency from **1477.4 ms to 711.8 ms** in the best tested configuration (**~2.1× speedup**).
- Maintained **100% transcription accuracy** on the project benchmark sample.
- Evaluated kernel fusion through ablation rather than assuming that more fusion is always faster.
- Investigated memory pressure and steady-state benchmarking, including warm-up before timed runs.

## Optimization Work

### 1. FlashAttention-style attention

The project replaced a multi-kernel attention path with a fused Triton implementation using blockwise computation and online softmax state. The goal was to reduce intermediate HBM traffic and avoid materializing the full attention matrix.

### 2. Fused Add + RMSNorm

Residual addition and RMSNorm were combined into one kernel. This removes an intermediate global-memory round trip and reduces kernel-launch overhead.

### 3. Decoder FP16 GEMV

For the autoregressive decode path where `M = 1`, a dedicated FP16 GEMV path was evaluated instead of treating the operation as a general tiled GEMM. The optimization targets weight-loading bandwidth and the shape characteristics of single-token decoding.

### 4. Tile / warp / stage tuning

The linear kernel was profiled with different launch configurations. The selected configuration used:

```text
BLOCK_M = 128
BLOCK_N = 64
BLOCK_K = 32
num_warps = 4
num_stages = 3
```

The configuration was selected empirically rather than assuming the largest degree of parallelism would be fastest.

## Fusion Ablation

One of the most useful findings was that **more fusion did not automatically produce better end-to-end performance**.

| Configuration | Latency (ms) |
|---|---:|
| Fully unfused | 819.9 |
| Encoder MLP fused | **711.8** |
| Decoder MLP fused | 719.2 |
| Encoder + Decoder MLP fused | 822.2 |

The best tested configuration fused the **Encoder MLP only**. This reinforced the importance of measuring the complete inference pipeline rather than judging an optimization only by its local kernel design.

## End-to-End Result

| Metric | Baseline | Best configuration |
|---|---:|---:|
| Inference latency | 1477.4 ms | **711.8 ms** |
| Relative speed | 1.0× | **~2.1×** |
| Benchmark transcription accuracy | 100% | **100%** |

## Benchmarking Methodology

A warm-up forward pass was executed before timed measurements to trigger Triton JIT compilation and CUDA initialization. This avoids mixing first-run compilation overhead with steady-state inference latency.

The workflow was:

```text
PyTorch / Triton baseline
        ↓
Operator and end-to-end profiling
        ↓
Kernel / fusion experiments
        ↓
Correctness checks
        ↓
Ablation comparison
        ↓
Best end-to-end configuration
```

## What I Learned

This project strengthened my understanding of the relationship between GPU kernel design and end-to-end model performance. In particular, it showed that reducing kernel launches or fusing operations is not automatically beneficial: tensor shapes, memory traffic, launch configuration, and resource pressure all affect the final result.

It also gave me practical experience with profiling, controlled ablation, debugging GPU memory constraints, and validating performance improvements without sacrificing model output correctness.

## Repository Scope

The original project was completed in an academic coursework setting. To avoid redistributing course materials or starter code, this public portfolio version focuses on the **optimization methodology, experimental results, and technical findings** rather than the original assignment repository.

## Technologies

`Python` · `PyTorch` · `Triton` · `GPU Profiling` · `CUDA-enabled NVIDIA GPU`
