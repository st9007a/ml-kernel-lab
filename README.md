# ML Kernel Lab

## Triton Kernel

### Implemented Kernel List

* RMS Norm
* RoPE
* SwiGLU
* Masked Softmax
* Flash Attention V1
* Paged K/V Attention
* Mixture of Experts: Grouped GEMM

### Performance

See `benchmarks/triton_kernel/results`

## Development

Install with editable mode + dev dependencies

```
pip install uv

# Install to the system
uv pip install --system --break-system-packages -e . --group dev


# Install to the venv
uv venv
uv pip install -e . --group dev
```

## TODO

* Triton
  * FA2
  * INT8 Inference
  * INT4 Inference
* CUTLASS/CuTe
  * FA3
  * FA4
* NCCL
* Harness
  * Correctness
  * Benchmark
  * Profiling
