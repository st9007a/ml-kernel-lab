# ML Kernel Lab

## Triton Kernel

### Implemented Kernel List

* RMS Norm
* RoPE
* SwiGLU
* Masked Softmax
* Flash Attention V1
* Paged K/V Attention
* Mixture of Experts
  * Grouped GEMM of expert networks

### Performance

See `benchmarks/triton_kernel/results`

### Development

Install with editable mode + dev dependencies

```
pip install uv
uv pip install --system -e . --group dev
```
