# Benchmark Result of Flash Attention V1

* CUDA Version: 12.8
* Triton Version: 3.4.0
* Torch Version: 2.8.0+cu128
* GPU: RTX 2000 Ada

```
flash-attn-v1-forward-latency-seq-len:
   seq_len    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA  Torch SDPA Compile
0    128.0  0.016032       0.027296               0.017856    0.017632            0.017952
1    256.0  0.021504       0.051456               0.031136    0.031392            0.031168
2    512.0  0.058560       0.124832               0.055584    0.057280            0.057728
3   1024.0  0.146560       0.503424               0.125248    0.125312            0.125344
flash-attn-v1-forward-latency-batch-size:
   batch_size    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA  Torch SDPA Compile
0         1.0  0.058368       0.124960               0.057312    0.057664            0.057312
1         2.0  0.088096       0.241424               0.086816    0.086880            0.086864
2         4.0  0.165056       0.528672               0.144832    0.144864            0.144896
3         8.0  0.299232       1.205216               0.262816    0.263168            0.263168
flash-attn-v1-forward-latency-head-dim:
   head_dim    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA  Torch SDPA Compile
0      64.0  0.033792       0.122864               0.528928    0.039552            0.039360
1     128.0  0.059072       0.124928               0.532128    0.057760            0.057376
```
