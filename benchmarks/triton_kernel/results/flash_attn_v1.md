# Benchmark Result of Flash Attention V1

* CUDA Version: 12.8
* Triton Version: 3.4.0
* Torch Version: 2.8.0+cu128

## RTX 2000 Ada

```
flash-attn-v1-forward-latency-seq-len:
   seq_len    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0    128.0  0.014848       0.026560               0.017344    0.017248
1    256.0  0.021056       0.050752               0.030240    0.030240
2    512.0  0.057024       0.122048               0.054240    0.056608
3   1024.0  0.136832       0.490464               0.124800    0.124864
flash-attn-v1-causal-forward-latency-seq-len:
   seq_len    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0    128.0  0.016064       0.034784               0.022624    0.014272
1    256.0  0.022016       0.064608               0.047488    0.020640
2    512.0  0.048256       0.153392               0.089472    0.042528
3   1024.0  0.110240       0.720368               0.338640    0.087744
flash-attn-v1-forward-latency-batch-size:
   batch_size    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0         1.0  0.057088       0.122272               0.056352    0.056608
1         2.0  0.083776       0.234944               0.085024    0.085568
2         4.0  0.155616       0.514256               0.143040    0.142992
3         8.0  0.284192       1.170208               0.260160    0.259616
flash-attn-v1-forward-latency-head-dim:
   head_dim    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0      64.0  0.030720       0.120160               0.531584    0.038624
1     128.0  0.057184       0.122112               0.533392    0.056544
```

## RTX 5090

```
flash-attn-v1-forward-latency-seq-len:
   seq_len    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0    128.0  0.014336       0.016384               0.014208    0.013920
1    256.0  0.020480       0.018944               0.014336    0.014336
2    512.0  0.034816       0.032672               0.022528    0.022528
3   1024.0  0.063488       0.075776               0.044640    0.044640
flash-attn-v1-causal-forward-latency-seq-len:
   seq_len    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0    128.0  0.014336       0.030304               0.012288    0.012704
1    256.0  0.020480       0.032832               0.016384    0.014336
2    512.0  0.035232       0.049152               0.028672    0.018432
3   1024.0  0.063488       0.110592               0.071264    0.032864
flash-attn-v1-forward-latency-batch-size:
   batch_size    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0         1.0  0.034816       0.032672               0.022528    0.022528
1         2.0  0.035312       0.047104               0.030720    0.030720
2         4.0  0.036928       0.075360               0.036864    0.036864
3         8.0  0.069632       0.137632               0.070240    0.070144
flash-attn-v1-forward-latency-head-dim:
   head_dim    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0      64.0  0.020480       0.028672               0.075744    0.014272
1     128.0  0.034816       0.032672               0.079456    0.022528
```
