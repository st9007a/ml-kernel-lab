# Benchmark Result of Flash Attention V1

* CUDA Version: 12.8
* Triton Version: 3.4.0
* Torch Version: 2.8.0+cu128

## RTX 2000 Ada

```
flash-attn-v1-forward-latency-seq-len:
   seq_len    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0    128.0  0.015312       0.026784               0.017024    0.016960
1    256.0  0.020512       0.050992               0.030240    0.030240
2    512.0  0.057152       0.122160               0.054240    0.056576
3   1024.0  0.136688       0.490560               0.124896    0.124800
flash-attn-v1-causal-forward-latency-seq-len:
   seq_len    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0    128.0  0.015936       0.034144               0.023040    0.014816
1    256.0  0.021952       0.064544               0.047456    0.020608
2    512.0  0.048224       0.153280               0.089728    0.042240
3   1024.0  0.110048       0.721312               0.339744    0.087904
flash-attn-v1-forward-latency-batch-size:
   batch_size    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0         1.0  0.056896       0.122320               0.056576    0.056256
1         2.0  0.084288       0.235072               0.085184    0.085728
2         4.0  0.155488       0.514368               0.143008    0.142976
3         8.0  0.284736       1.170400               0.259616    0.259648
flash-attn-v1-forward-latency-head-dim:
   head_dim    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0      64.0  0.031184       0.120000               0.532032    0.038784
1     128.0  0.056864       0.122432               0.534400    0.056480
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
