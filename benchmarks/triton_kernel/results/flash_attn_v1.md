# Benchmark Result of Flash Attention V1

* CUDA Version: 12.8
* Triton Version: 3.4.0
* Torch Version: 2.8.0+cu128

## RTX 2000 Ada

```
flash-attn-v1-forward-latency-seq-len:
   seq_len    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0    128.0  0.014528       0.024896               0.017568    0.017216
1    256.0  0.020752       0.051840               0.031008    0.030784
2    512.0  0.058656       0.125600               0.056128    0.057392
3   1024.0  0.145216       0.497440               0.123168    0.123104
flash-attn-v1-causal-forward-latency-seq-len:
   seq_len    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0    128.0  0.014816       0.032992               0.022752    0.012192
1    256.0  0.022080       0.063808               0.047200    0.020464
2    512.0  0.050240       0.154576               0.087264    0.041696
3   1024.0  0.120512       0.718832               0.340416    0.085984
flash-attn-v1-forward-latency-batch-size:
   batch_size    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0         1.0  0.058816       0.125248               0.057376    0.057216
1         2.0  0.087456       0.241952               0.086592    0.086496
2         4.0  0.162912       0.530048               0.142624    0.142752
3         8.0  0.298256       1.206304               0.258000    0.258144
flash-attn-v1-forward-latency-head-dim:
   head_dim    Triton  Torch Unfused  Torch Unfused Compile  Torch SDPA
0      64.0  0.028448       0.121888               0.515184    0.039072
1     128.0  0.058784       0.125344               0.515456    0.057408
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
