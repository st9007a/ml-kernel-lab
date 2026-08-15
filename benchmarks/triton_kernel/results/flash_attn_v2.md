# Benchmark Result of Flash Attention V2

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

# Benchmark Result of Flash Attention V2 GQA

## RTX 2000 Ada

```
flash-attn-v1-gqa-forward-latency-seq-len:
   seq_len    Triton  Torch SDPA GQA  Torch SDPA Expanded  Torch Unfused Expanded  Torch Unfused Expanded Compile
0    128.0  0.028448        0.024256             0.053024                0.076896                        0.038368
1    256.0  0.056176        0.058272             0.095808                0.140320                        0.070384
2    512.0  0.154928        0.128288             0.191040                0.563328                        0.163552
3   1024.0  0.514496        0.425200             0.559200                2.296352                        0.508752
W0720 00:53:42.387000 945 torch/_inductor/utils.py:1436] [0/2] Not enough SMs to use max_autotune_gemm mode
flash-attn-v1-gqa-causal-forward-latency-seq-len:
   seq_len    Triton  Torch SDPA GQA  Torch SDPA Expanded  Torch Unfused Expanded  Torch Unfused Expanded Compile
0    128.0  0.028576        0.024064             0.048672                0.091936                        0.063136
1    256.0  0.053584        0.046720             0.084160                0.180480                        0.115904
2    512.0  0.115168        0.084768             0.142048                0.778720                        0.370848
3   1024.0  0.331328        0.247936             0.379024                3.638656                        1.613440
flash-attn-v1-gqa-forward-latency-kv-heads:
   n_kv_heads    Triton  Torch SDPA GQA  Torch SDPA Expanded  Torch Unfused Expanded
0         1.0  0.156992        0.125728             0.170368                0.551840
1         2.0  0.157120        0.127008             0.179328                0.553216
2         4.0  0.155680        0.127952             0.188448                0.556416
3         8.0  0.156192        0.132832             0.204928                0.568848
4        16.0  0.156928        0.134304             0.218464                0.581808
5        32.0  0.159232        0.146848             0.209472                0.586688
flash-attn-v1-gqa-forward-latency-batch-size:
   batch_size    Triton  Torch SDPA GQA  Torch SDPA Expanded  Torch Unfused Expanded
0         1.0  0.156416        0.132832             0.204320                0.568512
1         2.0  0.287328        0.232048             0.370624                1.306656
2         4.0  0.540176        0.437136             0.703008                2.568848
3         8.0  1.049088        0.856224             1.369600                5.066304
flash-attn-v1-gqa-forward-latency-head-dim:
   head_dim   Triton  Torch SDPA GQA  Torch SDPA Expanded  Torch Unfused Expanded
0      64.0  0.07920        0.079456             0.119072                0.561248
1     128.0  0.15592        0.131712             0.204320                0.563360
```
