# Benchmark Result of Flash Attention V2

* CUDA Version: 13.0
* Triton Version: 3.7.1
* Torch Version: 2.13.0+cu132

## RTX 2000 Ada

```
flash-attn-v2-forward-latency-seq-len:
   seq_len  Triton Fixed (ms)  Triton Autotuned (ms)  Torch SDPA (ms)
0    128.0           0.014448               0.013792         0.017376
1    256.0           0.021024               0.020640         0.031296
2    512.0           0.054336               0.050432         0.055424
3   1024.0           0.133344               0.123904         0.127008
flash-attn-v2-causal-forward-latency-seq-len:
   seq_len  Triton Fixed (ms)  Triton Autotuned (ms)  Torch SDPA (ms)
0    128.0           0.015264               0.014656         0.013248
1    256.0           0.023360               0.023168         0.021472
2    512.0           0.048544               0.045248         0.043360
3   1024.0           0.107232               0.101984         0.091152
flash-attn-v2-mha-causal-prefill-forward-latency-seq-len:
   seq_len  Triton Fixed (ms)  Triton Autotuned (ms)  Torch SDPA (ms)
0   1024.0           0.308160               0.298688         0.267104
1   2048.0           1.006272               0.960848         0.893312
2   4096.0           3.948640               3.728256         3.663200
3   8192.0          15.459584              14.577824        14.714560
flash-attn-v2-forward-latency-batch-size:
   batch_size  Triton Fixed (ms)  Triton Autotuned (ms)  Torch SDPA (ms)
0         1.0           0.054976               0.050624         0.055504
1         2.0           0.082752               0.078144         0.086528
2         4.0           0.146848               0.141488         0.146464
3         8.0           0.273760               0.259680         0.266880
flash-attn-v2-forward-latency-head-dim:
   head_dim  Triton Fixed (ms)  Triton Autotuned (ms)  Torch SDPA (ms)
0      64.0           0.029760               0.024832         0.037792
1     128.0           0.054528               0.051072         0.055488
flash-attn-v2-gqa-forward-latency-seq-len:
   seq_len  Triton (ms)  Torch SDPA GQA (ms)
0    128.0     0.027008             0.024832
1    256.0     0.056960             0.057856
2    512.0     0.144160             0.132928
3   1024.0     0.473440             0.427200
flash-attn-v2-gqa-causal-forward-latency-seq-len:
   seq_len  Triton (ms)  Torch SDPA GQA (ms)
0    128.0     0.026976             0.023904
1    256.0     0.052464             0.049312
2    512.0     0.109056             0.091392
3   1024.0     0.303744             0.249568
flash-attn-v2-gqa-forward-latency-kv-heads:
   n_kv_heads  Triton (ms)  Torch SDPA GQA (ms)
0         1.0     0.145632             0.126272
1         2.0     0.145168             0.126912
2         4.0     0.142432             0.128288
3         8.0     0.144064             0.133024
4        16.0     0.144880             0.134816
5        32.0     0.146656             0.147936
flash-attn-v2-gqa-forward-latency-batch-size:
   batch_size  Triton (ms)  Torch SDPA GQA (ms)
0         1.0     0.144000             0.133152
1         2.0     0.265328             0.235920
2         4.0     0.492288             0.440592
3         8.0     0.979904             0.855840
flash-attn-v2-gqa-forward-latency-head-dim:
   head_dim  Triton (ms)  Torch SDPA GQA (ms)
0      64.0     0.077792             0.080832
1     128.0     0.144480             0.133408
```
