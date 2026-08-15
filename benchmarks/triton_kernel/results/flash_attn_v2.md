# Benchmark Result of Flash Attention V2

* CUDA Version: 12.8
* Triton Version: 3.4.0
* Torch Version: 2.8.0+cu128

## RTX 2000 Ada

```
flash-attn-v2-forward-latency-seq-len:
   seq_len  Triton (ms)  Torch Unfused (ms)  Torch Unfused Compile (ms)  Torch SDPA (ms)
0    128.0     0.014688            0.027040                    0.017472         0.017664
1    256.0     0.019488            0.051392                    0.031008         0.030944
2    512.0     0.054400            0.127008                    0.055648         0.055296
3   1024.0     0.125824            0.502592                    0.124160         0.124160
flash-attn-v2-causal-forward-latency-seq-len:
   seq_len  Triton (ms)  Torch Unfused (ms)  Torch Unfused Compile (ms)  Torch SDPA (ms)
0    128.0     0.014368            0.035200                    0.023808         0.014912
1    256.0     0.020096            0.064736                    0.049056         0.020480
2    512.0     0.046112            0.156480                    0.096704         0.041984
3   1024.0     0.102272            0.719456                    0.342304         0.086752
flash-attn-v2-forward-latency-batch-size:
   batch_size  Triton (ms)  Torch Unfused (ms)  Torch Unfused Compile (ms)  Torch SDPA (ms)
0         1.0     0.054560            0.126912                    0.055712         0.055200
1         2.0     0.080320            0.241152                    0.087040         0.087008
2         4.0     0.145248            0.527264                    0.143824         0.143872
3         8.0     0.264288            1.204608                    0.261216         0.261120
flash-attn-v2-forward-latency-head-dim:
   head_dim  Triton (ms)  Torch Unfused (ms)  Torch Unfused Compile (ms)  Torch SDPA (ms)
0      64.0     0.028320            0.121760                    0.295904         0.036992
1     128.0     0.054512            0.126848                    0.297664         0.055712
flash-attn-v2-gqa-forward-latency-seq-len:
   seq_len  Triton (ms)  Torch SDPA GQA (ms)  Torch SDPA Expanded (ms)  Torch Unfused Expanded (ms)  Torch Unfused Expanded Compile (ms)
0    128.0     0.026096             0.024448                  0.054144                     0.075584                             0.039488
1    256.0     0.053600             0.058144                  0.093824                     0.139904                             0.068096
2    512.0     0.135984             0.122496                  0.187488                     0.556384                             0.162496
3   1024.0     0.455552             0.408192                  0.544064                     2.295040                             0.495232
flash-attn-v2-gqa-causal-forward-latency-seq-len:
   seq_len  Triton (ms)  Torch SDPA GQA (ms)  Torch SDPA Expanded (ms)  Torch Unfused Expanded (ms)  Torch Unfused Expanded Compile (ms)
0    128.0     0.026560             0.023168                  0.048800                     0.090208                             0.066336
1    256.0     0.049712             0.044064                  0.082784                     0.180256                             0.117648
2    512.0     0.103712             0.080000                  0.134976                     0.750560                             0.369936
3   1024.0     0.290400             0.239360                  0.371248                     3.632736                             1.603424
flash-attn-v2-gqa-forward-latency-kv-heads:
   n_kv_heads  Triton (ms)  Torch SDPA GQA (ms)  Torch SDPA Expanded (ms)  Torch Unfused Expanded (ms)
0         1.0     0.140096             0.120320                  0.167520                     0.545888
1         2.0     0.140256             0.122272                  0.172768                     0.547680
2         4.0     0.137728             0.123168                  0.183488                     0.551520
3         8.0     0.139360             0.127344                  0.199200                     0.561280
4        16.0     0.141024             0.132352                  0.214016                     0.579584
5        32.0     0.144928             0.144048                  0.201360                     0.586784
flash-attn-v2-gqa-forward-latency-batch-size:
   batch_size  Triton (ms)  Torch SDPA GQA (ms)  Torch SDPA Expanded (ms)  Torch Unfused Expanded (ms)
0         1.0     0.139328             0.127360                  0.199872                     0.561184
1         2.0     0.256000             0.224800                  0.363008                     1.307680
2         4.0     0.473376             0.421216                  0.688160                     2.566576
3         8.0     0.929264             0.822080                  1.327072                     5.061792
flash-attn-v2-gqa-forward-latency-head-dim:
   head_dim  Triton (ms)  Torch SDPA GQA (ms)  Torch SDPA Expanded (ms)  Torch Unfused Expanded (ms)
0      64.0     0.075648             0.078144                  0.115648                     0.542464
1     128.0     0.141152             0.127888                  0.200576                     0.558016
```
