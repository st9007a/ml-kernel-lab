# Benchmark Result of Mixture-of-Experts

* CUDA Version: 12.8
* Triton Version: 3.4.0
* Torch Version: 2.8.0+cu128

## RTX 2000 Ada

```
moe-forward-latency-num-tokens:
   num_tokens  Functional MoE (ms)  Functional MoE Compile (ms)  Torch Dense (ms)  Torch Dense Compile (ms)
0       128.0             0.316896                     0.109760          0.155360                  0.127296
1       256.0             0.317664                     0.126688          0.194176                  0.162016
2       512.0             0.349920                     0.172832          0.302080                  0.255136
3      1024.0             0.338528                     0.216032          0.513888                  0.454688
4      2048.0             0.449888                     0.333440          0.955616                  0.886800
moe-forward-latency-num-experts:
   n_experts  Functional MoE (ms)  Functional MoE Compile (ms)  Torch Dense (ms)  Torch Dense Compile (ms)
0        4.0             0.325312                     0.153152          0.180320                  0.144544
1        8.0             0.324992                     0.171520          0.301776                  0.255104
2       16.0             0.333536                     0.218144          0.558592                  0.507392
3       32.0             0.437632                     0.318368          1.131104                  1.066640
moe-forward-latency-top-k:
   top_k  Functional MoE (ms)  Functional MoE Compile (ms)  Torch Dense (ms)  Torch Dense Compile (ms)
0    1.0             0.315840                     0.126016          0.293600                  0.249440
1    2.0             0.328032                     0.170624          0.302336                  0.255136
2    4.0             0.331616                     0.212160          0.308480                  0.258816
moe-forward-latency-model-size:
   d_model    d_ff  Functional MoE (ms)  Functional MoE Compile (ms)  Torch Dense (ms)  Torch Dense Compile (ms)
0    128.0   512.0             0.322336                     0.111264          0.129312                  0.105472
1    256.0  1024.0             0.328336                     0.171520          0.301856                  0.255808
2    512.0  2048.0             0.467712                     0.351472          0.849568                  0.791904
```
