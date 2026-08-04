# Benchmark Result of Mixture-of-Experts

* CUDA Version: 13.0
* Triton Version: 3.7.1
* Torch Version: 2.13.0+cu130

## RTX 2000 Ada

```
moe-forward-latency-num-tokens:
   num_tokens  Functional MoE (ms)  Functional MoE Compile (ms)  Torch Grouped MM MoE (ms)  Torch Grouped MM MoE Compile (ms)  Torch Dense (ms)  Torch Dense Compile (ms)
0       128.0             0.318720                     0.110352                   0.608128                           0.412000          0.155008                  0.127072
1       256.0             0.318128                     0.128640                   0.503744                           0.407456          0.194496                  0.162720
2       512.0             0.324576                     0.171520                   0.518240                           0.437664          0.301408                  0.254880
3      1024.0             0.335520                     0.216752                   0.514624                           0.427712          0.514144                  0.452688
4      2048.0             0.450016                     0.335424                   0.627808                           0.531648          0.954272                  0.885776
moe-forward-latency-num-experts:
   n_experts  Functional MoE (ms)  Functional MoE Compile (ms)  Torch Grouped MM MoE (ms)  Torch Grouped MM MoE Compile (ms)  Torch Dense (ms)  Torch Dense Compile (ms)
0        4.0             0.324192                     0.151488                   0.411280                           0.326944          0.179200                  0.143968
1        8.0             0.325600                     0.171424                   0.519824                           0.437728          0.302112                  0.254496
2       16.0             0.334496                     0.217072                   0.711520                           0.623808          0.557408                  0.506592
3       32.0             0.436128                     0.317408                   1.103168                           1.018928          1.130320                  1.066208
moe-forward-latency-top-k:
   top_k  Functional MoE (ms)  Functional MoE Compile (ms)  Torch Grouped MM MoE (ms)  Torch Grouped MM MoE Compile (ms)  Torch Dense (ms)  Torch Dense Compile (ms)
0    1.0             0.315968                     0.125472                   0.508320                           0.417568           0.29280                  0.247744
1    2.0             0.325664                     0.171168                   0.520128                           0.437696           0.30144                  0.254944
2    4.0             0.329120                     0.211392                   0.511600                           0.420512           0.30720                  0.257888
moe-forward-latency-model-size:
   d_model    d_ff  Functional MoE (ms)  Functional MoE Compile (ms)  Torch Grouped MM MoE (ms)  Torch Grouped MM MoE Compile (ms)  Torch Dense (ms)  Torch Dense Compile (ms)
0    128.0   512.0             0.321376                     0.109632                   0.493184                           0.411008          0.128576                  0.103872
1    256.0  1024.0             0.325344                     0.171296                   0.520832                           0.437136          0.301472                  0.254528
2    512.0  2048.0             0.467056                     0.351024                   0.575760                           0.492864          0.848832                  0.790976
```
