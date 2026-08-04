# Benchmark Result of Mixture-of-Experts

* CUDA Version: 12.8
* Triton Version: 3.4.0
* Torch Version: 2.8.0+cu128

## RTX 2000 Ada

```
moe-forward-latency-num-tokens:
   num_tokens  Functional MoE (ms)  Functional MoE Compile (ms)  Torch Grouped MM MoE (ms)  Torch Grouped MM MoE Compile (ms)  Torch Dense (ms)  Torch Dense Compile (ms)
0       128.0             0.315392                     0.113504                   0.503472                           0.406400          0.154752                  0.126768
1       256.0             0.316608                     0.129040                   0.511296                           0.420160          0.193920                  0.161568
2       512.0             0.321696                     0.170880                   0.521472                           0.433088          0.301184                  0.255040
3      1024.0             0.333760                     0.217760                   0.513312                           0.427808          0.513792                  0.452608
4      2048.0             0.451104                     0.333808                   0.626176                           0.528704          0.954816                  0.885728
moe-forward-latency-num-experts:
   n_experts  Functional MoE (ms)  Functional MoE Compile (ms)  Torch Grouped MM MoE (ms)  Torch Grouped MM MoE Compile (ms)  Torch Dense (ms)  Torch Dense Compile (ms)
0        4.0             0.322880                     0.152528                   0.409056                           0.326208          0.179264                  0.144320
1        8.0             0.323936                     0.170784                   0.517056                           0.435360          0.301408                  0.253856
2       16.0             0.334224                     0.215008                   0.755328                           0.649440          0.557440                  0.507488
3       32.0             0.437344                     0.318048                   1.103552                           1.029008          1.131424                  1.066848
moe-forward-latency-top-k:
   top_k  Functional MoE (ms)  Functional MoE Compile (ms)  Torch Grouped MM MoE (ms)  Torch Grouped MM MoE Compile (ms)  Torch Dense (ms)  Torch Dense Compile (ms)
0    1.0             0.313312                     0.124992                   0.512288                           0.420832          0.293184                  0.248992
1    2.0             0.324736                     0.171456                   0.512144                           0.443072          0.302464                  0.254112
2    4.0             0.328288                     0.212096                   0.509600                           0.419168          0.308048                  0.257856
moe-forward-latency-model-size:
   d_model    d_ff  Functional MoE (ms)  Functional MoE Compile (ms)  Torch Grouped MM MoE (ms)  Torch Grouped MM MoE Compile (ms)  Torch Dense (ms)  Torch Dense Compile (ms)
0    128.0   512.0             0.321248                     0.109888                   0.493824                           0.412720          0.130144                  0.105440
1    256.0  1024.0             0.325024                     0.171360                   0.521344                           0.437136          0.301696                  0.254144
2    512.0  2048.0             0.465872                     0.350432                   0.573888                           0.495648          0.849280                  0.790928
```
