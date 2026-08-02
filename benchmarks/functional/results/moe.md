# Benchmark Result of Mixture-of-Experts

* CUDA Version: 12.8
* Triton Version: 3.4.0
* Torch Version: 2.8.0+cu128

## RTX 2000 Ada

```
moe-forward-latency-num-tokens:
   num_tokens  Functional MoE  Functional MoE Compile  Torch Dense  Torch Dense Compile
0       128.0        0.323936                0.107776     0.152112             0.126496
1       256.0        0.319296                0.129792     0.190080             0.159232
2       512.0        0.326096                0.191392     0.298336             0.254784
3      1024.0        0.407904                0.287840     0.507264             0.449248
4      2048.0        0.610944                0.490928     0.948384             0.883456
moe-forward-latency-num-experts:
   n_experts  Functional MoE  Functional MoE Compile  Torch Dense  Torch Dense Compile
0        4.0        0.326272                0.182720     0.178608             0.139792
1        8.0        0.328384                0.191648     0.298400             0.254048
2       16.0        0.341568                0.221856     0.552080             0.501840
3       32.0        0.440496                0.318976     1.125584             1.062912
moe-forward-latency-top-k:
   top_k  Functional MoE  Functional MoE Compile  Torch Dense  Torch Dense Compile
0    1.0        0.319552                0.126496     0.289792             0.248448
1    2.0        0.327824                0.192512     0.299008             0.253984
2    4.0        0.400320                0.281312     0.304800             0.256704
moe-forward-latency-model-size:
   d_model    d_ff  Functional MoE  Functional MoE Compile  Torch Dense  Torch Dense Compile
0    128.0   512.0        0.322944                 0.11008     0.127392             0.105184
1    256.0  1024.0        0.328288                 0.19152     0.298752             0.253984
2    512.0  2048.0        0.553056                 0.43920     0.841888             0.781280
```
