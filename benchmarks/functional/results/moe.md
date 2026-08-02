# Benchmark Result of Mixture-of-Experts

* CUDA Version: 12.8
* Triton Version: 3.4.0
* Torch Version: 2.8.0+cu128

## RTX 2000 Ada

```
moe-forward-latency-num-tokens:
   num_tokens  Functional MoE  Functional MoE Compile  Torch Dense  Torch Dense Compile
0       128.0        0.318624                0.110080     0.152032             0.126896
1       256.0        0.319584                0.146816     0.191328             0.159776
2       512.0        0.323648                0.194784     0.298272             0.256096
3      1024.0        0.406304                0.291072     0.507008             0.453664
4      2048.0        0.609632                0.495840     0.948064             0.885824
moe-forward-latency-num-experts:
   n_experts  Functional MoE  Functional MoE Compile  Torch Dense  Torch Dense Compile
0        4.0        0.326336                0.185392     0.177824             0.130592
1        8.0        0.325392                0.195536     0.298144             0.234496
2       16.0        0.327296                0.209888     0.547520             0.516080
3       32.0        0.438528                0.322144     1.125312             1.059744
moe-forward-latency-top-k:
   top_k  Functional MoE  Functional MoE Compile  Torch Dense  Torch Dense Compile
0    1.0        0.316320                0.142944     0.290160             0.227200
1    2.0        0.330304                0.178416     0.256384             0.234592
2    4.0        0.395072                0.286336     0.305312             0.250272
moe-forward-latency-model-size:
   d_model    d_ff  Functional MoE  Functional MoE Compile  Torch Dense  Torch Dense Compile
0    128.0   512.0        0.323024                0.339872     0.126624             0.117632
1    256.0  1024.0        0.330912                0.196064     0.298592             0.289856
2    512.0  2048.0        0.560928                0.557920     0.843392             0.857216
```
