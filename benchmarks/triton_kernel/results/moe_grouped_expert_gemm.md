# Benchmark Result of MoE Grouped Expert GEMM

* CUDA Version: 13.0
* Triton Version: 3.7.1
* Torch Version: 2.13.0+cu130

## RTX 2000 Ada

```
moe-grouped-expert-gemm-forward-latency-num-assignments:
   num_assignments  Triton (ms)  Torch Grouped MM (ms)
0            128.0     0.035552               0.135280
1            256.0     0.038624               0.134112
2            512.0     0.045056               0.133376
3           1024.0     0.068064               0.134848
4           2048.0     0.099328               0.136768
5           4096.0     0.172096               0.178976
moe-grouped-expert-gemm-forward-latency-num-experts:
   n_experts  Triton (ms)  Torch Grouped MM (ms)
0        4.0     0.176896               0.135616
1        8.0     0.171392               0.178432
2       16.0     0.176448               0.225344
3       32.0     0.185536               0.395424
moe-grouped-expert-gemm-forward-latency-model-size:
   d_model    d_ff  Triton (ms)  Torch Grouped MM (ms)
0    128.0   512.0     0.023392               0.134144
1    256.0  1024.0     0.068128               0.133552
2    512.0  2048.0     0.175584               0.210784
moe-grouped-expert-gemm-forward-latency-routing-imbalance:
   hot_expert_share  Triton (ms)  Torch Grouped MM (ms)
0             0.125     0.172096               0.178752
1             0.250     0.177088               0.182160
2             0.500     0.183648               0.161856
3             0.750     0.191904               0.177184
4             0.875     0.197888               0.164432
```
