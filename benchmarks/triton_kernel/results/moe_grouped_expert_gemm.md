# Benchmark Result of MoE Grouped Expert GEMM

* CUDA Version: 13.0
* Triton Version: 3.7.1
* Torch Version: 2.13.0+cu130

## RTX 2000 Ada

```
moe-grouped-expert-gemm-forward-latency-num-assignments:
   num_assignments  Triton (ms)  Torch Grouped MM (ms)
0            128.0     0.036208               0.136640
1            256.0     0.036672               0.135760
2            512.0     0.038608               0.135072
3           1024.0     0.050528               0.135120
4           2048.0     0.074368               0.137664
5           4096.0     0.104800               0.178656
moe-grouped-expert-gemm-forward-latency-num-experts:
   n_experts  Triton (ms)  Torch Grouped MM (ms)
0        4.0     0.095712               0.135520
1        8.0     0.104736               0.178624
2       16.0     0.122656               0.224352
3       32.0     0.166368               0.400768
moe-grouped-expert-gemm-forward-latency-model-size:
   d_model    d_ff  Triton (ms)  Torch Grouped MM (ms)
0    128.0   512.0     0.017440               0.137120
1    256.0  1024.0     0.051264               0.135232
2    512.0  2048.0     0.142496               0.210944
moe-grouped-expert-gemm-forward-latency-routing-imbalance:
   hot_expert_share  Triton (ms)  Torch Grouped MM (ms)
0             0.125     0.104736               0.178448
1             0.250     0.106496               0.181920
2             0.500     0.108736               0.161728
3             0.750     0.114304               0.175968
4             0.875     0.114784               0.160928
moe-grouped-expert-gemm-forward-latency-expert-capacity:
   expert_capacity  Triton (ms)  Torch Grouped MM (ms)
0            512.0     0.105344               0.178496
1           1024.0     0.105856               0.178496
2           1536.0     0.106912               0.179232
3           2048.0     0.106592               0.178432
```
