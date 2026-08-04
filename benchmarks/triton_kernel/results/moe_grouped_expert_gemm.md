# Benchmark Result of MoE Grouped Expert GEMM

* CUDA Version: 13.0
* Triton Version: 3.7.1
* Torch Version: 2.13.0+cu130

## RTX 2000 Ada

```
moe-grouped-expert-gemm-forward-latency-num-assignments:
   num_assignments  Triton (ms)  Torch Grouped MM (ms)
0            128.0     0.036272               0.137280
1            256.0     0.036704               0.135808
2            512.0     0.038608               0.136288
3           1024.0     0.051104               0.135456
4           2048.0     0.074528               0.138944
5           4096.0     0.104640               0.177952
moe-grouped-expert-gemm-forward-latency-num-experts:
   n_experts  Triton (ms)  Torch Grouped MM (ms)
0        4.0     0.095872               0.135680
1        8.0     0.104704               0.178208
2       16.0     0.122656               0.226016
3       32.0     0.166912               0.400160
moe-grouped-expert-gemm-forward-latency-model-size:
   d_model    d_ff  Triton (ms)  Torch Grouped MM (ms)
0    128.0   512.0     0.017184               0.135552
1    256.0  1024.0     0.050592               0.135328
2    512.0  2048.0     0.143072               0.211168
moe-grouped-expert-gemm-forward-latency-routing-imbalance:
   hot_expert_share  Triton (ms)  Torch Grouped MM (ms)
0             0.125     0.104704               0.179744
1             0.250     0.106496               0.186992
2             0.500     0.108832               0.162112
3             0.750     0.114192               0.176160
4             0.875     0.113920               0.161376
```
