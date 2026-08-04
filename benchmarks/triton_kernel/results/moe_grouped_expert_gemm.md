# Benchmark Result of MoE Grouped Expert GEMM

* CUDA Version: 13.0
* Triton Version: 3.7.1
* Torch Version: 2.13.0+cu130

## RTX 2000 Ada

```
moe-grouped-expert-gemm-forward-latency-num-assignments:
   num_assignments  Triton Fixed (ms)  Triton Autotuned (ms)  Torch Grouped MM (ms)
0            128.0           0.035808               0.035680               0.110112
1            256.0           0.036480               0.036512               0.110784
2            512.0           0.037920               0.037792               0.112800
3           1024.0           0.051552               0.051392               0.110688
4           2048.0           0.073152               0.073408               0.128288
5           4096.0           0.104256               0.104176               0.174688
moe-grouped-expert-gemm-forward-latency-num-experts:
   n_experts  Triton Fixed (ms)  Triton Autotuned (ms)  Torch Grouped MM (ms)
0        4.0           0.094976               0.095072               0.129216
1        8.0           0.104608               0.104192               0.174720
2       16.0           0.121600               0.121344               0.221312
3       32.0           0.165728               0.162752               0.310752
moe-grouped-expert-gemm-forward-latency-model-size:
   d_model    d_ff  Triton Fixed (ms)  Triton Autotuned (ms)  Torch Grouped MM (ms)
0    128.0   512.0           0.016832               0.016896                0.11040
1    256.0  1024.0           0.051296               0.051392                0.11616
2    512.0  2048.0           0.141408               0.139296                0.21312
moe-grouped-expert-gemm-forward-latency-routing-imbalance:
   hot_expert_share  Triton Fixed (ms)  Triton Autotuned (ms)  Torch Grouped MM (ms)
0             0.125           0.104064               0.104768               0.175104
1             0.250           0.106592               0.106208               0.178272
2             0.500           0.109184               0.109120               0.159264
3             0.750           0.113888               0.113728               0.160352
4             0.875           0.113584               0.113664               0.146528
moe-grouped-expert-gemm-forward-latency-expert-capacity:
   expert_capacity  Triton Fixed (ms)  Triton Autotuned (ms)  Torch Grouped MM (ms)
0            512.0           0.104640               0.104224               0.174112
1           1024.0           0.105392               0.105952               0.174688
2           1536.0           0.106352               0.106144               0.174400
3           2048.0           0.107152               0.106656               0.175056
W0804 07:26:20.882000 1410 torch/_inductor/utils.py:1953] [0/0] Not enough SMs to use max_autotune_gemm mode
moe-grouped-expert-gemm-forward-latency-production-model-size:
   d_model     d_ff  Triton Fixed (ms)  Triton Autotuned (ms)  Torch Grouped MM (ms)  Torch Grouped MM Compile (ms)  Torch Grouped MM Max Autotune (ms)
0   1024.0   4096.0           0.875296               0.856288               0.954112                       0.961248                            0.957024
1   2048.0   8192.0           3.969440               3.798032               3.828320                       3.804192                            3.828064
2   4096.0  14336.0          28.118176              23.354239              13.699152                      13.706080                           13.646800
moe-grouped-expert-gemm-forward-latency-production-num-assignments:
   num_assignments  Triton Fixed (ms)  Triton Autotuned (ms)  Torch Grouped MM (ms)  Torch Grouped MM Compile (ms)  Torch Grouped MM Max Autotune (ms)
0           4096.0          28.058624              23.412623              13.698272                      13.719856                           13.730416
1           8192.0          57.022528              42.273857              27.537856                      27.382816                           27.319103
2          16384.0         111.461342              79.911522              47.819551                      47.823696                           47.809278
```
