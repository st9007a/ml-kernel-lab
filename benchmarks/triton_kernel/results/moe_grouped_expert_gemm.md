# Benchmark Result of MoE Grouped Expert GEMM

* CUDA Version: 13.0
* Triton Version: 3.7.1
* Torch Version: 2.13.0+cu130

## RTX 2000 Ada

```
moe-grouped-expert-gemm-forward-latency-num-assignments:
   num_assignments  Triton v1 Fixed (ms)  Triton v1 Autotuned (ms)  Triton v2 (ms)  Triton v2 Autotuned (ms)  Torch Grouped MM (ms)
0            128.0              0.035968                  0.035872        0.036032                  0.036032               0.139552
1            256.0              0.037088                  0.037056        0.036512                  0.036544               0.139456
2            512.0              0.038048                  0.037984        0.038208                  0.038176               0.138912
3           1024.0              0.051648                  0.051584        0.047520                  0.047552               0.139232
4           2048.0              0.072992                  0.073152        0.067936                  0.065504               0.134928
5           4096.0              0.104448                  0.104416        0.099200                  0.095616               0.179648
moe-grouped-expert-gemm-forward-latency-num-experts:
   n_experts  Triton v1 Fixed (ms)  Triton v1 Autotuned (ms)  Triton v2 (ms)  Triton v2 Autotuned (ms)  Torch Grouped MM (ms)
0        4.0              0.094976                  0.094816        0.088032                  0.088096               0.132208
1        8.0              0.104480                  0.103872        0.098752                  0.096032               0.178656
2       16.0              0.121232                  0.121280        0.113440                  0.113440               0.225632
3       32.0              0.165600                  0.162848        0.156992                  0.151680               0.406976
moe-grouped-expert-gemm-forward-latency-model-size:
   d_model    d_ff  Triton v1 Fixed (ms)  Triton v1 Autotuned (ms)  Triton v2 (ms)  Triton v2 Autotuned (ms)  Torch Grouped MM (ms)
0    128.0   512.0              0.017792                  0.017664        0.017536                  0.017440               0.137120
1    256.0  1024.0              0.051456                  0.051392        0.047904                  0.047808               0.139312
2    512.0  2048.0              0.141776                  0.139616        0.136224                  0.133312               0.216160
moe-grouped-expert-gemm-forward-latency-routing-imbalance:
   hot_expert_share  Triton v1 Fixed (ms)  Triton v1 Autotuned (ms)  Triton v2 (ms)  Triton v2 Autotuned (ms)  Torch Grouped MM (ms)
0             0.125              0.103776                  0.104512        0.099360                  0.095584               0.178432
1             0.250              0.106912                  0.106912        0.095744                  0.093056               0.180912
2             0.500              0.109024                  0.108960        0.096640                  0.096160               0.162880
3             0.750              0.113760                  0.113760        0.096544                  0.095616               0.163552
4             0.875              0.114208                  0.114080        0.096032                  0.095584               0.150688
moe-grouped-expert-gemm-forward-latency-expert-capacity:
   expert_capacity  Triton v1 Fixed (ms)  Triton v1 Autotuned (ms)  Triton v2 (ms)  Triton v2 Autotuned (ms)  Torch Grouped MM (ms)
0            512.0              0.104608                  0.103840        0.098496                  0.096128               0.178592
1           1024.0              0.105824                  0.105216        0.101120                  0.097408               0.178416
2           1536.0              0.106624                  0.106560        0.102592                  0.099520               0.178880
3           2048.0              0.107008                  0.106336        0.102112                  0.101760               0.178512
moe-grouped-expert-gemm-forward-latency-production-model-size:
   d_model     d_ff  Triton v1 Fixed (ms)  Triton v1 Autotuned (ms)  Triton v2 (ms)  Triton v2 Autotuned (ms)  Torch Grouped MM (ms)  Torch Grouped MM Compile (ms)  Torch Grouped MM Max Autotune (ms)
0   1024.0   4096.0              0.869664                  0.876096        0.826832                  0.836576               0.961344                       0.968224                            0.961968
1   2048.0   8192.0              3.924256                  3.750464        3.707200                  3.543456               3.832736                       3.802624                            3.846864
2   4096.0  14336.0             28.068720                 23.317489       13.498272                 12.827072              13.715808                      13.728800                           13.733168
moe-grouped-expert-gemm-forward-latency-production-num-assignments:
   num_assignments  Triton v1 Fixed (ms)  Triton v1 Autotuned (ms)  Triton v2 (ms)  Triton v2 Autotuned (ms)  Torch Grouped MM (ms)  Torch Grouped MM Compile (ms)  Torch Grouped MM Max Autotune (ms)
0           4096.0             28.083103                 23.303841       13.500800                 12.830448              13.727584                      13.737792                           13.730288
1           8192.0             57.021137                 42.280354       27.406240                 25.023104              27.303776                      27.312897                           27.295296
2          16384.0            111.500801                 79.902016       55.079519                 50.020386              47.814465                      47.815361                           47.822992
moe-grouped-expert-gemm-v2-forward-latency-group-size-production:
   num_assignments  GROUP_SIZE_M=1 (ms)  GROUP_SIZE_M=2 (ms)  GROUP_SIZE_M=4 (ms)  GROUP_SIZE_M=8 (ms)  GROUP_SIZE_M=16 (ms)  GROUP_SIZE_M=32 (ms)
0           4096.0            36.997601            19.088256            14.098112            13.519488             13.599200             13.578368
1           8192.0            73.935215            38.044912            28.551488            27.443296             26.567871             26.574944
2          16384.0           147.808197            75.984177            57.588192            54.750031             53.321182             52.378464
moe-grouped-expert-gemm-v2-forward-latency-group-size-imbalance:
   hot_expert_share  GROUP_SIZE_M=1 (ms)  GROUP_SIZE_M=2 (ms)  GROUP_SIZE_M=4 (ms)  GROUP_SIZE_M=8 (ms)  GROUP_SIZE_M=16 (ms)  GROUP_SIZE_M=32 (ms)
0             0.125             0.094400             0.094592             0.095584             0.098496              0.098528              0.098656
1             0.250             0.092944             0.092416             0.093216             0.095264              0.099936              0.100096
2             0.500             0.095712             0.095888             0.095648             0.096688              0.098912              0.102848
3             0.750             0.095648             0.095552             0.095552             0.096192              0.102336              0.106784
4             0.875             0.097088             0.096352             0.095648             0.095680              0.100928              0.109856
```
