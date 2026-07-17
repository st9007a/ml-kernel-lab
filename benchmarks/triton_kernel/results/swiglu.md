# Benchmark Result of SwiGLU

* CUDA Version: 12.8
* Triton Version: 3.4.0
* Torch Version: 2.8.0+cu128
* GPU: RTX 2000 Ada

```
swiglu-forward-latency-intermediate-size:
   intermediate_size    Triton     Torch  Torch Compile
0             8192.0  0.006208  0.008032       0.005568
1            11008.0  0.005632  0.008640       0.006272
2            14336.0  0.006400  0.008672       0.006432
3            18944.0  0.005856  0.008928       0.007136
4            28672.0  0.008336  0.010592       0.007936
5            57344.0  0.009472  0.012544       0.008592
swiglu-forward-latency-batch-size:
   batch_size    Triton     Torch  Torch Compile
0         1.0  0.006400  0.010448       0.007872
1         2.0  0.009440  0.012448       0.009408
2         4.0  0.011168  0.013184       0.010992
3         8.0  0.014336  0.016416       0.015136
4        16.0  0.022464  0.026112       0.024672
```
