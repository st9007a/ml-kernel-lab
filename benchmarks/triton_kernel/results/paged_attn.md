# Benchmark Result of Paged K/V Attention

* CUDA Version: 12.8
* Triton Version: 3.4.0
* Torch Version: 2.8.0+cu128

## RTX 2000 Ada

```
paged-attn-decode-forward-latency-seq-len:
   seq_len    Triton     Torch  Torch Compile
0    128.0  0.097856  0.059712       0.047392
1    256.0  0.135744  0.086336       0.076640
2    512.0  0.182176  0.146400       0.134976
3   1024.0  0.299168  0.265152       0.248512
4   2048.0  0.470528  0.437424       0.425888
5   4096.0  0.790016  0.769216       0.754512
paged-attn-decode-forward-latency-batch-size:
   batch_size    Triton     Torch  Torch Compile
0         1.0  0.167648  0.059616       0.048512
1         2.0  0.174400  0.096016       0.084320
2         4.0  0.200160  0.143600       0.131472
3         8.0  0.300032  0.265280       0.247072
4        16.0  0.468864  0.437120       0.423520
5        32.0  0.786208  0.767472       0.749376
paged-attn-decode-forward-latency-block-size:
   block_size    Triton     Torch  Torch Compile
0         8.0  0.314432  0.263296       0.247072
1        16.0  0.300416  0.263104       0.246944
2        32.0  0.283808  0.263008       0.246976
3        64.0  0.283776  0.263136       0.246912
paged-attn-decode-forward-latency-head-dim:
   head_dim    Triton    Torch  Torch Compile
0      64.0  0.184128  0.14848       0.146368
1     128.0  0.300352  0.26528       0.261824
```
