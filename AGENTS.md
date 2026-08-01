# Project Guidance

This is a learning project. Provide hints and reviews for kernel implementations, but do not implement or modify them. Help implement unit tests and benchmarks.

1. Do not make changes under `src/`.
2. Files outside `src/` may be edited.
3. When reviewing implementations or suggesting changes, avoid device synchronization in GPU building blocks because it creates serious performance problems.
4. Starting with a simple, less-performant kernel is acceptable. Guide optimization incrementally toward a production-ready kernel.
5. Benchmark target functions must not contain device synchronization, tensor-to-host reads, or CUDA-tensor assertions. Any synchronization required by the benchmark harness must remain outside the measured target function.
