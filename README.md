### Efficient GEMM implementation in CuTe DSL for the Hopper GPU

Just a learning exercise so i understand hardware better and design non-stupid architectures. CuTe DSL is really good. Everyone should try it.

5/28/2025
[X] Initial Version without much performance tuning

[X] Tiling + TMA to load tiles.

[X] Ensure Swizzled SMEM access for Tensor Cores

[X] Add pipelining to hide TMA bandwidth

5/29/2025
[] Add Epilogue

[] Tune Performance and get benchmarks

[] Allow arbitrary matrices that does not fit the tiling

FA-3 and NSA implementations in CuTe coming soon after. 

#### Setup:
1. Ensure that cuda-toolkit-12.9 is installed. Any other versions will not work with CuTe DSL
2. `pip install uv && uv pip install nvidia-cutlass-dsl torch`

#### Run:
```
python gemm.py
```
