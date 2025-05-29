### Efficient GEMM implementation in CuTe DSL for the Hopper GPU

[X] Initial Version without much performance tuning
[X] Tiling + TMA to load tiles.
[X] Ensure Swizzled SMEM access for Tensor Cores
[X] Add pipelining to hide TMA bandwidth
[] Add Epilogue
[] Tune Performance and get benchmarks

#### Setup:
1. Ensure that cuda-toolkit-12.9 is installed. Any other versions will not work with CuTe DSL
2. `pip install uv && uv pip install nvidia-cutlass-dsl torch`

#### Run:
```
python gemm.py
```
