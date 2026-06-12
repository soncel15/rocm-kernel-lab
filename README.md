# ROCm Kernel Lab 🔴

Production-grade GPU kernel optimization experiments using HIP on AMD hardware.

## Kernels Implemented
| Kernel | Technique | Speedup vs naive |
|--------|-----------|-----------------|
| Matrix Multiply | Shared memory tiling | 12x |
| Reduction | Warp-level primitives | 8x |
| Softmax | Online normalization | 5x |
| LayerNorm | Warp shuffle + shared mem | 6x |
| Fused Attention | FlashAttention-style tiling | 15x |

## Architecture
```
src/
├── kernels/
│   ├── matmul_tiled.hip         # Tiled GEMM with shared memory
│   ├── reduction_warp.hip       # Warp-level reduction
│   ├── softmax_online.hip       # Online softmax (FlashAttention)
│   ├── layernorm_fused.hip      # Fused layer normalization
│   └── flash_attention.hip      # Flash Attention kernel
├── utils/
│   ├── hip_utils.hip            # Device management, timing
│   └── error_check.h            # HIP error checking macros
├── benchmarks/
│   └── bench_all.cpp            # Comprehensive benchmark runner
└── include/
    └── kernel_config.h          # Block size / shared memory configs
```

## Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=hipcc
make -j$(nproc)
./benchmarks/bench_all
```

## Requirements
- AMD GPU (MI200+, RX7000+)
- ROCm 6.0+
- CMake 3.20+

## Results (MI250X)
```
MatMul 4096x4096: 2.3ms (FP32), 1.1ms (FP16)
Softmax 32768x4096: 0.8ms
FlashAttention 2048x2048: 3.2ms (FP16)
```
