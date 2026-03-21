# Profiling Notes

Use profiling only on the Linux + NVIDIA server.

## Nsight Compute

```bash
bash scripts/profile_ncu.sh benchmarks/bench_softmax.py --backend cuda --shape llm_prefill
```

## Nsight Systems

```bash
bash scripts/profile_nsys.sh benchmarks/bench_rmsnorm.py --backend cuda --shape llm_prefill
```

## What to record

- GPU model
- CUDA version
- PyTorch version
- input shape
- dtype
- average latency
- kernel occupancy
- achieved memory throughput
