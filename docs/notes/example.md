# RMSNorm Framework Example

This note shows how to use the repository scaffold with `RMSNorm` as the first study operator.

## 1. Put the Torch baseline in the reference directory

File:

- `kernel_lab/ops/references/rmsnorm_ref.py`

This is the simplest readable implementation and should stay easy to verify:

```python
def rmsnorm_reference(x, weight, eps=1e-6):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_hat = x * torch.rsqrt(variance + eps)
    return x_hat * weight
```

Why start here:

1. It gives the correctness target for every accelerated backend.
2. It keeps the math visible while learning the operator.
3. It makes debugging Triton and CUDA versions much easier.

## 2. Register the operator once and reuse the same call path

Files:

- `kernel_lab/ops/references/rmsnorm_ref.py`
- `kernel_lab/ops/registry.py`

The reference implementation registers itself under the shared operator name `rmsnorm`. This means the rest of the repo can call:

```python
from kernel_lab.ops import registry

output = registry.run("rmsnorm", x, weight, preferred=("cuda", "triton", "reference"))
```

The call site does not need to know which backend is currently available. The registry selects the first available backend from the preference order.

## 3. Use the smoke script to see the framework in action

File:

- `scripts/rmsnorm_smoke.py`

Run on CPU:

```bash
python scripts/rmsnorm_smoke.py --backend reference --device cpu
```

Run on a CUDA machine and let the registry pick the best available backend:

```bash
python scripts/rmsnorm_smoke.py --backend auto --device cuda --dtype bfloat16
```

What the script demonstrates:

1. Build an input tensor and weight tensor.
2. Dispatch through the shared registry API.
3. Print the chosen backend and output shape.

## 4. Add correctness tests before optimizing

Files:

- `tests/test_correctness.py`
- `tests/test_grad.py`

Use these tests as the minimum bar before writing any Triton or CUDA kernel:

```bash
pytest tests/test_correctness.py -k rmsnorm
pytest tests/test_grad.py -k rmsnorm
```

The current tests cover:

1. Numerical equivalence of the reference implementation.
2. Gradient correctness through `gradcheck`.
3. Backend-to-reference comparison when an accelerated backend becomes available.

## 5. Benchmark through a dedicated benchmark script

File:

- `benchmarks/bench_rmsnorm.py`

Run a local CPU benchmark:

```bash
python benchmarks/bench_rmsnorm.py --backend reference --device cpu --shape tiny
```

Run on the NVIDIA server:

```bash
python benchmarks/bench_rmsnorm.py --backend auto --device cuda --shape llm_prefill --dtype bfloat16
```

Keep the benchmark script stable while changing the kernel implementation. That way you compare different versions with the same input generation and timing logic.

## 6. Add the Triton implementation next

File:

- `kernel_lab/ops/triton/rmsnorm.py`

Suggested workflow:

1. Keep the function signature aligned with `rmsnorm_reference`.
2. Implement the Triton kernel in this file.
3. Change `is_available` from `False` to a real runtime check.
4. Re-run the RMSNorm correctness tests.
5. Re-run `bench_rmsnorm.py` with `--backend triton`.

The important rule is that the Triton version should not invent a new interface. The framework works best when every backend shares one operator name and one Python-level call signature.

## 7. Add the CUDA implementation last

Files:

- `kernel_lab/ops/cuda/python_api.py`
- `kernel_lab/ops/cuda/csrc/rmsnorm.cu`
- `kernel_lab/ops/cuda/csrc/bindings.cpp`

Suggested workflow:

1. Keep the Python API thin and let it call into the built extension.
2. Put kernel code and binding code in `csrc/`.
3. Build on the Linux + NVIDIA machine:

```bash
python setup.py build_ext --inplace
```

4. Re-run:

```bash
pytest tests/test_correctness.py -k rmsnorm
pytest tests/test_grad.py -k rmsnorm
python benchmarks/bench_rmsnorm.py --backend cuda --device cuda --shape llm_prefill --dtype bfloat16
```

## 8. Profile only after correctness and timing look reasonable

Files:

- `scripts/profile_ncu.sh`
- `scripts/profile_nsys.sh`

Example:

```bash
bash scripts/profile_ncu.sh benchmarks/bench_rmsnorm.py --backend cuda --shape llm_prefill --dtype bfloat16
```

At this stage, record:

1. GPU model
2. input shape
3. dtype
4. average latency
5. occupancy or memory throughput

## 9. Reuse the same pattern for the next operator

For the next operator such as `softmax`, `rope`, or `swiglu`, repeat the same sequence:

1. reference
2. registry
3. correctness
4. grad
5. benchmark
6. Triton
7. CUDA
8. profile

This is the main value of the framework: every new operator follows the same structure, so the learning cost stays low.

