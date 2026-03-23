# Triton RMSNorm Walkthrough

This note demonstrates how a Triton `RMSNorm` implementation fits into the current framework.

## Goal

Keep the first Triton kernel simple enough to read:

1. one program handles one row
2. reduce over the last dimension
3. reuse the same Python signature as the reference implementation
4. let the registry dispatch to Triton only when the runtime is actually available

## Step 1: Start from the reference contract

Reference file:

- `kernel_lab/ops/references/rmsnorm_ref.py`

The Triton backend keeps the same interface:

```python
def rmsnorm_triton(x, weight, eps=1e-6):
    ...
```

This matters because the shared call site stays unchanged:

```python
registry.run("rmsnorm", x, weight, preferred=("cuda", "triton", "reference"))
```

## Step 2: Guard Triton imports

File:

- `kernel_lab/ops/triton/rmsnorm.py`

The module first tries to import Triton. If Triton is missing, the file still imports cleanly and the backend simply reports itself as unavailable.

That is the reason the registry can safely load all backends on macOS and CPU-only environments.

## Step 3: Implement a minimal row-wise RMSNorm kernel

The kernel uses one Triton program per row:

1. load one row of `x`
2. compute `sum(x * x)` in fp32
3. compute `inv_rms = rsqrt(mean_square + eps)`
4. multiply by `weight`
5. store the result

The implementation keeps the code intentionally small. It is a study kernel, not an optimized production kernel.

## Step 4: Flatten to 2D in Python

The Python wrapper does:

1. validate device and shape
2. flatten input to `[rows, hidden_size]`
3. launch the Triton kernel with `grid = (rows,)`
4. reshape the output back to the original tensor shape

This is a useful pattern for many row-wise operators such as:

- RMSNorm
- LayerNorm
- Softmax

## Step 5: Register the Triton backend

The Triton file ends with:

```python
register(
    OperatorImplementation(
        name="rmsnorm",
        backend="triton",
        fn=rmsnorm_triton,
        is_available=_triton_available,
        description="Minimal Triton RMSNorm kernel for framework study.",
    )
)
```

This means:

1. the backend participates in the shared registry
2. `registry.run("rmsnorm", ...)` can dispatch to it
3. tests and benchmarks do not need a special Triton-only call path

## Step 6: Keep auto-dispatch device-aware

One framework detail matters here: `auto` dispatch should not choose Triton or CUDA for CPU tensors.

The helper in `kernel_lab/ops/common/utils.py` now uses:

```python
preferred_backends_for_device(torch.device("cpu")) == ("reference",)
preferred_backends_for_device(torch.device("cuda")) == ("cuda", "triton", "reference")
```

That change is used by:

- `benchmarks/bench_rmsnorm.py`
- `benchmarks/bench_softmax.py`
- `scripts/rmsnorm_smoke.py`

## Step 7: Validate correctness

Run on a CUDA machine with Triton installed:

```bash
pytest tests/test_correctness.py -k rmsnorm
pytest tests/test_grad.py -k rmsnorm
python benchmarks/bench_rmsnorm.py --backend triton --device cuda --shape tiny
```

The optional backend correctness test compares the Triton output against the reference implementation on the correct device.

## Step 8: What this example does not optimize yet

This first Triton kernel is intentionally conservative. It does not yet include:

- autotuning
- vectorized loads
- fused residual or bias
- split reductions for very large hidden sizes
- architecture-specific heuristics

That is the right tradeoff for the first framework example. Readability comes first, then optimization work can happen in measured steps.
