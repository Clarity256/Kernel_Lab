# Kernel Lab

Kernel Lab is a small research playground for learning and comparing operator implementations across three stages:

1. Torch reference implementation
2. Triton implementation
3. CUDA extension implementation

Every operator follows the same workflow: correctness first, then benchmarks, then profiling.

## Design goals

- Keep the code readable enough for study and iteration.
- Make local macOS development possible without requiring a GPU.
- Keep the remote Linux + NVIDIA validation path obvious and repeatable.

## Repository layout

```text
kernel_lab/
├── ops/
│   ├── registry.py
│   ├── references/
│   ├── triton/
│   ├── cuda/
│   └── common/
├── tests/
├── benchmarks/
├── scripts/
└── docs/
```

## Development loop

Implement one operator at a time.

1. Start in `kernel_lab/ops/references/` with the Torch baseline.
2. Add the Triton version in `kernel_lab/ops/triton/`.
3. Add the CUDA binding and kernels in `kernel_lab/ops/cuda/`.
4. Validate with `pytest`.
5. Compare with the benchmark scripts.
6. Profile on the Linux + NVIDIA server with `ncu` or `nsys`.

## Quick start

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

Run the default tests:

```bash
pytest
```

Run a baseline benchmark on CPU:

```bash
python benchmarks/bench_softmax.py --backend reference --device cpu
```

When you are on a Linux + NVIDIA machine, you can build the CUDA extension with:

```bash
python setup.py build_ext --inplace
```

## Current sample operators

- `softmax`
- `rmsnorm`
- `rope`
- `swiglu` placeholder
- `attention_toy` placeholder

The Triton and CUDA directories currently provide templates and integration points, not finished optimized kernels.

