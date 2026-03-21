# Roadmap

Use the same checklist for every operator:

1. Write the Torch reference implementation.
2. Add correctness coverage in `tests/test_correctness.py`.
3. Add gradient coverage if the op is differentiable.
4. Add or extend the benchmark script.
5. Add the Triton implementation.
6. Add the CUDA implementation and bindings.
7. Profile the final version on the target GPU.

Suggested first operators:

- softmax
- rmsnorm
- rope
- swiglu
- attention

