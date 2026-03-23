import sys
from pathlib import Path

from setuptools import setup


def build_cuda_extensions():
    from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA toolkit was not found. Build the extension on a Linux host with CUDA installed."
        )

    root = Path(__file__).parent
    csrc = root / "kernel_lab" / "ops" / "cuda" / "csrc"

    return [
        CUDAExtension(
            name="kernel_lab_cuda",
            sources=[
                str(csrc / "bindings.cpp"),
                str(csrc / "softmax.cu"),
                str(csrc / "rmsnorm.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ]


def should_build_cuda_extension() -> bool:
    return "build_ext" in sys.argv


if __name__ == "__main__":
    setup_kwargs = {"name": "kernel_lab_cuda"}

    if should_build_cuda_extension():
        try:
            from torch.utils.cpp_extension import BuildExtension
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Building the CUDA extension requires torch in the active environment."
            ) from exc

        setup_kwargs["ext_modules"] = build_cuda_extensions()
        setup_kwargs["cmdclass"] = {"build_ext": BuildExtension}

    setup(**setup_kwargs)
