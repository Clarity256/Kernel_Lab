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


if __name__ == "__main__":
    from torch.utils.cpp_extension import BuildExtension

    setup(
        name="kernel_lab_cuda",
        ext_modules=build_cuda_extensions(),
        cmdclass={"build_ext": BuildExtension},
    )

