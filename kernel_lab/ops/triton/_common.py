from __future__ import annotations

import importlib.util
import platform

import torch


def triton_runtime_available() -> bool:
    return (
        platform.system() == "Linux"
        and torch.cuda.is_available()
        and importlib.util.find_spec("triton") is not None
    )

