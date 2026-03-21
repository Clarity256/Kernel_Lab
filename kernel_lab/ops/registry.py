from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

Backend = Literal["reference", "triton", "cuda"]


@dataclass(frozen=True)
class OperatorImplementation:
    name: str
    backend: Backend
    fn: Callable[..., object]
    is_available: Callable[[], bool]
    description: str


_REGISTRY: dict[str, dict[Backend, OperatorImplementation]] = {}
_DEFAULTS_LOADED = False


def register(implementation: OperatorImplementation) -> None:
    bucket = _REGISTRY.setdefault(implementation.name, {})
    bucket[implementation.backend] = implementation


def load_default_registry() -> None:
    global _DEFAULTS_LOADED

    if _DEFAULTS_LOADED:
        return

    from kernel_lab.ops.references import rope_ref, rmsnorm_ref, softmax_ref  # noqa: F401
    from kernel_lab.ops.triton import attention_toy, rmsnorm, softmax, swiglu  # noqa: F401
    from kernel_lab.ops.cuda import python_api  # noqa: F401

    _DEFAULTS_LOADED = True


def list_ops() -> list[str]:
    load_default_registry()
    return sorted(_REGISTRY)


def implementations(name: str) -> dict[Backend, OperatorImplementation]:
    load_default_registry()
    if name not in _REGISTRY:
        raise KeyError(f"Operator '{name}' is not registered.")
    return dict(_REGISTRY[name])


def available_backends(name: str) -> list[Backend]:
    return [
        backend
        for backend, implementation in implementations(name).items()
        if implementation.is_available()
    ]


def get(
    name: str,
    *,
    backend: Backend | None = None,
    preferred: tuple[Backend, ...] = ("cuda", "triton", "reference"),
    require_available: bool = True,
) -> OperatorImplementation:
    choices = implementations(name)

    if backend is not None:
        implementation = choices.get(backend)
        if implementation is None:
            raise KeyError(f"Operator '{name}' does not have backend '{backend}'.")
        if require_available and not implementation.is_available():
            raise RuntimeError(f"Backend '{backend}' for '{name}' is registered but unavailable.")
        return implementation

    for candidate in preferred:
        implementation = choices.get(candidate)
        if implementation is None:
            continue
        if implementation.is_available():
            return implementation

    if require_available:
        raise RuntimeError(f"No available backend found for operator '{name}'.")

    first_backend = next(iter(choices))
    return choices[first_backend]


def run(
    name: str,
    *args,
    backend: Backend | None = None,
    preferred: tuple[Backend, ...] = ("cuda", "triton", "reference"),
    **kwargs,
):
    implementation = get(name, backend=backend, preferred=preferred)
    return implementation.fn(*args, **kwargs)

