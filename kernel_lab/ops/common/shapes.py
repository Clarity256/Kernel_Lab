from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpShape:
    batch: int
    seq_len: int
    hidden_size: int


SHAPE_PRESETS: dict[str, OpShape] = {
    "tiny": OpShape(batch=1, seq_len=128, hidden_size=256),
    "llm_prefill": OpShape(batch=2, seq_len=2048, hidden_size=4096),
    "llm_decode": OpShape(batch=8, seq_len=1, hidden_size=4096),
}


def get_shape(name: str) -> OpShape:
    try:
        return SHAPE_PRESETS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(SHAPE_PRESETS))
        raise KeyError(f"Unknown shape preset '{name}'. Expected one of: {valid}.") from exc

