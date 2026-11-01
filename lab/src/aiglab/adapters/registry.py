"""Known runtimes. Only adapters with real implementations can run; the rest are detected and
reported honestly as not implemented, so a manifest naming them fails at dry-run, not mid-run."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any

from aiglab.adapters.base import RuntimeAdapter, RuntimeStatus


@dataclass(frozen=True)
class RuntimeInfo:
    id: str
    description: str
    binaries: tuple[str, ...]
    implemented: bool
    test_only: bool = False


RUNTIMES: dict[str, RuntimeInfo] = {
    "ollama": RuntimeInfo("ollama", "Ollama (llama.cpp engine) via its local HTTP API", ("ollama",), True),
    "llama_cpp": RuntimeInfo(
        "llama_cpp", "llama.cpp llama-server / llama-bench", ("llama-server", "llama-bench"), False
    ),
    "mlx_lm": RuntimeInfo("mlx_lm", "Apple MLX-LM", ("mlx_lm.generate", "mlx_lm.server"), False),
    "lmstudio": RuntimeInfo("lmstudio", "LM Studio (lms CLI / local server)", ("lms",), False),
    "vllm": RuntimeInfo("vllm", "vLLM OpenAI-compatible server", ("vllm",), False),
    "fixture": RuntimeInfo("fixture", "Deterministic scripted responses (tests only; never evidence)", (), True, True),
}


def runtime_ids() -> frozenset[str]:
    return frozenset(RUNTIMES)


def detect_all() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for info in RUNTIMES.values():
        if info.id == "ollama":
            status = get_adapter("ollama").detect()
        elif info.test_only:
            status = RuntimeStatus(info.id, True, True, True, "fixture", "test-only adapter; never evidence")
        else:
            found = [b for b in info.binaries if shutil.which(b)]
            status = RuntimeStatus(
                id=info.id,
                implemented=False,
                installed=bool(found),
                running=False,
                version=None,
                reason=(
                    "detected but no adapter is implemented yet"
                    if found
                    else "not installed on this host; no adapter implemented"
                ),
                detail={"binaries_found": found},
            )
        out.append({**status.as_dict(), "description": info.description, "test_only": info.test_only})
    return out


def get_adapter(runtime_id: str, **kwargs: Any) -> RuntimeAdapter:
    if runtime_id == "ollama":
        from aiglab.adapters.ollama import OllamaAdapter

        return OllamaAdapter(**kwargs)
    if runtime_id == "fixture":
        from aiglab.adapters.fixture import FixtureAdapter

        return FixtureAdapter(**kwargs)
    info = RUNTIMES.get(runtime_id)
    reason = "unknown runtime" if info is None else "no adapter implemented for this runtime yet"
    raise NotImplementedError(f"{runtime_id}: {reason}")
