"""Runtime adapter contract.

An adapter turns a manifest's model/runtime into real requests and reports what the runtime
itself says about the loaded configuration. Adapters never invent a number: anything they
cannot observe is returned as None, and the caller records it as unavailable with a reason.

``executes_real_runtime`` is a class-level fact, not a parameter. It is True only for adapters
whose every request reaches a real inference runtime on the host being measured, and the
adapter id must also be listed in ``aiglab.evidence.EXECUTING_ADAPTERS``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from aiglab.manifest import Manifest


@dataclass(frozen=True)
class RuntimeStatus:
    id: str
    implemented: bool
    installed: bool
    running: bool
    version: str | None
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.implemented and self.installed and self.running

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "usable": self.usable}


@dataclass
class ChatRequest:
    id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    format: Any = None


@dataclass
class ChatResult:
    """One request/response. Durations are milliseconds; token counts as reported by the runtime."""

    id: str
    origin: str
    ok: bool
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    done_reason: str | None = None
    started_at: str | None = None
    #: client-side wall clock: request sent → first content or tool-call chunk received.
    ttft_ms: float | None = None
    #: "content" or "tool_calls". Runtimes may emit tool calls only once complete, so a first chunk
    #: of kind tool_calls is not a first-token latency; TTFT statistics use "content" requests only.
    first_chunk_kind: str | None = None
    #: client-side wall clock: request sent → final chunk received.
    wall_ms: float | None = None
    prompt_tokens: int | None = None
    #: prompt tokens the runtime served from its prefix cache (reported by the runtime).
    prompt_cached_tokens: int | None = None
    completion_tokens: int | None = None
    #: runtime-reported phase durations.
    prompt_eval_ms: float | None = None
    decode_ms: float | None = None
    load_ms: float | None = None
    runtime_total_ms: float | None = None

    @property
    def decode_tokens_per_s(self) -> float | None:
        if not self.completion_tokens or not self.decode_ms or self.decode_ms <= 0:
            return None
        return self.completion_tokens / (self.decode_ms / 1000.0)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "decode_tokens_per_s": self.decode_tokens_per_s}


@dataclass
class LoadedState:
    """What the runtime reports about the configuration actually loaded for the run."""

    model_resident: bool
    effective_context: int | None
    server_parallel_slots: int | None
    runtime_reported_model_bytes: int | None
    runtime_reported_gpu_bytes: int | None
    sample_pid: int | None
    sample_process: str | None
    artifact_path: str | None
    artifact_sha256_from_runtime: str | None
    other_resident_models: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


class RuntimeAdapter(Protocol):
    id: str
    adapter_version: str
    executes_real_runtime: bool

    def detect(self) -> RuntimeStatus: ...

    def resident_models(self) -> list[str]: ...

    def model_identity(self, manifest: Manifest) -> dict[str, Any]:
        """Runtime-reported identity of the requested model, without loading it."""
        ...

    def load(self, manifest: Manifest) -> ChatResult:
        """Load the model with the manifest's settings (a tiny request). Recorded as cold start."""
        ...

    def loaded_state(self, manifest: Manifest) -> LoadedState: ...

    def unload(self, manifest: Manifest) -> bool:
        """Release the manifest's model. The runner calls this only for a model the run itself
        loaded, restoring the host's prior state; it never unloads anything else."""
        ...

    def chat(self, manifest: Manifest, request: ChatRequest) -> ChatResult: ...

    def log_marker(self) -> Any:
        """Opaque position in the runtime's log, taken before a run."""
        ...

    def log_excerpt(self, marker: Any) -> str | None:
        """Memory-relevant runtime log lines written since ``marker``, or None."""
        ...
