"""Fixture adapter: deterministic scripted responses for tests and pipeline dry-runs.

It never contacts a runtime. Every result it produces carries ``origin="fixture"``, the adapter
is not in EXECUTING_ADAPTERS, and ``executes_real_runtime`` is False, so three independent
checks keep its output out of aig_tested.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aiglab.adapters.base import ChatRequest, ChatResult, LoadedState, RuntimeStatus
from aiglab.manifest import Manifest

Responder = Callable[[ChatRequest], ChatResult]


def default_responder(request: ChatRequest) -> ChatResult:
    return ChatResult(
        id=request.id,
        origin="fixture",
        ok=True,
        content="fixture",
        done_reason="stop",
        ttft_ms=100.0,
        first_chunk_kind="content",
        wall_ms=500.0,
        prompt_tokens=50,
        completion_tokens=20,
        prompt_eval_ms=40.0,
        decode_ms=400.0,
        load_ms=0.0,
        runtime_total_ms=480.0,
    )


class FixtureAdapter:
    id = "fixture"
    adapter_version = "1.0.0"
    executes_real_runtime = False

    def __init__(self, responder: Responder | None = None) -> None:
        self.responder = responder or default_responder
        self.requests: list[ChatRequest] = []

    def detect(self) -> RuntimeStatus:
        return RuntimeStatus(
            id=self.id,
            implemented=True,
            installed=True,
            running=True,
            version="fixture",
            reason="test-only adapter; never evidence",
        )

    def resident_models(self) -> list[str]:
        return []

    def model_identity(self, manifest: Manifest) -> dict[str, Any]:
        return {"present": True, "ref": manifest.model.runtime_ref, "fixture": True}

    def verify_artifact(self, identity: dict[str, Any]) -> dict[str, Any]:
        return {"ok": False, "detail": "fixture adapter has no artifact"}

    def load(self, manifest: Manifest) -> ChatResult:
        return self.chat(manifest, ChatRequest(id="load", messages=[]))

    def loaded_state(self, manifest: Manifest) -> LoadedState:
        return LoadedState(
            model_resident=True,
            effective_context=manifest.context,
            server_parallel_slots=manifest.concurrency,
            runtime_reported_model_bytes=None,
            runtime_reported_gpu_bytes=None,
            sample_pid=None,
            sample_process=None,
            artifact_path=None,
            artifact_sha256_from_runtime=None,
        )

    def unload(self, manifest: Manifest) -> bool:
        return True

    def chat(self, manifest: Manifest, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        result = self.responder(request)
        result.origin = "fixture"  # a responder cannot launder fixture output into another origin
        return result

    def log_marker(self) -> Any:
        return None

    def log_excerpt(self, marker: Any) -> str | None:
        return None
