"""Ollama adapter (real runtime).

Verified against Ollama 0.35.0 on macOS, where the server spawns a bundled ``llama-server``
child per loaded model with ``--model <blob> -c <ctx*slots> -np <slots>``. Identity is checked
end to end from files on disk, not from API claims alone:

  /api/tags digest == sha256(manifest file)  → manifest layer digest == blob in the process argv
  → sha256(blob) == layer digest (re-hashed unless the operator skips it).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiglab.adapters.base import ChatRequest, ChatResult, LoadedState, RuntimeStatus
from aiglab.manifest import Manifest

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
KEEP_ALIVE = "15m"
LOG_PATTERNS = re.compile(
    r"buffer size|KV buffer|llama_kv_cache: size|gpu memory|offloaded|n_ctx_seq|n_seq_max|"
    r"out of memory|failed to allocate|error",
    re.I,
)

ProcessLister = Callable[[], list[tuple[int, int, str]]]


def list_processes() -> list[tuple[int, int, str]]:
    try:
        out = subprocess.run(
            ["ps", "-axww", "-o", "pid=,ppid=,args="], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows: list[tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


def split_ref(ref: str) -> tuple[str, str, str]:
    """'qwen2.5:7b-instruct' -> ('library', 'qwen2.5', '7b-instruct'); 'user/m' -> ('user', 'm', 'latest')."""
    name, _, tag = ref.partition(":")
    namespace, _, model = name.rpartition("/")
    return (namespace or "library"), model, (tag or "latest")


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _flag(argv: Sequence[str], *names: str) -> str | None:
    for i, token in enumerate(argv[:-1]):
        if token in names:
            return argv[i + 1]
    return None


class OllamaAdapter:
    id = "ollama"
    adapter_version = "1.0.0"
    executes_real_runtime = True

    def __init__(
        self,
        endpoint: str | None = None,
        models_dir: Path | None = None,
        log_path: Path | None = None,
        process_lister: ProcessLister = list_processes,
        verify_artifact_hash: bool = True,
    ) -> None:
        self.endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
        self.models_dir = models_dir or Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama/models")
        self.log_path = log_path or Path.home() / ".ollama/logs/server.log"
        self.process_lister = process_lister
        self.verify_artifact_hash = verify_artifact_hash
        self._identity_cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ http
    def _get(self, path: str, timeout: float = 5.0) -> Any:
        with urllib.request.urlopen(self.endpoint + path, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, body: dict[str, Any], timeout: float = 30.0) -> Any:
        req = urllib.request.Request(
            self.endpoint + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _stream(self, path: str, body: dict[str, Any], timeout: float) -> Iterator[dict[str, Any]]:
        req = urllib.request.Request(
            self.endpoint + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if line.strip():
                    yield json.loads(line)

    # ------------------------------------------------------------------ detection
    def detect(self) -> RuntimeStatus:
        binary = shutil.which("ollama")
        try:
            version = str(self._get("/api/version", timeout=3).get("version"))
            running = True
            reason = None
        except (OSError, urllib.error.URLError, ValueError) as exc:
            version, running = None, False
            reason = f"Ollama API not reachable at {self.endpoint}: {type(exc).__name__}"
        if binary is None and not running:
            reason = "ollama binary not on PATH and API not reachable"
        return RuntimeStatus(
            id=self.id,
            implemented=True,
            installed=binary is not None or running,
            running=running,
            version=version,
            reason=reason,
            detail={"binary": binary, "endpoint": self.endpoint},
        )

    def available_models(self) -> list[dict[str, Any]]:
        return list(self._get("/api/tags").get("models") or [])

    def resident_models(self) -> list[str]:
        return [str(m.get("name")) for m in self._get("/api/ps").get("models") or []]

    # ------------------------------------------------------------------ identity
    def _manifest_path(self, ref: str) -> Path:
        namespace, model, tag = split_ref(ref)
        return self.models_dir / "manifests" / "registry.ollama.ai" / namespace / model / tag

    def model_identity(self, manifest: Manifest) -> dict[str, Any]:
        ref = manifest.model.runtime_ref
        if ref in self._identity_cache:
            return self._identity_cache[ref]
        tags = {str(m.get("name")): m for m in self.available_models()}
        tag = tags.get(ref) or tags.get(f"{ref}:latest")
        if tag is None:
            return {"present": False, "reason": f"model {ref!r} is not pulled in this Ollama instance"}
        show = self._post("/api/show", {"model": ref})
        details = show.get("details") or {}
        info = show.get("model_info") or {}
        ctx_keys = [k for k in info if k.endswith(".context_length")]
        identity: dict[str, Any] = {
            "present": True,
            "ref": ref,
            "api_digest": f"sha256:{tag.get('digest')}",
            "api_size_bytes": tag.get("size"),
            "format": details.get("format"),
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "parameter_count": info.get("general.parameter_count"),
            "trained_context_length": info.get(ctx_keys[0]) if ctx_keys else None,
            "capabilities": show.get("capabilities") or [],
            "parent_model": details.get("parent_model") or None,
        }
        mpath = self._manifest_path(ref)
        if mpath.is_file():
            raw = mpath.read_bytes()
            identity["manifest_file_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
            layers = json.loads(raw).get("layers") or []
            model_layers = [lyr for lyr in layers if lyr.get("mediaType") == "application/vnd.ollama.image.model"]
            if model_layers:
                identity["artifact_digest"] = model_layers[0]["digest"]
                identity["artifact_size_bytes"] = model_layers[0].get("size")
                identity["artifact_path"] = str(self.models_dir / "blobs" / model_layers[0]["digest"].replace(":", "-"))
        else:
            identity["manifest_file_sha256"] = None
            identity["manifest_reason"] = f"manifest file not found at {mpath}"
        self._identity_cache[ref] = identity
        return identity

    def verify_artifact(self, identity: dict[str, Any]) -> dict[str, Any]:
        path = identity.get("artifact_path")
        if not path or not Path(path).is_file():
            return {"ok": False, "detail": "artifact blob not found on disk"}
        if not self.verify_artifact_hash:
            return {"ok": True, "detail": "not re-hashed (operator skipped); blob is content-addressed by name"}
        started = time.perf_counter()
        actual = "sha256:" + sha256_file(Path(path))
        took = round(time.perf_counter() - started, 2)
        return {
            "ok": actual == identity.get("artifact_digest"),
            "detail": f"re-hashed {identity.get('artifact_size_bytes')} bytes in {took}s: {actual}",
        }

    # ------------------------------------------------------------------ requests
    def _options(self, manifest: Manifest) -> dict[str, Any]:
        options: dict[str, Any] = dict(manifest.runtime.options)
        options.update(
            {
                "num_ctx": manifest.context,
                "temperature": manifest.temperature,
                "seed": manifest.seed,
                "num_predict": manifest.max_output_tokens,
            }
        )
        return options

    def load(self, manifest: Manifest) -> ChatResult:
        return self.chat(
            manifest,
            ChatRequest(id="load", messages=[{"role": "user", "content": "Reply with the single word: ready"}]),
        )

    def unload(self, manifest: Manifest) -> bool:
        try:
            self._post("/api/generate", {"model": manifest.model.runtime_ref, "keep_alive": 0}, timeout=60)
        except (OSError, urllib.error.URLError, ValueError):
            return False
        return True

    def chat(self, manifest: Manifest, request: ChatRequest) -> ChatResult:
        body: dict[str, Any] = {
            "model": manifest.model.runtime_ref,
            "messages": request.messages,
            "stream": True,
            "keep_alive": KEEP_ALIVE,
            "options": self._options(manifest),
        }
        if request.tools:
            body["tools"] = request.tools
        if request.format is not None:
            body["format"] = request.format
        result = ChatResult(id=request.id, origin="executed", ok=False)
        result.started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
        t0 = time.perf_counter()
        content: list[str] = []
        final: dict[str, Any] = {}
        try:
            for chunk in self._stream("/api/chat", body, timeout=manifest.request_timeout_s):
                if "error" in chunk:
                    result.error = f"runtime error: {chunk['error']}"
                    break
                msg = chunk.get("message") or {}
                if result.ttft_ms is None and (msg.get("content") or msg.get("tool_calls")):
                    result.ttft_ms = (time.perf_counter() - t0) * 1000
                    result.first_chunk_kind = "content" if msg.get("content") else "tool_calls"
                if msg.get("content"):
                    content.append(msg["content"])
                if msg.get("tool_calls"):
                    result.tool_calls.extend(msg["tool_calls"])
                if chunk.get("done"):
                    final = chunk
                if time.perf_counter() - t0 > manifest.request_timeout_s:
                    result.error = f"request exceeded {manifest.request_timeout_s}s"
                    break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            result.error = f"HTTP {exc.code}: {detail}"
        except (OSError, urllib.error.URLError, ValueError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        result.wall_ms = (time.perf_counter() - t0) * 1000
        result.content = "".join(content)
        if final:
            result.done_reason = final.get("done_reason")
            result.prompt_tokens = final.get("prompt_eval_count")
            result.prompt_cached_tokens = final.get("prompt_eval_cached_count")
            result.completion_tokens = final.get("eval_count")
            result.prompt_eval_ms = _ns_to_ms(final.get("prompt_eval_duration"))
            result.decode_ms = _ns_to_ms(final.get("eval_duration"))
            result.load_ms = _ns_to_ms(final.get("load_duration"))
            result.runtime_total_ms = _ns_to_ms(final.get("total_duration"))
        result.ok = result.error is None and bool(final)
        if not final and result.error is None:
            result.error = "stream ended without a final chunk"
        return result

    # ------------------------------------------------------------------ loaded state
    def loaded_state(self, manifest: Manifest) -> LoadedState:
        ref = manifest.model.runtime_ref
        resident = self._get("/api/ps").get("models") or []
        mine = [m for m in resident if m.get("name") in (ref, f"{ref}:latest")]
        others = [str(m.get("name")) for m in resident if m not in mine]
        identity = self.model_identity(manifest)
        blob = identity.get("artifact_path")
        servers = [
            (pid, args) for pid, _ppid, args in self.process_lister() if "llama-server" in args or " runner " in args
        ]
        target = [(pid, args) for pid, args in servers if blob and blob in args]
        pid = target[0][0] if len(target) == 1 else None
        args = target[0][1].split() if len(target) == 1 else []
        total_ctx = _int(_flag(args, "-c", "--ctx-size"))
        slots = _int(_flag(args, "-np", "--parallel"))
        model_arg = _flag(args, "--model", "-m")
        m0 = mine[0] if mine else {}
        ps_ctx = m0.get("context_length")
        effective = ps_ctx if isinstance(ps_ctx, int) else (total_ctx // slots if total_ctx and slots else None)
        return LoadedState(
            model_resident=bool(mine),
            effective_context=effective,
            server_parallel_slots=slots,
            runtime_reported_model_bytes=m0.get("size"),
            runtime_reported_gpu_bytes=m0.get("size_vram"),
            sample_pid=pid,
            sample_process="llama-server" if pid else None,
            artifact_path=model_arg,
            artifact_sha256_from_runtime=(
                "sha256:" + Path(model_arg).name.removeprefix("sha256-") if model_arg else None
            ),
            other_resident_models=others,
            detail={
                "api_ps_digest": f"sha256:{m0.get('digest')}" if m0.get("digest") else None,
                "server_total_ctx": total_ctx,
                "matching_server_processes": len(target),
                "server_processes_seen": len(servers),
            },
        )

    # ------------------------------------------------------------------ logs
    def log_marker(self) -> int | None:
        try:
            return self.log_path.stat().st_size
        except OSError:
            return None

    def log_excerpt(self, marker: Any) -> str | None:
        if not isinstance(marker, int):
            return None
        try:
            with self.log_path.open("rb") as fh:
                fh.seek(marker)
                text = fh.read(4 * 1024 * 1024).decode("utf-8", "replace")
        except OSError:
            return None
        lines = [line for line in text.splitlines() if LOG_PATTERNS.search(line)]
        return "\n".join(lines[-400:])


def _ns_to_ms(value: Any) -> float | None:
    return value / 1e6 if isinstance(value, (int, float)) else None


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def parse_runtime_memory(log: str | None) -> dict[str, Any]:
    """Runtime-reported memory breakdown (MiB) from llama.cpp log lines; last occurrence wins."""
    if not log:
        return {}
    out: dict[str, Any] = {}
    kv = re.findall(r"KV buffer size\s*=\s*([\d.]+) MiB", log)
    if kv:
        out["kv_cache_mib"] = float(kv[-1])
    model = re.findall(r"(\S+) model buffer size\s*=\s*([\d.]+) MiB", log)
    if model:
        out["model_buffers_mib"] = {name: float(v) for name, v in model[-4:]}
    compute = re.findall(r"(\S+) compute buffer size\s*=\s*([\d.]+) MiB", log)
    if compute:
        out["compute_buffers_mib"] = {name: float(v) for name, v in compute[-4:]}
    gpu = re.findall(r'msg="gpu memory".*?available="([^"]+)".*?free="([^"]+)"', log)
    if gpu:
        out["gpu_memory_reported"] = {"available": gpu[-1][0], "free": gpu[-1][1]}
    if re.search(r"out of memory|failed to allocate", log, re.I):
        out["allocation_failure_logged"] = True
    return out
