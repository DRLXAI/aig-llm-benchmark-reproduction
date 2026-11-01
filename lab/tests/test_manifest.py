from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiglab.manifest import ManifestError, load_manifest, manifest_hash, parse_manifest
from tests.conftest import manifest_dict

MANIFESTS = Path(__file__).resolve().parents[1] / "manifests"


def errors(raw: object) -> list[str]:
    with pytest.raises(ManifestError) as exc:
        parse_manifest(raw)
    return exc.value.errors


def test_valid_manifest_parses() -> None:
    m = parse_manifest(manifest_dict())
    assert (m.context, m.concurrency, m.warmup, m.measured) == (8192, 1, 1, 3)
    assert m.model.quantization == "Q4_K_M"
    assert len(m.sha256) == 64


def test_hash_ignores_key_order() -> None:
    raw = manifest_dict()
    reordered = dict(reversed(list(raw.items())))
    assert manifest_hash(raw) == manifest_hash(reordered)


def test_reports_every_problem_at_once() -> None:
    raw = manifest_dict()
    del raw["model"]["family"]
    raw["config"]["context"] = 10
    raw["config"]["concurrency"] = 0
    raw["repetitions"]["measured"] = 0
    errs = errors(raw)
    assert len(errs) >= 4
    assert any("model.family" in e for e in errs)


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"runtime__id": "tensorrt"}, "not a known runtime"),
        ({"workload__id": "poetry"}, "not a known workload"),
        ({"workload__version": "0.9.0"}, "pins '0.9.0'"),
        ({"methodology_version": "2.0.0"}, "not supported"),
        ({"runtime__endpoint": "https://api.example.com"}, "loopback"),
        ({"target__kind": "cloud"}, "target.provider is required"),
        ({"target__hourly_usd": 1.2}, "price_source_url"),
        ({"target__memory_kind": "hbm9"}, "memory_kind"),
        ({"config__temperature": 3}, "temperature"),
        ({"model__expected_artifact_digest": "md5:abc"}, "sha256:"),
        ({"id": "bad id!"}, "id may contain"),
        ({"surprise": True}, "unknown top-level keys"),
        ({"manifest_version": 2}, "manifest_version"),
    ],
)
def test_rejections(overrides: dict[str, object], fragment: str) -> None:
    assert any(fragment in e for e in errors(manifest_dict(**overrides))), overrides


def test_non_object_rejected() -> None:
    assert errors([1, 2]) == ["manifest must be a JSON object"]


def test_unreadable_file(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text("{not json")
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_every_committed_manifest_is_valid() -> None:
    paths = sorted(MANIFESTS.glob("*.json"))
    assert paths, "expected committed manifests"
    for p in paths:
        m = load_manifest(p)
        assert m.id == p.stem
        assert json.loads(p.read_text())["runtime"]["id"] != "fixture", "fixture manifests must not be committed"
