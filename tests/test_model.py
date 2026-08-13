from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from kriti.model import KritiASR, logistic_probability, restore_danda


def test_restore_danda() -> None:
    assert restore_danda(" नमस्ते  नेपाल ", True) == "नमस्ते नेपाल।"
    assert restore_danda("नमस्ते।", True) == "नमस्ते।"
    assert restore_danda("नमस्ते", False) == "नमस्ते"


def test_logistic_probability() -> None:
    assert logistic_probability([0.0], [1.0], 0.0) == 0.5
    assert logistic_probability([1000.0], [1.0], 0.0) == 1.0
    assert logistic_probability([-1000.0], [1.0], 0.0) == 0.0
    assert math.isclose(logistic_probability([1.0, 2.0], [0.5, -0.25], 1.0), 0.7310585786)


def test_logistic_probability_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="aligned"):
        logistic_probability([1.0], [], 0.0)


def test_public_benchmark_snapshot_is_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    benchmark = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["scope"]["evaluated_systems"] == 19
    assert benchmark["scope"]["view_sha256"] == (
        "2374cac54831ce9c69282503763d7f1e12ada0404ae34ed471a7538cdae6c61f"
    )
    systems = benchmark["systems"]
    assert len(systems) == 19
    assert systems[0]["key"] == "kriti"
    assert systems[0]["rank"] == systems[1]["rank"] == 1
    assert systems[0]["raw_wer"] < min(system["raw_wer"] for system in systems[1:])


def test_encoder_features_are_extracted_before_decode() -> None:
    events: list[str] = []

    class FakeModel:
        def transcribe(self, *_args: object, **_kwargs: object) -> list[str]:
            events.append("decode")
            return ["नेपाल"]

    head = {
        "feature_count": 1_024,
        "coefficients": [0.0] * 1_024,
        "intercept": 0.0,
        "threshold": 1.0,
    }
    model = KritiASR(model=FakeModel(), head=head, device="cpu")

    def features(_paths: list[str], _batch_size: int) -> list[list[float]]:
        events.append("features")
        return [[0.0] * 1_024]

    model._encoder_features = features  # type: ignore[method-assign]
    assert model.transcribe("audio.wav") == "नेपाल"
    assert events == ["features", "decode"]
