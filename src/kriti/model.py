"""Inference loader for the Kriti Nepali RNNT package."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_REPO_ID = "harrrshall/kriti"
MODEL_FILENAME = "kriti.nemo"
HEAD_FILENAME = "punctuation_head.json"
MODEL_SHA256 = "0144854f0cc78f4b6115b75089fad632c39207d5256e53f92da996b9bbe43582"
HEAD_SHA256 = "5874b6fc6b4f1172dffa249a42f5054ffe196cff9b97854fe180eafc4134e9bb"
EXPECTED_ASR_PARAMETERS = 119_461_121
EXPECTED_HEAD_PARAMETERS = 1_025
EXPECTED_LIVE_PARAMETERS = EXPECTED_ASR_PARAMETERS + EXPECTED_HEAD_PARAMETERS
TERMINAL_PUNCTUATION = frozenset("।.!?！？\"'”’)]}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_danda(text: str, should_restore: bool) -> str:
    normalized = " ".join(text.split())
    if not should_restore or not normalized or normalized[-1] in TERMINAL_PUNCTUATION:
        return normalized
    return f"{normalized}।"


def logistic_probability(
    features: list[float], coefficients: list[float], intercept: float
) -> float:
    if len(features) != len(coefficients) or not features:
        raise ValueError("features and coefficients must be non-empty and aligned")
    logit = math.fsum(value * weight for value, weight in zip(features, coefficients, strict=True))
    logit += intercept
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _load_head(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "model_type": "binary_logistic_regression",
        "feature_extractor": "conformer_encoder_masked_mean_and_std_v1",
        "feature_count": 1_024,
        "parameter_count": EXPECTED_HEAD_PARAMETERS,
        "classes": [0, 1],
        "positive_class": 1,
        "seed": 20_260_813,
        "action": "append_devanagari_danda",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(f"punctuation head has an invalid {field}")
    coefficients = payload.get("coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != 1_024:
        raise RuntimeError("punctuation head has invalid coefficients")
    numeric = [float(value) for value in coefficients]
    intercept = float(payload.get("intercept"))
    threshold = float(payload.get("threshold"))
    if not all(math.isfinite(value) for value in [*numeric, intercept, threshold]):
        raise RuntimeError("punctuation head contains non-finite values")
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError("punctuation head threshold is outside [0, 1]")
    normalized = payload.copy()
    normalized.update(coefficients=numeric, intercept=intercept, threshold=threshold)
    return normalized


def _prediction_texts(result: object, expected: int) -> list[str]:
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, list) or len(result) != expected:
        raise RuntimeError("the rnnt decoder returned an unsupported result")
    texts: list[str] = []
    for item in result:
        text = getattr(item, "text", item)
        if not isinstance(text, str):
            raise RuntimeError("the rnnt decoder returned a non-text result")
        texts.append(text)
    return texts


def configure_deterministic_runtime(seed: int = 20_260_813) -> None:
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_compact_asr(
    model_path: str | Path,
    *,
    device: str | None = None,
    expected_sha256: str | None = MODEL_SHA256,
) -> tuple[Any, str]:
    import nemo.collections.asr as nemo_asr
    import torch

    model_path = Path(model_path)
    if expected_sha256 is not None and _sha256(model_path) != expected_sha256:
        raise RuntimeError("kriti.nemo sha-256 mismatch")
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    configure_deterministic_runtime()
    model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location="cpu", strict=False)
    if hasattr(model, "ctc_decoder"):
        del model.ctc_decoder
    output_heads = model.joint.joint_net[2]
    if "ne" not in output_heads:
        raise RuntimeError("kriti.nemo does not contain a Nepali joint head")
    for language in list(output_heads):
        if language != "ne":
            del output_heads[language]
    retained_rows = int(output_heads["ne"].out_features)
    if retained_rows != 257:
        raise RuntimeError("the Nepali rnnt head has an unexpected vocabulary size")
    embedding = model.decoder.prediction["embed"]
    if embedding.num_embeddings != retained_rows:
        compact = torch.nn.Embedding(retained_rows, embedding.embedding_dim)
        compact.weight.data.copy_(embedding.weight.data[:retained_rows])
        model.decoder.prediction["embed"] = compact
    model.cur_decoder = "rnnt"
    model.freeze()
    model = model.to(target)
    model.eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_ASR_PARAMETERS:
        raise RuntimeError(f"unexpected live asr parameter count: {parameters}")
    return model, target


def encoder_features(
    model: Any,
    paths: list[str],
    device: str,
    *,
    batch_size: int = 32,
) -> list[list[float]]:
    import numpy as np
    import soundfile as sf
    import torch

    if not paths or batch_size <= 0:
        raise ValueError("paths must be non-empty and batch_size must be positive")
    indexed = [(index, path, sf.info(path).frames) for index, path in enumerate(paths)]
    indexed.sort(key=lambda item: item[2])
    pooled: list[tuple[int, list[float]]] = []
    with torch.inference_mode():
        for offset in range(0, len(indexed), batch_size):
            batch = indexed[offset : offset + batch_size]
            waveforms = []
            for _, path, _ in batch:
                audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
                if sample_rate != 16_000:
                    raise ValueError(f"audio must be 16 khz: {path}")
                waveform = np.mean(audio, axis=1)
                if not waveform.size:
                    raise ValueError(f"audio is empty: {path}")
                waveforms.append(torch.from_numpy(waveform))
            lengths = torch.tensor([waveform.numel() for waveform in waveforms], device=device)
            signals = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True).to(device)
            processed, processed_lengths = model.preprocessor(
                input_signal=signals,
                length=lengths,
            )
            encoded, encoded_lengths = model.encoder(
                audio_signal=processed,
                length=processed_lengths,
            )
            steps = torch.arange(encoded.shape[-1], device=encoded.device)[None, :]
            mask = steps < encoded_lengths[:, None]
            weights = mask[:, None, :].to(encoded.dtype)
            denominator = encoded_lengths[:, None].to(encoded.dtype)
            mean = (encoded * weights).sum(dim=-1) / denominator
            variance = ((encoded - mean[:, :, None]).square() * weights).sum(dim=-1) / denominator
            vectors = torch.cat((mean, variance.sqrt()), dim=1).float().cpu().tolist()
            pooled.extend(
                (index, vector) for (index, _, _), vector in zip(batch, vectors, strict=True)
            )
    pooled.sort(key=lambda item: item[0])
    output = [vector for _, vector in pooled]
    if any(len(vector) != 1_024 for vector in output):
        raise RuntimeError("encoder feature width differs from the punctuation head")
    return output


@dataclass
class KritiASR:
    """Kriti model with Nepali RNNT decoding and acoustic danda restoration."""

    model: Any
    head: dict[str, Any]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        *,
        revision: str | None = None,
        device: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
    ) -> KritiASR:
        from huggingface_hub import hf_hub_download

        model_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=MODEL_FILENAME,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
            )
        )
        head_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=HEAD_FILENAME,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
            )
        )
        return cls.from_files(model_path, head_path, device=device)

    @classmethod
    def from_files(
        cls,
        model_path: str | Path,
        head_path: str | Path,
        *,
        device: str | None = None,
        verify_release_hashes: bool = True,
    ) -> KritiASR:
        model_path = Path(model_path)
        head_path = Path(head_path)
        if verify_release_hashes and _sha256(head_path) != HEAD_SHA256:
            raise RuntimeError("punctuation_head.json sha-256 mismatch")
        model, target = load_compact_asr(
            model_path,
            device=device,
            expected_sha256=MODEL_SHA256 if verify_release_hashes else None,
        )
        return cls(model=model, head=_load_head(head_path), device=target)

    def _encoder_features(self, paths: list[str], batch_size: int) -> list[list[float]]:
        pooled = encoder_features(self.model, paths, self.device, batch_size=batch_size)
        if any(len(vector) != self.head["feature_count"] for vector in pooled):
            raise RuntimeError("encoder feature width differs from the punctuation head")
        return pooled

    def transcribe(
        self, audio: str | Path | list[str | Path], *, batch_size: int = 32
    ) -> str | list[str]:
        single = isinstance(audio, (str, Path))
        paths = [str(audio)] if single else [str(path) for path in audio]
        if not paths or batch_size <= 0:
            raise ValueError("audio must be non-empty and batch_size must be positive")
        features = self._encoder_features(paths, batch_size)
        decoded: list[str] = []
        for offset in range(0, len(paths), batch_size):
            batch = paths[offset : offset + batch_size]
            decoded.extend(
                _prediction_texts(
                    self.model.transcribe(
                        batch,
                        batch_size=len(batch),
                        logprobs=False,
                        language_id="ne",
                    ),
                    len(batch),
                )
            )
        output = []
        for text, vector in zip(decoded, features, strict=True):
            probability = logistic_probability(
                vector, self.head["coefficients"], self.head["intercept"]
            )
            output.append(restore_danda(text, probability >= self.head["threshold"]))
        return output[0] if single else output


def load_model(
    repo_id: str = DEFAULT_REPO_ID,
    *,
    revision: str | None = None,
    device: str | None = None,
) -> KritiASR:
    return KritiASR.from_pretrained(repo_id, revision=revision, device=device)
