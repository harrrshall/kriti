"""Export the Nepali-only Kriti NeMo archive from the pinned base model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

BASE_SHA256 = "8c75fee5d82e61d31ddc4e891875617f730b4dfef0f4b226ff1e576bf4143039"
EXPECTED_SAVED_PARAMETERS = 122_901_761


def sanitize_config(model: object) -> None:
    """Remove source-machine paths while preserving registered NeMo artifacts."""
    from omegaconf import open_dict

    with open_dict(model.cfg):
        for dataset_key in ("train_ds", "validation_ds", "test_ds"):
            dataset = model.cfg.get(dataset_key)
            if dataset is not None and "manifest_filepath" in dataset:
                dataset.manifest_filepath = None
        for tokenizer in model.cfg.tokenizer.langs.values():
            if tokenizer.get("dir"):
                tokenizer.dir = Path(str(tokenizer.dir)).name


def sanitize_archive(path: Path) -> None:
    """Remove tokenizer trainer paths and normalize archive metadata."""
    from sentencepiece import sentencepiece_model_pb2

    temporary = path.with_name(f".{path.name}.sanitized.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with tarfile.open(path, "r:*") as source, tarfile.open(temporary, "w") as target:
            for member in source:
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise RuntimeError("unsafe member name in NeMo archive")
                if member.issym() or member.islnk():
                    raise RuntimeError("links are not allowed in the NeMo archive")
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                member.mtime = 0
                if not member.isfile():
                    target.addfile(member)
                    continue
                handle = source.extractfile(member)
                if handle is None:
                    raise RuntimeError(f"cannot read archive member: {member.name}")
                data = handle.read()
                if member.name.endswith("_tokenizer.model"):
                    model = sentencepiece_model_pb2.ModelProto()
                    model.ParseFromString(data)
                    del model.trainer_spec.input[:]
                    model.trainer_spec.model_prefix = "tokenizer"
                    data = model.SerializeToString()
                member.size = len(data)
                target.addfile(member, io.BytesIO(data))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if sha256(source) != BASE_SHA256:
        raise RuntimeError("base archive sha-256 mismatch")

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(str(source), map_location="cpu")
    output_heads = model.joint.joint_net[2]
    if "ne" not in output_heads:
        raise RuntimeError("base model has no Nepali rnnt head")
    for language in list(output_heads):
        if language != "ne":
            del output_heads[language]
    del model.ctc_decoder
    sanitize_config(model)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_SAVED_PARAMETERS:
        raise RuntimeError(f"unexpected saved parameter count: {parameters}")
    model.save_to(str(output))
    sanitize_archive(output)
    print(
        json.dumps(
            {
                "output": output.name,
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "parameters_before_runtime_compaction": parameters,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
