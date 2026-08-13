"""Run a complete deterministic Kriti evaluation over a frozen JSONL view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from kriti import KritiASR
from kriti.metrics import metric_set

EXPECTED_VIEW_SHA256 = "2374cac54831ce9c69282503763d7f1e12ada0404ae34ed471a7538cdae6c61f"
EXPECTED_RECORDS = 3_630
EXPECTED_SOURCE_COUNTS = {
    "fleurs_ne_np": 304,
    "indicvoices_nepali": 2_569,
    "openslr54": 757,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_view(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    sample_ids = [str(row["sample_id"]) for row in rows]
    if not rows or len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("view must contain unique ordered sample ids")
    for row in rows:
        if not all(field in row for field in ("audio", "reference", "source_id")):
            raise RuntimeError("view row lacks a required field")
    if len(rows) != EXPECTED_RECORDS:
        raise RuntimeError("view record count differs from the frozen benchmark")
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row["source_id"])
        source_counts[source] = source_counts.get(source, 0) + 1
    if source_counts != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError("view source composition differs from the frozen benchmark")
    if sha256(path) != EXPECTED_VIEW_SHA256:
        raise RuntimeError("view sha-256 differs from the frozen benchmark")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="harrrshall/kriti")
    parser.add_argument("--revision")
    parser.add_argument("--model-file")
    parser.add_argument("--head-file")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    view_path = Path(args.view).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = read_view(view_path)
    if bool(args.model_file) != bool(args.head_file):
        raise ValueError("model-file and head-file must be provided together")
    if args.model_file:
        model = KritiASR.from_files(args.model_file, args.head_file, device=args.device)
    else:
        model = KritiASR.from_pretrained(args.model, revision=args.revision, device=args.device)
    hypotheses = model.transcribe([row["audio"] for row in rows], batch_size=args.batch_size)
    predictions_path = output_dir / "predictions.jsonl"
    temporary = output_dir / ".predictions.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for row, hypothesis in zip(rows, hypotheses, strict=True):
            prediction = {
                "sample_id": row["sample_id"],
                "source_id": row["source_id"],
                "reference": row["reference"],
                "hypothesis": hypothesis,
                "error_type": "",
            }
            handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, predictions_path)
    summary = metric_set(
        [str(row["reference"]) for row in rows],
        [str(hypothesis) for hypothesis in hypotheses],
    )
    summary.update(
        {
            "view_sha256": sha256(view_path),
            "predictions_sha256": sha256(predictions_path),
            "model": args.model,
            "revision": args.revision,
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
