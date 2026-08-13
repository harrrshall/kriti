"""Train Kriti's portable acoustic terminal-danda head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from kriti.model import encoder_features, load_compact_asr

SEED = 20_260_813


def validation_bucket(sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big") % 5


def selection_key(sample_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{sample_id}".encode()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if not rows:
        raise RuntimeError("training view is empty")
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    *,
    validation: bool,
    target_source: str,
    negatives_per_source: int,
) -> list[dict[str, Any]]:
    positives: list[dict[str, Any]] = []
    negatives: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if (validation_bucket(str(row["sample_id"])) == 0) is not validation:
            continue
        if row["source_id"] == target_source:
            positives.append(row)
        else:
            negatives.setdefault(str(row["source_id"]), []).append(row)
    selected = positives.copy()
    for source_rows in negatives.values():
        selected.extend(
            sorted(source_rows, key=lambda row: selection_key(str(row["sample_id"])))[
                :negatives_per_source
            ]
        )
    return sorted(selected, key=lambda row: selection_key(str(row["sample_id"])))


def choose_threshold(
    probabilities: list[float],
    labels: list[int],
    *,
    minimum_precision: float = 0.75,
    minimum_recall: float = 0.75,
    maximum_false_positive_rate: float = 0.005,
) -> tuple[float, dict[str, float | int]]:
    pairs = list(zip(probabilities, labels, strict=True))
    reports = []
    for index in range(1, 1000):
        threshold = round(index / 1000, 3)
        tp = sum(probability >= threshold and label == 1 for probability, label in pairs)
        fp = sum(probability >= threshold and label == 0 for probability, label in pairs)
        fn = sum(probability < threshold and label == 1 for probability, label in pairs)
        tn = sum(probability < threshold and label == 0 for probability, label in pairs)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
        report = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": false_positive_rate,
        }
        if (
            precision >= minimum_precision
            and recall >= minimum_recall
            and false_positive_rate <= maximum_false_positive_rate
        ):
            reports.append(((recall, precision, threshold), threshold, report))
    if not reports:
        raise RuntimeError("no threshold satisfies the frozen constraints")
    _, threshold, report = max(reports, key=lambda item: item[0])
    return threshold, report


def extract(
    model: Any, rows: list[dict[str, Any]], device: str, batch_size: int
) -> list[list[float]]:
    return encoder_features(
        model,
        [str(row["audio"]) for row in rows],
        device,
        batch_size=batch_size,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-view", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rows = read_rows(Path(args.train_view))
    train_rows = select_rows(
        rows,
        validation=False,
        target_source="fleurs_ne_np",
        negatives_per_source=3_000,
    )
    validation_rows = select_rows(
        rows,
        validation=True,
        target_source="fleurs_ne_np",
        negatives_per_source=1_000,
    )
    model, device = load_compact_asr(
        args.model,
        device=args.device,
        expected_sha256=None,
    )
    train_features = np.asarray(extract(model, train_rows, device, args.batch_size))
    validation_features = np.asarray(extract(model, validation_rows, device, args.batch_size))
    train_labels = np.asarray([int(row["source_id"] == "fleurs_ne_np") for row in train_rows])
    validation_labels = [int(row["source_id"] == "fleurs_ne_np") for row in validation_rows]
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2_000,
        random_state=SEED,
        solver="lbfgs",
    )
    classifier.fit(train_features, train_labels)
    probabilities = classifier.predict_proba(validation_features)[:, 1].tolist()
    threshold, validation = choose_threshold(probabilities, validation_labels)
    coefficients = [float(value) for value in classifier.coef_[0]]
    intercept = float(classifier.intercept_[0])
    if len(coefficients) != 1_024 or not all(
        math.isfinite(value) for value in [*coefficients, intercept, threshold]
    ):
        raise RuntimeError("trained punctuation head is invalid")
    payload = {
        "schema_version": 1,
        "model_type": "binary_logistic_regression",
        "feature_extractor": "conformer_encoder_masked_mean_and_std_v1",
        "feature_count": 1_024,
        "parameter_count": 1_025,
        "positive_class": 1,
        "classes": [0, 1],
        "coefficients": coefficients,
        "intercept": intercept,
        "threshold": threshold,
        "seed": SEED,
        "action": "append_devanagari_danda",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "validation": validation,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"output": output.name, "threshold": threshold}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
