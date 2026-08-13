from __future__ import annotations

from scripts.train_punctuation_head import choose_threshold, select_rows, validation_bucket


def test_validation_bucket_is_stable() -> None:
    assert validation_bucket("sample-a") == validation_bucket("sample-a")
    assert 0 <= validation_bucket("sample-a") < 5


def test_select_rows_keeps_domains_separate() -> None:
    rows = [
        {"sample_id": f"id-{index}", "source_id": "fleurs_ne_np" if index < 10 else "other"}
        for index in range(30)
    ]
    train = select_rows(
        rows,
        validation=False,
        target_source="fleurs_ne_np",
        negatives_per_source=3,
    )
    validation = select_rows(
        rows,
        validation=True,
        target_source="fleurs_ne_np",
        negatives_per_source=3,
    )
    assert {row["sample_id"] for row in train}.isdisjoint({row["sample_id"] for row in validation})
    assert sum(row["source_id"] == "other" for row in train) <= 3
    assert sum(row["source_id"] == "other" for row in validation) <= 3


def test_choose_threshold_uses_frozen_constraints() -> None:
    threshold, report = choose_threshold([0.9, 0.8, 0.1, 0.05], [1, 1, 0, 0])
    assert threshold == 0.8
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
