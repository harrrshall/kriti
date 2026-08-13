"""Exact text normalization and aggregate metrics used by the Kriti benchmark."""

from __future__ import annotations

import unicodedata

from jiwer import cer, process_characters, process_words, wer


def normalize_raw(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalize_punctuation_insensitive(text: str) -> str:
    normalized = normalize_raw(text).casefold()
    characters = [
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def metric_set(references: list[str], hypotheses: list[str]) -> dict[str, float | int]:
    if not references or len(references) != len(hypotheses):
        raise ValueError("metric inputs must be non-empty and aligned")
    raw_references = [normalize_raw(item) for item in references]
    raw_hypotheses = [normalize_raw(item) for item in hypotheses]
    pi_references = [normalize_punctuation_insensitive(item) for item in references]
    pi_hypotheses = [normalize_punctuation_insensitive(item) for item in hypotheses]
    raw_words = process_words(raw_references, raw_hypotheses)
    pi_words = process_words(pi_references, pi_hypotheses)
    raw_characters = process_characters(raw_references, raw_hypotheses)
    pi_characters = process_characters(pi_references, pi_hypotheses)
    return {
        "records": len(references),
        "raw_wer": wer(raw_references, raw_hypotheses),
        "raw_cer": cer(raw_references, raw_hypotheses),
        "punctuation_insensitive_wer": wer(pi_references, pi_hypotheses),
        "punctuation_insensitive_cer": cer(pi_references, pi_hypotheses),
        "raw_word_errors": raw_words.substitutions + raw_words.deletions + raw_words.insertions,
        "raw_reference_words": raw_words.hits + raw_words.substitutions + raw_words.deletions,
        "raw_character_errors": (
            raw_characters.substitutions + raw_characters.deletions + raw_characters.insertions
        ),
        "raw_reference_characters": (
            raw_characters.hits + raw_characters.substitutions + raw_characters.deletions
        ),
        "punctuation_insensitive_word_errors": (
            pi_words.substitutions + pi_words.deletions + pi_words.insertions
        ),
        "punctuation_insensitive_reference_words": (
            pi_words.hits + pi_words.substitutions + pi_words.deletions
        ),
        "punctuation_insensitive_character_errors": (
            pi_characters.substitutions + pi_characters.deletions + pi_characters.insertions
        ),
        "punctuation_insensitive_reference_characters": (
            pi_characters.hits + pi_characters.substitutions + pi_characters.deletions
        ),
    }
