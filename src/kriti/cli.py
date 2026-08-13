"""Command line inference for Kriti."""

from __future__ import annotations

import argparse

from kriti.model import DEFAULT_REPO_ID, load_model


def main() -> int:
    parser = argparse.ArgumentParser(description="transcribe 16 khz Nepali audio with Kriti")
    parser.add_argument("audio", nargs="+")
    parser.add_argument("--model", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision")
    parser.add_argument("--device")
    args = parser.parse_args()
    model = load_model(args.model, revision=args.revision, device=args.device)
    predictions = model.transcribe(args.audio)
    for prediction in predictions:
        print(prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
