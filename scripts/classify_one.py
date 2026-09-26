#!/usr/bin/env python
"""Classify one ticket from the command line and print the result as JSON.

Usage: uv run python scripts/classify_one.py "I was charged twice, refund now!"
"""

import argparse

from support_ai.classifier.classify import classify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="The raw ticket text to classify.")
    args = parser.parse_args()

    result = classify(args.text)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
