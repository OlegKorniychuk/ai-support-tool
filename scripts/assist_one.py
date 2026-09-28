#!/usr/bin/env python
"""Run the Reply Assistant on one ticket from the command line and print the result as JSON.

Usage: uv run python scripts/assist_one.py "How do I change my birth time?"
"""

import argparse

from support_ai.assistant.assist import assist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="The raw ticket text to answer.")
    args = parser.parse_args()

    result = assist(args.text)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
