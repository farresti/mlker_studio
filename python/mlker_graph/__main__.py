"""Validate a model and print a JSON array of diagnostics."""

import argparse
from dataclasses import asdict
import json
import sys

from . import ModelError, load_model, validate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a .yaml model")
    args = parser.parse_args()
    try:
        diagnostics = validate_model(load_model(args.path))
    except ModelError as error:
        diagnostics = error.diagnostics
    except OSError as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps([asdict(d) for d in diagnostics], indent=2, ensure_ascii=False))
    return int(any(d.severity == "error" for d in diagnostics))


if __name__ == "__main__":
    sys.exit(main())
