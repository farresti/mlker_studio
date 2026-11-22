"""Compile a validated model into a standalone PyTorch project."""

import argparse
from dataclasses import asdict
import json
import sys

from .. import ModelError, load_model
from . import CompileError, compile_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("destination")
    args = parser.parse_args()
    try:
        project = compile_model(load_model(args.model), args.destination)
    except ModelError as error:
        stage = "compilation" if isinstance(error, CompileError) else "validation"
        print(json.dumps({"stage": stage, "diagnostics": [asdict(d) for d in error.diagnostics]}, indent=2),
              file=sys.stderr)
        return 1
    except OSError as error:
        parser.exit(2, f"{error}\n")
    print(project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
