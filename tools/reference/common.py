"""Shared writer for the reference fixture files under tools/reference.

Standard library only: selftest.sh imports this module without the venv.
A fixture file is a TSV with the header `key<TAB>value<TAB>source`.
"""

import filecmp
import os
import shutil
import sys
import tempfile


def _format_scalar(value):
    # float() first: numpy float64 is a float subclass, but its repr is not a plain number.
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, str):
        return value
    return str(value)


def _format_value(value):
    if isinstance(value, str):
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return ",".join(_format_scalar(item) for item in value)
    return _format_scalar(value)


def write_tsv(path, rows):
    """Write rows of (key, value, source) to path with a header line."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("key\tvalue\tsource\n")
        for key, value, source in rows:
            handle.write(f"{key}\t{_format_value(value)}\t{source}\n")


def main(generators):
    """Write each file, or with --check compare the generated files to the committed ones.

    generators maps a path relative to the working directory (the repository
    root) to a function that returns the rows of that file.
    """
    if "--check" not in sys.argv[1:]:
        for path, generate in generators.items():
            write_tsv(path, generate())
        return

    scratch = tempfile.mkdtemp()
    failed = False
    try:
        for path, generate in generators.items():
            fresh = os.path.join(scratch, path)
            write_tsv(fresh, generate())
            if not os.path.exists(path):
                print(f"missing: {path}")
                failed = True
            elif not filecmp.cmp(fresh, path, shallow=False):
                print(f"differs: {path}")
                failed = True
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if failed:
        sys.exit(1)
