"""Shared writer for the reference fixture files under tools/reference.

Standard library only: selftest.sh imports this module without the venv.
A fixture file is a TSV with the header `key<TAB>value<TAB>source`.
"""

import math
import os
import shutil
import sys
import tempfile


def stable(value, digits=12):
    """Round a CPU-dependent float to `digits` significant digits, so a 1 ulp difference between BLAS kernels does not change the fixture text."""
    return float(f"{value:.{digits}g}")


def stable_value(value, digits=12):
    """Round a float, or each float of a list or array, with `stable`; leave other values unchanged."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [stable(item, digits) if isinstance(item, float) else item for item in value]
    if isinstance(value, float):
        return stable(value, digits)
    return value


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


def values_close(committed_text, fresh_text, rel=1e-11, abs_tol=1e-11):
    """Compare two comma-separated value strings; floats may differ by floating point noise.

    The committed value is the reference: |committed - fresh| must be at most
    abs_tol + rel * |committed|. Any other difference, a different count or a
    non-number, is a change.
    """
    if committed_text == fresh_text:
        return True
    committed = committed_text.split(",")
    fresh = fresh_text.split(",")
    if len(committed) != len(fresh):
        return False
    for old, new in zip(committed, fresh):
        if old == new:
            continue
        try:
            a = float(old)
            b = float(new)
        except ValueError:
            return False
        if math.isnan(a) and math.isnan(b):
            continue
        if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
            return False
        if not abs(a - b) <= abs_tol + rel * abs(a):
            return False
    return True


def rows_match(committed_line, fresh_line):
    """Compare one TSV line. A data row (key, value, source) may differ in its value by noise only."""
    committed = committed_line.split("\t")
    fresh = fresh_line.split("\t")
    if len(committed) != len(fresh):
        return False
    if len(committed) == 3 and committed[0] != "key":
        return (
            committed[0] == fresh[0]
            and committed[2] == fresh[2]
            and values_close(committed[1], fresh[1])
        )
    return committed_line == fresh_line


def files_match(committed_path, fresh_path):
    """True when the two TSV files have the same rows, with values compared by values_close."""
    with open(committed_path, encoding="utf-8") as handle:
        committed = handle.read().splitlines()
    with open(fresh_path, encoding="utf-8") as handle:
        fresh = handle.read().splitlines()
    if len(committed) != len(fresh):
        return False
    return all(rows_match(old, new) for old, new in zip(committed, fresh))


def _print_first_diffs(fresh, committed, limit=5):
    """Print up to `limit` rows whose text differs, to show what the check found."""
    with open(committed, encoding="utf-8") as handle:
        old = handle.read().splitlines()
    with open(fresh, encoding="utf-8") as handle:
        new = handle.read().splitlines()
    shown = 0
    for number, (old_line, new_line) in enumerate(zip(old, new), start=1):
        if not rows_match(old_line, new_line) and shown < limit:
            print(f"  line {number}:")
            print(f"    committed: {old_line}")
            print(f"    generated: {new_line}")
            shown += 1
    if len(old) != len(new):
        print(f"  line count: committed {len(old)}, generated {len(new)}")


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
            elif not files_match(path, fresh):
                print(f"differs: {path}")
                _print_first_diffs(fresh, path)
                failed = True
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if failed:
        sys.exit(1)
