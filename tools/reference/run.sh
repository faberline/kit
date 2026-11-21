#!/bin/sh
# Runs tools/reference/<crate>/generate.py for each crate directory that has one.
# Arguments are passed on, so `run.sh --check` checks the committed fixtures.
# REF_VENV names the virtualenv that has the pins of requirements.txt.
set -u
cd "$(dirname "$0")/../.." || exit 1
REF_VENV=${REF_VENV:-.venv-reference}

found=0
rc=0
for d in tools/reference/*/; do
  [ -f "$d/generate.py" ] || continue
  found=1
  "$REF_VENV/bin/python" "$d/generate.py" "$@" || rc=1
done

if [ "$found" -eq 0 ]; then
  echo "no generator yet"
fi
exit "$rc"
