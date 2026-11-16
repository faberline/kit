#!/bin/sh
# Self-test of compare.sh with fake criterion results. Prints one ok: or FAIL: line per case.
# Exit 0 when every case passes, 1 otherwise.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
COMPARE="$HERE/compare.sh"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

fail=0

# mkres DIR ID NS: write a fake criterion result.
mkres() {
  mkdir -p "$1/$2/new"
  printf '{"mean":{"point_estimate":%s}}\n' "$3" > "$1/$2/new/estimates.json"
}

# check NAME EXPECTED ACTUAL
check() {
  if [ "$2" = "$3" ]; then
    echo "ok: $1"
  else
    echo "FAIL: $1 (expected exit $2, got $3)"
    fail=1
  fi
}

# run ARGS...: run compare.sh, print its exit status.
run() {
  rc=0
  sh "$COMPARE" "$@" > /dev/null 2>&1 || rc=$?
  echo "$rc"
}

# 1. equal result: exit 0
mkres "$T/c1" matmul/n100 1000
printf '{"matmul/n100":1000}\n' > "$T/b1.json"
check "equal result exits 0" 0 "$(run --criterion-dir "$T/c1" --baseline "$T/b1.json")"

# 2. 9 percent slower: exit 0
mkres "$T/c2" matmul/n100 1090
check "9 percent slower exits 0" 0 "$(run --criterion-dir "$T/c2" --baseline "$T/b1.json")"

# 3. 20 percent slower: exit 1
mkres "$T/c3" matmul/n100 1200
check "20 percent slower exits 1" 1 "$(run --criterion-dir "$T/c3" --baseline "$T/b1.json")"

# 4. baseline id with no result: exit 1
printf '{"matmul/n100":1000,"gone/x":500}\n' > "$T/b4.json"
check "baseline id missing from results exits 1" 1 "$(run --criterion-dir "$T/c1" --baseline "$T/b4.json")"

# 5. result id not in the baseline: exit 0
mkres "$T/c5" matmul/n100 1000
mkres "$T/c5" extra/y 700
check "extra id not in baseline exits 0" 0 "$(run --criterion-dir "$T/c5" --baseline "$T/b1.json")"

# 6. empty criterion dir: exit 2
mkdir -p "$T/c6"
check "empty criterion dir exits 2" 2 "$(run --criterion-dir "$T/c6" --baseline "$T/b1.json")"

# 7. --update writes the baseline from the results
rc=0
sh "$COMPARE" --criterion-dir "$T/c1" --baseline "$T/u7.json" --update > /dev/null 2>&1 || rc=$?
check "--update exits 0" 0 "$rc"
if jq -e --argjson want '{"matmul/n100":1000}' '. == $want' "$T/u7.json" > /dev/null 2>&1; then
  echo "ok: --update writes the expected baseline"
else
  echo "FAIL: --update writes the expected baseline"
  fail=1
fi

exit "$fail"
