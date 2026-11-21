#!/bin/sh
# Self-test of the reference tooling. Prints `ok <check>` for each check and
# exits 1 on the first failure. It needs only python3 and no venv.
set -u
PYTHONDONTWRITEBYTECODE=1
export PYTHONDONTWRITEBYTECODE
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
tmp=$(mktemp -d) || exit 1

fail() {
  echo "FAIL: $1"
  rm -rf "$tmp"
  exit 1
}

# (a) write_tsv writes the exact header and one row.
printf 'key\tvalue\tsource\nk\t1.5\tsrc\n' > "$tmp/a.expected"
python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import common; common.write_tsv(sys.argv[2], [("k", 1.5, "src")])' \
  "$here" "$tmp/a.tsv" || fail "write_tsv"
cmp -s "$tmp/a.expected" "$tmp/a.tsv" || fail "write_tsv output"
echo "ok write_tsv"

# (b) main --check exits 1 when the committed file differs by one byte, 0 when equal.
mkdir "$tmp/b" || fail "mkdir b"
cd "$tmp/b" || fail "cd b"
gen='import sys; sys.path.insert(0, sys.argv[1]); import common; common.main({"x.tsv": lambda: [("k", 1.5, "src")]})'

printf 'key\tvalue\tsource\nk\t1.5\tsrc\n' > x.tsv
python3 -c "$gen" "$here" --check || fail "main --check exits 0 when equal"

printf 'key\tvalue\tsource\nk\t1.6\tsrc\n' > x.tsv
if python3 -c "$gen" "$here" --check; then
  fail "main --check exits 1 when changed"
fi
echo "ok main --check"
cd "$root" || fail "cd root"

# (d) stable rounds a CPU-dependent float to 12 significant digits; write_tsv does not round.
printf 'key\tvalue\tsource\na\t0.976007092444\ts\nb\t0.976007092444\ts\nc\t1.5\ts\nd\t126410606437752.0\ts\n' > "$tmp/d.expected"
python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import common; common.write_tsv(sys.argv[2], [("a", common.stable(0.9760070924438541), "s"), ("b", common.stable(0.976007092443854), "s"), ("c", 1.5, "s"), ("d", 126410606437752.0, "s")])' \
  "$here" "$tmp/d.tsv" || fail "stable"
cmp -s "$tmp/d.expected" "$tmp/d.tsv" || fail "stable"
echo "ok stable"

# (c) check_table.sh exits 1 on a table with a `not covered` row, 0 on a table with none.
header='| Crate | Function | Reference expression | Tolerance | Test | Status |'
sep='|---|---|---|---|---|---|'
printf '%s\n%s\n| frame | frame::x::y | r | 1e-9 | t | not covered |\n' "$header" "$sep" > "$tmp/bad.md"
if sh "$here/check_table.sh" "$tmp/bad.md" > /dev/null; then
  fail "check_table exits 1 on a not covered row"
fi
sed 's/not covered/covered/g' "$root/docs/reference-coverage.md" > "$tmp/good.md" || fail "copy table"
sh "$here/check_table.sh" "$tmp/good.md" > /dev/null || fail "check_table exits 0 with no not covered row"
echo "ok check_table"

rm -rf "$tmp"
echo "ok all"
