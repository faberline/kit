#!/bin/sh
# Checks the reference coverage table (default docs/reference-coverage.md,
# or the file given as the first argument).
# Exit 1 and print each row when a row has the Status `not covered`.
# Exit 1 and print the key when a function printed by inventory.sh has no row
# (the Function column of either table holds the key).
# Exit 0 otherwise.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
table=${1:-$root/docs/reference-coverage.md}
tmp=$(mktemp -d) || exit 1
status=0

# Rows whose status (the second-last cell) is `not covered`.
if awk -F'|' '
  /^\|/ {
    s = $(NF - 1)
    gsub(/^[ \t]+|[ \t]+$/, "", s)
    if (s == "not covered") { print; bad = 1 }
  }
  END { exit bad }
' "$table"; then
  :
else
  status=1
fi

# Function cells (third cell) of every row of both tables.
awk -F'|' '
  /^\|/ {
    f = $3
    gsub(/^[ \t]+|[ \t]+$/, "", f)
    print f
  }
' "$table" | sort -u > "$tmp/cells"

sh "$here/inventory.sh" > "$tmp/inventory"
while IFS='|' read -r crate path name; do
  rel=${path#crates/$crate/src/}
  rel=${rel%.rs}
  key="$crate::$rel::$name"
  if ! grep -qxF "$key" "$tmp/cells"; then
    echo "no row: $key"
    status=1
  fi
done < "$tmp/inventory"

rm -rf "$tmp"
exit "$status"
