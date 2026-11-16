#!/bin/sh
# Compare criterion results with a stored baseline.
#
# Exit 0: no benchmark is more than THRESHOLD percent slower than its baseline.
# Exit 1: a benchmark is slower than the threshold, or a baseline id has no result.
# Exit 2: usage error, no baseline, or no criterion result found.
#
# Usage: compare.sh [--criterion-dir DIR] [--baseline FILE] [--threshold PERCENT] [--update]
#   --update  write the baseline from the results and exit 0 (no comparison).
set -eu

CRITERION_DIR=target/criterion
BASELINE=benches/baseline/baseline.json
PERCENT=10
UPDATE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --criterion-dir) CRITERION_DIR=$2; shift 2 ;;
    --baseline) BASELINE=$2; shift 2 ;;
    --threshold) PERCENT=$2; shift 2 ;;
    --update) UPDATE=1; shift ;;
    *) echo "compare.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

TMP=$(mktemp)
RES=$(mktemp)
trap 'rm -f "$TMP" "$RES"' EXIT

# One line per result: "<id><TAB><mean nanoseconds>".
# The id is the path under CRITERION_DIR without "/new/estimates.json".
find "$CRITERION_DIR" -path '*/new/estimates.json' -type f 2>/dev/null | sort | while IFS= read -r f; do
  rel=${f#"$CRITERION_DIR"/}
  id=${rel%/new/estimates.json}
  case "$id" in
    report|report/*) continue ;;
  esac
  ns=$(jq -r '.mean.point_estimate' "$f")
  printf '%s\t%s\n' "$id" "$ns"
done > "$TMP"

if [ ! -s "$TMP" ]; then
  echo "compare.sh: no criterion result under $CRITERION_DIR" >&2
  exit 2
fi

# The results as one JSON object: id -> nanoseconds.
jq -R -s -S 'split("\n") | map(select(length > 0) | split("\t"))
  | map({key: .[0], value: (.[1] | tonumber)}) | from_entries' "$TMP" > "$RES"

if [ "$UPDATE" = 1 ]; then
  mkdir -p "$(dirname "$BASELINE")"
  jq -S --indent 2 . "$RES" > "$BASELINE"
  echo "compare.sh: wrote $(jq 'length' "$RES") baseline entries to $BASELINE"
  exit 0
fi

if [ ! -f "$BASELINE" ]; then
  echo "compare.sh: no baseline file $BASELINE" >&2
  exit 2
fi

printf 'id\tbaseline_ns\tnew_ns\tchange_percent\tstatus\n'
jq -r --argjson pct "$PERCENT" --slurpfile res "$RES" --slurpfile base "$BASELINE" '
  ($res[0]) as $r
  | ($base[0]) as $b
  | def row($id):
      if $r[$id] == null then
        "\($id)\t\($b[$id])\t-\t-\tMISSING"
      elif $b[$id] == null then
        "\($id)\t-\t\($r[$id])\t-\tnew"
      else
        (((($r[$id] / $b[$id]) * 100 - 100) * 100 | round) / 100) as $chg
        | (if $r[$id] <= $b[$id] * (1 + $pct / 100) then "ok" else "SLOWER" end) as $st
        | "\($id)\t\($b[$id])\t\($r[$id])\t\($chg)\t\($st)"
      end;
    ([$r | keys[]] + [$b | keys[]] | unique | .[] | row(.))
' -n > "$TMP.table" 2>/dev/null || {
  echo "compare.sh: cannot compare with $BASELINE" >&2
  rm -f "$TMP.table"
  exit 2
}
cat "$TMP.table"

BAD=$(awk -F '\t' '$5 == "SLOWER" || $5 == "MISSING" { n++ } END { print n + 0 }' "$TMP.table")
rm -f "$TMP.table"
if [ "$BAD" -eq 0 ]; then
  exit 0
fi
echo "compare.sh: $BAD benchmark(s) slower than ${PERCENT}% or missing" >&2
exit 1
