#!/bin/sh
# Lists every `pub fn` and `pub const fn` of the numeric crates, one per line:
#   <crate>|<file path>|<fn name>
# Output is sorted with LC_ALL=C.
# Skipped: files named tests.rs and files under a tests/ directory. In any
# other file, reading stops at the first line that is exactly `#[cfg(test)]`,
# so the test module at the end of a file is not listed.
set -eu
cd "$(dirname "$0")/../.."

for crate in frame arraykit scikit; do
  find "crates/$crate/src" -name '*.rs' ! -name tests.rs ! -path '*/tests/*' \
    | LC_ALL=C sort \
    | while IFS= read -r file; do
        awk -v crate="$crate" -v path="$file" '
          $0 == "#[cfg(test)]" { exit }
          /^[ \t]*pub (const )?fn / {
            name = $0
            sub(/^[ \t]*pub (const )?fn /, "", name)
            sub(/[^A-Za-z0-9_].*/, "", name)
            print crate "|" path "|" name
          }
        ' "$file"
      done
done | LC_ALL=C sort
