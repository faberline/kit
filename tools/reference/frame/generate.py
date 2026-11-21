"""Reference values for the frame Series methods (docs/reference-coverage.md).

Writes crates/frame/tests/fixtures/series_reference.tsv from pandas. The inputs are
integer formulas that the Rust tests rebuild exactly; no np.random is used.

Run from the repository root:
    .venv-reference/bin/python tools/reference/frame/generate.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

# Small case: 8 values with one null, a tie (4, 4) and a negative (-1).
SMALL = [3, None, -1, 4, 4, 2, 7, 0]

# Large case: 10,000 rows, a null on every 7th row where i % 7 == 3.
N = 10000


def large_values():
    return [None if i % 7 == 3 else ((i * 37 + 11) % 101) / 4.0 for i in range(N)]


def large_keys():
    return [i % 50 for i in range(N)]


def corr_x():
    return [(i * 13 + 5) % 37 for i in range(N)]


def corr_y():
    return [2 * ((i * 13 + 5) % 37) + i % 11 for i in range(N)]


def series_rows():
    s = pd.Series(SMALL, dtype="float64")
    big = pd.Series(large_values(), dtype="float64")
    keys = pd.Series(large_keys())
    rows = []

    def add(key, value, source):
        rows.append((key, float(value), source))

    # Aggregates that skip nulls.
    add("series.sum.small.sum", s.sum(), "pd.Series([3, None, -1, 4, 4, 2, 7, 0]).sum()")
    add("series.mean.small.mean", s.mean(), "s.mean()")
    add("series.std.small.std", s.std(), "s.std()")
    add("series.var.small.var", s.var(), "s.var()")
    add("series.median.small.median", s.median(), "s.median()")
    add("series.min.small.min", s.min(), "s.min()")
    add("series.max.small.max", s.max(), "s.max()")
    add("series.count.small.count", s.count(), "s.count()")
    add("series.sum.large.sum", big.sum(), "big.sum()")
    add("series.mean.large.mean", big.mean(), "big.mean()")
    add("series.std.large.std", big.std(), "big.std()")
    add("series.var.large.var", big.var(), "big.var()")
    add("series.median.large.median", big.median(), "big.median()")
    add("series.min.large.min", big.min(), "big.min()")
    add("series.max.large.max", big.max(), "big.max()")
    add("series.count.large.count", big.count(), "big.count()")
    add("series.quantile.small.q25", s.quantile(0.25), "s.quantile(0.25)")
    add("series.quantile.large.q25", big.quantile(0.25), "big.quantile(0.25)")
    add("series.quantile.large.q90", big.quantile(0.9), "big.quantile(0.9)")

    # Two series with no nulls (corr does not skip nulls in frame).
    add(
        "series.corr.large.corr",
        pd.Series(corr_x()).corr(pd.Series(corr_y())),
        "pd.Series(x).corr(pd.Series(y))",
    )

    # Transformations and selections.
    add("series.abs.small.sum", s.abs().sum(), "s.abs().sum()")
    add("series.between.small.count", s.between(0.0, 3.0).sum(), "s.between(0.0, 3.0).sum()")
    add("series.clip.small.sum", s.clip(0.0, 3.0).sum(), "s.clip(0.0, 3.0).sum()")
    add(
        "series.drop_duplicates.small.len",
        len(s.drop_duplicates(keep="first")),
        "len(s.drop_duplicates(keep='first'))",
    )
    add(
        "series.duplicated.small.count",
        s.duplicated(keep="first").sum(),
        "s.duplicated(keep='first').sum()",
    )
    mask = [i % 2 == 0 for i in range(len(SMALL))]
    add("series.filter.small.sum", s[mask].sum(), "s[i % 2 == 0].sum()")
    add("series.isin.small.count", s.isin([4, 0]).sum(), "s.isin([4, 0]).sum()")
    add("series.map.small.sum", (s * 2 + 1).sum(), "(s * 2 + 1).sum()")
    cond_mask = [i % 3 == 0 for i in range(len(SMALL))]
    add("series.mask.small.sum", s.mask(cond_mask, 0).sum(), "s.mask(i % 3 == 0, 0).sum()")
    add("series.nlargest.small.sum", s.nlargest(3).sum(), "s.nlargest(3).sum()")
    add("series.nlargest.large.sum", big.nlargest(5).sum(), "big.nlargest(5).sum()")
    add("series.nsmallest.small.sum", s.nsmallest(3).sum(), "s.nsmallest(3).sum()")
    add("series.nsmallest.large.sum", big.nsmallest(5).sum(), "big.nsmallest(5).sum()")

    # rank: a null gets no rank (NaN); the others get their average rank.
    add("series.rank.small.sum", s.rank().sum(), "s.rank().sum()")
    ties = pd.Series([1, 1, 2]).rank()
    add("series.rank.ties.r0", ties.iloc[0], "pd.Series([1, 1, 2]).rank().iloc[0]")
    add("series.rank.ties.r1", ties.iloc[1], "pd.Series([1, 1, 2]).rank().iloc[1]")
    add("series.rank.ties.r2", ties.iloc[2], "pd.Series([1, 1, 2]).rank().iloc[2]")
    add("series.rank.ties.len", len(ties), "len(pd.Series([1, 1, 2]).rank())")
    add("series.rank.small.nulls", s.rank().isna().sum(), "s.rank().isna().sum()")
    add("series.rank.large.sum", big.rank().sum(), "big.rank().sum()")
    add("series.rank.large.count", big.rank().count(), "big.rank().count()")

    add("series.replace.small.sum", s.replace(4, 40).sum(), "s.replace(4, 40).sum()")
    add(
        "series.replace_map.small.sum",
        s.replace({4: 40, -1: -10}).sum(),
        "s.replace({4: 40, -1: -10}).sum()",
    )
    # Labels 10..17 before the reset, so that label 7 is absent until the reset.
    labelled = pd.Series(SMALL, dtype="float64", index=range(10, 10 + len(SMALL)))
    add(
        "series.reset_index.small.loc7",
        labelled.reset_index(drop=True).loc[7],
        "pd.Series(SMALL, index=range(10, 18)).reset_index(drop=True).loc[7]",
    )
    rounded = pd.Series([1.234, 5.678, -2.3456, 0.1, 6.789]).round(2)
    add("series.round.fixed.sum", rounded.sum(), "pd.Series([1.234, 5.678, -2.3456, 0.1, 6.789]).round(2).sum()")
    add("series.set_name.small.sum", s.rename("renamed").sum(), "s.rename('renamed').sum()")
    add("series.sort.small.first", s.sort_values(ascending=True).iloc[0], "s.sort_values().iloc[0]")
    add("series.sort.small.first_desc", s.sort_values(ascending=False).iloc[0], "s.sort_values(ascending=False).iloc[0]")
    add("series.unique.small.len", len(s.unique()), "len(s.unique())")
    counts = keys.value_counts()
    add("series.value_counts.large.key0", counts[0], "keys.value_counts()[0]")
    add("series.value_counts.large.keys", len(counts), "len(keys.value_counts())")
    keep_mask = [i % 3 != 0 for i in range(len(SMALL))]
    add("series.where_cond.small.sum", s.where(keep_mask, 0).sum(), "s.where(i % 3 != 0, 0).sum()")

    # The two tests moved from tests/it/reference.rs.
    add("series.sum.nulls.sum", pd.Series([1, None, 5], dtype="float64").sum(), "pd.Series([1, None, 5]).sum()")
    add("series.mean.nulls.mean", pd.Series([1, None, 5], dtype="float64").mean(), "pd.Series([1, None, 5]).mean()")
    return rows


common.main({"crates/frame/tests/fixtures/series_reference.tsv": series_rows})
