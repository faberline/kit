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
        common.stable(float(pd.Series(corr_x()).corr(pd.Series(corr_y())))),
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


# Window and null inputs. SMALL_W has nulls at positions 0, 3, 4 and 9, and every
# value is above zero, so pct_change has no zero divisor. LARGE_W has a null where
# i % 7 == 3; it is not large_values(), which can give 0.0.
SMALL_W = [None if i in (0, 3, 4, 9) else ((i * 3) % 7 + 1) / 2 for i in range(10)]
LARGE_W = [None if i % 7 == 3 else (((i * 37 + 11) % 101) + 1) / 4 for i in range(N)]


def large_stats(result):
    """Count, sum and position-weighted sum of the non-null values of a result."""
    weights = pd.Series(range(1, len(result) + 1), index=result.index, dtype="float64")
    return float(result.count()), float(result.sum()), float((result * weights).sum())


def series_window_rows():
    s = pd.Series(SMALL_W, dtype="float64")
    big = pd.Series(LARGE_W, dtype="float64")
    rows = []

    def add(key, value, source):
        rows.append((key, value, source))

    def add_result(prefix, expr, small_result, large_result):
        # expr is a template: {x} is the input series, s or big.
        if small_result is not None:
            add(f"{prefix}.small", small_result, expr.format(x="s"))
        if large_result is not None:
            count, total, wsum = large_stats(large_result)
            add(f"{prefix}.large.count", count, expr.format(x="big"))
            add(f"{prefix}.large.sum", total, expr.format(x="big"))
            add(f"{prefix}.large.wsum", wsum, expr.format(x="big"))

    def both(prefix, expr, fn, with_large=True):
        add_result(prefix, expr, fn(s), fn(big) if with_large else None)

    # Cumulative operations. cumprod is small only: the large product overflows.
    both("window.cumsum", "{x}.cumsum()", lambda x: x.cumsum())
    both("window.cummax", "{x}.cummax()", lambda x: x.cummax())
    both("window.cummin", "{x}.cummin()", lambda x: x.cummin())
    both("window.cumprod", "{x}.cumprod()", lambda x: x.cumprod(), with_large=False)

    # Differences, percentage changes and shifts.
    both("window.diff1", "{x}.diff(1)", lambda x: x.diff(1))
    both("window.diff2", "{x}.diff(2)", lambda x: x.diff(2))
    both("window.pct_change1", "{x}.pct_change(1)", lambda x: x.pct_change(1))
    both("window.pct_change2", "{x}.pct_change(2)", lambda x: x.pct_change(2))
    both("window.shift2", "{x}.shift(2)", lambda x: x.shift(2))
    both("window.shift_neg2", "{x}.shift(-2)", lambda x: x.shift(-2))

    # Rolling windows: default min_periods (= window), min_periods 2, and min_periods 40 of 50.
    for name in ["sum", "mean", "min", "max", "std", "var"]:
        both(
            f"rolling.{name}.w3",
            "{x}.rolling(3)." + name + "()",
            lambda x, n=name: getattr(x.rolling(3), n)(),
        )
        add_result(
            f"rolling.{name}.w3mp2",
            "{x}.rolling(3, min_periods=2)." + name + "()",
            getattr(s.rolling(3, min_periods=2), name)(),
            None,
        )
        add_result(
            f"rolling.{name}.w50",
            "{x}.rolling(50, min_periods=40)." + name + "()",
            None,
            getattr(big.rolling(50, min_periods=40), name)(),
        )

    # Null handling.
    both("null.ffill", "{x}.ffill()", lambda x: x.ffill())
    both("null.bfill", "{x}.bfill()", lambda x: x.bfill())
    both("null.fillna", "{x}.fillna(-1.5)", lambda x: x.fillna(-1.5))
    both("null.apply", "{x} * 2 + 1", lambda x: x * 2 + 1)
    add("null.dropna.small", s.dropna(), "s.dropna()")
    add("null.dropna.large.count", float(big.dropna().count()), "big.dropna().count()")
    add("null.dropna.large.sum", float(big.dropna().sum()), "big.dropna().sum()")
    add("null.null_count.small", float(s.isna().sum()), "s.isna().sum()")
    add("null.null_count.large", float(big.isna().sum()), "big.isna().sum()")
    add("null.isna.small", s.isna().astype(float), "s.isna().astype(float)")
    add("null.isna.large.count", float(big.isna().sum()), "big.isna().sum()")
    add("null.notna.small", s.notna().astype(float), "s.notna().astype(float)")
    add("null.notna.large.count", float(big.notna().sum()), "big.notna().sum()")
    both("null.interpolate", "{x}.interpolate()", lambda x: x.interpolate())
    nearest = s.interpolate(method="nearest")
    add(
        "null.interpolate_nearest.small",
        [float(nearest.iloc[3]), float(nearest.iloc[4])],
        's.interpolate(method="nearest").iloc[3:5]',
    )
    return rows


# Ewm inputs have no nulls: pandas carries a value over a null, frame gives Null
# (a separate defect), so Ewm is checked on dense inputs only.
DENSE_SMALL = [((i * 3) % 7 + 1) / 2 for i in range(10)]
DENSE_LARGE = [(((i * 37 + 11) % 101) + 1) / 4 for i in range(N)]


def add_large(rows, prefix, result, expr):
    """Add the count, sum and position-weighted sum of a large result."""
    count, total, wsum = large_stats(result)
    rows.append((f"{prefix}.count", count, expr))
    rows.append((f"{prefix}.sum", total, expr))
    rows.append((f"{prefix}.wsum", wsum, expr))


def window_rows():
    s = pd.Series(SMALL_W, dtype="float64")
    big = pd.Series(LARGE_W, dtype="float64")
    d = pd.Series(DENSE_SMALL, dtype="float64")
    D = pd.Series(DENSE_LARGE, dtype="float64")
    rows = []

    def add(key, value, source):
        rows.append((key, value, source))

    # Expanding windows, with nulls. min_periods 3 counts non-null values.
    for name in ["sum", "mean", "min", "max", "std", "var"]:
        add(f"expanding.{name}.small", getattr(s.expanding(), name)(), f"s.expanding().{name}()")
        add_large(
        rows,
            f"expanding.{name}.large",
            getattr(big.expanding(), name)(),
            f"big.expanding().{name}()",
        )
    add("expanding.sum.mp3.small", s.expanding(min_periods=3).sum(), "s.expanding(min_periods=3).sum()")
    add("expanding.mean.mp3.small", s.expanding(min_periods=3).mean(), "s.expanding(min_periods=3).mean()")

    # Ewm windows on dense inputs: each parameter, adjust True and False, mean, std and var.
    for param, value in [("alpha", 0.3), ("halflife", 4.0), ("span", 10.0)]:
        for adjust in (True, False):
            label = "adj" if adjust else "noadj"
            for name in ["mean", "std", "var"]:
                add(
                    f"ewm.{param}.{label}.{name}.small",
                    getattr(d.ewm(**{param: value}, adjust=adjust, min_periods=1), name)(),
                    f"d.ewm({param}={value}, adjust={adjust}, min_periods=1).{name}()",
                )
                add_large(
        rows,
                    f"ewm.{param}.{label}.{name}.large",
                    getattr(D.ewm(**{param: value}, adjust=adjust, min_periods=1), name)(),
                    f"D.ewm({param}={value}, adjust={adjust}, min_periods=1).{name}()",
                )
    add(
        "ewm.span.min_periods3.mean.small",
        d.ewm(span=10.0, min_periods=3).mean(),
        "d.ewm(span=10.0, min_periods=3).mean()",
    )
    return rows


def groupby_rows():
    keys = pd.Series(
        [None if k % 97 == 5 else k % 50 for k in range(N)], dtype="float64"
    )
    df = pd.DataFrame({"g": keys, "v": pd.Series(LARGE_W, dtype="float64")})
    gb = df.groupby("g")["v"]
    rows = []

    def add(key, value, source):
        rows.append((key, value, source))

    def add_groups(prefix, result, expr):
        # The first five groups by key, the sum over all groups, and the group count.
        for pos in range(5):
            add(f"{prefix}.k{pos}", result.iloc[pos], expr)
        add(f"{prefix}.sum", result.sum(), expr)
        add(f"{prefix}.len", len(result), expr)

    for name in ["sum", "mean", "median", "min", "max", "std", "var", "count", "first", "last"]:
        add_groups(f"groupby.{name}.large", getattr(gb, name)(), f"df.groupby('g')['v'].{name}()")
    for name in ["sum", "mean", "count", "min", "max", "first", "last", "std", "median"]:
        add_groups(
            f"groupby.agg.{name}.large",
            gb.agg(name),
            f"df.groupby('g')['v'].agg('{name}')",
        )
    add_groups(
        "groupby.aggregate.range.large",
        gb.agg(lambda x: x.max() - x.min()),
        "df.groupby('g')['v'].agg(lambda x: x.max() - x.min())",
    )
    transformed = gb.transform(lambda x: x / x.max())
    add_large(
        rows,
        "groupby.transform.div_max.large",
        transformed,
        "df.groupby('g')['v'].transform(lambda x: x / x.max())",
    )
    kept = df.groupby("g").filter(lambda x: x["v"].mean() > 12.5)
    add("groupby.filter.mean_gt.large.rows", len(kept), "df.groupby('g').filter(mean > 12.5)")
    add("groupby.filter.mean_gt.large.sum", kept["v"].sum(), "df.groupby('g').filter(mean > 12.5)['v'].sum()")
    add(
        "groupby.filter.mean_gt.large.groups",
        kept["g"].nunique(),
        "df.groupby('g').filter(mean > 12.5)['g'].nunique()",
    )

    # The two tests moved from tests/it/reference.rs.
    t = pd.DataFrame({"g": ["a", "a", "b", None], "v": [1.0, None, 4.0, 9.0]})
    means = t.groupby("g")["v"].mean()
    add("groupby.mean.nulls.a", means.loc["a"], "t.groupby('g')['v'].mean().loc['a']")
    add("groupby.mean.nulls.b", means.loc["b"], "t.groupby('g')['v'].mean().loc['b']")
    add("groupby.mean.nulls.len", len(means), "len(t.groupby('g')['v'].mean())")
    u = pd.DataFrame({"g": ["a", "a", "a", "b", "b"], "v": [1.0, 2.0, 4.0, 10.0, 20.0]})
    stds = u.groupby("g")["v"].std()
    medians = u.groupby("g")["v"].median()
    add("groupby.agg_std.small.a", stds.loc["a"], "u.groupby('g')['v'].std().loc['a']")
    add("groupby.agg_std.small.b", stds.loc["b"], "u.groupby('g')['v'].std().loc['b']")
    add("groupby.agg_median.small.a", medians.loc["a"], "u.groupby('g')['v'].median().loc['a']")
    add("groupby.agg_median.small.b", medians.loc["b"], "u.groupby('g')['v'].median().loc['b']")
    return rows


# Join inputs. Left: 10,000 rows, key null where i % 97 == 5, else i % 50 (200 rows per key),
# lv = ((i * 37 + 11) % 101) / 4. Right: 8,000 rows, key null where j % 89 == 7, else
# j % 4000 + 10 (two rows per key, so the left keys 0 to 9 have no right row), rv = ((j * 53 + 7) % 103) / 2. Every value is a multiple of
# 0.25, so the sums are exact in pandas and in Rust.
RIGHT_N = 8000
JOIN_HOWS = ["inner", "left", "right", "outer"]


def join_left():
    keys = [None if i % 97 == 5 else i % 50 for i in range(N)]
    vals = [((i * 37 + 11) % 101) / 4.0 for i in range(N)]
    return pd.DataFrame(
        {"k": pd.Series(keys, dtype="float64"), "lv": pd.Series(vals, dtype="float64")}
    )


def join_right():
    keys = [None if j % 89 == 7 else j % 4000 + 10 for j in range(RIGHT_N)]
    vals = [((j * 53 + 7) % 103) / 2.0 for j in range(RIGHT_N)]
    return pd.DataFrame(
        {"k": pd.Series(keys, dtype="float64"), "rv": pd.Series(vals, dtype="float64")}
    )


def join_rows():
    rows = []

    def add_frame(prefix, out, columns, expr):
        # The row count, and for each column its non-null count and its sum.
        rows.append((f"{prefix}.rows", float(len(out)), expr))
        for column in columns:
            rows.append((f"{prefix}.{column}_count", float(out[column].count()), expr))
            rows.append((f"{prefix}.{column}_sum", float(out[column].sum()), expr))

    left, right = join_left(), join_right()
    for how in JOIN_HOWS:
        add_frame(
            f"join.{how}.large",
            left.merge(right, on="k", how=how),
            ["k", "lv", "rv"],
            f"left.merge(right, on='k', how='{how}')",
        )

    # The three tests moved from tests/it/reference.rs.
    small_left = pd.DataFrame({"id": [1, 2, 3], "lv": [10, 20, 30]})
    small_right = pd.DataFrame({"id": [2, 3, 3, 4], "rv": [200, 300, 301, 400]})
    for how in JOIN_HOWS:
        add_frame(
            f"join.{how}.small",
            small_left.merge(small_right, on="id", how=how),
            ["id", "lv", "rv"],
            f"l.merge(r, on='id', how='{how}')",
        )
    int_left = pd.DataFrame({"k": pd.Series([1, 2, 3], dtype="int64")})
    float_right = pd.DataFrame(
        {"k": pd.Series([2.0, 3.0], dtype="float64"), "rv": pd.Series([5, 6], dtype="int64")}
    )
    add_frame(
        "join.intfloat.inner",
        int_left.merge(float_right, on="k"),
        ["rv"],
        "l.merge(r, on='k') with l.k int64 [1, 2, 3] and r.k float64 [2.0, 3.0]",
    )

    # The row order of the moved tests: sorted by id, then rv (NaN last), as the Rust test sorts.
    for how in ["right", "outer"]:
        ordered = small_left.merge(small_right, on="id", how=how).sort_values(["id", "rv"])
        expr = f"l.merge(r, on='id', how='{how}').sort_values(['id', 'rv'])"
        rows.append((f"join.{how}.small.id_sorted", ordered["id"].tolist(), expr))
        rows.append((f"join.{how}.small.lv_by_id", ordered["lv"].tolist(), expr))
    rv_sorted = int_left.merge(float_right, on="k").sort_values("rv")["rv"]
    rows.append(
        (
            "join.intfloat.inner.rv_sorted",
            rv_sorted.tolist(),
            "l.merge(r, on='k').sort_values('rv')['rv']",
        )
    )
    return rows


# Reshape inputs. S (6 rows) has no (w, b) pair, so pivot_table has a missing cell and
# cross_tab has a zero. U (5 rows) has no (3, y) pair. W and T are wide frames. BIG has
# 10,000 rows, and BIGW has 10,000 rows with one null every 7th cell.
S_REGION = ["e", "e", "w", "e", "e", "w"]
S_PRODUCT = ["a", "b", "a", "a", "a", "a"]
S_SALES = [10.0, 20.0, 30.0, 40.0, 5.0, 15.0]


def reshape_small():
    return pd.DataFrame(
        {
            "region": S_REGION,
            "product": S_PRODUCT,
            "sales": pd.Series(S_SALES, dtype="float64"),
        }
    )


def reshape_big():
    region = pd.Series([i % 50 for i in range(N)], dtype="int64")
    product = [f"p{i % 4}" for i in range(N)]
    sales = [None if i % 7 == 3 else ((i * 37 + 11) % 101) / 4.0 for i in range(N)]
    return pd.DataFrame(
        {"region": region, "product": product, "sales": pd.Series(sales, dtype="float64")}
    )


def reshape_wide_big():
    frame = {"id": pd.Series(range(N), dtype="int64")}
    for k in (1, 2, 3):
        values = [None if (i + k) % 7 == 3 else ((i * 37 + k * 11) % 101) / 4.0 for i in range(N)]
        frame[f"score_{k}"] = pd.Series(values, dtype="float64")
    return pd.DataFrame(frame)


def reshape_rows():
    rows = []
    small = reshape_small()
    big = reshape_big()

    # pivot_table on the small frame, one aggregate at a time.
    for agg in ["sum", "mean", "count", "min", "max", "first", "last"]:
        table = small.pivot_table(values="sales", index="region", columns="product", aggfunc=agg)
        expr = f"df.pivot_table(values='sales', index='region', columns='product', aggfunc='{agg}')"
        rows.append((f"pivot_table.{agg}.small.index", table.index.tolist(), expr))
        rows.append((f"pivot_table.{agg}.small.a", table["a"].tolist(), expr))
        rows.append((f"pivot_table.{agg}.small.b", table["b"].tolist(), expr))

    # pivot_table on the large frame: the row count, the non-null cells and their sum.
    for agg in ["sum", "mean", "count"]:
        table = big.pivot_table(values="sales", index="region", columns="product", aggfunc=agg)
        expr = f"big.pivot_table(values='sales', index='region', columns='product', aggfunc='{agg}')"
        total = float(table.sum().sum())
        if agg == "mean":
            total = common.stable(total)
        rows.append((f"pivot_table.{agg}.large.rows", float(len(table)), expr))
        rows.append((f"pivot_table.{agg}.large.cells", float(table.count().sum()), expr))
        rows.append((f"pivot_table.{agg}.large.sum", total, expr))

    # cross_tab: a count per pair of values; a pair with no row is 0.
    ct = pd.crosstab(small["region"], small["product"])
    expr = "pd.crosstab(df['region'], df['product'])"
    rows.append(("cross_tab.small.index", ct.index.tolist(), expr))
    rows.append(("cross_tab.small.a", ct["a"].tolist(), expr))
    rows.append(("cross_tab.small.b", ct["b"].tolist(), expr))
    bct = pd.crosstab(big["region"], big["product"])
    expr = "pd.crosstab(big['region'], big['product'])"
    rows.append(("cross_tab.large.rows", float(len(bct)), expr))
    rows.append(("cross_tab.large.sum", float(bct.sum().sum()), expr))
    rows.append(("cross_tab.large.p0", float(bct["p0"].sum()), expr))

    # stack is melt in pandas: variable-major rows, a null is kept.
    wide = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3], dtype="int64"),
            "a": pd.Series([1.5, 2.5, 3.5], dtype="float64"),
            "b": pd.Series([4.5, 5.5, None], dtype="float64"),
        }
    )
    melted = wide.melt(id_vars=["id"], value_vars=["a", "b"])
    expr = "wide.melt(id_vars=['id'], value_vars=['a', 'b'])"
    rows.append(("stack.small.id", melted["id"].tolist(), expr))
    rows.append(("stack.small.variable", melted["variable"].tolist(), expr))
    rows.append(("stack.small.value", melted["value"].tolist(), expr))
    wide_big = reshape_wide_big()
    melted = wide_big.melt(id_vars=["id"], value_vars=["score_1", "score_2", "score_3"])
    expr = "wide_big.melt(id_vars=['id'], value_vars=['score_1', 'score_2', 'score_3'])"
    rows.append(("stack.large.rows", float(len(melted)), expr))
    rows.append(("stack.large.value_count", float(melted["value"].count()), expr))
    rows.append(("stack.large.value_sum", float(melted["value"].sum()), expr))
    rows.append(("stack.large.id_sum", float(melted["id"].sum()), expr))

    # unstack is pivot in pandas; the index comes back as the id column.
    long = pd.DataFrame(
        {
            "id": pd.Series([1, 1, 2, 2, 3], dtype="int64"),
            "key": ["x", "y", "x", "y", "x"],
            "val": pd.Series([1.5, 2.5, 3.5, 4.5, 5.5], dtype="float64"),
        }
    )
    pivoted = long.pivot(index="id", columns="key", values="val").reset_index()
    expr = "long.pivot(index='id', columns='key', values='val').reset_index()"
    rows.append(("unstack.small.id", pivoted["id"].tolist(), expr))
    rows.append(("unstack.small.x", pivoted["x"].tolist(), expr))
    rows.append(("unstack.small.y", pivoted["y"].tolist(), expr))

    # wide_to_long: the suffix is text in frame, so pandas' int suffix becomes text here.
    stubbed = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3], dtype="int64"),
            "score_1": pd.Series([1.5, 2.5, 3.5], dtype="float64"),
            "score_2": pd.Series([4.5, 5.5, 6.5], dtype="float64"),
            "bonus_1": pd.Series([10.0, 20.0, 30.0], dtype="float64"),
            "bonus_2": pd.Series([40.0, 50.0, 60.0], dtype="float64"),
        }
    )
    long_small = (
        pd.wide_to_long(stubbed, stubnames=["score", "bonus"], i="id", j="t", sep="_")
        .reset_index()
        .sort_values(["id", "t"])
    )
    long_small["t"] = long_small["t"].astype(str)
    expr = "pd.wide_to_long(T, stubnames=['score', 'bonus'], i='id', j='t', sep='_').reset_index().sort_values(['id', 't'])"
    rows.append(("wide_to_long.small.id", long_small["id"].tolist(), expr))
    rows.append(("wide_to_long.small.t", long_small["t"].tolist(), expr))
    rows.append(("wide_to_long.small.score", long_small["score"].tolist(), expr))
    rows.append(("wide_to_long.small.bonus", long_small["bonus"].tolist(), expr))
    long_big = pd.wide_to_long(wide_big, stubnames="score", i="id", j="t", sep="_").reset_index()
    expr = "pd.wide_to_long(wide_big, stubnames='score', i='id', j='t', sep='_').reset_index()"
    rows.append(("wide_to_long.large.rows", float(len(long_big)), expr))
    rows.append(("wide_to_long.large.score_count", float(long_big["score"].count()), expr))
    rows.append(("wide_to_long.large.score_sum", float(long_big["score"].sum()), expr))
    rows.append(("wide_to_long.large.id_sum", float(long_big["id"].sum()), expr))
    return rows


common.main(
    {
        "crates/frame/tests/fixtures/series_reference.tsv": series_rows,
        "crates/frame/tests/fixtures/series_window_reference.tsv": series_window_rows,
        "crates/frame/tests/fixtures/window_reference.tsv": window_rows,
        "crates/frame/tests/fixtures/groupby_reference.tsv": groupby_rows,
        "crates/frame/tests/fixtures/join_reference.tsv": join_rows,
        "crates/frame/tests/fixtures/reshape_reference.tsv": reshape_rows,
    }
)
