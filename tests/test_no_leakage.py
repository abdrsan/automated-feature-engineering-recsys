from datetime import datetime, timedelta
import numpy as np
import polars as pl

from src.feature_engine import generate_features
from src.feature_spec import FeatureSpec
from src.split import temporal_split


def example():
    return pl.DataFrame({
        "row_id": [0, 1, 2, 3], "user_id": [1, 1, 1, 1], "item_id": [5, 6, 7, 8],
        "timestamp": [datetime(2020, 1, d) for d in [1, 2, 2, 5]],
        "event": ["negative", "positive", "positive", "negative"],
        "label": [0, 1, 1, 0], "rating": [1., 4., 5., 2.],
    })


def test_jan2_sees_only_jan1_and_excludes_ties():
    df = example()
    specs = [FeatureSpec(("user_id",)), FeatureSpec(("user_id",), "positive"),
             FeatureSpec(("user_id",), "all", "count", 1)]
    f, meta = generate_features(df, specs)
    assert f[specs[0].name].to_list() == [0, 1, 1, 3]
    assert f[specs[1].name].to_list() == [0, 0, 0, 2]
    assert f[specs[2].name].to_list() == [0, 1, 1, 0]
    assert meta["audit_passed"]


def test_future_and_current_target_mutation_cannot_change_prefix_features():
    df = example()
    specs = [FeatureSpec(("user_id",), event, agg, window,
                         value_column="rating" if agg == "mean" else None)
             for event in ["all", "positive", "negative"]
             for agg in ["count", "mean", "recency", "event_rate"]
             for window in [None, 1, 3]]
    before, _ = generate_features(df, specs)
    changed = df.with_columns(
        pl.when(pl.col("row_id") >= 1).then(pl.lit(1000.)).otherwise(pl.col("rating")).alias("rating"),
        pl.when(pl.col("row_id") >= 1).then(pl.lit("negative")).otherwise(pl.col("event")).alias("event"),
    )
    # Сохраняем оба типа событий в схеме. Проверяем влияние измененных исходов,
    # а не ошибку валидации из-за исчезнувшего названия события.
    changed = changed.with_columns(pl.when(pl.col("row_id") == 3).then(pl.lit("positive")).otherwise(pl.col("event")).alias("event"))
    after, _ = generate_features(changed, specs)
    np.testing.assert_allclose(before.head(3).drop("row_id").to_numpy(),
                               after.head(3).drop("row_id").to_numpy(), equal_nan=True)


def test_prefix_generation_matches_full_generation():
    df = example()
    specs = [FeatureSpec(("user_id",), "all", agg, w)
             for agg in ["count", "recency"] for w in [None, 1, 3]]
    full, _ = generate_features(df, specs)
    prefix, _ = generate_features(df.head(3), specs)
    np.testing.assert_allclose(full.head(3).drop("row_id").to_numpy(),
                               prefix.drop("row_id").to_numpy(), equal_nan=True)


def test_split_never_separates_equal_timestamps():
    df = pl.DataFrame({"timestamp": [datetime(2020, 1, 1) + timedelta(days=d)
                                      for d in range(30) for _ in range(3)]})
    s = temporal_split(df)
    for b in [s.train_end, s.search_end, s.validation_end]:
        assert df["timestamp"][b - 1] < df["timestamp"][b]
