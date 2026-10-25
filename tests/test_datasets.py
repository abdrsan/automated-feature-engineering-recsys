from datetime import datetime
import numpy as np
import polars as pl
import pytest

from src.datasets import kion_interactions, yambda_interactions, positive_event
from src.feature_engine import generate_features
from src.feature_spec import FeatureSpec
from src.model import base_matrix, fit_encoder


def kion_fixture():
    interactions = pl.DataFrame({"user_id": [1, 1, 1, 2], "item_id": [10, 11, 12, 10],
        "last_watch_dt": ["2021-01-01", "2021-01-02", "2021-01-02", "2021-01-03"],
        "watched_pct": [10., 75., 60., None], "total_dur": [10, 750, 600, 10]})
    users = pl.DataFrame({"user_id": [1, 2], "age": ["age_25_34", None], "income": [None, None],
                          "sex": ["M", None], "kids_flg": [1, None]}, schema_overrides={"income": pl.String})
    items = pl.DataFrame({"item_id": [10, 11, 12], "content_type": ["film"] * 3,
                          "release_year": [2000., 2001., 2002.], "genres": ["drama, action", "comedy", None],
                          "countries": [None] * 3, "for_kids": [None] * 3, "age_rating": [12., 16., 18.],
                          "directors": ["A, B", "C", None]}, schema_overrides={"countries": pl.String})
    return interactions, users, items


def test_kion_real_schema_and_target_are_safe():
    df = kion_interactions(*kion_fixture())
    assert df.height == 3
    assert df["label"].to_list() == [0, 1, 1]
    assert set(df["event"]) == {"watch_high", "watch_low"}
    assert positive_event(set(df["event"])) == "watch_high"
    assert "watched_pct" not in df.columns and "total_dur" not in df.columns
    assert df["category_id"][0] == "drama" and df["author"][0] == "A"
    x, names, _ = base_matrix(df, fit_encoder(df))
    assert not {"history_value", "label", "event", "total_dur", "watched_pct"} & set(names)
    changed = df.with_columns((100 - pl.col("history_value")).alias("history_value"))
    np.testing.assert_array_equal(x, base_matrix(changed, fit_encoder(df))[0])


def test_kion_all_current_day_outcomes_are_excluded():
    df = kion_interactions(*kion_fixture())
    specs = [FeatureSpec(("user_id",)), FeatureSpec(("user_id",), "watch_high"),
             FeatureSpec(("user_id",), "all", "mean", None, "history_value")]
    values, _ = generate_features(df, specs)
    assert values[specs[0].name].to_list() == [0, 1, 1]
    assert values[specs[1].name].to_list() == [0, 0, 0]
    assert values[specs[2].name].to_list()[1:] == [10., 10.]


def yambda_fixture():
    likes = pl.DataFrame({"uid": [7, 7], "item_id": [100, 101], "timestamp": [0, 86400], "is_organic": [0, 1]})
    dislikes = pl.DataFrame({"uid": [7], "item_id": [102], "timestamp": [86400], "is_organic": [0]})
    artists = pl.DataFrame({"item_id": [100, 100, 101, 102], "artist_id": [5, 3, 2, 6]})
    albums = pl.DataFrame({"item_id": [100, 101, 102], "album_id": [10, 11, 12]})
    return likes, dislikes, artists, albums


def test_yambda_seconds_are_not_multiplied_by_five_and_no_calendar():
    df = yambda_interactions(*yambda_fixture())
    assert df["timestamp"][1] == datetime(1970, 1, 2)
    assert df.height == 3 and df["author"][0] == "3"
    assert positive_event(set(df["event"])) == "like"
    _, names, _ = base_matrix(df, fit_encoder(df))
    assert "hour" not in names and "weekday" not in names
    spec = FeatureSpec(("user_id",), "like", "count", 1)
    f, _ = generate_features(df, [spec])
    assert f[spec.name].to_list() == [0, 1, 1]


def test_yambda_invalid_timestamp_granularity_rejected():
    likes, dislikes, artists, albums = yambda_fixture()
    likes = likes.with_columns((pl.col("timestamp") + 1).alias("timestamp"))
    with pytest.raises(ValueError):
        yambda_interactions(likes, dislikes, artists, albums)
