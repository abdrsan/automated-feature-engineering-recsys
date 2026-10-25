from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.feature_engine import generate_features
from src.feature_spec import FeatureSpec


@pytest.fixture
def events():
    start = datetime(2020, 1, 1)
    return pl.DataFrame({
        "row_id": list(range(7)), "user_id": [1, 1, 1, 2, 1, 1, 2],
        "item_id": [10, 11, 12, 10, 10, 10, 11],
        "timestamp": [start + timedelta(days=d) for d in [0, 1, 1, 1, 4, 5, 6]],
        "event": ["negative", "positive", "positive", "positive", "negative", "positive", "negative"],
        "rating": [2., 4., 5., 4., 3., 5., 1.], "label": [0, 1, 1, 1, 0, 1, 0],
    })


def reference(df, spec):
    result = []
    for row in df.iter_rows(named=True):
        history = [h for h in df.iter_rows(named=True)
                   if h["timestamp"] < row["timestamp"]
                   and all(h[k] == row[k] for k in spec.keys)
                   and (spec.window_days is None or
                        h["timestamp"] >= row["timestamp"] - timedelta(days=spec.window_days))]
        selected = [h for h in history if spec.event == "all" or h["event"] == spec.event]
        vals = [h[spec.value_column] for h in selected] if spec.value_column else []
        agg = spec.aggregation
        if agg == "count": value = len(selected)
        elif agg == "sum": value = sum(vals)
        elif agg == "mean": value = np.mean(vals) if vals else np.nan
        elif agg == "min": value = min(vals) if vals else np.nan
        elif agg == "max": value = max(vals) if vals else np.nan
        elif agg == "nunique": value = len(set(vals))
        elif agg == "recency":
            value = (row["timestamp"] - max(h["timestamp"] for h in selected)).total_seconds() / 86400 if selected else np.nan
        elif agg in {"event_rate", "conversion"}:
            denom = len([h for h in history if spec.denominator_event == "all" or h["event"] == spec.denominator_event])
            value = (len(selected) + spec.smoothing) / (denom + 2 * spec.smoothing)
        elif agg == "frequency":
            duration = spec.window_days or (max(1, (row["timestamp"] - min(h["timestamp"] for h in history)).total_seconds() / 86400) if history else 1)
            value = len(selected) / duration
        elif agg == "trend":
            longer = [h for h in df.iter_rows(named=True)
                      if h["timestamp"] < row["timestamp"]
                      and all(h[k] == row[k] for k in spec.keys)
                      and (spec.event == "all" or h["event"] == spec.event)
                      and (spec.long_window_days is None or h["timestamp"] >= row["timestamp"] - timedelta(days=spec.long_window_days))]
            value = (len(selected) + spec.smoothing) / (len(longer) + spec.smoothing)
        result.append(value)
    return result


@pytest.mark.parametrize("keys", [(), ("user_id",), ("item_id",), ("user_id", "item_id")])
@pytest.mark.parametrize("event", ["all", "positive", "negative"])
@pytest.mark.parametrize("window", [None, 1, 3])
def test_all_aggregations_against_slow_oracle(events, keys, event, window):
    specs = [FeatureSpec(keys, event, agg, window,
                         value_column="item_id" if agg == "nunique" else
                         "rating" if agg in {"sum", "mean", "min", "max"} else None)
             for agg in ["count", "sum", "mean", "min", "max", "nunique", "recency", "event_rate", "conversion", "frequency"]]
    specs.append(FeatureSpec(keys, event, "trend", 1, long_window_days=3))
    actual, _ = generate_features(events, specs)
    for spec in specs:
        np.testing.assert_allclose(actual[spec.name].to_numpy(), reference(events, spec),
                                   atol=1e-6, equal_nan=True, err_msg=spec.name)


def test_invalid_spec_is_rejected(events):
    for spec in [FeatureSpec(("label",)), FeatureSpec(("user_id",), "DROP_TABLE"),
                 FeatureSpec(("user_id",), window_days=0),
                 FeatureSpec(("user_id",), aggregation="mean"),
                 FeatureSpec(("user_id;DELETE",))]:
        with pytest.raises(ValueError):
            generate_features(events, [spec])
