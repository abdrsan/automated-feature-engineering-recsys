from datetime import datetime, timedelta
import numpy as np
import polars as pl
from src.representations import temporal_svd_features


def test_frozen_basis_and_user_profile_ignore_current_and_future_targets():
    df = pl.DataFrame({"row_id": list(range(12)),
                       "user_id": [1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2],
                       "item_id": [10, 11, 12, 11, 12, 10, 10, 11, 12, 11, 12, 10],
                       "timestamp": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(12)],
                       "label": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]})
    f, _ = temporal_svd_features(df, 6, dimensions=2)
    assert f.head(6).drop("row_id").null_count().row(0) == (6, 6, 6, 6)
    mutated = df.with_columns(pl.when(pl.col("row_id") >= 7).then(1 - pl.col("label")).otherwise(pl.col("label")).alias("label"))
    g, _ = temporal_svd_features(mutated, 6, dimensions=2)
    np.testing.assert_allclose(f.head(8).drop("row_id").to_numpy(),
                               g.head(8).drop("row_id").to_numpy(), equal_nan=True)
