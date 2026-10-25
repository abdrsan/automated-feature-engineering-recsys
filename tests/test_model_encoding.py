from datetime import datetime, timedelta
import numpy as np
import polars as pl
from src.model import fit_encoder, base_matrix, matrix_from_features, take_rows


def test_unseen_categories_are_unknown_and_current_targets_are_not_inputs():
    df = pl.DataFrame({"user_id": [1, 1, 99], "item_id": [10, 11, 999],
                       "gender": ["F", "F", "M"], "rating": [5, 1, 4], "label": [1, 0, 1],
                       "timestamp": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3)]})
    encoder = fit_encoder(df.head(2))
    x, names, categorical = base_matrix(df, encoder)
    assert "rating" not in names and "label" not in names
    assert np.all(x[2, categorical] == -1)
    mutated = df.with_columns((10 - pl.col("rating")).alias("rating"), (1 - pl.col("label")).alias("label"))
    other, _, _ = base_matrix(mutated, encoder)
    np.testing.assert_array_equal(x, other)


def test_wide_matrix_preserves_column_order_missing_values_and_row_selection():
    base = np.array([[1., 10.], [2., 20.], [3., 30.]], dtype=np.float32)
    features = pl.DataFrame({"b": [20., None, 60.], "a": [10., 20., 30.]})
    matrix = matrix_from_features(base, features, ["a", "b"])
    np.testing.assert_allclose(matrix, [[1., 10., 10., 20.], [2., 20., 20., np.nan], [3., 30., 30., 60.]], equal_nan=True)
    assert np.shares_memory(take_rows(matrix, np.arange(1, 3)), matrix)
    np.testing.assert_allclose(take_rows(matrix, np.array([2, 0])), matrix[[2, 0]], equal_nan=True)
