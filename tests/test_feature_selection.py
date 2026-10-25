import numpy as np
import polars as pl
from src.feature_selection import prune_candidates


def test_pruning_uses_training_prefix_and_handles_one_retained_feature():
    features = pl.DataFrame({'row_id': list(range(10)), 'a': np.arange(10, dtype=float),
                             'duplicate': np.arange(10, dtype=float),
                             'only_future': [0.] * 5 + [1.,2.,3.,4.,5.]})
    config = {'max_missing_fraction': .99, 'near_constant_fraction': .999,
              'correlation_threshold': .999}
    keep, reasons = prune_candidates(features, 5, config)
    assert keep == ['a']
    assert set(reasons['feature']) == {'duplicate','only_future'}
