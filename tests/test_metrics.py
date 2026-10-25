from datetime import datetime
import numpy as np
import polars as pl
from src.metrics import ranking_metrics, paired_user_bootstrap


def test_perfect_and_reversed_rankings():
    df = pl.DataFrame({"row_id": list(range(5)), "user_id": [1] * 5,
                       "timestamp": [datetime(2020, 1, 1)] * 5, "label": [1, 1, 0, 0, 0]})
    perfect = ranking_metrics(df, np.array([.9, .8, .3, .2, .1]))
    reversed_ = ranking_metrics(df, np.array([.1, .2, .7, .8, .9]))
    assert np.isclose(perfect["ndcg_3"], 1)
    assert perfect["recall_3"] == 1
    assert perfect["map_3"] == 1
    assert reversed_["recall_3"] == 0
    assert perfect["ndcg_5"] > reversed_["ndcg_5"]


def test_different_times_do_not_form_one_ranking_query():
    df = pl.DataFrame({"row_id": [0, 1], "user_id": [1, 1],
                       "timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 2)], "label": [1, 0]})
    assert ranking_metrics(df, [.9, .1])["ranking_groups"] == 0


def test_identical_models_have_zero_paired_difference():
    df = pl.DataFrame({"user_id": [1, 1, 2, 2], "label": [0, 1, 0, 1]})
    result = paired_user_bootstrap(df, np.array([.2, .8, .4, .6]), np.array([.2, .8, .4, .6]), repeats=10)
    assert result["logloss_ci_95"] == [0., 0.]
    assert result["auc_ci_95"] == [0., 0.]


def test_repeated_item_in_same_time_is_not_a_ranking_candidate_set():
    df = pl.DataFrame({"row_id": [0, 1, 2, 3], "user_id": [1, 1, 2, 2],
                       "item_id": [10, 10, 20, 21], "timestamp": [datetime(2020, 1, 1)] * 4,
                       "label": [1, 0, 1, 0]})
    result = ranking_metrics(df, np.array([.5, .5, .9, .1]))
    assert result["ranking_groups"] == 1
    assert result["ranking_groups_excluded_duplicates"] == 1
    assert np.isclose(result["ndcg_5"], 1)
