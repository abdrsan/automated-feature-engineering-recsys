"""Temporal feature and metric checks against direct calculations on small data."""
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.feature_engine import generate_features
from src.feature_spec import FeatureSpec
from src.metrics import ranking_metrics
from src.split import temporal_split


def toy():
    return pl.DataFrame({"row_id": range(7), "user_id": [1, 1, 1, 1, 2, 1, 2],
        "item_id": [10, 10, 10, 10, 10, 11, 11], "category_id": [1, 1, 1, 1, 1, 2, 2],
        "timestamp": [datetime(2020, 1, 1) + timedelta(days=d) for d in [0, 1, 1, 4, 4, 7, 9]],
        "event": ["click", "click", "purchase", "purchase", "click", "view", "click"],
        "label": [0, 0, 1, 1, 0, 0, 0], "amount": [2., None, 4., 8., 1., 3., 9.]})


def expected_value(frame, spec, query):
    """Filter raw records directly; never uses engine primitives or model outputs."""
    qtime = query["timestamp"]
    condition = pl.col("timestamp") < qtime
    for key in spec.keys:
        condition &= pl.col(key).is_null() if query[key] is None else pl.col(key) == query[key]
    if spec.window_days is not None:
        condition &= pl.col("timestamp") >= qtime - timedelta(days=spec.window_days)
    history = frame.filter(condition)
    event_mask = np.ones(history.height, bool) if spec.event == "all" else history["event"].to_numpy() == spec.event
    chosen = history.filter(pl.Series(event_mask))
    count = chosen.height
    agg = spec.aggregation
    values = chosen[spec.value_column].drop_nulls().to_numpy() if spec.value_column else None
    if agg == "count": return count
    if agg == "sum": return float(np.sum(values))
    if agg == "nunique": return len(np.unique(values))
    if agg in {"mean", "min", "max"}:
        return float(getattr(np, agg)(values)) if len(values) else np.nan
    if agg == "recency":
        return (qtime - chosen["timestamp"].max()).total_seconds() / 86400 if count else np.nan
    if agg in {"event_rate", "conversion"}:
        denominator = history.height if spec.denominator_event == "all" else history.filter(pl.col("event") == spec.denominator_event).height
        return (count + spec.smoothing) / (denominator + 2 * spec.smoothing)
    if agg == "frequency":
        duration = spec.window_days or (max(1., (qtime - history["timestamp"].min()).total_seconds()/86400) if history.height else 1.)
        return count / duration
    if agg == "trend":
        long_spec = FeatureSpec(spec.keys, spec.event, "count", spec.long_window_days)
        return (count + spec.smoothing) / (expected_value(frame, long_spec, query) + spec.smoothing)
    raise AssertionError(agg)


@pytest.mark.parametrize("keys", [(), ("user_id",), ("item_id",), ("user_id", "item_id"), ("user_id", "category_id")])
@pytest.mark.parametrize("window", [None, 1, 3])
def test_raw_history_oracle_with_ties_and_null_values(keys, window):
    data = toy()
    specs = [FeatureSpec(keys, e, a, window, value_column="amount" if a in {"sum","mean","min","max"} else "item_id" if a == "nunique" else None)
             for e in ["all", "click", "purchase"] for a in ["count", "sum", "mean", "min", "max", "nunique", "recency", "event_rate", "frequency"]]
    specs += [FeatureSpec(keys, "purchase", "conversion", window, denominator_event="click", smoothing=2),
              FeatureSpec(keys, "click", "trend", 1, long_window_days=7)]
    out, _ = generate_features(data, specs)
    assert out["row_id"].to_list() == data["row_id"].to_list()
    for spec in specs:
        expected = [expected_value(data, spec, row) for row in data.iter_rows(named=True)]
        np.testing.assert_allclose(out[spec.name].to_numpy(), expected, rtol=2e-6, atol=1e-6, equal_nan=True, err_msg=spec.name)


def test_requested_jan_1_jan_2_jan_5_example():
    data = toy().filter(pl.col("row_id").is_in([0, 1, 3])).with_row_index("new").drop("row_id").rename({"new":"row_id"})
    spec = FeatureSpec(("user_id", "item_id"), "click", "count")
    result, _ = generate_features(data, [spec])
    assert result[spec.name].to_list() == [0., 1., 2.]


def test_current_and_future_outcomes_do_not_change_any_earlier_feature():
    data = toy()
    specs = [FeatureSpec(k, e, a, w, "amount" if a == "mean" else None)
             for k in [("user_id",), ("item_id",), ("user_id", "item_id"), ("user_id", "category_id")]
             for e in ["all", "click", "purchase"] for a in ["count", "mean", "recency", "event_rate"] for w in [None, 3]]
    cutoff = datetime(2020, 1, 5)
    poisoned = data.with_columns(
        pl.when(pl.col("timestamp") >= cutoff).then(pl.lit("purchase")).otherwise(pl.col("event")).alias("event"),
        pl.when(pl.col("timestamp") >= cutoff).then(1e8).otherwise(pl.col("amount")).alias("amount"))
    before, _ = generate_features(data, specs)
    after, _ = generate_features(poisoned, specs)
    ids = data.filter(pl.col("timestamp") <= cutoff)["row_id"]
    np.testing.assert_allclose(before.filter(pl.col("row_id").is_in(ids.implode())).drop("row_id").to_numpy(),
                               after.filter(pl.col("row_id").is_in(ids.implode())).drop("row_id").to_numpy(), equal_nan=True)


def test_null_entity_join_does_not_expand_or_drop_queries():
    data = toy().with_columns(pl.when(pl.col("row_id") % 2 == 0).then(None).otherwise(pl.col("category_id")).alias("category_id"))
    specs = [FeatureSpec(("user_id", "category_id"), "all", "count", w) for w in [None, 3]]
    out, _ = generate_features(data, specs)
    assert out.height == data.height and out["row_id"].n_unique() == data.height
    assert not np.isinf(out.drop("row_id").to_numpy()).any()
    assert not out.drop("row_id").null_count().to_numpy().any()


def test_ranking_formula_single_positive_at_rank_two():
    data = pl.DataFrame({"user_id":[1,1,1], "item_id":[10,11,12], "row_id":[0,1,2],
                         "timestamp":[datetime(2020,1,1)]*3, "label":[0,1,0]})
    metrics = ranking_metrics(data, np.array([.9,.8,.7]), ks=(1,2,3))
    assert metrics["ndcg_1"] == 0 and metrics["recall_1"] == 0 and metrics["map_1"] == 0
    assert metrics["ndcg_3"] == pytest.approx(1/np.log2(3))
    assert metrics["recall_3"] == 1 and metrics["map_3"] == .5


def test_rank_duplicate_item_query_rejected():
    data = pl.DataFrame({"user_id":[1,1], "item_id":[10,10], "row_id":[0,1],
                         "timestamp":[datetime(2020,1,1)]*2, "label":[0,1]})
    result = ranking_metrics(data, np.array([.2,.9]))
    assert result["ranking_groups"] == 0 and result["ranking_groups_excluded_duplicates"] == 1


def test_temporal_parts_exclude_equal_timestamp_and_future():
    data = pl.DataFrame({"timestamp":[datetime(2020,1,1)+timedelta(hours=i//4) for i in range(200)]})
    split = temporal_split(data)
    parts = [data[split.train], data[split.search], data[split.audit], data[split.test]]
    assert sum(p.height for p in parts) == data.height
    assert all(a["timestamp"].max() < b["timestamp"].min() for a,b in zip(parts,parts[1:]))


def test_manual_reference_cannot_ignore_a_better_validated_ablation(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    import scripts.run_study as pipeline
    data=pl.concat([toy().with_columns(pl.col("timestamp")+pl.duration(days=14*i)) for i in range(8)]).drop("row_id").with_row_index("row_id")
    losses={"plus_user_temporal":.10,"manual":.30,"manual_plus_trend":.25}
    class StubRunner:
        def __init__(self,*args): pass
        def matrix(self,names): return np.zeros((data.height,len(names)))
        def run(self,name,*args,**kwargs): return {"logloss":losses.get(name,.4)},{}
    monkeypatch.setattr(pipeline,"ExperimentRunner",StubRunner)
    monkeypatch.setattr(pipeline,"tune_model",lambda *a,**k: ({},SimpleNamespace(to_csv=lambda *a,**k:None)))
    config={"seed":42,"dataset_name":"synthetic"}
    pipeline.develop(data,temporal_split(data),config,{},tmp_path,only_baseline=True)
    selected=json.loads((tmp_path/"baseline_selection.json").read_text())
    assert selected["name"]=="plus_user_temporal"


def test_cache_invalidated_by_engine_version(monkeypatch,tmp_path):
    import scripts.run_study as pipeline
    data=toy()
    calls=[]
    def generate(frame,specs,**kwargs):
        calls.append(1)
        return frame.select("row_id").with_columns(pl.lit(float(len(calls))).alias(specs[0].name)),{}
    monkeypatch.setattr(pipeline,"generate_features",generate)
    spec=FeatureSpec(("user_id",))
    monkeypatch.setattr(pipeline,"FEATURE_ENGINE_VERSION","audit-original",raising=False)
    a,_=pipeline.cached_features(data,[spec],tmp_path,"audit")
    monkeypatch.setattr(pipeline,"FEATURE_ENGINE_VERSION","audit-changed",raising=False)
    b,_=pipeline.cached_features(data,[spec],tmp_path,"audit")
    assert len(calls)==2 and not a.equals(b)


def test_tpe_resume_matches_uninterrupted_history(tmp_path):
    from src.model import create_tuning_study, save_tuning_sampler
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    def objective(trial):
        return (trial.suggest_float("x",-2,3)-.25)**2 + trial.suggest_int("y",1,4)/10
    def run(database,count):
        study,path=create_tuning_study(f"sqlite:///{database}","audit",47)
        study.optimize(objective,n_trials=count,callbacks=[lambda s,t:save_tuning_sampler(s,path)])
        return study
    continuous=run(tmp_path/"continuous.sqlite3",14)
    run(tmp_path/"resumed.sqlite3",6)
    resumed=run(tmp_path/"resumed.sqlite3",8)
    assert [t.params for t in continuous.trials]==[t.params for t in resumed.trials]
    assert [t.value for t in continuous.trials]==[t.value for t in resumed.trials]


def test_split_rejects_unsorted_input():
    data=toy().reverse()
    with pytest.raises(ValueError,match="sorted"):
        temporal_split(data)


def test_full_eda_generated_with_all_expected_partitions():
    from src.data import study_statistics
    data=pl.concat([toy().with_columns(pl.col("timestamp")+pl.duration(days=14*i)) for i in range(8)]).drop("row_id").with_row_index("row_id")
    result=study_statistics(data,temporal_split(data))
    assert result["rows"]==data.height
    assert set(result["splits"])=={"train","validation_search","validation_audit","test"}
    assert sum(row["rows"] for row in result["splits"].values())==data.height
    assert result["splits"]["test"]["positive_rate"] is None


def test_changed_engine_helper_changes_cache_signature(monkeypatch):
    import src.feature_engine as engine
    before=engine.feature_engine_signature()
    def different_mask(event):
        return pl.lit(False)
    monkeypatch.setattr(engine,"_mask",different_mask)
    assert engine.feature_engine_signature()!=before


def test_model_resume_rejects_stale_implementation(tmp_path):
    import json
    from types import SimpleNamespace
    from src.experiments import ExperimentRunner
    from src.feature_engine import feature_engine_signature
    from src.provenance import STUDY_PROTOCOL_VERSION
    (tmp_path/"specs").mkdir()
    pl.DataFrame({"experiment_name":["audit"],"seed":[42],"split":["validation_search"],"logloss":[.5]}).write_csv(tmp_path/"experiments.csv")
    (tmp_path/"specs/audit_seed42.json").write_text(json.dumps({"features":[],"params":{},"train_start":0,
        "protocol_version":STUDY_PROTOCOL_VERSION,"engine_signature":feature_engine_signature(),"fit_signature":"stale"}))
    data=toy()
    runner=ExperimentRunner(data,data.select("row_id"),np.zeros((data.height,1)),["user_id"],[0],
                            SimpleNamespace(train=np.arange(3)),{"seed":42},tmp_path)
    with pytest.raises(ValueError,match="implementation differs"):
        runner.run("audit",[],{})


def test_reference_sum_sql_matches_polars_for_empty_history():
    import sqlite3
    from src.feature_spec import to_sql
    data=toy()
    database=sqlite3.connect(":memory:")
    database.execute("CREATE TABLE interactions(row_id INT,user_id INT,timestamp TEXT,event TEXT,amount REAL)")
    database.executemany("INSERT INTO interactions VALUES(?,?,?,?,?)",data.select("row_id","user_id","timestamp","event","amount").with_columns(pl.col("timestamp").cast(pl.String)).iter_rows())
    spec=FeatureSpec(("user_id",),"all","sum",value_column="amount")
    sql_values=dict(database.execute(to_sql(spec)).fetchall())
    actual,_=generate_features(data,[spec])
    assert [sql_values[i] for i in data["row_id"]]==actual[spec.name].to_list()
    database.close()


def test_fixed_pool_keeps_fixed_alias_of_adaptive_only_duplicate():
    from src.feature_selection import prune_candidates, prune_fixed_candidates
    data=pl.DataFrame({"row_id":range(8),"user_id":[1]*8,"item_id":range(8),
        "event":["click"]*8,"label":[0]*8,
        "timestamp":[datetime(2020,1,1)+timedelta(hours=i) for i in range(6)]
                    +[datetime(2020,1,4),datetime(2020,1,5)]})
    specs=[FeatureSpec(("user_id",),"click","count",w) for w in [1,7,None]]
    matrix,_=generate_features(data,specs)
    config={"max_missing_fraction":.999,"near_constant_fraction":.9999,"correlation_threshold":.9999}
    adaptive,_=prune_candidates(matrix,6,config)
    assert adaptive==[specs[0].name]  # all three columns equal on the training prefix
    fixed,_=prune_fixed_candidates(matrix,specs,6,config)
    assert specs[1].name in fixed and specs[0].name not in fixed
    assert matrix[specs[1].name][6]==6 and matrix[specs[0].name][6]==0


def test_reference_sql_matches_null_entity_keys():
    import sqlite3
    from src.feature_spec import to_sql
    data=toy().with_columns(pl.when(pl.col("row_id")<4).then(None).otherwise(pl.col("user_id")).alias("user_id"))
    database=sqlite3.connect(":memory:")
    database.execute("CREATE TABLE interactions(row_id INT,user_id INT,timestamp TEXT,event TEXT)")
    database.executemany("INSERT INTO interactions VALUES(?,?,?,?)",data.select(
        "row_id","user_id","timestamp","event").with_columns(pl.col("timestamp").cast(pl.String)).iter_rows())
    spec=FeatureSpec(("user_id",),"all","count")
    sql_values=dict(database.execute(to_sql(spec)).fetchall())
    actual,_=generate_features(data,[spec])
    assert [sql_values[i] for i in data["row_id"]]==actual[spec.name].to_list()
    database.close()
