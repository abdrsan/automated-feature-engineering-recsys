"""Independent evidence collection; does not write research metrics or select models."""
import argparse
from collections import defaultdict
import gc
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
import zipfile

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import rankdata

from audit.test_independent import expected_value
from src.data import save_json
from src.datasets import load_kion, load_yambda
from src.feature_engine import generate_features
from src.feature_spec import FeatureSpec
from src.model import base_matrix, matrix_from_features
from src.representations import temporal_svd_features


def independent_metrics(frame, probabilities):
    labels = frame["label"].to_numpy().astype(float)
    p = np.clip(probabilities, np.finfo(float).eps, 1-np.finfo(float).eps)
    positives = labels.sum()
    auc = (rankdata(probabilities)[labels == 1].sum() - positives*(positives+1)/2) / (positives*(len(labels)-positives))
    output = {"auc": float(auc), "logloss": float(np.mean(-labels*np.log(p)-(1-labels)*np.log1p(-p)))}
    groups = defaultdict(list)
    for row, probability in zip(frame.select("user_id", "timestamp", "item_id", "row_id", "label").iter_rows(), probabilities):
        user, time, item, row_id, label = row
        groups[(user,time)].append((float(probability), row_id, item, label))
    values = defaultdict(list)
    eligible_groups, eligible_rows, duplicate_groups = 0, 0, 0
    for entries in groups.values():
        npos = sum(e[3] for e in entries)
        if not 0 < npos < len(entries):
            continue
        if len({e[2] for e in entries}) != len(entries):
            duplicate_groups += 1
            continue
        eligible_groups += 1
        eligible_rows += len(entries)
        ranked = sorted(entries, key=lambda e: (-e[0], e[1]))
        for k in [3,5,10]:
            hits, dcg, ap = 0, 0., 0.
            for rank, entry in enumerate(ranked[:k],1):
                if entry[3]:
                    hits += 1
                    dcg += 1/np.log2(rank+1)
                    ap += hits/rank
            ideal = sum(1/np.log2(rank+1) for rank in range(1,min(k,npos)+1))
            values[f"ndcg_{k}"].append(dcg/ideal)
            values[f"recall_{k}"].append(hits/npos)
            values[f"map_{k}"].append(ap/min(k,npos))
    output.update({key: float(np.mean(value)) for key,value in values.items()})
    output.update(ranking_groups=eligible_groups, ranking_rows=eligible_rows,
                  ranking_groups_excluded_duplicates=duplicate_groups)
    return output


def independent_user_bootstrap(frame, baseline, automatic, repeats=300, seed=2026):
    labels = frame["label"].to_numpy()
    def point_loss(probabilities):
        probabilities = np.clip(probabilities, 1e-15, 1-1e-15)
        return -labels*np.log(probabilities)-(1-labels)*np.log1p(-probabilities)
    delta = point_loss(automatic)-point_loss(baseline)
    # Порядок group_by может меняться между процессами. Фиксируем его до random draw,
    # чтобы одному seed соответствовала одна и та же выборка пользователей.
    per_user = frame.select("user_id").with_columns(pl.Series("delta", delta)).group_by("user_id").agg(
        pl.col("delta").sum(), pl.len().alias("n")).sort("user_id")
    sums, counts = per_user["delta"].to_numpy(), per_user["n"].to_numpy()
    random = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(repeats):
        draw = random.integers(0, len(sums), len(sums))
        bootstrap.append(float(sums[draw].sum()/counts[draw].sum()))
    return {"delta_logloss": float(delta.mean()), "cluster": "user_id", "random_seed": seed,
            "repeats": repeats, "ci_95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "conditional_on_fixed_models": True, "cluster_order": "user_id ascending"}


def stats(frame):
    counts = {}
    for entity in ["user_id", "item_id"]:
        c = frame.group_by(entity).len()["len"]
        counts[entity] = {"min":c.min(),"mean":c.mean(),"median":c.median(),"p95":c.quantile(.95),"max":c.max()}
    return {"rows":frame.height,"users":frame["user_id"].n_unique(),"items":frame["item_id"].n_unique(),
        "start":str(frame["timestamp"].min()),"end":str(frame["timestamp"].max()),
        "labels":frame.group_by("label").len().sort("label").to_dicts(),"positive_rate":frame["label"].mean(),
        "missing":frame.null_count().row(0,named=True),"cardinality":{c:frame[c].n_unique() for c in frame.columns},
        "duplicate_events":frame.height-frame.unique(["user_id","item_id","timestamp","label"]).height,
        "duplicate_pairs":frame.height-frame.unique(["user_id","item_id"]).height,"interaction_counts":counts}


def check_dataset(name, output, regenerate=True, source_root="artifacts"):
    tick = perf_counter()
    root = Path(source_root)/name
    output = output/name
    output.mkdir(parents=True,exist_ok=True)
    df = load_kion() if name=="kion" else load_yambda()
    manifest = json.loads((root/"run_manifest.json").read_text())
    lock = json.loads((root/"locked_selection.json").read_text())
    bounds = manifest["split"]
    assert bounds["n_rows"] == df.height
    train_end, search_end, valid_end = [bounds[k] for k in ["train_end","search_end","validation_end"]]
    # Reconstruct parts directly from committed boundaries, without importing split.py.
    split_stats = {}
    for part,start,end in [("train",0,train_end),("search",train_end,search_end),("audit",search_end,valid_end),("test",valid_end,df.height)]:
        subset = df.slice(start,end-start)
        split_stats[part] = {"start":str(subset["timestamp"].min()),"end":str(subset["timestamp"].max()),"rows":subset.height,"positive_rate":subset["label"].mean()}
        if start:
            assert df["timestamp"][start-1] < subset["timestamp"].min()
    result = stats(df)
    if name == "kion":
        with zipfile.ZipFile("data/raw/kion/interactions.csv.zip") as archive:
            raw = pl.read_csv(io.BytesIO(archive.read("interactions.csv")), infer_schema_length=10000)
        result["raw_rows"] = raw.height
        result["raw_missing_target"] = raw["watched_pct"].null_count()
        assert raw.height - raw["watched_pct"].null_count() == df.height
        assert raw.filter(pl.col("watched_pct")>=50).height == int(df["label"].sum())
        aligned=df.select("user_id","item_id","label").join(
            raw.select("user_id","item_id",(pl.col("watched_pct")>=50).cast(pl.Int8).alias("raw_label")),
            on=["user_id","item_id"],validate="1:1")
        assert aligned.height==df.height and (aligned["label"]==aligned["raw_label"]).all()
        del raw
    else:
        result["raw_event_rows"] = {event:pl.scan_parquet(f"data/raw/yambda/{event}.parquet").select(pl.len()).collect().item() for event in ["likes","dislikes"]}
        assert sum(result["raw_event_rows"].values()) == df.height
        assert result["raw_event_rows"]["likes"] == int(df["label"].sum())
        raw=pl.concat([pl.read_parquet(f"data/raw/yambda/{event}.parquet").rename({"uid":"user_id"}).with_columns(
            pl.lit(label,pl.Int8).alias("label"),pl.from_epoch("timestamp",time_unit="s"))
            for event,label in [("likes",1),("dislikes",0)]])
        identity=["user_id","item_id","timestamp","is_organic","label"]
        assert raw.select(identity).sort(identity).equals(df.select(identity).sort(identity))
        del raw
    result["raw_target_alignment_verified_all_rows"]=True
    old_eda = json.loads((root/"eda_full.json").read_text())
    for key in ["rows","users","items","start","end","duplicate_events"]:
        assert old_eda[key] == result[key], (name,key,old_eda[key],result[key])
    save_json(output/"data_statistics.json",{**result,"splits":split_stats})

    initial = root/"source_manifest.json"
    for source in json.loads(initial.read_text())["files"]:
        raw_path=Path("data/raw")/name/source["name"]
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == source["sha256"]
    all_specs = {s.name:s for s in map(FeatureSpec.from_dict,lock["all_specs"])}
    names = list(dict.fromkeys(lock["manual_features"]+lock["auto_features"]))
    final_meta = json.loads((root/"final_results.json").read_text())["final_feature_profile"]
    final_paths = list((root/"cache").glob(f"final_features_{final_meta['key']}.parquet"))
    assert len(final_paths)==1
    cache_path = final_paths[0]
    assert pl.scan_parquet(cache_path).select(pl.len()).collect().item() == df.height
    assert pl.read_parquet(cache_path,columns=["row_id"])["row_id"].to_list() == df["row_id"].to_list()
    missing, checks = [], []
    groups = defaultdict(list)
    for n in names: groups[all_specs[n].keys].append(all_specs[n])
    rng = np.random.default_rng(527)
    query_indices = sorted({0,1,train_end,search_end,valid_end,df.height-1,*rng.integers(2,df.height,size=10).tolist()})
    for keys,specs in groups.items():
        cached=pl.read_parquet(cache_path,columns=["row_id",*[s.name for s in specs]])
        if regenerate:
            generated,_=generate_features(df,specs)
            assert generated.height==df.height
        for s in specs:
            values=cached[s.name].to_numpy()
            assert not np.isinf(values).any(),s.name
            missing.append({"feature":s.name,"null_or_nan_fraction":float(np.isnan(values).mean()),"infinite":0})
            if regenerate:
                np.testing.assert_allclose(generated[s.name].to_numpy(),values,rtol=2e-5,atol=1e-5,equal_nan=True,err_msg=s.name)
        # Independent direct-history reference on actual rows across every family.
        for query_index in query_indices:
            query=df.row(query_index,named=True)
            condition=pl.lit(True)
            for key in keys: condition &= pl.col(key).is_null() if query[key] is None else pl.col(key)==query[key]
            entity_history=df.filter(condition)
            for s in specs:
                expected=expected_value(entity_history,s,query)
                actual=cached[s.name][query_index]
                actual=np.nan if actual is None else actual
                if s.aggregation in {"count","nunique"}:
                    assert actual==expected,(name,query_index,s.name,actual,expected)
                else:
                    tolerance=abs(float(np.spacing(np.float32(expected))))*4 if np.isfinite(expected) else 0
                    np.testing.assert_allclose(actual,np.float32(expected),rtol=0,atol=tolerance,equal_nan=True,
                                               err_msg=f"{name}:{query_index}:{s.name}")
                checks.append({"row_id":query_index,"feature":s.name,"expected":expected,"actual":actual,"status":"PASS"})
        print(name,"feature family verified",keys,len(specs),flush=True)
        del cached
        if regenerate:
            del generated
    pl.DataFrame(checks).write_csv(output/"feature_spot_checks.csv")
    pl.DataFrame(missing).write_csv(output/"feature_missingness.csv")
    gc.collect()
    features=pl.read_parquet(cache_path)
    if lock["warmup_end"] is not None:
        embedding,_=temporal_svd_features(df,lock["warmup_end"])
        features=features.hstack(embedding.drop("row_id"))
    test=df.slice(valid_end)
    encoder=json.loads((root/"encoder_final.json").read_text())
    base,base_names,_=base_matrix(test,encoder)
    features=features.slice(valid_end)
    experiments=pl.read_csv(root/"experiments.csv")
    metric_rows,model_checks=[],[]
    for row in experiments.filter(pl.col("split")=="test").to_dicts():
        arm,seed=row["experiment_name"],row["seed"]
        predictions=pl.read_parquet(root/"predictions"/f"{arm}_seed{seed}.parquet")
        assert predictions["row_id"].to_list()==test["row_id"].to_list()
        p=predictions["prediction"].to_numpy()
        model=lgb.Booster(model_file=str(root/"models"/f"final_{arm}_seed{seed}.txt"))
        assert model.feature_name()[:len(base_names)]==base_names
        temporal_names=model.feature_name()[len(base_names):]
        if arm=="manual": assert temporal_names==lock["manual_features"]
        if arm=="automatic": assert temporal_names==lock["auto_features"]
        matrix=matrix_from_features(base,features,temporal_names)
        replay=model.predict(matrix,num_threads=4)
        np.testing.assert_allclose(p,replay,rtol=0,atol=1e-13)
        assert model.current_iteration()==row["best_iteration"]
        model_checks.append({"model":arm,"seed":seed,"features":len(model.feature_name()),"prediction_max_abs_difference":float(np.max(abs(p-replay))),"iterations":model.current_iteration()})
        for metric,reproduced in independent_metrics(test,p).items():
            difference=reproduced-row[metric]
            assert abs(difference)<1e-10,(name,arm,metric,difference)
            metric_rows.append({"dataset":name,"experiment":arm,"seed":seed,"metric":metric,"reported":row[metric],"reproduced":reproduced,"difference":difference})
        print(name,"model and metrics verified",arm,seed,flush=True)
        del matrix,model
    pl.DataFrame(metric_rows).write_csv(output/"metrics_reproduced.csv")
    pl.DataFrame(model_checks).write_csv(output/"models_replayed.csv")
    # Independently resample complete users, without the project's bootstrap helper.
    baseline=pl.read_parquet(root/"predictions/manual_seed42.parquet")["prediction"].to_numpy()
    automatic=pl.read_parquet(root/"predictions/automatic_seed42.parquet")["prediction"].to_numpy()
    save_json(output/"independent_bootstrap.json", independent_user_bootstrap(test, baseline, automatic))
    selected=pl.read_csv(root/"selected_features.csv")
    assert set(selected["feature"])==set(lock["auto_features"])
    windows=pl.read_csv(root/"selected_windows.csv")
    assert windows.sort(windows.columns).equals(selected.select(windows.columns).unique().sort(windows.columns))
    source=lgb.Booster(model_file=str(root/"models"/f"{lock['auto_name']}_seed42.txt"))
    gains=dict(zip(source.feature_name(),source.feature_importance("gain")))
    for row in pl.read_csv(root/"feature_importance.csv").to_dicts():
        assert row["feature"] in gains
        assert abs(row["importance_gain"]-gains[row["feature"]])<1e-7
    profiles=json.loads((root/"generation_profile.json").read_text())
    candidate=pl.read_csv(root/"candidate_features.csv")
    pruning=pl.read_csv(root/"pruned_features.csv")
    assert profiles["n_candidates"]-pruning.height==profiles["n_after_pruning"]
    assert set(selected["feature"])<=set(candidate["feature"])
    save_json(output/"summary.json",{"status":"PASS","rows":df.height,"initial_candidates":profiles["n_candidates"],"all_candidates_including_agent":candidate.height,"after_pruning":profiles["n_after_pruning"],"selected":selected.height,"spot_checks":len(checks),"regenerated_feature_cells":len(names)*df.height if regenerate else 0,"model_prediction_replays":len(model_checks),"seconds":perf_counter()-tick})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["before","after"],required=True)
    parser.add_argument("--dataset",choices=["kion","yambda"],required=True)
    parser.add_argument("--skip-regeneration",action="store_true")
    args=parser.parse_args()
    source="artifacts/pre_audit" if args.phase=="before" and Path("artifacts/pre_audit").exists() else "artifacts"
    check_dataset(args.dataset,Path("artifacts/audit")/args.phase,not args.skip_regeneration,source)
