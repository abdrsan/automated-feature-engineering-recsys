"""Replay every saved development checkpoint and the best HPO objectives."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sqlite3

import lightgbm as lgb
import numpy as np
import polars as pl

from audit.run_checks import independent_metrics
from src.data import save_json
from src.datasets import load_kion, load_yambda
from src.model import base_matrix, matrix_from_features, fit_model
from src.representations import temporal_svd_features


def read_blocks(root, row_ids, names):
    output=pl.DataFrame({"row_id":row_ids})
    for path in sorted((root/"cache").glob("*.parquet"),key=lambda p:(not p.name.startswith("final_features"),p.name)):
        needed=[n for n in names if n not in output.columns and n in pl.read_parquet_schema(path)]
        if needed:
            block=pl.scan_parquet(path).select("row_id",*needed).filter(pl.col("row_id").is_in(output["row_id"].implode())).collect()
            output=output.join(block,on="row_id",how="left",validate="1:1")
    assert set(names)<=set(output.columns)
    return output


def replay(name,phase):
    parent=Path("artifacts/pre_audit") if phase=="before" else Path("artifacts")
    root=parent/name
    out=Path("artifacts/audit")/phase/name
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/"run_manifest.json").read_text())
    lock=json.loads((root/"locked_selection.json").read_text())
    config=manifest["config"]
    df=load_kion() if name=="kion" else load_yambda()
    a,b,c=[manifest["split"][key] for key in ["train_end","search_end","validation_end"]]
    experiments=pl.read_csv(root/"experiments.csv").filter(pl.col("split")!="test")
    spec_files={}
    all_names=set()
    for row in experiments.to_dicts():
        key=f"{row.get('source_experiment') or row['experiment_name']}_seed{row['seed']}"
        spec=json.loads((root/"specs"/(key+".json")).read_text())
        assert (root/"models"/(key+".txt")).exists()
        spec_files[key]=spec
        all_names.update(n for n in spec["features"] if not n.startswith("svd_"))
    query=df.slice(a,c-a)
    features=read_blocks(root,query["row_id"],all_names)
    if lock["warmup_end"] is not None:
        embedding,_=temporal_svd_features(df.head(c),lock["warmup_end"])
        features=features.join(embedding,on="row_id",how="left",validate="1:1")
    encoder=json.loads((root/"encoder_development.json").read_text())
    base,base_names,categorical=base_matrix(query,encoder)
    differences=[]
    for row in experiments.to_dicts():
        key=f"{row.get('source_experiment') or row['experiment_name']}_seed{row['seed']}"
        spec=spec_files[key]
        model=lgb.Booster(model_file=str(root/"models"/(key+".txt")))
        assert model.feature_name()==spec["base_names"]+spec["features"]
        assert len(model.feature_name())==row["n_features"]
        if row.get("feature_hash"):
            assert hashlib.sha256("\n".join(model.feature_name()).encode()).hexdigest()==row["feature_hash"]
        start,end=(0,b-a) if row["split"]=="validation_search" else (b-a,c-a)
        part=query.slice(start,end-start)
        x=matrix_from_features(base[start:end],features.slice(start,end-start),spec["features"])
        p=model.predict(x,num_threads=4)
        for metric,value in independent_metrics(part,p).items():
            delta=value-row[metric]
            assert abs(delta)<1e-10,(name,key,metric,delta)
            differences.append({"experiment":row["experiment_name"],"source":key,"seed":row["seed"],
                "split":row["split"],"metric":metric,"reported":row[metric],"reproduced":value,"difference":delta})
        del x,model
    pl.DataFrame(differences).write_csv(out/"search_metrics_reproduced.csv")
    del features,base,query
    gc.collect()
    # Compare CSV ledgers with the actual SQLite trial records, then refit best trials.
    trial_replays=[]
    connection=sqlite3.connect(f"file:{(root/'tuning.sqlite3').resolve()}?mode=ro",uri=True)
    for arm,study_name,source in [("manual","manual","manual"),("auto","automatic","auto_tuned")]:
        csv=pl.read_csv(root/(arm+"_tuning.csv"))
        ledger=connection.execute("SELECT t.number,t.state,v.value FROM trials t JOIN studies s ON t.study_id=s.study_id LEFT JOIN trial_values v ON t.trial_id=v.trial_id WHERE s.study_name=? ORDER BY t.number",(study_name,)).fetchall()
        assert len(ledger)==csv.height
        for number,state,value in ledger:
            record=csv.filter(pl.col("number")==number).row(0,named=True)
            assert state==record["state"]
            if state=="COMPLETE": assert abs(value-record["value"])<1e-13
        best=csv.filter(pl.col("state")=="COMPLETE").sort("value").row(0,named=True)
        params={k.removeprefix("params_"):v for k,v in best.items() if k.startswith("params_")}
        params["num_leaves"]=int(params["num_leaves"])
        params["min_child_samples"]=int(params["min_child_samples"])
        spec=json.loads((root/"specs"/(source+"_seed42.json")).read_text())
        sample=np.sort(np.random.default_rng(config["seed"]).choice(np.arange(a),min(config["max_tuning_train_rows"],a),replace=False))
        ids=np.r_[sample,np.arange(a,b)]
        part=df[ids]
        values=read_blocks(root,part["row_id"],spec["features"])
        base,names,categorical=base_matrix(part,encoder)
        matrix=matrix_from_features(base,values,spec["features"])
        train=np.arange(len(sample))
        valid=np.arange(len(sample),len(ids))
        model,seconds=fit_model(matrix,part["label"].to_numpy(),train,valid,names+spec["features"],categorical,params,config,seed=config["seed"])
        p=model.booster_.predict(matrix[valid],num_threads=4)
        labels=part["label"].to_numpy()[valid]
        loss=float(np.mean(-labels*np.log(p)-(1-labels)*np.log1p(-p)))
        assert abs(loss-best["value"])<1e-10,(name,arm,loss,best["value"])
        trial_replays.append({"arm":arm,"trial":best["number"],"reported":best["value"],"reproduced":loss,
            "difference":loss-best["value"],"training_seconds":seconds,"completed_trials":csv.filter(pl.col("state")=="COMPLETE").height})
        del values,base,matrix,model
        gc.collect()
    connection.close()
    pl.DataFrame(trial_replays).write_csv(out/"best_hpo_replayed.csv")
    save_json(out/"search_replay_summary.json",{"status":"PASS","checkpoint_evaluations":experiments.height,
        "independent_metric_comparisons":len(differences),"best_hpo_trials_refitted":len(trial_replays),"sqlite_ledgers_match_csv":True})
    print(name,phase,"PASS every development checkpoint, ledger and best HPO replay",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["kion","yambda"],required=True)
    parser.add_argument("--phase",choices=["before","after"],required=True)
    args=parser.parse_args()
    replay(args.dataset,args.phase)
