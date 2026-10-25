"""Independent rescoring and booster replay of the new controlled comparison."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import rankdata

from src.datasets import load_kion, load_yambda
from src.feature_engine import generate_features
from src.feature_spec import FeatureSpec
from src.model import fit_encoder, base_matrix, matrix_from_features
from src.split import temporal_split


def independent_scores(labels, probabilities):
    labels = np.asarray(labels, dtype=float)
    positives = labels.sum()
    ranks = rankdata(probabilities)
    auc = (ranks[labels == 1].sum()-positives*(positives+1)/2)/(positives*(len(labels)-positives))
    p = np.clip(probabilities, np.finfo(float).eps, 1-np.finfo(float).eps)
    loss = np.mean(-labels*np.log(p)-(1-labels)*np.log1p(-p))
    order = np.argsort(-probabilities, kind="stable")
    scores, y = probabilities[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(y)-1]
    hits = np.cumsum(y)[ends]
    ap = np.sum(hits/(ends+1)*np.diff(np.r_[0., hits/positives]))
    return {"auc": float(auc), "logloss": float(loss), "average_precision": float(ap)}


def verify(folder, output, replay=True, raw_dir="data/raw"):
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, differences = [], []
    for dataset in ["kion", "yambda"]:
        root = folder/dataset
        if not root.exists():
            continue
        lock = json.loads((root/"locked_protocol.json").read_text())
        data = load_kion(raw_dir) if dataset == "kion" else load_yambda(raw_dir)
        if lock["mode"] == "demo":
            data = data.head(100000)
        split = temporal_split(data)
        assert vars(split) == lock["split"]
        assert lock["selection_data"] == "search only" and lock["seed"] == 42
        common = json.loads((Path("artifacts")/dataset/"manual_params.json").read_text())
        assert lock["params"] == common
        original = {s.name: s for s in map(FeatureSpec.from_dict, lock["all_specs"])}
        names = list(dict.fromkeys(n for method in lock["methods"].values() for n in method))
        if replay:
            features, _ = generate_features(data, [original[n] for n in names])
            encoder = fit_encoder(data.head(split.validation_end))
            base, base_names, categorical = base_matrix(data[split.test], encoder)
            features = features[split.test]
        table = pl.read_csv(root/"comparison.csv")
        test = data[split.test]
        seen = set()
        for row in table.to_dicts():
            method = row["method"]
            source = row["shared_with"] or method
            path = root/("predictions_"+source+".parquet")
            saved = pl.read_parquet(path)
            assert saved["row_id"].equals(test["row_id"])
            assert saved["label"].equals(test["label"])
            p = saved["prediction"].to_numpy()
            for metric, value in independent_scores(test["label"].to_numpy(), p).items():
                delta = value-row[metric]
                assert abs(delta) < 1e-12, (dataset, method, metric, delta)
                differences.append({"dataset": dataset, "method": method, "metric": metric,
                    "reported": row[metric], "reproduced": value, "difference": delta})
            if replay and source not in seen:
                booster = lgb.Booster(model_file=str(root/("final_"+source+".txt")))
                columns = lock["methods"][source]
                assert booster.feature_name() == base_names+columns
                assert booster.num_feature() == row["inputs"]
                assert booster.current_iteration() == row["iterations"]
                aliases = {"min_child_samples": "min_data_in_leaf", "colsample_bytree": "feature_fraction",
                           "reg_lambda": "lambda_l2", "reg_alpha": "lambda_l1"}
                for param, value in common.items():
                    actual = booster.params.get(param, booster.params.get(aliases.get(param)))
                    # LightGBM prints configuration floats with limited precision in model text.
                    assert np.isclose(float(actual), value, rtol=5e-6, atol=1e-9), (param, actual, value)
                assert booster.params["seed"] == 42
                x = matrix_from_features(base, features, columns)
                predicted = booster.predict(x, num_threads=4)
                np.testing.assert_allclose(predicted, p, rtol=0, atol=1e-13)
                seen.add(source)
                del x, booster
        checksums = json.loads((root/"files_sha256.json").read_text())
        assert all(hashlib.sha256((root/name).read_bytes()).hexdigest() == digest for name, digest in checksums.items())
        summaries.append({"dataset": dataset, "mode": lock["mode"], "rows": data.height,
            "methods": table.height, "unique_final_models_replayed": len(seen),
            "same_parameters_verified": True, "same_test_population_verified": True,
            "independent_classification_metrics": 3, "status": "PASS"})
        if replay:
            del features, base
        del data
        gc.collect()
        print(dataset, summaries[-1], flush=True)
    pl.DataFrame(differences).write_csv(output/"controlled_metrics_reproduced.csv")
    result = {"status": "PASS", "runs": summaries, "metric_comparisons": len(differences),
        "max_metric_difference": max(abs(r["difference"]) for r in differences)}
    (output/"controlled_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()
    verify(args.run, args.output, replay=not args.skip_replay, raw_dir=args.raw_dir)
