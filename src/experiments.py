import gc
import hashlib
import json
from pathlib import Path
import threading
from time import perf_counter

import numpy as np
import polars as pl
import psutil
import lightgbm as lgb

from src.data import save_json
from src.metrics import evaluate
from src.model import fit_model, matrix_from_features
from src.feature_engine import feature_engine_signature
from src.provenance import STUDY_PROTOCOL_VERSION, implementation_signature


class MemoryTracker:
    def __init__(self):
        self.peak = psutil.Process().memory_info().rss / 2**20
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while not self.stop.wait(0.2):
            self.peak = max(self.peak, process.memory_info().rss / 2**20)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, psutil.Process().memory_info().rss / 2**20)
        self.stop.set()
        self.thread.join()


class ExperimentRunner:
    def __init__(self, df, features, base, base_names, categorical, split, config, root):
        self.df, self.features, self.base = df, features, base
        self.base_names, self.categorical = base_names, categorical
        self.split, self.config, self.root = split, config, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in ["models", "predictions", "specs"]:
            (self.root / folder).mkdir(exist_ok=True)
        self.rows = pl.read_csv(self.root / "experiments.csv").to_dicts() if (self.root / "experiments.csv").exists() else []

    def matrix(self, feature_names):
        return matrix_from_features(self.base, self.features, feature_names)

    def run(self, name, feature_names, params, seed=None, generation_seconds=0.,
            train=None, details=None, save_model=True):
        seed = self.config["seed"] if seed is None else seed
        model_key = f"{name}_seed{seed}"
        spec_path = self.root / "specs" / f"{model_key}.json"
        previous = next((r for r in self.rows if r["experiment_name"] == name and r["seed"] == seed
                         and r["split"] == "validation_search"), None)
        if previous and spec_path.exists():
            saved = json.loads(spec_path.read_text())
            if (saved.get("protocol_version") != STUDY_PROTOCOL_VERSION or
                saved.get("engine_signature") != feature_engine_signature() or
                saved.get("fit_signature") != implementation_signature(fit_model)):
                raise ValueError("Cached model implementation differs; use a fresh output directory")
            start = int((self.split.train if train is None else train)[0])
            if saved["features"] != feature_names or saved["params"] != params or saved["train_start"] != start:
                raise ValueError(f"Cached experiment {model_key} conflicts with new configuration")
            booster = lgb.Booster(model_file=str(self.root / "models" / f"{model_key}.txt"))
            gains = dict(zip(booster.feature_name(), map(float, booster.feature_importance("gain"))))
            print(f"RESUME {model_key} logloss={previous['logloss']:.6f}", flush=True)
            return previous, gains
        print(f"START {model_key} features={len(self.base_names) + len(feature_names)}", flush=True)
        with MemoryTracker() as memory:
            x = self.matrix(feature_names)
            y = self.df["label"].to_numpy()
            names = [*self.base_names, *feature_names]
            model, seconds = fit_model(x, y, self.split.train if train is None else train,
                                        self.split.search, names, self.categorical, params,
                                        self.config, seed)
            search_predictions = model.booster_.predict(x[self.split.search], num_threads=self.config["threads"])
            search_metrics = evaluate(self.df[self.split.search], search_predictions)
            # Здесь считаем только search. Audit откроем после фиксации выбора,
            # иначе он превратится в еще одну выборку для подбора признаков.
            row = {"experiment_name": name, "seed": seed, "split": "validation_search",
                   "n_features": len(names), "training_time": seconds,
                   "feature_generation_time": generation_seconds,
                   "cache_hit": generation_seconds == 0.,
                   "best_iteration": model.best_iteration_,
                   "n_train": len(self.split.train if train is None else train),
                   "feature_hash": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                   **search_metrics}
            if details:
                row.update(details)
            gains = dict(zip(names, map(float, model.booster_.feature_importance("gain"))))
            if save_model:
                model.booster_.save_model(str(self.root / "models" / f"{model_key}.txt"))
                save_json(self.root / "specs" / f"{model_key}.json", {
                    "features": feature_names, "base_names": self.base_names, "params": params,
                    "iterations": model.best_iteration_, "seed": seed,
                    "train_start": int((self.split.train if train is None else train)[0]),
                    "protocol_version": STUDY_PROTOCOL_VERSION,
                    "engine_signature": feature_engine_signature(),
                    "fit_signature": implementation_signature(fit_model),
                })
            del x, model
            gc.collect()
        row["memory_usage"] = memory.peak
        self.rows.append(row)
        self.flush()
        print(f"DONE {model_key} logloss={row['logloss']:.6f} auc={row['auc']:.6f} seconds={seconds:.1f}", flush=True)
        return row, gains

    def flush(self):
        if self.rows:
            pl.DataFrame(self.rows, infer_schema_length=None).write_csv(self.root / "experiments.csv")


def candidate_metadata(specs, selected):
    chosen = set(selected)
    return pl.DataFrame([{"feature": s.name, "entity": s.entity, "event": s.event,
                          "aggregation": s.aggregation,
                          "window_days": s.window_days,
                          "long_window_days": s.long_window_days,
                          "window": "all_history" if s.window_days is None else f"{s.window_days}d",
                          "selected": s.name in chosen,
                          "spec_json": json.dumps(s.to_dict())} for s in specs])
