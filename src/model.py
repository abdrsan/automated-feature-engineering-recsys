import gc
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.metrics import log_loss


BASE_CATEGORICAL = ["user_id", "item_id", "gender", "age", "occupation", "category_id", "genres",
                    "author", "income", "kids_flg", "content_type", "countries", "for_kids", "age_rating", "is_organic"]


def fit_encoder(df):
    # Словарь строим на обучающем префиксе. Новые ID не должны заранее попасть в него.
    columns = [c for c in BASE_CATEGORICAL if c in df.columns]
    return {c: {str(value): i for i, value in enumerate(sorted(df[c].drop_nulls().cast(pl.String).unique().to_list()))}
            for c in columns}


def base_matrix(df, encoder):
    arrays, names = [], []
    for c, mapping in encoder.items():
        arrays.append(df[c].cast(pl.String).replace_strict(mapping, default=-1, return_dtype=pl.Int32).to_numpy())
        names.append(c)
    categorical = list(range(len(names)))
    if "release_year" in df.columns:
        arrays.append(df["release_year"].cast(pl.Float32).to_numpy())
        names.append("release_year")
    # Час и день недели известны в момент запроса. У Yambda время относительное,
    # поэтому календарные признаки из искусственной даты 1970 года не добавляем.
    if "time_is_relative" not in df.columns or not df["time_is_relative"].any():
        arrays.extend([df["timestamp"].dt.hour().to_numpy(), df["timestamp"].dt.weekday().to_numpy()])
        names.extend(["hour", "weekday"])
    return np.column_stack(arrays).astype(np.float32), names, categorical


def matrix_from_features(base, features, names):
    if not names:
        return base
    if len(base) != features.height:
        raise ValueError("Base/temporal feature rows differ")
    # Заполняем одну float32-матрицу по колонкам, чтобы не держать в памяти
    # еще одну полную копию широкого набора признаков.
    matrix = np.empty((len(base), base.shape[1] + len(names)), dtype=np.float32)
    matrix[:, :base.shape[1]] = base
    for column, name in enumerate(names, start=base.shape[1]):
        matrix[:, column] = features[name].to_numpy()
    return matrix


def take_rows(matrix, indices):
    if len(indices) and np.all(np.diff(indices) == 1):
        return matrix[int(indices[0]):int(indices[-1]) + 1]
    return matrix[indices]


def fit_model(x, y, train, valid, feature_names, categorical, params, config, seed=42,
              fixed_iterations=None):
    tick = perf_counter()
    options = {"objective": "binary", "n_estimators": fixed_iterations or config["max_estimators"],
               "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1,
               "min_child_samples": 100, "reg_lambda": 5., "colsample_bytree": 0.9,
               "subsample": 0.9, "subsample_freq": 1, "max_bin": 63,
               "n_jobs": config["threads"], "random_state": seed,
               "deterministic": True, "force_col_wise": True, "verbosity": -1}
    options.update(params)
    model = lgb.LGBMClassifier(**options)
    extra = {}
    if fixed_iterations is None:
        extra = {"eval_set": [(take_rows(x, valid), y[valid])], "eval_metric": "binary_logloss",
                 "callbacks": [lgb.early_stopping(config["early_stopping_rounds"], verbose=False)]}
    model.fit(take_rows(x, train), y[train], feature_name=feature_names,
              categorical_feature=categorical, **extra)
    return model, perf_counter() - tick


def tuning_history_signature(study):
    history = [(t.number, str(t.state), t.params) for t in study.trials]
    return hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest()


def save_tuning_sampler(study, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps({"sampler": study.sampler,
                                      "history": tuning_history_signature(study)}))
    temporary.replace(path)


def create_tuning_study(storage, study_name, seed):
    if not storage.startswith("sqlite:///"):
        raise ValueError("Reproducible local tuning requires SQLite storage")
    path = Path(storage.removeprefix("sqlite:///")).with_name(study_name + "_sampler.pkl")
    study = optuna.create_study(direction="minimize", study_name=study_name,
        storage=storage, load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=seed))
    if study.trials:
        if not path.exists():
            raise ValueError("Study has trials but no saved sampler state; use a fresh output directory")
        # Здесь читается собственный локальный checkpoint. Чужие pickle загружать нельзя:
        # их десериализация может выполнить код.
        state = pickle.loads(path.read_bytes())
        if state["history"] != tuning_history_signature(study):
            raise ValueError("Sampler/trial history differs after interruption; use a fresh output directory")
        study.sampler = state["sampler"]
    return study, path


def tune_model(x, y, train, valid, names, categorical, config, storage, study_name):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(config["seed"])
    max_rows = config["max_tuning_train_rows"]
    tuning_train = np.sort(rng.choice(train, min(max_rows, len(train)), replace=False))
    # Для HPO сокращаем число обучающих строк, но не историю их признаков.
    # Иначе сравнивали бы другую задачу, где большая часть прошлого пропала.
    study, sampler_path = create_tuning_study(storage, study_name, config["seed"])
    def objective(trial):
        params = {"num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63]),
                  "min_child_samples": trial.suggest_int("min_child_samples", 50, 500, log=True),
                  "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.10, log=True),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                  "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30., log=True),
                  "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 3., log=True)}
        model, seconds = fit_model(x, y, tuning_train, valid, names, categorical, params, config,
                                    seed=config["seed"])
        loss = log_loss(y[valid], model.booster_.predict(x[valid], num_threads=config["threads"]), labels=[0, 1])
        trial.set_user_attr("training_seconds", seconds)
        trial.set_user_attr("best_iteration", model.best_iteration_)
        print(f"tune {study_name} trial={trial.number} logloss={loss:.6f} seconds={seconds:.1f}", flush=True)
        del model
        gc.collect()
        return loss
    completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    remaining = max(0, config["hyperparameter_trials"] - completed)
    try:
        study.optimize(objective, n_trials=remaining,
                       callbacks=[lambda state, trial: save_tuning_sampler(state, sampler_path)])
    finally:
        save_tuning_sampler(study, sampler_path)
    return study.best_params, study.trials_dataframe()
