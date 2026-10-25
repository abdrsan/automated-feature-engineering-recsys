import hashlib

import numpy as np
import polars as pl


def prune_candidates(features, train_end, config, seed=42):
    train = features.head(train_end).drop("row_id")
    reasons, keep, hashes = [], [], {}
    rng = np.random.default_rng(seed)
    # Для грубого отсева корреляции оцениваем на 20k train-строк.
    # Вычисление на всех событиях дороже; выборка может пропустить отдельные различия.
    sampled_rows = np.sort(rng.choice(train.height, min(20000, train.height), replace=False))
    samples = []
    for name in train.columns:
        values = train[name].to_numpy()
        finite = np.isfinite(values)
        missing = 1 - np.mean(finite)
        reason = None
        if missing > config["max_missing_fraction"]:
            reason = "missing_or_sparse"
        else:
            valid = values[finite]
            _, counts = np.unique(valid, return_counts=True)
            largest = max(counts.max(initial=0), len(values) - len(valid)) / len(values)
            if largest >= config["near_constant_fraction"]:
                reason = "constant_or_near_constant"
        digest = hashlib.sha256(np.nan_to_num(values, nan=-999999).tobytes()).hexdigest()
        if reason is None and digest in hashes and np.array_equal(
                values, train[hashes[digest]].to_numpy(), equal_nan=True):
            reason = "exact_duplicate_of:" + hashes[digest]
        if reason is None:
            hashes[digest] = name
            keep.append(name)
            sample = values[sampled_rows].copy()
            # Подстановка медианы нужна только для оценки корреляции.
            # В матрице для LightGBM пропуски остаются как были.
            sample[~np.isfinite(sample)] = np.nanmedian(sample) if np.isfinite(sample).any() else 0
            samples.append(sample)
        else:
            reasons.append({"feature": name, "reason": reason})
    if samples:
        matrix = np.column_stack(samples)
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.atleast_2d(np.corrcoef(matrix.T))
        accepted = []
        for j, name in enumerate(keep):
            duplicate = next((i for i in accepted if abs(correlation[i, j]) >= config["correlation_threshold"]), None)
            if duplicate is None:
                accepted.append(j)
            else:
                reasons.append({"feature": name, "reason": "correlated_with:" + keep[duplicate]})
        keep = [keep[j] for j in accepted]
    return keep, pl.DataFrame(reasons, schema={"feature": pl.String, "reason": pl.String})


def ranked_features(model, feature_names, base_names):
    gains = model.booster_.feature_importance(importance_type="gain")
    importance = dict(zip(feature_names, map(float, gains)))
    ranked = sorted((n for n in feature_names if n not in base_names), key=lambda n: (-importance[n], n))
    return ranked, importance


def prune_fixed_candidates(features, specs, train_end, config, seed=42):
    # Сначала оставляем разрешенные окна, потом ищем дубликаты внутри них.
    # Иначе all-history может уступить место 180d, после чего фильтр окон удалит и 180d.
    allowed = [s.name for s in specs if s.window_days in [None, 7, 30]
               and (s.aggregation != "trend" or s.long_window_days == 30)]
    return prune_candidates(features.select("row_id", *allowed), train_end, config, seed)
