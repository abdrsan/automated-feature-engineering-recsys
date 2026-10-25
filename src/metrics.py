import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score


def ranking_metrics(df, predictions, ks=(3, 5, 10)):
    # В одной query сравниваем items одного пользователя в один момент времени.
    # Это порядок среди наблюдавшихся событий, а не выдача по всему каталогу.
    candidate = "item_id" if "item_id" in df.columns else "row_id"
    scored = df.select(list(dict.fromkeys(["user_id", "timestamp", "row_id", "label", candidate]))).with_columns(
        pl.Series("score", predictions))
    queries = scored.group_by(["user_id", "timestamp"]).agg(
        pl.len().alias("size"), pl.col("label").sum().alias("positives"),
        pl.col(candidate).n_unique().alias("unique_items"))
    mixed = queries.filter((pl.col("size") >= 2) & (pl.col("positives") > 0) &
                            (pl.col("positives") < pl.col("size")))
    eligible = mixed.filter(pl.col("unique_items") == pl.col("size"))
    # Повтор одного item сделал бы ranking неоднозначным: исключаем всю такую группу.
    excluded = mixed.height - eligible.height
    if eligible.height == 0:
        return {"ranking_groups": 0, "ranking_rows": 0, "ranking_groups_excluded_duplicates": excluded,
                **{f"{metric}_{k}": float("nan")
                for k in ks for metric in ["ndcg", "recall", "map"]}}
    ranked = scored.join(eligible, on=["user_id", "timestamp"], how="inner").sort(
        ["user_id", "timestamp", "score", "row_id"], descending=[False, False, True, False])
    keys = ["user_id", "timestamp"]
    ranked = ranked.with_columns(pl.col("row_id").cum_count().over(keys).alias("rank"),
                                  pl.col("label").cum_sum().over(keys).alias("hits"))
    result = {"ranking_groups": eligible.height, "ranking_rows": ranked.height,
              "ranking_groups_excluded_duplicates": excluded}
    max_size = int(eligible["positives"].max())
    for k in ks:
        ideal = np.r_[0., np.cumsum(1 / np.log2(np.arange(1, min(k, max_size) + 1) + 1))]
        by_query = ranked.group_by(keys).agg(
            pl.col("positives").first(),
            pl.when(pl.col("rank") <= k).then(pl.col("label") / (pl.col("rank") + 1).log(2)).otherwise(0).sum().alias("dcg"),
            pl.when(pl.col("rank") <= k).then(pl.col("label")).otherwise(0).sum().alias("top_hits"),
            pl.when(pl.col("rank") <= k).then(pl.col("label") * pl.col("hits") / pl.col("rank")).otherwise(0).sum().alias("ap"),
        )
        npos = by_query["positives"].to_numpy()
        result[f"ndcg_{k}"] = float(np.mean(by_query["dcg"].to_numpy() / ideal[np.minimum(npos, k)]))
        result[f"recall_{k}"] = float(np.mean(by_query["top_hits"].to_numpy() / npos))
        result[f"map_{k}"] = float(np.mean(by_query["ap"].to_numpy() / np.minimum(npos, k)))
    return result


def evaluate(df, predictions):
    y = df["label"].to_numpy()
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("Invalid probability prediction")
    return {"auc": float(roc_auc_score(y, predictions)) if len(np.unique(y)) == 2 else float("nan"),
            "logloss": float(log_loss(y, predictions, labels=[0, 1])),
            **ranking_metrics(df, predictions)}


def paired_user_bootstrap(df, baseline, candidate, repeats=300, seed=42):
    # Пересэмплируем пользователей целиком: события одного человека зависимы.
    # В каждой итерации обе модели получают те же веса, поэтому сравнение парное.
    # Модели здесь не переобучаются; разброс всего feature search этот интервал не оценивает.
    y = df["label"].to_numpy()
    users, inverse = np.unique(df["user_id"].to_numpy(), return_inverse=True)
    baseline = np.clip(baseline, 1e-7, 1 - 1e-7)
    candidate = np.clip(candidate, 1e-7, 1 - 1e-7)
    b = -(y * np.log(baseline) + (1 - y) * np.log1p(-baseline))
    c = -(y * np.log(candidate) + (1 - y) * np.log1p(-candidate))
    sums = np.bincount(inverse, weights=c - b)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = []
    auc_deltas = []
    for _ in range(repeats):
        weights = rng.multinomial(len(users), np.full(len(users), 1 / len(users)))
        estimates.append(float(weights @ sums / (weights @ counts)))
        row_weights = weights[inverse]
        if np.sum(row_weights[y == 0]) > 0 and np.sum(row_weights[y == 1]) > 0:
            auc_deltas.append(float(roc_auc_score(y, candidate, sample_weight=row_weights) -
                                    roc_auc_score(y, baseline, sample_weight=row_weights)))
    return {"delta_logloss": float(np.mean(c - b)),
            "logloss_ci_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "delta_auc": float(roc_auc_score(y, candidate) - roc_auc_score(y, baseline)),
            "auc_ci_95": np.quantile(auc_deltas, [0.025, 0.975]).tolist(),
            "bootstrap_users": len(users), "repeats": repeats,
            "interpretation": "Fixed models; does not include search/training variability"}
