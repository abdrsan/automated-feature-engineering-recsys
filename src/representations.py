import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD


def temporal_svd_features(df, warmup_end, dimensions=16, seed=42):
    """Freeze basis on a warmup prefix; emit valid features only AFTER that prefix."""
    warmup = df.head(warmup_end)
    # Фиксируем пространство item-векторов на раннем участке истории.
    # Более поздние события смогут менять профиль пользователя, но не сам basis.
    if warmup["timestamp"].max() >= df["timestamp"][warmup_end]:
        raise ValueError("Warmup shares a timestamp with prediction rows")
    users = warmup["user_id"].unique().sort().to_numpy()
    items = warmup["item_id"].unique().sort().to_numpy()
    user_index = np.searchsorted(users, warmup["user_id"].to_numpy())
    item_index = np.searchsorted(items, warmup["item_id"].to_numpy())
    weights = warmup["label"].to_numpy().astype(float) * 2 - 1
    # Положительная реакция дает +1, отрицательная -1. Повторы суммируются в sparse-матрице.
    matrix = coo_matrix((weights, (user_index, item_index)), shape=(len(users), len(items))).tocsr()
    dimensions = min(dimensions, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=dimensions, random_state=seed)
    svd.fit(matrix)
    vectors = svd.components_.T.astype(np.float32)
    columns = [f"__emb{i}" for i in range(dimensions)]
    basis = pl.DataFrame({"item_id": items, **{c: vectors[:, i] for i, c in enumerate(columns)}})
    history = df.join(basis, on="item_id", how="left", validate="m:1").with_columns(
        [pl.col(c).fill_null(0) for c in columns])
    blocks = history.group_by(["user_id", "timestamp"]).agg([
        (pl.col(c) * (pl.col("label").cast(pl.Float32) * 2 - 1)).sum().alias(c + "s") for c in columns
    ] + [pl.len().alias("__n")]).sort(["user_id", "timestamp"])
    # Сдвиг по временным блокам не дает профилю увидеть текущую реакцию или ее соседей по timestamp.
    blocks = blocks.with_columns(
        pl.col("__n").cum_sum().shift(1).over("user_id").fill_null(0).alias("__prev_n"),
        *[pl.col(c + "s").cum_sum().shift(1).over("user_id").fill_null(0).alias(c + "p") for c in columns],
    ).with_columns([(pl.col(c + "p") / pl.col("__prev_n").clip(1, None)).alias(c + "p") for c in columns])
    joined = history.join(blocks.select("user_id", "timestamp", "__prev_n", *[c + "p" for c in columns]),
                          on=["user_id", "timestamp"], how="left", validate="m:1").sort("row_id")
    item_vectors = joined.select(columns).to_numpy()
    profiles = joined.select([c + "p" for c in columns]).to_numpy()
    dot = np.sum(profiles * item_vectors, axis=1)
    profile_norm = np.linalg.norm(profiles, axis=1)
    item_norm = np.linalg.norm(item_vectors, axis=1)
    output = pl.DataFrame({"row_id": joined["row_id"], "svd_dot": dot,
                           "svd_cosine": dot / np.maximum(profile_norm * item_norm, 1e-8),
                           "svd_profile_norm": profile_norm, "svd_item_norm": item_norm})
    # Для строк warmup сам basis уже знает их будущие реакции. Поэтому эти строки
    # исключаются из fit и в SVD-модели, и в сравниваемых с ней контролях.
    output = output.with_columns([pl.when(pl.col("row_id") < warmup_end).then(None).otherwise(
        pl.col(c)).cast(pl.Float32).alias(c) for c in output.columns if c != "row_id"])
    return output, {"warmup_end": warmup_end, "warmup_last_timestamp": str(warmup["timestamp"].max()),
                    "dimensions": dimensions, "basis_items": len(items),
                    "unknown_item_fraction": float(np.mean(item_norm[warmup_end:] == 0)),
                    "explained_variance": float(svd.explained_variance_ratio_.sum())}
