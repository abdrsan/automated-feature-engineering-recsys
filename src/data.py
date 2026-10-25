from pathlib import Path
import hashlib
import io
import json
import zipfile

import polars as pl


def _dat(path, columns):
    content = Path(path).read_bytes().decode("latin-1").replace("::", "\t")
    return pl.read_csv(io.StringIO(content), separator="\t", has_header=False,
                       new_columns=columns, infer_schema_length=10000)


def load_movielens(raw_dir="data/raw"):
    raw = Path(raw_dir)
    folder = raw / "ml-1m"
    if not folder.exists():
        if hashlib.md5((raw / "ml-1m.zip").read_bytes()).hexdigest() != "c4d9eecfca2ab87c1945afe126590906":
            raise ValueError("MovieLens archive is incomplete or checksum differs")
        with zipfile.ZipFile(raw / "ml-1m.zip") as archive:
            for name in ("ratings.dat", "movies.dat", "users.dat", "README"):
                member = f"ml-1m/{name}"
                if member in archive.namelist():
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / name).write_bytes(archive.read(member))
    ratings = _dat(folder / "ratings.dat", ["user_id", "item_id", "rating", "timestamp"])
    users = _dat(folder / "users.dat", ["user_id", "gender", "age", "occupation", "zip"])
    items = _dat(folder / "movies.dat", ["item_id", "title", "genres"])
    items = items.with_columns(
        pl.col("genres").str.split("|").list.first().alias("category_id"),
        pl.col("title").str.extract(r"\((\d{4})\)$", 1).cast(pl.Int32).alias("release_year"),
    ).drop("title")
    df = ratings.join(users, on="user_id", how="left", validate="m:1").join(
        items, on="item_id", how="left", validate="m:1"
    ).with_columns(
        (pl.col("rating") >= 4).cast(pl.Int8).alias("label"),
        pl.when(pl.col("rating") >= 4).then(pl.lit("positive")).otherwise(
            pl.lit("negative")).alias("event"),
        pl.from_epoch("timestamp", time_unit="s").alias("timestamp"),
    ).sort(["timestamp", "user_id", "item_id"]).with_row_index("row_id")
    validate_data(df)
    return df


def load_interactions(path, event_map=None, positive_label=None):
    path = Path(path)
    df = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)
    if df.schema["timestamp"].is_numeric():
        df = df.with_columns(pl.from_epoch("timestamp", time_unit="s"))
    elif df.schema["timestamp"] == pl.String:
        df = df.with_columns(pl.col("timestamp").str.to_datetime())
    if "event" not in df.columns:
        if event_map:
            df = df.with_columns(pl.col("label").replace_strict(
                {int(k): v for k, v in event_map.items()}).alias("event"))
        else:
            df = df.with_columns(pl.col("label").cast(pl.String).alias("event"))
    if positive_label is not None:
        df = df.with_columns((pl.col("label") == positive_label).cast(pl.Int8).alias("label"))
    if set(df["label"].unique().to_list()) != {0, 1}:
        raise ValueError("Classifier requires both binary labels; specify positive_label explicitly")
    df = df.sort(["timestamp", "user_id", "item_id"]).with_row_index("row_id")
    validate_data(df)
    return df


def validate_data(df):
    required = {"user_id", "item_id", "timestamp", "label", "event", "row_id"}
    if not required <= set(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    if df.select(pl.any_horizontal(pl.col(c).is_null() for c in required).any()).item():
        raise ValueError("Required identifiers, times, events and targets cannot be null")
    if df["row_id"].n_unique() != df.height:
        raise ValueError("row_id must be unique")
    if not df["timestamp"].is_sorted():
        raise ValueError("Interactions must be sorted by timestamp")


def describe_data(df):
    return {
        "rows": df.height, "users": df["user_id"].n_unique(),
        "items": df["item_id"].n_unique(),
        "start": str(df["timestamp"].min()), "end": str(df["timestamp"].max()),
        "schema": {k: str(v) for k, v in df.schema.items()},
        "nulls": df.null_count().row(0, named=True),
        "label_distribution": df.group_by("label").len().sort("label").to_dicts(),
        "event_distribution": df.group_by("event").len().sort("event").to_dicts(),
        "duplicate_events": df.height - df.unique(
            ["user_id", "item_id", "timestamp", "label"]).height,
        "repeat_user_item_rows": df.height - df.unique(["user_id", "item_id"]).height,
        "distinct_timestamps": df["timestamp"].n_unique(),
        "same_timestamp_rows": df.height - df["timestamp"].n_unique(),
        "memory_mb": df.estimated_size("mb"),
    }


def dataset_fingerprint(df):
    return hashlib.sha256(df.hash_rows(seed=42).to_numpy().tobytes()).hexdigest()


def study_statistics(df, split):
    stats = describe_data(df)
    stats["splits"] = {}
    train = df.head(split.train_end)
    users, items = train["user_id"].unique().implode(), train["item_id"].unique().implode()
    for name, indices in [("train", split.train), ("validation_search", split.search),
                          ("validation_audit", split.audit), ("test", split.test)]:
        part = df[indices]
        stats["splits"][name] = {"rows": len(indices), "start": str(part["timestamp"].min()),
            "end": str(part["timestamp"].max()),
            "positive_rate": part["label"].mean() if name != "test" else None,
            "cold_users_fraction": 1-part["user_id"].is_in(users).mean(),
            "cold_items_fraction": 1-part["item_id"].is_in(items).mean()}
    return stats


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
