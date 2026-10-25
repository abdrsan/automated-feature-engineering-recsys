import io
import json
from pathlib import Path
import zipfile

import polars as pl

from src.data import validate_data


def positive_event(events):
    for event in ["positive", "like", "watch_high"]:
        if event in events:
            return event
    return sorted(events)[-1]


def finish_interactions(df):
    df = df.sort(["timestamp", "user_id", "item_id", "event"]).with_row_index("row_id")
    validate_data(df)
    if set(df["label"].unique().to_list()) != {0, 1}:
        raise ValueError("Both binary labels are required")
    return df


def kion_interactions(interactions, users, items, user_modulus=1):
    """Predict reported aggregate watch percentage; same-day records are atomic."""
    required = {"user_id", "item_id", "last_watch_dt", "watched_pct"}
    if not required <= set(interactions.columns):
        raise ValueError(f"KION columns missing: {required - set(interactions.columns)}")
    if user_modulus < 1:
        raise ValueError("user_modulus must be >= 1")
    df = interactions.filter(pl.col("user_id") % user_modulus == 0)
    df = df.with_columns(pl.col("watched_pct").cast(pl.Float64, strict=False))
    df = df.filter(pl.col("watched_pct").is_not_null() & (pl.col("watched_pct") >= 0))
    users = users.select("user_id", "age", "income", "sex", "kids_flg").rename({"sex": "gender"})
    items = items.select("item_id", "content_type", "release_year", "genres", "countries",
                         "for_kids", "age_rating", "directors").with_columns(
        pl.col("genres").str.split(",").list.first().str.strip_chars().alias("category_id"),
        pl.col("directors").str.split(",").list.first().str.strip_chars().alias("author"),
    ).drop("directors")
    df = df.join(users, on="user_id", how="left", validate="m:1").join(
        items, on="item_id", how="left", validate="m:1")
    df = df.with_columns(
        pl.col("last_watch_dt").cast(pl.String).str.to_date().cast(pl.Datetime("us")).alias("timestamp"),
        (pl.col("watched_pct") >= 50).cast(pl.Int8).alias("label"),
        pl.when(pl.col("watched_pct") >= 50).then(pl.lit("watch_high")).otherwise(
            pl.lit("watch_low")).alias("event"),
        pl.col("watched_pct").alias("history_value"),
        pl.lit(False).alias("time_is_relative"),
    ).drop("last_watch_dt", "watched_pct")
    if "total_dur" in df.columns:
        df = df.drop("total_dur")  # Длительность текущего просмотра стала бы подсказкой о target.
    df = df.with_columns([pl.col(c).fill_null("__unknown__") for c in
                          ["category_id", "author", "gender", "genres", "countries", "content_type", "income", "age"]])
    return finish_interactions(df)


def load_kion(raw_dir="data/raw", user_modulus=1):
    folder = Path(raw_dir) / "kion"
    with zipfile.ZipFile(folder / "interactions.csv.zip") as archive:
        members = [name for name in archive.namelist() if Path(name).name == "interactions.csv"]
        if len(members) != 1:
            raise ValueError("Expected one interactions.csv in official KION archive")
        interactions = pl.read_csv(io.BytesIO(archive.read(members[0])), infer_schema_length=10000)
    return kion_interactions(interactions, pl.read_csv(folder / "users.csv", infer_schema_length=10000),
                             pl.read_csv(folder / "items.csv", infer_schema_length=10000), user_modulus)


def yambda_interactions(likes, dislikes, artists, albums):
    # В статье T' = floor((T-T0)/5)*5: это уже секунды, округленные до 5s.
    # Повторное умножение на пять растянуло бы все временные окна.
    df = pl.concat([likes.with_columns(pl.lit("like").alias("event"), pl.lit(1, pl.Int8).alias("label")),
                    dislikes.with_columns(pl.lit("dislike").alias("event"), pl.lit(0, pl.Int8).alias("label"))])
    df = df.rename({"uid": "user_id", "timestamp": "relative_seconds"})
    if df.filter(pl.col("relative_seconds") % 5 != 0).height:
        raise ValueError("Yambda timestamps must be multiples of 5 seconds")
    if not df["is_organic"].is_in([0, 1]).all():
        raise ValueError("Invalid is_organic flag")
    # У item может быть несколько артистов и альбомов. Для этого эксперимента берем
    # минимальный ID, чтобы join не размножил события; часть информации при этом теряется.
    artists = artists.group_by("item_id").agg(pl.col("artist_id").min().alias("author"))
    albums = albums.group_by("item_id").agg(pl.col("album_id").min().alias("category_id"))
    df = df.join(artists, on="item_id", how="left", validate="m:1").join(
        albums, on="item_id", how="left", validate="m:1").with_columns(
        pl.from_epoch(pl.col("relative_seconds").cast(pl.Int64), time_unit="s").alias("timestamp"),
        pl.col("author").cast(pl.String).fill_null("__unknown__"),
        pl.col("category_id").cast(pl.String).fill_null("__unknown__"),
        pl.lit(True).alias("time_is_relative"),
    )
    return finish_interactions(df)


def load_yambda(raw_dir="data/raw"):
    folder = Path(raw_dir) / "yambda"
    return yambda_interactions(*[pl.read_parquet(folder / name) for name in
                                 ["likes.parquet", "dislikes.parquet", "artist_item_mapping.parquet", "album_item_mapping.parquet"]])


def dataset_protocol(name, user_modulus=1):
    if name == "kion":
        return {"name": "KION", "target": "reported watched_pct >= 50 among observed aggregate watches",
                "events": ["watch_high", "watch_low"], "cohort": f"user_id % {user_modulus} == 0; all valid-target records; no temporal row truncation",
                "time": "last_watch_dt at day precision; exclude entire target day",
                "history_value": "reported watched_pct (past records only)",
                "category_id": "first listed genre", "author": "first listed director",
                "limitations": "One aggregate user-item record, not session/impression log; no CTR or full-catalog retrieval claim; static metadata assumed"}
    if name == "yambda":
        return {"name": "Yambda-50M explicit-feedback subset", "target": "like (1) vs dislike (0), conditional on an explicit reaction",
                "events": ["like", "dislike"], "cohort": "all likes and dislikes from official flat/50m files; no row sampling",
                "time": "relative seconds, rounded DOWN to multiples of 5; epoch is an internal coordinate only; no calendar features",
                "category_id": "minimum album_id per item", "author": "minimum artist_id per item",
                "limitations": "No listen/unlike/undislike context in this run; event-history counts are not current preference state; action-conditioned probabilities are not CTR or retrieval metrics"}
    return {"name": name}
