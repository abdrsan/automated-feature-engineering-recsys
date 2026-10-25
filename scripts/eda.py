import json
from src.data import describe_data, load_movielens, save_json
from src.split import temporal_split

if __name__ == "__main__":
    df = load_movielens()
    split = temporal_split(df)
    stats = describe_data(df)
    stats["splits"] = {}
    for name, indices in [("train", split.train), ("validation_search", split.search),
                          ("validation_audit", split.audit), ("test", split.test)]:
        part = df[indices]
        stats["splits"][name] = {
            "rows": len(indices), "start": str(part["timestamp"].min()),
            "end": str(part["timestamp"].max()),
            "positive_rate": part["label"].mean() if name != "test" else None,
            "cold_users_fraction": 1 - part["user_id"].is_in(df[split.train]["user_id"].unique().implode()).mean(),
            "cold_items_fraction": 1 - part["item_id"].is_in(df[split.train]["item_id"].unique().implode()).mean(),
        }
    save_json("artifacts/eda_full.json", stats)
    print(json.dumps(stats, indent=2))
    df.write_parquet("data/processed/interactions.parquet")
