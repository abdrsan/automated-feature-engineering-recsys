from src.feature_spec import FeatureSpec
from src.datasets import positive_event


def manual_features(schema, events):
    relations = [("user_id",), ("item_id",), ("user_id", "item_id")]
    if "category_id" in schema:
        relations.append(("user_id", "category_id"))
    if "author" in schema:
        relations.append(("user_id", "author"))
    specs = []
    for keys in relations:
        for event in ["all", *sorted(events)]:
            for window in [None, 7, 30]:
                specs.append(FeatureSpec(keys, event, "count", window))
            specs.append(FeatureSpec(keys, event, "recency"))
        # В KION есть доля просмотра, в архивном MovieLens — оценка.
        # Их среднее по прошлым событиям дает более тонкий сигнал, чем один бинарный count.
        value_column = "rating" if "rating" in schema else "history_value" if "history_value" in schema else None
        if value_column:
            for window in [None, 7, 30]:
                specs.append(FeatureSpec(keys, "all", "mean", window, value_column))
        numerator = positive_event(events)
        for window in [None, 7, 30]:
            specs.append(FeatureSpec(keys, numerator, "event_rate", window))
    return list(dict.fromkeys(specs))


def manual_trends(schema, events):
    relations = [("user_id",), ("item_id",)]
    if "category_id" in schema:
        relations.append(("user_id", "category_id"))
    if "author" in schema:
        relations.append(("user_id", "author"))
    return [FeatureSpec(keys, event, "trend", 7, long_window_days=30)
            for keys in relations for event in ["all", *sorted(events)]]


def ablation_groups(specs):
    def stage(spec):
        if spec.aggregation == "trend": return 7
        if spec.aggregation in {"event_rate", "conversion"}: return 6
        return {("user_id",): 1, ("item_id",): 2, ("user_id", "item_id"): 3,
                ("user_id", "category_id"): 4}.get(spec.keys, 5)
    names = ["base_categorical", "plus_user_temporal", "plus_item_temporal",
             "plus_user_item", "plus_user_category", "plus_metadata_relations",
             "manual", "manual_plus_trend"]
    # Если author отсутствует, этот этап ничего не добавляет к user-category.
    # Повторно обучать ту же модель под другим именем нет смысла.
    return [(name, [s for s in specs if stage(s) <= i])
            for i, name in enumerate(names) if i != 5 or any(s.keys == ("user_id", "author") for s in specs)]
