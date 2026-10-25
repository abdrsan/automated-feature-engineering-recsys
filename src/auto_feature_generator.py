from src.feature_spec import FeatureSpec
from src.datasets import positive_event


def discover_entities(schema):
    # Связь добавляется, только если соответствующее поле действительно есть в данных.
    entities = [(), ("user_id",), ("item_id",), ("user_id", "item_id")]
    for column in ["category_id", "brand", "author", "merchant"]:
        if column in schema:
            entities.extend([(column,), ("user_id", column)])
    if "gender" in schema:
        entities.append(("item_id", "gender"))
    return entities


def generate_candidates(schema, events, windows, max_features=700):
    specs = []
    numerator = positive_event(events)
    value_column = "rating" if "rating" in schema else "history_value" if "history_value" in schema else None
    for keys in discover_entities(schema):
        # Одна схема порождает семейство признаков: меняются событие и длина истории.
        # Полезные сочетания дальше выбирает validation, а не этот генератор.
        for event in ["all", *sorted(events)]:
            specs.extend(FeatureSpec(keys, event, "count", w) for w in windows)
            specs.append(FeatureSpec(keys, event, "recency"))
        for w in windows:
            specs.append(FeatureSpec(keys, numerator, "event_rate", w))
            if value_column:
                specs.append(FeatureSpec(keys, "all", "mean", w, value_column))
        if value_column:
            specs.extend(FeatureSpec(keys, "all", agg, None, value_column) for agg in ["min", "max", "sum"])
        for w in [7, 30, None]:
            if w in windows and (w is None or (keys and keys != ("category_id",))):
                specs.append(FeatureSpec(keys, "all", "nunique", w, "item_id" if "item_id" not in keys else "user_id"))
        for short, long in [(1, 7), (3, 14), (7, 30), (14, 60), (30, 90)]:
            if short in windows and long in windows:
                specs.extend(FeatureSpec(keys, e, "trend", short, long_window_days=long)
                             for e in ["all", numerator])
    specs = list(dict.fromkeys(specs))
    if len(specs) > max_features:
        # Лучше остановить слишком широкий поиск здесь, до расчета большой матрицы.
        raise ValueError(f"Candidate budget exceeded: {len(specs)} > {max_features}")
    for spec in specs:
        spec.validate(schema, events, allowed_keys=discover_entities(schema))
    return specs
