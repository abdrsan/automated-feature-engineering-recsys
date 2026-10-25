from src.auto_feature_generator import discover_entities
from src.feature_spec import FeatureSpec


def propose_features(schema, events, current_features, previous_experiments,
                     feature_importance, limit=12, llm_provider=None):
    """Propose intermediate windows around useful families; never reads test metrics."""
    if llm_provider is not None:
        payload = llm_provider({"schema": {k: str(v) for k, v in schema.items()},
                                "entities": [list(keys) for keys in discover_entities(schema)],
                                "events": sorted(events),
                                "current_features": [s.to_dict() for s in current_features],
                                "previous_experiments": previous_experiments,
                                "feature_importance": feature_importance,
                                "max_proposals": limit})
        return [s for s in validate_llm_proposals(payload, schema, events, limit)
                if s not in set(current_features)]
    current = set(current_features)
    tried = {name for experiment in previous_experiments for name in experiment.get("proposed", [])}
    by_name = {s.name: s for s in current}
    proposals = []
    # У полезных семейств пробуем соседние длины окна: 0.65× и 1.4× от текущей.
    # Это простая эвристика предложения; решение оставить признак принимает validation.
    for name in sorted(feature_importance, key=lambda n: (-feature_importance[n], n)):
        spec = by_name.get(name)
        if not spec or spec.window_days is None or spec.aggregation == "trend":
            continue
        for factor in [0.65, 1.4]:
            window = max(1, min(365, round(spec.window_days * factor)))
            candidate = FeatureSpec(spec.keys, spec.event, spec.aggregation, window,
                                    spec.value_column, spec.denominator_event)
            candidate.validate(schema, events, discover_entities(schema))
            if candidate not in current and candidate.name not in tried and candidate not in proposals:
                proposals.append(candidate)
                if len(proposals) >= limit:
                    return proposals
    return proposals


def validate_llm_proposals(payload, schema, events, max_features=12):
    # Внешняя модель может вернуть список описаний признаков в JSON.
    # Проверяем их той же грамматикой; SQL или Python от модели не исполняем.
    if not isinstance(payload, list) or len(payload) > max_features:
        raise ValueError("Expected a bounded JSON list of FeatureSpec objects")
    specs = [FeatureSpec.from_dict(value) for value in payload]
    for spec in specs:
        spec.validate(schema, events, discover_entities(schema))
    return list(dict.fromkeys(specs))
