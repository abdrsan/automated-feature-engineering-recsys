from collections import defaultdict
from time import perf_counter

import polars as pl

from src.feature_spec import FeatureSpec, count_spec
from src.provenance import implementation_signature


FEATURE_ENGINE_VERSION = "2"


def feature_engine_signature():
    return implementation_signature(_mask, _count, _rolling_expr, _all_history,
                                    generate_features, count_spec, FeatureSpec.validate,
                                    FeatureSpec.name.fget)


def _mask(event):
    return pl.lit(True) if event == "all" else pl.col("event") == event


def _count(event):
    return pl.col("__event_" + event).sum()


def _rolling_expr(spec):
    mask = _mask(spec.event)
    aggregation = spec.aggregation
    if aggregation == "count":
        value = _count(spec.event)
    elif aggregation in {"sum", "mean", "min", "max"}:
        value = getattr(pl.col(f"__value_{spec.event}_{spec.value_column}"), aggregation)()
    elif aggregation == "nunique":
        value = pl.col(f"__value_{spec.event}_{spec.value_column}").drop_nulls().n_unique()
    elif aggregation == "recency":
        value = pl.col("__time_" + spec.event).max()
    elif aggregation in {"event_rate", "conversion"}:
        value = (_count(spec.event) + spec.smoothing) / (
            _count(spec.denominator_event) + 2 * spec.smoothing)
    elif aggregation == "frequency":
        value = _count(spec.event) / spec.window_days
    else:
        raise ValueError(aggregation)
    return value.alias(spec.name)


def _all_history(raw, keys, specs):
    # События с одним timestamp сначала объединяем. Обычный shift по строкам
    # позволил бы второй строке увидеть первую, хотя их порядок во времени неизвестен.
    expressions = []
    transforms = []
    for i, spec in enumerate(specs):
        alias = f"__a{i}"
        mask = _mask(spec.event)
        agg = spec.aggregation
        def previous_sum(name):
            # Сдвигаем уже сумму по временным блокам: весь текущий блок остается за границей истории.
            return pl.col(name).cum_sum().shift(1).over(keys).fill_null(0)
        if agg in {"count", "frequency"}:
            expressions.append(_count(spec.event).alias(alias))
            value = previous_sum(alias)
            if agg == "frequency":
                expressions.append(pl.col("timestamp").min().dt.epoch("us").alias(alias + "t"))
                start = pl.col(alias + "t").cum_min().shift(1).over(keys)
                days = ((pl.col("timestamp").dt.epoch("us") - start) / 86_400_000_000).clip(1, None)
                value = (value / days).fill_null(0)
        elif agg == "nunique":
            marker = alias + "first"
            identity = list(dict.fromkeys([*keys, spec.value_column]))
            raw = raw.with_columns(
                pl.when(mask & pl.col(spec.value_column).is_not_null()).then(
                    pl.struct(identity).is_first_distinct()
                ).otherwise(False).alias(marker)
            ) if spec.event == "all" else raw.with_columns(
                pl.when(mask & pl.col(spec.value_column).is_not_null()).then(
                    pl.struct([*identity, "event"]).is_first_distinct()
                ).otherwise(False).alias(marker)
            )
            expressions.append(pl.col(marker).sum().alias(alias))
            value = previous_sum(alias)
        elif agg in {"sum", "mean", "min", "max"}:
            col = pl.col(spec.value_column).filter(mask)
            if agg == "mean":
                expressions.extend([col.sum().alias(alias), col.count().alias(alias + "n")])
                denominator = previous_sum(alias + "n")
                value = pl.when(denominator > 0).then(previous_sum(alias) / denominator)
            else:
                expressions.append(getattr(col, agg)().alias(alias))
                if agg == "sum":
                    value = previous_sum(alias)
                else:
                    value = getattr(pl.col(alias), f"cum_{agg}")().forward_fill().shift(1).over(keys)
        elif agg == "recency":
            expressions.append(pl.col("timestamp").filter(mask).max().dt.epoch("us").alias(alias))
            previous = pl.col(alias).cum_max().forward_fill().shift(1).over(keys)
            value = (pl.col("timestamp").dt.epoch("us") - previous) / 86_400_000_000
        elif agg in {"event_rate", "conversion"}:
            expressions.extend([_count(spec.event).alias(alias),
                                _count(spec.denominator_event).alias(alias + "n")])
            value = (previous_sum(alias) + spec.smoothing) / (
                previous_sum(alias + "n") + 2 * spec.smoothing)
        else:
            raise ValueError(agg)
        transforms.append(value.cast(pl.Float32).alias(spec.name))
    blocks = raw.group_by([*keys, "timestamp"]).agg(expressions).sort([*keys, "timestamp"])
    audit = blocks.select((pl.col("timestamp").shift(1).over(keys) >= pl.col("timestamp")).any()).item()
    if audit:
        raise AssertionError("History cutoff invariant failed")
    return blocks.with_columns(transforms).select([*keys, "timestamp", *[s.name for s in specs]])


def generate_features(df, specs, progress=None):
    started = perf_counter()
    if df["row_id"].n_unique() != df.height:
        raise ValueError("Query row identifiers are not unique")
    events = set(df["event"].unique().to_list())
    requested = list(dict.fromkeys(specs))
    if not requested:
        return df.select("row_id").sort("row_id"), {"seconds": perf_counter() - started,
                "memory_mb": 0., "n_features": 0, "groups": [], "audit_passed": True}
    expanded = list(requested)
    # Для short/long ratio нужны два счетчика. Считаем каждый один раз,
    # даже если он используется сразу в нескольких признаках.
    for spec in requested:
        spec.validate(df.schema, events)
        if spec.aggregation == "trend":
            expanded.extend([count_spec(spec, spec.window_days), count_spec(spec, spec.long_window_days)])
    expanded = list(dict.fromkeys(expanded))
    groups = defaultdict(list)
    for spec in expanded:
        if spec.aggregation != "trend":
            groups[spec.keys].append(spec)
    output = df.select("row_id").sort("row_id")
    timings = []
    for original_keys, group in groups.items():
        tick = perf_counter()
        # Фиктивный ключ позволяет тем же кодом посчитать статистику по всему потоку.
        keys = list(original_keys) or ["__global"]
        raw = df.with_columns(pl.lit(0).alias("__global")) if not original_keys else df
        raw = raw.sort([*keys, "timestamp", "row_id"])
        needed_events = {s.event for s in group} | {s.denominator_event for s in group}
        primitives = []
        for event in needed_events:
            primitives.extend([
                _mask(event).cast(pl.Int64).alias("__event_" + event),
                pl.when(_mask(event)).then(pl.col("timestamp").dt.epoch("us")).otherwise(None).alias("__time_" + event),
            ])
        values_needed = {(s.event, s.value_column) for s in group if s.value_column}
        primitives.extend(pl.when(_mask(event)).then(pl.col(column)).otherwise(None).alias(
            f"__value_{event}_{column}") for event, column in values_needed)
        raw = raw.with_columns(primitives)
        query = raw.select("row_id", *keys, "timestamp")
        windows = defaultdict(list)
        for spec in group:
            windows[spec.window_days].append(spec)
        for window, chunk in windows.items():
            if window is None:
                values = _all_history(raw, keys, chunk)
            else:
                # closed="left" задает [T-W, T): старую границу включаем, событие в T исключаем.
                values = raw.rolling("timestamp", period=f"{window}d", closed="left", group_by=keys).agg(
                    [_rolling_expr(s) for s in chunk] + [pl.col("timestamp").max().alias("__last_source")]
                )
                if values.select((pl.col("__last_source") >= pl.col("timestamp")).any()).item():
                    raise AssertionError("Feature uses an event at or after its prediction time")
                values = values.drop("__last_source").unique([*keys, "timestamp"], maintain_order=True)
                values = values.with_columns([
                    ((pl.col("timestamp").dt.epoch("us") - pl.col(s.name)) / 86_400_000_000
                     if s.aggregation == "recency" else pl.col(s.name)).cast(pl.Float32).alias(s.name)
                    for s in chunk
                ])
            # После группировки возвращаем значения исходным строкам. validate и проверка
            # числа строк ловят many-to-many join, который мог бы незаметно изменить выборку.
            aligned = query.join(values, on=[*keys, "timestamp"], how="left", validate="m:1",
                                 nulls_equal=True).select("row_id", *[s.name for s in chunk]).sort("row_id")
            if aligned.height != output.height:
                raise AssertionError("Feature join changed query cardinality")
            output = output.hstack(aligned.drop("row_id"))
        timings.append({"entity": group[0].entity, "seconds": perf_counter() - tick,
                        "n_features": len(group)})
        if progress:
            progress(timings[-1])
    for spec in requested:
        if spec.aggregation == "trend":
            short, long = count_spec(spec, spec.window_days), count_spec(spec, spec.long_window_days)
            output = output.with_columns(((pl.col(short.name) + spec.smoothing) / (
                pl.col(long.name) + spec.smoothing)).cast(pl.Float32).alias(spec.name))
    output = output.select("row_id", *[s.name for s in requested])
    if output.select(pl.any_horizontal(pl.col(s.name).is_infinite() for s in requested).any()).item():
        raise AssertionError("Infinite feature value")
    return output, {"seconds": perf_counter() - started, "memory_mb": output.estimated_size("mb"),
                    "n_features": len(requested), "groups": timings,
                    "cutoff": "[T - window, T); all_history < T", "audit_passed": True}
