from dataclasses import asdict, dataclass
import re


AGGREGATIONS = {"count", "sum", "mean", "min", "max", "nunique", "recency",
                "event_rate", "conversion", "frequency", "trend"}


@dataclass(frozen=True)
class FeatureSpec:
    keys: tuple[str, ...]
    event: str = "all"
    aggregation: str = "count"
    window_days: int | None = None
    value_column: str | None = None
    denominator_event: str = "all"
    long_window_days: int | None = None
    smoothing: float = 1.0

    @property
    def entity(self):
        return "_".join(k.removesuffix("_id") for k in self.keys) or "global"

    @property
    def name(self):
        window = "all" if self.window_days is None else f"{self.window_days}d"
        parts = [self.entity, self.event, self.aggregation, window]
        if self.value_column:
            parts.append(self.value_column)
        if self.aggregation in {"event_rate", "conversion"}:
            parts.append(self.denominator_event)
        if self.aggregation == "trend":
            parts.append("all" if self.long_window_days is None else f"{self.long_window_days}d")
        if self.smoothing != 1.0:
            parts.append(f"s{self.smoothing:g}".replace(".", "p"))
        return "__".join(parts)

    def validate(self, schema, events, allowed_keys=None):
        # Описание может прийти от proposer или внешней модели. До расчета проверяем,
        # что поля существуют, окно осмысленно, а вместо имени не передан исполняемый код.
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")
        names = [*self.keys, self.event, self.denominator_event]
        if self.value_column:
            names.append(self.value_column)
        if any(not re.fullmatch(r"[A-Za-z0-9_]+", n) for n in names):
            raise ValueError("FeatureSpec accepts identifiers, never executable code")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("Repeated keys")
        if any(k not in schema for k in self.keys):
            raise ValueError("Unknown entity key")
        if {"label", "event", "timestamp", "rating", "row_id", "history_value", "watched_pct", "total_dur"} & set(self.keys):
            raise ValueError("Targets, event values and times cannot be entity keys")
        if allowed_keys is not None and self.keys not in allowed_keys:
            raise ValueError("Entity relationship is not allowlisted")
        for event in (self.event, self.denominator_event):
            if event != "all" and event not in events:
                raise ValueError(f"Unknown event: {event}")
        for window in (self.window_days, self.long_window_days):
            if window is not None and (type(window) is not int or not 1 <= window <= 3650):
                raise ValueError("Windows must be integer days in [1, 3650]")
        if not 0 < self.smoothing <= 100:
            raise ValueError("Smoothing must be positive and bounded")
        if self.aggregation in {"sum", "mean", "min", "max", "nunique"}:
            if self.value_column not in schema:
                raise ValueError("Aggregation requires an existing value_column")
            if self.aggregation != "nunique" and not schema[self.value_column].is_numeric():
                raise ValueError("Numeric aggregation needs a numeric column")
        elif self.value_column is not None:
            raise ValueError("value_column does not apply to this aggregation")
        if self.aggregation == "trend":
            if self.window_days is None:
                raise ValueError("Trend needs a finite short window")
            if self.long_window_days is not None and self.long_window_days <= self.window_days:
                raise ValueError("Trend needs long window > short window")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["keys"] = tuple(value["keys"])
        return cls(**value)


def count_spec(spec, window):
    return FeatureSpec(spec.keys, spec.event, "count", window)


def to_sql(spec):
    # SQL нужен, чтобы показать смысл признака привычными GROUP BY/JOIN операциями.
    # В самом эксперименте эти запросы не выполняются: вычисления делает Polars.
    match = " AND ".join(f'h."{k}" IS NOT DISTINCT FROM q."{k}"' for k in spec.keys) or "TRUE"
    time = 'h.timestamp < q.timestamp'
    if spec.window_days is not None:
        time += f" AND h.timestamp >= q.timestamp - INTERVAL '{spec.window_days} days'"
    event = "TRUE" if spec.event == "all" else f"h.event = '{spec.event}'"
    expressions = {"count": "COUNT(h.row_id)", "sum": f'COALESCE(SUM(h."{spec.value_column}"), 0)',
                   "mean": f'AVG(h."{spec.value_column}")', "min": f'MIN(h."{spec.value_column}")',
                   "max": f'MAX(h."{spec.value_column}")',
                   "nunique": f'COUNT(DISTINCT h."{spec.value_column}")',
                   "recency": "EXTRACT(EPOCH FROM (q.timestamp - MAX(h.timestamp))) / 86400"}
    if spec.aggregation not in expressions:
        raise ValueError("Derived features compile from count/recency SQL primitives")
    return (f'SELECT q.row_id, {expressions[spec.aggregation]} AS "{spec.name}"\n'
            f'FROM interactions q LEFT JOIN interactions h ON {match} AND {time} AND {event}\n'
            'GROUP BY q.row_id, q.timestamp;')
