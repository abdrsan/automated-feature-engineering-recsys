from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalSplit:
    train_end: int
    search_end: int
    validation_end: int
    n_rows: int

    @property
    def train(self):
        return np.arange(self.train_end)

    @property
    def search(self):
        return np.arange(self.train_end, self.search_end)

    @property
    def audit(self):
        return np.arange(self.search_end, self.validation_end)

    @property
    def test(self):
        return np.arange(self.validation_end, self.n_rows)


def temporal_split(df, fractions=(0.70, 0.775, 0.85)):
    if not df.height or not df["timestamp"].is_sorted() or df["timestamp"].null_count():
        raise ValueError("Temporal split requires nonempty, sorted, non-null timestamps")
    times = df["timestamp"].to_numpy()
    if not 0 < fractions[0] < fractions[1] < fractions[2] < 1:
        raise ValueError("Expected train < search validation < audit validation < test")
    # Номинальная граница может попасть внутрь дня или 5s bin. Сдвигаем ее
    # к первому событию этого timestamp, чтобы одинаковое время не разделилось между частями.
    bounds = [int(np.searchsorted(times, times[int(len(times) * f)], side="left"))
              for f in fractions]
    if not 0 < bounds[0] < bounds[1] < bounds[2] < len(times):
        raise ValueError("Too few distinct timestamps for temporal split")
    for bound in bounds:
        assert times[bound - 1] < times[bound], "Equal timestamps crossed a split"
    return TemporalSplit(*bounds, len(times))
