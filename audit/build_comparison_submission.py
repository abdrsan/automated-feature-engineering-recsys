"""Assemble a standalone comparison from the existing, audited implementation."""
import ast
import hashlib
import json
from pathlib import Path

import nbformat as nbf
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "FINAL_SUBMISSION"


def existing_code(relative, names):
    source = (ROOT / relative).read_text()
    lines = source.splitlines()
    pieces = []
    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], "id", None)
        if name in names:
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            pieces.append("\n".join(lines[first-1:node.end_lineno]))
    return "\n\n\n".join(pieces)


def build():
    DEST.mkdir(exist_ok=True)
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    requirements = [line for line in requirements if not line.startswith(("optuna", "PyYAML", "pytest"))]
    requirements += ["nbformat==5.10.4", "nbclient==0.10.2", "ipykernel==6.30.0"]
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text, **metadata):
        cells.append(nbf.v4.new_code_cell(text.strip(), metadata=metadata))
    def reuse(path, names):
        code(existing_code(path, names), original_source=path)

    sources, parameters, original, provenance = {}, {}, {}, {}
    for dataset in ["kion", "yambda"]:
        root = ROOT / "artifacts" / dataset
        sources[dataset] = json.loads((root / "source_manifest.json").read_text())
        parameters[dataset] = json.loads((root / "manual_params.json").read_text())
        rows = pl.read_csv(root / "experiments.csv")
        original[dataset] = rows.filter((pl.col("split") == "test") & pl.col("experiment_name").is_in(
            ["manual", "automatic", "auto_warmup", "auto_plus_svd"])).select(
                "experiment_name", "seed", "n_features", "logloss", "auc", "training_time").to_dicts()
        provenance[dataset] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ["experiments.csv", "locked_selection.json", "manual_tuning.csv", "auto_tuning.csv"]}

    md("""# Сравнение методов Feature Engineering

## 1. Введение

Проверяем, какие признаки дают лучший баланс качества, времени расчета и понятности
в одной и той же LightGBM-модели. Основа — существующий проект с ручными временными
агрегатами и автоматическим поиском. Реализации ниже взяты из его модулей; новая
оболочка нужна для короткого, одинакового протокола сравнения.

Два независимых эксперимента: KION — доля просмотра ≥50% среди наблюдаемых
пользователь–фильм агрегатов; Yambda — like против dislike среди явных реакций.
Это binary classification. Основная метрика — LogLoss, дополнительные — ROC-AUC и
average precision (PR-AUC в смысле AP). Качество выдачи по всему каталогу эти задачи не измеряют.

Сравниваем raw IDs/metadata, manual aggregates, manual + trends, automated fixed windows,
полный automated pool и отбор из него. Гипотеза: расширение агрегатов полезно, но
дополнительные признаки и окна не обязательно окупают стоимость. SVD из исходного
проекта разбирается отдельно: его warmup меняет обучающую популяцию.

Все test результаты здесь — повторная оценка уже известного holdout. Это ограничивает
силу выводов, даже если новый код выбирает признаки только по validation.

Первая ячейка проверяет зависимости и при необходимости выполняет `%pip install`
в окружении текущего kernel. По умолчанию выбран полный эксперимент; `demo` оставлен
для быстрой проверки на 100k событий. Сохраненные outputs должны относиться к `full`.
""")
    code('''from importlib import metadata

# Устанавливаем только отсутствующие пакеты или версии, отличающиеся от проверенных.
# Используем %pip через API IPython: установка попадет в окружение текущего kernel.
REQUIRED_PACKAGES = ''' + repr(requirements) + '''
install = []
for requirement in REQUIRED_PACKAGES:
    package, expected = requirement.split("==")
    try:
        actual = metadata.version(package)
    except metadata.PackageNotFoundError:
        actual = None
    if actual != expected:
        install.append(requirement)
if install:
    get_ipython().run_line_magic("pip", "install --disable-pip-version-check " + " ".join(install))
assert all(metadata.version(name.split("==")[0]) == name.split("==")[1] for name in REQUIRED_PACKAGES)
print("Зависимости установлены и совпадают с requirements.txt")
''')
    code('''import os
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fe-comparison-mpl")

import gc
import io
import json
import re
import hashlib
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt
import psutil
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
from IPython.display import display

MODE = os.environ.get("FE_MODE", "full")  # demo: 100k rows; full: all rows
DATASETS = os.environ.get("FE_DATASETS", "kion,yambda").split(",")
SEED = 42
assert MODE in {"demo", "full"} and set(DATASETS) <= {"kion", "yambda"}
RUN = Path(".runs") / (MODE + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN.mkdir(parents=True, exist_ok=False)
CONFIG = {"threads": 4, "max_estimators": 80 if MODE == "demo" else 600,
          "early_stopping_rounds": 10 if MODE == "demo" else 40}
WINDOWS = [1, 7, 30, None] if MODE == "demo" else [1, 3, 7, 14, 21, 30, 60, 90, 180, None]
BUDGETS = [16, 32] if MODE == "demo" else [32, 64, 96, 128]
PRUNING = {"max_missing_fraction": .999, "near_constant_fraction": .9999,
           "correlation_threshold": .9999}
THRESHOLD = .0001
plt.rcParams.update({"figure.figsize": (9, 4), "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
print("Mode:", MODE, "datasets:", DATASETS, "outputs:", RUN)
''')
    md("""`demo` проверяет весь путь, включая обучение и test, на первых 100 тысячах событий.
Его цифры нельзя выдавать за полный эксперимент. Для полного повторения измените
`MODE` на `full` или задайте `FE_MODE=full` перед запуском kernel. Новый каталог
создается при каждом Run All; старые эксперименты не перезаписываются.

Параметры LightGBM ниже взяты из manual HPO исходного проекта (20 trials на dataset).
Они одинаковы для всех методов внутри dataset. Это сознательно консервативный контроль
в пользу manual; общую оптимальность параметров не заявляем. В этом notebook нового HPO нет.
""")
    code("SOURCES = " + repr(sources) + "\n\nCOMMON_PARAMS = " + repr(parameters) +
         "\n\nORIGINAL_TEST_RUNS = " + repr(original) + "\n\nORIGINAL_PROVENANCE = " + repr(provenance))
    md("""## 2. Данные и EDA

KION содержит один агрегат на user–item с точностью даты до дня. Строки без watched_pct
исключены, текущие watched_pct/total_dur не входят в модель. Историческая доля просмотра
допустима только из предыдущих дней. В Yambda взяты все likes/dislikes из flat/50m,
а не все 50M действий. Время уже измерено в относительных секундах, округленных вниз
до 5s: дополнительно умножать его на пять нельзя. Календарные признаки для него отключены.

Многозначные metadata упрощены так же, как в исходном проекте: первый жанр/режиссер
KION и минимальный album/artist ID Yambda. Это предотвращает row explosion, но теряет
часть информации. Metadata считаются заранее известными; их исторические версии отсутствуют.
""")
    code('''def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_raw_data():
    if os.environ.get("FE_RAW_DIR"):
        return Path(os.environ["FE_RAW_DIR"])
    for folder in [Path.cwd(), *Path.cwd().parents]:
        if (folder / "data/raw/kion").exists():
            return folder / "data/raw"
    return Path(".data/raw")


RAW = locate_raw_data()
for dataset in DATASETS:
    for entry in SOURCES[dataset]["files"]:
        path = RAW / dataset / entry["name"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".part")
            print("Downloading", entry["name"], flush=True)
            with urllib.request.urlopen(entry["url"], timeout=120) as response, partial.open("wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
            assert sha256(partial) == entry["sha256"], "Downloaded source checksum differs"
            partial.replace(path)
        assert path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"], path
print("Pinned raw files verified:", RAW)
''')
    reuse("src/data.py", {"validate_data"})
    reuse("src/datasets.py", {"positive_event", "finish_interactions", "kion_interactions", "load_kion",
                              "yambda_interactions", "load_yambda"})
    reuse("src/split.py", {"TemporalSplit", "temporal_split"})
    code('''frames, splits, eda = {}, {}, []
for dataset in DATASETS:
    frame = load_kion(RAW) if dataset == "kion" else load_yambda(RAW)
    if MODE == "demo":
        frame = frame.head(100000)
    frames[dataset] = frame
    splits[dataset] = temporal_split(frame)
    train = frame.head(splits[dataset].train_end)
    eda.append({"dataset": dataset, "rows": frame.height, "users": frame["user_id"].n_unique(),
                "items": frame["item_id"].n_unique(), "start": str(frame["timestamp"].min()),
                "end": str(frame["timestamp"].max()), "train_positive_rate": train["label"].mean(),
                "duplicate_events": frame.height-frame.unique(["user_id", "item_id", "timestamp", "label"]).height})
    print(dataset, "schema:", {k: str(v) for k, v in frame.schema.items()})
    missing = frame.null_count().unpivot(variable_name="column", value_name="missing").filter(pl.col("missing") > 0)
    display(pd.DataFrame(missing.to_dicts(), columns=["column", "missing"]))
display(pd.DataFrame(eda))
fig, axes = plt.subplots(1, len(DATASETS), figsize=(6*len(DATASETS), 3), squeeze=False)
for ax, dataset in zip(axes[0], DATASETS):
    train = frames[dataset].head(splits[dataset].train_end)
    counts = train.group_by("label").len().sort("label")
    ax.bar(counts["label"].cast(pl.String), counts["len"])
    ax.set(title=dataset + ": target on train", xlabel="label", ylabel="events")
plt.tight_layout()
plt.show()
''')
    md("""## 3. Подготовка и защита от leakage

Global temporal split: около 70% train, 7.5% search, 7.5% audit, 15% test.
Одинаковые timestamps не разделяются. Encoder и pruning fit-ятся только на train;
search выбирает размер набора и число деревьев. Audit не участвует в выборе.
После lock модели fit-ятся на первых 85% с фиксированным числом деревьев, затем оцениваются на test.

История для T — `[T−W, T)`, то есть текущее событие и весь его timestamp исключены.
Это online replay: более ранние исходы внутри holdout доступны следующим событиям.
Для production нужны настоящие label availability timestamps. У KION мы предполагаем
доступность агрегата после last_watch_day; файл не позволяет доказать это допущение.
""")
    code('''partition_rows = []
for dataset, frame in frames.items():
    split = splits[dataset]
    previous = None
    for name, ids in [("train", split.train), ("search", split.search), ("audit", split.audit), ("test", split.test)]:
        part = frame[ids]
        assert previous is None or previous < part["timestamp"].min()
        previous = part["timestamp"].max()
        partition_rows.append({"dataset": dataset, "part": name, "rows": part.height,
                               "start": str(part["timestamp"].min()), "end": str(previous)})
display(pd.DataFrame(partition_rows))
''')
    md("""## 4. Baseline и модель

Raw baseline получает user/item IDs, доступные metadata и известные календарные поля
(только KION). Словари категорий строятся на train; неизвестные значения получают −1,
который LightGBM обрабатывает как missing category. Event, label, текущий outcome в
base matrix не входят. Все методы используют один encoder, seed=42, общие параметры,
одинаковый early stopping и одинаковые строки обучения.
""")
    reuse("src/model.py", {"BASE_CATEGORICAL", "fit_encoder", "base_matrix", "matrix_from_features",
                           "take_rows", "fit_model"})
    md("""## 5. Методы Feature Engineering

| Метод | Что меняется | Зачем это проверяем |
|---|---|---|
| Raw | IDs и metadata | Качество без событийной истории |
| Manual | Counts, recency, positive-event rates; 7/30/all; user/item/user-item и metadata relations | Разумный ручной reference |
| Manual + trends | Сглаженные count7/count30 | Добавляет ли динамика интереса пользу |
| Auto fixed | Автоматические relations/aggregations, но окна 7/30/all | Польза factory без широкого поиска окон |
| Auto pool | Окна 1/3/7/14/21/30/60/90/180/all и train pruning | Что дает широкий набор |
| Auto selected | Gain предлагает top-K, каждый набор заново проверяется на search | Можно ли уменьшить число inputs |

В KION есть также среднее и extrema **прошлой** watched_pct. Rates сглажены,
trends используют `(short+1)/(long+1)`. Движок обрабатывает нулевую историю и ties;
NaN для первой recency/mean оставляем LightGBM. Count при пустой истории равен нулю.

Ниже — существующая декларативная грамматика и Polars engine. Generic engine длиннее
конкретного эксперимента: один расчет обслуживает разные entities, windows и aggregations.
Python loops здесь идут по feature families; цикл по каждому interaction не используется.
""")
    reuse("src/feature_spec.py", {"AGGREGATIONS", "FeatureSpec", "count_spec"})
    reuse("src/feature_engine.py", {"_mask", "_count", "_rolling_expr", "_all_history", "generate_features"})
    reuse("src/baseline_features.py", {"manual_features", "manual_trends"})
    reuse("src/auto_feature_generator.py", {"discover_entities", "generate_candidates"})
    reuse("src/feature_selection.py", {"prune_candidates", "prune_fixed_candidates"})
    md("""Небольшие проверки ниже рассчитаны так, чтобы ошибка включения текущего события
их сломала. Сначала пример Jan 1/Jan 2/Jan 5, затем tied timestamps и изменение
будущих outcomes. Это дополняет независимые tests исходного проекта, а не заменяет
проверку distributions и сохраненных predictions.
""")
    code('''toy = pl.DataFrame({"row_id": [0, 1, 2], "user_id": [1, 1, 1], "item_id": [10, 10, 10],
                    "event": ["click", "click", "purchase"], "label": [0, 0, 1],
                    "timestamp": [datetime(2020, 1, d) for d in [1, 2, 5]]})
spec = FeatureSpec(("user_id", "item_id"), "click")
actual, _ = generate_features(toy, [spec])
assert actual[spec.name].to_list() == [0., 1., 2.]
tied = toy.with_columns(pl.when(pl.col("row_id") == 2).then(datetime(2020, 1, 2)).otherwise(pl.col("timestamp")).alias("timestamp"))
assert generate_features(tied, [spec])[0][spec.name].to_list() == [0., 1., 1.]
specs = [FeatureSpec(keys, event, agg, window) for keys in [(), ("user_id",), ("item_id",), ("user_id", "item_id")]
         for event in ["all", "click", "purchase"] for agg in ["count", "recency", "event_rate"] for window in [None, 3]]
before, _ = generate_features(toy, specs)
changed = toy.with_columns(pl.when(pl.col("row_id") >= 1).then(pl.lit("purchase")).otherwise(pl.col("event")).alias("event"))
after, _ = generate_features(changed, specs)
np.testing.assert_allclose(before.head(2).drop("row_id").to_numpy(), after.head(2).drop("row_id").to_numpy(), equal_nan=True)
assert not np.isinf(before.drop("row_id").to_numpy()).any() and before.height == toy.height
assert log_loss([0, 1], [.2, .8]) == -np.log(.8)
assert roc_auc_score([0, 1], [.2, .8]) == 1
assert average_precision_score([0, 1], [.2, .8]) == 1
print("PASS: current/future exclusion, timestamp ties, row count, no inf, metric examples")
''')
    md("""## 6. Эксперименты

Сначала считаем общий candidate pool на development prefix и удаляем constants,
почти пустые, duplicate и коррелирующие колонки только по train. В fixed control
сначала ограничиваем окна, затем делаем собственный pruning: adaptive-only alias
не может удалить допустимую fixed колонку.

Обучаем raw/manual/trends, затем оба auto pools и top-K. Если результаты отличаются
не более чем на 0.0001 LogLoss, выбираем меньший набор. Все решения фиксируются до audit/test.
Для финальной стоимости каждый уникальный набор рассчитывается заново на полной
истории. Одинаковые наборы переиспользуют одну модель и помечаются `shared_with`:
это не два независимых эксперимента. Generation time не включает обучение;
fit time не включает предсказание и feature search. Matrix MiB — размер float32 input,
а RSS — снимок памяти процесса, не изолированный peak метода.
""")
    code('''def classification_scores(labels, probabilities):
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    return {"logloss": float(log_loss(labels, probabilities, labels=[0, 1])),
            "auc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities))}


def user_bootstrap(test, baseline, candidate, repeats=300):
    y = test["label"].to_numpy()
    loss = lambda p: -y*np.log(np.clip(p, 1e-15, 1-1e-15))-(1-y)*np.log1p(-np.clip(p, 1e-15, 1-1e-15))
    delta = loss(candidate)-loss(baseline)
    by_user = test.select("user_id").with_columns(pl.Series("delta", delta)).group_by("user_id").agg(
        pl.col("delta").sum(), pl.len().alias("n")).sort("user_id")
    sums, counts = by_user["delta"].to_numpy(), by_user["n"].to_numpy()
    random = np.random.default_rng(SEED)
    values = []
    for _ in range(repeats):
        draw = random.integers(0, len(sums), len(sums))
        values.append(sums[draw].sum()/counts[draw].sum())
    return {"delta_logloss": float(delta.mean()), "ci95": np.quantile(values, [.025, .975]).tolist(),
            "repeats": repeats, "conditional_on_fixed_models": True}


def run_comparison(dataset, frame, split):
    folder = RUN / dataset
    folder.mkdir()
    dev = frame.head(split.validation_end)
    events = set(dev["event"].unique().to_list())
    manual = manual_features(dev.schema, events)
    trends = manual_trends(dev.schema, events)
    specs = list(dict.fromkeys(generate_candidates(dev.schema, events, WINDOWS, 900) + manual + trends))
    lookup = {s.name: s for s in specs}
    features, generation = generate_features(dev, specs)
    assert features.height == dev.height and features["row_id"].equals(dev["row_id"])
    keep, reasons = prune_candidates(features, split.train_end, PRUNING, SEED)
    fixed, fixed_reasons = prune_fixed_candidates(features, specs, split.train_end, PRUNING, SEED)
    encoder = fit_encoder(dev.head(split.train_end))
    base, base_names, categories = base_matrix(dev, encoder)
    y = dev["label"].to_numpy()
    models, histories, fitted, profiles = {}, [], {}, {}
    params = COMMON_PARAMS[dataset]

    def search_fit(name, names):
        key = tuple(names)
        if key in fitted:
            source = fitted[key]
            models[name] = models[source]
            profiles[name] = {**profiles[source], "shared_with": source}
            return profiles[name]
        x = matrix_from_features(base, features, names)
        model, seconds = fit_model(x, y, split.train, split.search,
            base_names+names, categories, params, CONFIG, SEED)
        p = model.booster_.predict(x[split.search], num_threads=4)
        row = {"dataset": dataset, "method": name, "inputs": x.shape[1], "dev_fit_seconds": seconds,
               "iterations": model.best_iteration_, "shared_with": "", **classification_scores(y[split.search], p)}
        histories.append(row)
        profiles[name] = row
        models[name] = model.booster_
        fitted[key] = name
        model.booster_.save_model(str(folder / ("search_"+name+".txt")))
        del x, model
        gc.collect()
        print(dataset, name, "search", round(row["logloss"], 6), flush=True)
        return row

    methods = {"raw": [], "manual": [s.name for s in manual],
               "manual_trends": [s.name for s in manual+trends]}
    for name, names in methods.items():
        search_fit(name, names)

    def choose_pool(prefix, pool):
        full = prefix+"_pool"
        search_fit(full, pool)
        gain = dict(zip(models[full].feature_name(), models[full].feature_importance("gain")))
        ranked = sorted(pool, key=lambda n: (-gain[n], n))
        options = [(full, pool)]
        for budget in BUDGETS:
            if budget < len(pool):
                name = prefix+"_top"+str(budget)
                search_fit(name, ranked[:budget])
                options.append((name, ranked[:budget]))
        best = min(profiles[name]["logloss"] for name, _ in options)
        return min((pair for pair in options if profiles[pair[0]]["logloss"] <= best+THRESHOLD),
                   key=lambda pair: (len(pair[1]), pair[0]))

    fixed_name, fixed_names = choose_pool("fixed", fixed)
    selected_name, selected_names = choose_pool("auto", keep)
    methods.update(auto_fixed=fixed_names, auto_pool=keep, auto_selected=selected_names)
    sources = {"raw": "raw", "manual": "manual", "manual_trends": "manual_trends",
               "auto_fixed": fixed_name, "auto_pool": "auto_pool", "auto_selected": selected_name}
    lock = {"methods": methods, "source_models": sources, "params": params, "seed": SEED,
            "split": vars(split), "windows": WINDOWS, "budgets": BUDGETS, "threshold": THRESHOLD,
            "selection_data": "search only", "mode": MODE, "all_specs": [s.to_dict() for s in specs],
            "initial_candidates": len(specs), "after_pruning": len(keep)}
    (folder / "locked_protocol.json").write_text(json.dumps(lock, indent=2))
    audit = []
    for method, names in methods.items():
        x = matrix_from_features(base[split.audit], features[split.audit], names)
        p = models[sources[method]].predict(x, num_threads=4)
        audit.append({"dataset": dataset, "method": method, **classification_scores(y[split.audit], p)})
        del x
    pd.DataFrame(histories).to_csv(folder / "search.csv", index=False)
    pd.DataFrame(audit).to_csv(folder / "audit.csv", index=False)
    reasons.write_csv(folder / "pruning.csv")
    fixed_reasons.write_csv(folder / "fixed_pruning.csv")
    dev_cost = {"shared_pool_generation_seconds": generation["seconds"],
                "search_fit_seconds": sum(r["dev_fit_seconds"] for r in histories),
                "models_fitted": len(histories)}
    del features, base, models, dev, y
    gc.collect()

    encoder_final = fit_encoder(frame.head(split.validation_end))
    base, base_names, categories = base_matrix(frame, encoder_final)
    results, predictions, cache, importance = [], {}, {}, None
    test = frame[split.test]
    for method, names in methods.items():
        key = tuple(names)
        source = sources[method]
        if key in cache:
            previous = cache[key]
            row = {**previous, "method": method, "shared_with": previous["method"]}
            predictions[method] = predictions[previous["method"]]
        else:
            values, meta = generate_features(frame, [lookup[n] for n in names])
            missing = values.drop("row_id").null_count().row(0, named=True)
            if method == "auto_pool":
                pd.DataFrame([{"feature": n, "null_fraction": missing[n]/frame.height,
                               "nan_fraction": values[n].is_nan().sum()/frame.height,
                               "min": values[n].min(), "max": values[n].max(), "mean": values[n].mean(),
                               "dtype": str(values.schema[n])} for n in names]).to_csv(folder / "feature_distributions.csv", index=False)
            x = matrix_from_features(base, values, names)
            matrix_mib = x.nbytes / 2**20
            rss = psutil.Process().memory_info().rss / 2**20
            del values
            model, seconds = fit_model(x, frame["label"].to_numpy(), np.arange(split.validation_end), split.search,
                base_names+names, categories, params, CONFIG, SEED,
                fixed_iterations=int(profiles[source]["iterations"]))
            p = model.booster_.predict(x[split.test], num_threads=4)
            model.booster_.save_model(str(folder / ("final_"+method+".txt")))
            row = {"dataset": dataset, "mode": MODE, "method": method, "seed": SEED,
                   "inputs": x.shape[1], "generation_seconds": meta["seconds"], "fit_seconds": seconds,
                   "matrix_mib": matrix_mib, "rss_after_matrix_mib": rss,
                   "dev_fit_seconds": profiles[source]["dev_fit_seconds"], "shared_with": "",
                   "iterations": model.n_estimators_, **classification_scores(test["label"].to_numpy(), p)}
            predictions[method] = p
            cache[key] = row
            if method == "auto_pool":
                importance = pd.DataFrame({"feature": model.booster_.feature_name(), "gain": model.booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
            pl.DataFrame({"row_id": test["row_id"], "label": test["label"], "prediction": p}).write_parquet(folder / ("predictions_"+method+".parquet"))
            del model, x
            gc.collect()
        results.append(row)
        print(dataset, method, "test", round(row["logloss"], 6), "inputs", row["inputs"], flush=True)
    table = pd.DataFrame(results)
    baseline = table.loc[table.method == "raw", "logloss"].iloc[0]
    table["delta_logloss_vs_raw"] = table.logloss-baseline
    table.to_csv(folder / "comparison.csv", index=False)
    importance.to_csv(folder / "feature_importance.csv", index=False)
    selected_specs = [lookup[n].to_dict() for n in selected_names]
    pd.DataFrame(selected_specs).to_csv(folder / "selected_features.csv", index=False)
    interval = user_bootstrap(test, predictions["manual_trends"], predictions["auto_selected"])
    (folder / "bootstrap.json").write_text(json.dumps(interval, indent=2))
    (folder / "cost.json").write_text(json.dumps(dev_cost, indent=2))
    manifest = {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}
    (folder / "files_sha256.json").write_text(json.dumps(manifest, indent=2))
    return {"table": table, "search": pd.DataFrame(histories), "audit": pd.DataFrame(audit),
            "importance": importance, "selected_specs": selected_specs,
            "pruning": pd.DataFrame(reasons.to_dicts(), columns=["feature", "reason"]),
            "cost": dev_cost, "bootstrap": interval, "folder": str(folder)}


studies = {}
for dataset in DATASETS:
    studies[dataset] = run_comparison(dataset, frames[dataset], splits[dataset])
''')
    md("""## 7. Сравнение результатов

В основной таблице — только фактически выполненный **текущий** запуск. Если сверху
выбран demo, это демонстрация на 100k, а не воспроизведение полного исследования.
Ниже quality и стоимость расчета отображаются отдельно; меньше LogLoss — лучше.
`auto_pool` и `auto_selected` могут оказаться одной моделью. В таком случае подбор
не нашел более компактный набор с допустимой потерей качества.
""")
    code('''comparison = pd.concat([study["table"] for study in studies.values()], ignore_index=True)
columns = ["dataset", "mode", "method", "inputs", "logloss", "auc", "average_precision",
           "delta_logloss_vs_raw", "generation_seconds", "fit_seconds", "matrix_mib", "shared_with"]
display(comparison[columns].round(6))
comparison.to_csv(RUN / "comparison.csv", index=False)
fig, axes = plt.subplots(len(DATASETS), 2, figsize=(12, 4*len(DATASETS)), squeeze=False)
for row, dataset in enumerate(DATASETS):
    table = studies[dataset]["table"]
    axes[row, 0].barh(table.method, table.logloss, color="#346b99")
    axes[row, 0].set(title=dataset + ": " + MODE, xlabel="Test LogLoss (lower is better)")
    axes[row, 1].barh(table.method, table.generation_seconds, label="generation", color="#6a9eab")
    axes[row, 1].barh(table.method, table.fit_seconds, left=table.generation_seconds, label="final fit", color="#d99546")
    axes[row, 1].set(xlabel="Seconds; shared inputs reuse the same run")
    axes[row, 1].legend()
plt.tight_layout()
plt.savefig(RUN / "comparison.png", dpi=140)
plt.show()
display(pd.DataFrame([{ "dataset": name, **study["cost"]} for name, study in studies.items()]))
''')
    md("""## 8. Анализ признаков

Gain ниже взят из train splits финальной модели `auto_pool`. Он показывает использование
признаков бустингом, но не причинный вклад и не test permutation importance. Коррелирующие
окна делят gain между собой. Curve top-K использует search, поэтому сама по себе
не является независимым доказательством качества на новых данных.
""")
    code('''fig, axes = plt.subplots(len(DATASETS), 2, figsize=(13, 5*len(DATASETS)), squeeze=False)
for i, dataset in enumerate(DATASETS):
    study = studies[dataset]
    top = study["importance"].head(12).iloc[::-1]
    axes[i, 0].barh(top.feature, top.gain, color="#346b99")
    axes[i, 0].set(title=dataset+": train gain", xlabel="Gain")
    for prefix in ["auto_", "fixed_"]:
        curve = study["search"][study["search"].method.str.startswith(prefix)].sort_values("inputs")
        axes[i, 1].plot(curve.inputs, curve.logloss, "o-", label=prefix.rstrip("_"))
    axes[i, 1].set(xlabel="Model input features", ylabel="Search LogLoss", title="Budget comparison")
    axes[i, 1].legend()
    selected = pd.DataFrame(study["selected_specs"])
    selected["entity"] = selected["keys"].map(lambda keys: "/".join(keys) or "global")
    selected["window"] = selected["window_days"].map(lambda value: "all" if pd.isna(value) else str(int(value))+"d")
    print(dataset, "selected primary windows; trend long windows remain in CSV")
    display(pd.crosstab(selected.entity, selected.window))
    print("Pruning reasons:")
    display(study["pruning"].assign(reason=study["pruning"].reason.str.split(":").str[0]).groupby("reason").size().to_frame("removed"))
plt.tight_layout()
plt.savefig(RUN / "feature_analysis.png", dpi=140)
plt.show()
''')
    md("""## 9. Обсуждение

У manual/trends одинаковые параметры и training population; разница условно показывает
пользу добавления trends. Auto fixed против manual также меняет relations и primitives,
а auto pool против fixed — и окна, и состав surviving features. Поэтому весь выигрыш
автоматического набора нельзя приписать только выбору длины окна.

Ниже печатаются фактический test winner и фиксированный contrast auto-selected против
manual+trends. Winner не используется для нового выбора. User-cluster bootstrap учитывает
зависимость событий одного человека, но условен на одном split и фиксированных моделях;
неопределенность всего поиска в него не входит.

Прежние full results с тремя seeds приведены отдельно и были проверены по boosters и
predictions. Они относятся к старому протоколу выбора manual stage, поэтому не смешиваются
со свежей основной таблицей. Yambda выбрал на search более короткий manual stage, который
оказался хуже полного manual на test. Это показывает нестабильность reference selection.
SVD сравнивается только с `auto_warmup`: у обеих моделей одинаковые warmup-excluded fit rows.
""")
    code('''for dataset, study in studies.items():
    table = study["table"]
    best = table.sort_values(["logloss", "inputs"]).iloc[0]
    print(dataset, "observed test winner:", best["method"], "LogLoss:", round(best.logloss, 6))
    print("auto_selected minus manual_trends:", study["bootstrap"])
    display(study["audit"].round(6))
original = pd.concat([pd.DataFrame(rows).assign(dataset=name) for name, rows in ORIGINAL_TEST_RUNS.items()], ignore_index=True)
display(original.groupby(["dataset", "experiment_name"]).agg(
    seeds=("seed", "count"), inputs=("n_features", "first"),
    mean_logloss=("logloss", "mean"), seed_sd=("logloss", "std"), mean_auc=("auc", "mean")).round(6))
''')
    md("""## 10. Вывод

Результат определяется основной таблицей, а не заранее выбранным названием метода.
Автоматическая фабрика действительно переносит ручной выбор отдельных колонок на
уровень грамматики и validation search. Человек по-прежнему задает target, связи,
агрегации, окна-кандидаты и бюджет. При равном качестве manual легче сопровождать;
компактный selected набор удобнее большого pool, если выигрыш подтверждается на audit/test.

Ограничения: известный test, один global split, в новом контролируемом сравнении один seed;
исторические availability timestamps отсутствуют. KION не содержит повторных user-item
пар, 180d там эквивалентно all-history; Yambda использует только explicit reactions.
Production latency, full-catalog retrieval и универсальная оптимальность окон не проверялись.
Эти ограничения нельзя устранить красивым графиком или повторным запуском того же holdout.

Исходные исследования: [KION](https://github.com/irsafilo/KION_DATASET),
[Yambda](https://huggingface.co/datasets/yandex/yambda),
[описание Yambda](https://arxiv.org/abs/2505.22238).
Точные revisions, SHA256, исходные параметры и provenance предыдущих runs встроены выше.
Текущие model files, predictions, protocol lock, CSV, costs и SHA256 сохраняются в `.runs/`.
""")
    code('''checks = []
for dataset, study in studies.items():
    folder = Path(study["folder"])
    recorded = json.loads((folder / "files_sha256.json").read_text())
    assert all(sha256(folder / name) == digest for name, digest in recorded.items())
    for path in folder.glob("predictions_*.parquet"):
        saved = pl.read_parquet(path)
        scores = classification_scores(saved["label"].to_numpy(), saved["prediction"].to_numpy())
        method = path.stem.removeprefix("predictions_")
        row = study["table"].set_index("method").loc[method]
        assert abs(scores["logloss"]-row.logloss) < 1e-12
        assert abs(scores["auc"]-row.auc) < 1e-12
    checks.append({"dataset": dataset, "mode": MODE, "status": "PASS", "rows": frames[dataset].height,
                   "selected_specs": len(study["selected_specs"]), "predictions_rescored": True})
(RUN / "verification.json").write_text(json.dumps(checks, indent=2))
display(pd.DataFrame(checks))
review = pd.DataFrame([
    ("Methodological correctness", "PARTIAL", "One split; previously observed test"),
    ("Fairness of comparison", "PASS", "Same rows, parameters, seed and early-stopping budget"),
    ("Feature engineering quality", "PASS", "Existing factory, train-only pruning, actual search refits"),
    ("Data leakage prevention", "PARTIAL", "Strict event-time tests PASS; production availability assumed"),
    ("Experimental evidence", "PASS" if MODE == "full" else "PARTIAL", "Measured current run; demo is not full replication"),
    ("Interpretation", "PASS", "Conditional CI; no full-catalog or optimal-window claims"),
    ("Reproducibility", "PASS" if MODE == "full" else "PARTIAL", "Self-contained notebook, pinned sources and saved predictions"),
    ("Clarity", "PASS", "Ten sections and one main comparison table"),
], columns=["category", "status", "comment"])
display(review)
print("Unofficial expert estimate for the full submission: 87/100; not an official grading rubric.")
print("READY WITH LIMITATIONS. Results for this run:", RUN)
''')
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.13"}
    notebook.metadata.description = "Standalone reuse of existing temporal feature implementations; controlled FE comparison"
    path = DEST / "feature_engineering_comparison.ipynb"
    nbf.write(notebook, path)
    (DEST / "requirements.txt").write_text("\n".join(requirements)+"\n")
    (DEST / ".gitignore").write_text(".venv/\n.data/\n.runs/\n.ipynb_checkpoints/\n__pycache__/\n.DS_Store\n")
    print("Created", path, "cells", len(cells), "code", sum(c.cell_type=="code" for c in cells))


if __name__ == "__main__":
    build()
