"""Build a readable, standalone notebook with actual implementation cells."""
import ast
from pathlib import Path
import pprint

import nbformat as nbf
import yaml


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "00_full_research.ipynb"


def implementation(path, names=None, strip_decorators=False):
    source = (ROOT / path).read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    pieces = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            continue
        if names is not None and getattr(node, "name", None) not in names:
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            first = node.lineno
            if not strip_decorators and node.decorator_list:
                first = min(first, *(decorator.lineno for decorator in node.decorator_list))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            first = node.lineno
        else:
            continue
        pieces.append("\n".join(lines[first - 1:node.end_lineno]))
    return "\n\n\n".join(pieces)


def build():
    cells = []
    def md(content):
        cells.append(nbf.v4.new_markdown_cell(content.strip()))
    def code(content, **metadata):
        cells.append(nbf.v4.new_code_cell(content.strip(), metadata=metadata))
    def section(title, explanation, path, names=None):
        md(f"## {title}\n\n{explanation}")
        code(implementation(path, names), implementation_source=path)

    md("""# Поиск временных признаков для рекомендательной модели

Один и тот же счетчик можно посчитать за день, неделю или месяц. Например, число
реакций на исполнителя за последние три дня говорит о недавнем интересе, а за
полгода — о более устойчивом предпочтении. Обычно такие окна выбирает человек.
Здесь проверяем, можно ли передать этот выбор автоматическому поиску и получить
результат не хуже разумно настроенного ручного baseline.

Сначала разберем данные и построим LightGBM с ручными признаками. Затем зададим
правила генерации кандидатов: по кому группировать события, что считать и за какой
период. Система рассчитает эти варианты и сравнит наборы на будущей по времени
validation. Модель во всех основных сравнениях остается бустингом: так проще понять,
что изменилось именно из-за признаков.

Используем два источника. В KION предсказываем, достигла ли доля просмотра 50%,
среди уже наблюдаемых пар пользователь–фильм. В Yambda предсказываем like или dislike
среди явных реакций. Эти данные позволяют проверить исторические признаки, но
не дают оценить выдачу рекомендаций по всему каталогу. Ограничения разберем ниже.

Вся вычислительная часть находится в ячейках этого файла. Папки `src` и `scripts`
для его выполнения не нужны. На диск сохраняются исходные данные, рассчитанные
матрицы и результаты, чтобы повторный запуск не начинал все сначала.

Перед запуском выберите один из режимов:

- `saved` — прочитать результаты уже выполненного полного исследования и построить графики;
- `smoke` — проверить весь код на первых 100 тысячах событий с коротким обучением;
- `full` — обучить модели на полном выбранном наборе, провести по 20 HPO trials
  для manual и auto и повторить основные модели с тремя seeds.

При наличии результатов по умолчанию включается `saved`, иначе — `smoke`.
Для запуска в отдельной папке нужны этот `.ipynb` и установленные библиотеки.
Недостающие данные скачиваются из официальных источников. Новые расчеты сохраняются
в собственный каталог; его имя задается через `RUN_NAME`.

В проекте уже прошел независимый аудит: после него исправлены выбор ручного baseline
и несколько ошибок воспроизводимости. Тот же test был виден до исправлений,
поэтому текущие цифры — повторная оценка на известном holdout. Выбор признаков и
параметров по-прежнему делается только на search validation; нового слепого
подтверждения эти результаты не заменяют. MovieLens сохранен как архивный запуск.
""")
    dependencies = (ROOT / "requirements-notebooks.txt").read_text().splitlines()[1:]
    requirements = (ROOT / "requirements.txt").read_text().splitlines() + dependencies
    md("""## 1. Библиотеки и окружение

Расчеты проверены на Python 3.13. Polars нужен для исторических агрегаций, LightGBM —
для обучения, Optuna — для подбора параметров. Остальные библиотеки используются
для метрик, SVD, тестов и графиков. Версии зафиксированы, чтобы изменение библиотеки
не стало незаметной причиной другого результата.

Если пакеты еще не установлены, раскомментируйте `%pip install` в следующей ячейке.
Лимит в четыре потока уменьшает конкуренцию за память на обычном ноутбуке.
""")
    code('from importlib import metadata\n\n# %pip устанавливает пакеты в окружение текущего kernel.\nREQUIRED_PACKAGES = [\'numpy==2.3.2\', \'polars==1.43.2\', \'pandas==2.3.1\', \'lightgbm==4.6.0\', \'optuna==4.9.0\', \'scikit-learn==1.7.2\', \'scipy==1.16.3\', \'matplotlib==3.10.9\', \'psutil==7.0.0\', \'PyYAML==6.0.2\', \'pytest==9.0.2\', \'nbformat==5.10.4\', \'nbclient==0.10.2\', \'ipykernel==6.30.0\']\ninstall = []\nfor requirement in REQUIRED_PACKAGES:\n    package, version = requirement.split("==")\n    try:\n        actual = metadata.version(package)\n    except metadata.PackageNotFoundError:\n        actual = None\n    if actual != version:\n        install.append(requirement)\nif install:\n    get_ipython().run_line_magic("pip", "install --disable-pip-version-check " + " ".join(install))\nassert all(metadata.version(r.split("==")[0]) == r.split("==")[1] for r in REQUIRED_PACKAGES)\nprint("Проверенные зависимости доступны")')
    code('''import os
import ast
import sys
import gc
import io
import json
import re
import hashlib
import inspect
import textwrap
import pickle
import threading
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/yandex-notebook-mpl")

import numpy as np
import polars as pl
import lightgbm as lgb
import optuna
import psutil
import pytest
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

print("Python:", sys.version.split()[0])
print("Versions:", {m.__name__: m.__version__ for m in [np, pl, lgb, optuna]})''')
    config = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
    auto = yaml.safe_load((ROOT / "configs/auto_features.yaml").read_text())
    md("""## 2. Настройки запуска

Выберите `DATASET`: `kion` или `yambda`. Режим `MODE` определяет, будем ли мы читать
готовые результаты или заново обучать модели. Для отдельного эксперимента измените
`RUN_NAME`, чтобы его файлы было легко отличить от предыдущих.

`CONFIG` задает бюджет обучения, seeds и границы выборок. `AUTO_CONFIG` описывает
пространство признаков: возможные окна, размеры проверяемых наборов и порог KEEP/DROP.
Эти настройки задает исследователь. Автоматизация начинается внутри этого пространства,
а не с самостоятельного выбора target или смысла событий.

В `smoke` бюджет намеренно уменьшен. Его метрики нужны для проверки выполнения,
а не для выводов о качестве метода. Переменные `TEMPORAL_NOTEBOOK_*` позволяют
выбирать те же настройки при автоматическом запуске.
""")
    code('''WORK_DIR = Path.cwd()
if WORK_DIR.name == "notebooks":
    WORK_DIR = WORK_DIR.parent
os.chdir(WORK_DIR)
DATASET = os.environ.get("TEMPORAL_NOTEBOOK_DATASET", "kion")  # kion | yambda | movielens
assert DATASET in {"kion", "yambda", "movielens"}
REFERENCE_ROOT = WORK_DIR / "artifacts" / DATASET
MODE = os.environ.get("TEMPORAL_NOTEBOOK_MODE", "saved" if (REFERENCE_ROOT / "final_results.json").exists() else "smoke")
RUN_NAME = os.environ.get("TEMPORAL_NOTEBOOK_RUN", "notebook_" + MODE)
assert MODE in {"saved", "smoke", "full"}
OUTPUT_ROOT = REFERENCE_ROOT if MODE == "saved" else WORK_DIR / "artifacts" / (DATASET + "_" + RUN_NAME)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RAW_DIR = WORK_DIR / "data" / "raw"
CUSTOM_DATA_PATH = None  # Путь к своей таблице с user_id, item_id, timestamp и бинарным label.
POSITIVE_LABEL = None   # Для многоклассовой таблицы: какое значение считать положительным исходом.
'''
         + "\nCONFIG = " + pprint.pformat(config, sort_dicts=False)
         + "\nAUTO_CONFIG = " + pprint.pformat(auto, sort_dicts=False)
         + '''
if MODE == "smoke":
    CONFIG.update(hyperparameter_trials=2, max_estimators=80, early_stopping_rounds=10,
                  seeds=[42], bootstrap_repeats=20)
    AUTO_CONFIG.update(windows=[1, 7, 30, None], selection_sizes=[16, 32], agent_rounds=1)
if DATASET != "movielens":
    CONFIG.update(dataset_name=DATASET, run_representations=True)
    AUTO_CONFIG["max_features"] = 900
print("Mode:", MODE, "| Output:", OUTPUT_ROOT)
print("HPO trials per model:", CONFIG["hyperparameter_trials"], "| Seeds:", CONFIG["seeds"])''')
    section("3. Загрузка данных", """Скачиваем конкретные версии KION и Yambda, а не содержимое меняющейся ссылки
«на сегодня». Для каждого файла сохраняем размер и SHA256. Это позволяет проверить,
что повторный запуск действительно использует те же данные.

Для Yambda берем likes/dislikes из части flat/50m и таблицы связей с исполнителями
и альбомами. Полный Yambda-5B и аудиоэмбеддинги для этого эксперимента не нужны.
Загрузчик умеет продолжить недокачанный файл и повторить запрос при обрыве сети.""", "src/dataset_sources.py")
    code(implementation("scripts/download_data.py"), implementation_source="scripts/download_data.py")
    section("4. Подготовка общей таблицы событий", """Два источника устроены по-разному, поэтому приводим их к общей схеме:
`user_id`, `item_id`, `timestamp`, `event`, `label` и доступные metadata. `label`
нужен модели как ответ, `event` — движку для отбора прошлых действий нужного типа.
Они не подаются в модель как признаки текущей строки.

Каждое событие получает уникальный `row_id`. По нему потом возвращаем агрегаты
в исходный порядок и сверяем predictions с labels. Проверки пустых ID, времени
и повторных row_id нужны до обучения: хороший score не поможет заметить перепутанные
строки. Функции MovieLens здесь оставлены для чтения архивного эксперимента.""", "src/data.py")
    code(implementation("src/datasets.py"), implementation_source="src/datasets.py")
    md("""### Реальная семантика источников

**KION.** В скачанном архиве 1 594 787 строк за 13 марта — 22 августа 2021 года;
268 строк без доли просмотра исключаются. Эти числа расходятся с описанием репозитория,
поэтому ориентируемся на сам файл. Каждая строка — агрегат по user–item с датой
последнего просмотра. По нему нельзя восстановить полную последовательность сессий.

Положительный исход — `watched_pct >= 50`. Прошлую долю просмотра используем
как `history_value`, текущую — нельзя: из нее напрямую получен ответ. По той же причине
не передаем модели текущий `total_dur`. Время известно до дня, поэтому исключаем из
истории весь текущий день. Доступность прошлых агрегатов после их `last_watch_dt`
принимается как offline допущение; реального времени готовности labels в файле нет.

**Yambda.** Используем 881 456 likes и 107 776 dislikes — всего 989 232 реакции.
Это все такие события из flat/50m, но не все 50 миллионов событий: прослушивания
и отмены реакций в run не входят. Предсказываем знак уже выраженной реакции.
Вероятность like среди всех прослушиваний по этому набору оценить нельзя.

У item бывает несколько артистов или альбомов. Для простой связи выбираем минимальный
ID; это теряет часть информации, зато не размножает строки при join. Дубликаты реакций
сохраняются: при точности 5 секунд нельзя надежно решить, где повторное действие,
а где повтор логирования. Статические metadata в обоих источниках считаем заранее
доступными; их историю изменений данные не дают.

**Единицы времени Yambda.** Формула в статье — `floor((T−T0)/5)*5`.
Значения уже выражены в секундах и округлены вниз до 5 секунд. Умножать их еще раз
на пять нельзя. Дата 1970 года нужна только для работы с elapsed time в Polars;
реальные час и день недели неизвестны, поэтому такие признаки отключены.

Первичные источники: [KION](https://github.com/irsafilo/KION_DATASET),
[Yambda dataset card](https://huggingface.co/datasets/yandex/yambda),
[Yambda timestamp formula](https://arxiv.org/html/2505.22238v1).
""")
    section("5. Разбиение по времени и первый взгляд на данные", """Модель должна предсказывать будущее по прошлому, поэтому случайное разбиение
здесь не подходит. Берем около 70% ранних событий для train, следующие 7.5% —
для выбора признаков и параметров (`search`), еще 7.5% — для проверки уже выбранных
вариантов (`audit`), последние 15% — для final test.

Два validation участка нужны из-за многократного поиска: если постоянно выбирать
лучший вариант по одной выборке, к ней тоже можно переобучиться. Audit дает следующую
по времени проверку и не меняет выбор модели. События с одинаковым timestamp
остаются в одной части, даже если из-за этого проценты слегка сдвигаются.

После загрузки смотрим размер, пропуски, повторные пары и долю новых пользователей.
Это помогает понять, какие истории вообще могут быть полезны. Например, если пара
user–item не повторяется, ее счетчик будет нулевым почти для каждой строки.""", "src/split.py")
    code('''if CUSTOM_DATA_PATH is None and DATASET in {"kion", "yambda"}:
    folder = RAW_DIR / DATASET
    needed = ["interactions.csv.zip", "users.csv", "items.csv"] if DATASET == "kion" else [
        "likes.parquet", "dislikes.parquet", "artist_item_mapping.parquet", "album_item_mapping.parquet"]
    if not all((folder / name).exists() for name in needed):
        fetch_sources(DATASET, RAW_DIR)
    data_df = load_kion(RAW_DIR) if DATASET == "kion" else load_yambda(RAW_DIR)
elif CUSTOM_DATA_PATH is None:
    extracted = RAW_DIR / "ml-1m"
    if not all((extracted / name).exists() for name in ["ratings.dat", "movies.dat", "users.dat"]):
        download()
    data_df = load_movielens(RAW_DIR)
else:
    assert MODE != "saved", "Custom data require an independent run directory"
    data_df = load_interactions(CUSTOM_DATA_PATH, positive_label=POSITIVE_LABEL)
if MODE == "smoke":
    data_df = data_df.head(100000)
split = temporal_split(data_df, tuple(CONFIG["split_fractions"]))
development_df = data_df.head(split.validation_end)
eda = describe_data(development_df)
print("Whole dataset:", data_df.shape)
print("Development prefix:", development_df.shape, "| memory MiB:", round(eda["memory_mb"], 2))
print("Missing values:", eda["nulls"])
print("Duplicate events:", eda["duplicate_events"], "| repeat user-item rows:", eda["repeat_user_item_rows"])
display(pl.DataFrame([{"column": name, "dtype": dtype} for name, dtype in eda["schema"].items()]))
display(development_df.head(10))
if MODE != "saved":
    save_json(OUTPUT_ROOT / "eda.json", eda)
    save_json(OUTPUT_ROOT / "dataset_protocol.json", dataset_protocol(DATASET))''')
    code('''source_manifest_path = RAW_DIR / DATASET / "source_manifest.json"
if MODE != "saved" and source_manifest_path.exists():
    save_json(OUTPUT_ROOT / "source_manifest.json", json.loads(source_manifest_path.read_text()))''')
    code('''split_rows = []
train_users = data_df[split.train]["user_id"].unique().implode()
for name, indices in [("train", split.train), ("validation_search", split.search),
                      ("validation_audit", split.audit), ("test", split.test)]:
    part = data_df[indices]
    split_rows.append({"split": name, "rows": len(indices),
                       "start_time_coordinate": str(part["timestamp"].min()), "end_time_coordinate": str(part["timestamp"].max()),
                       "positive_rate": part["label"].mean() if name != "test" else None,
                       "unknown_user_id_fraction": 1 - part["user_id"].is_in(train_users).mean()})
display(pl.DataFrame(split_rows))
print("For Yambda these are relative coordinates, not calendar dates.")''')
    md("""## 6. Почему выбран поиск признаков поверх бустинга

Можно заменить всю модель на GNN или Transformer, но тогда трудно понять, какая
часть разницы связана с признаками, а какая — с новой моделью и другой задачей.
Для основного эксперимента оставляем LightGBM и меняем способ построения его входов.

Featuretools подсказывает удобную идею: описывать признак через связь и агрегацию.
Добавляем к этому явное временное окно и проверяем наборы на temporal validation.
Ниже сравнение подходов и источники, на которые опирается этот выбор. Это обзор,
а не таблица измеренного качества GNN/Transformers: они здесь не обучались.
SVD реализован отдельно как недорогая проверка другого источника сигнала.
В разделах 7–18 определяем функции; само обучение начнется в разделе 19.
""")
    for part in (ROOT / "RESEARCH.md").read_text().split("\n## "):
        md(part if part.startswith("#") else "### " + part)
    code(implementation("src/provenance.py"), implementation_source="src/provenance.py")
    section("7. Как описывается один признак: FeatureSpec", """Вместо отдельной функции на каждую колонку создаем небольшое описание.
Например, `keys=(user_id, category_id)`, `event=like`, `aggregation=count`,
`window_days=14` означает: посчитать прошлые like этого пользователя в этой категории
за последние 14 дней. `None` вместо окна означает всю доступную историю.

Одно описание используется генератором, движком, поиском и журналом выбранных признаков.
Так проще проследить, откуда взялась колонка модели. Валидация не пропустит неизвестное
поле, отрицательное окно или исполняемый код вместо имени. Следующая ячейка печатает
SQL того же признака для наглядности; сам расчет выполняет Polars.""", "src/feature_spec.py")
    code('''example_spec = FeatureSpec(("user_id", "category_id"), positive_event(set(development_df["event"].unique())), "count", 14)
print(to_sql(example_spec))''')
    section("8. Расчет признаков без будущих событий", """Для строки в момент `T` используем только события с временем меньше `T`.
Окно длины `W` — это `[T-W, T)`: нижняя граница включена, текущий момент исключен.
Например, для второго события 2 января счетчик не видит ни это событие, ни другое
событие с той же датой. Если доступна только дата, порядок внутри дня нам неизвестен.

Оконные агрегаты считаются через Polars rolling с `closed=left`. Для всей истории
сначала объединяем события одинакового времени, потом берем накопленную статистику
со сдвигом. Простого `shift(1)` по исходным строкам было бы недостаточно из-за ties.

Trend рассчитывается из двух прошлых счетчиков: `(short + smoothing) / (long + smoothing)`.
Rates используют те же временные границы для числителя и знаменателя. У первой строки
count равен нулю, а recency и mean могут быть пропущены: прошлых наблюдений еще нет.
Движок также проверяет число строк после join и отсутствие бесконечных значений.""", "src/feature_engine.py")
    section("9. Ручной baseline", """Сначала строим обычный набор, который разумно было бы подготовить вручную:
число прошлых событий, их типы, recency, долю положительных исходов и доступные средние.
Считаем их по пользователю, item, паре user–item и связям пользователя с категорией
и автором. В KION автором считаем первого режиссера, в Yambda — выбранного исполнителя.

Ручные окна фиксированы: 7 дней, 30 дней и вся история. Отношение счетчиков 7d/30d
добавляем отдельным этапом, чтобы увидеть вклад трендов. Функция `ablation_groups`
постепенно добавляет семьи признаков. Позже baseline выбирается по search среди всех
этих выполненных этапов: полный набор не обязан быть самым сильным.""", "src/baseline_features.py")
    section("10. Откуда берутся автоматические кандидаты", """Генератор перебирает допустимые связи, события, агрегации и окна из настроек.
Он не знает, какой вариант окажется лучше: одинаковое правило применяется ко всем
доступным семействам. Связь user–brand, например, появится только при наличии brand.

Число колонок ограничено бюджетом, чтобы поиск не превратился в материализацию
десятков тысяч почти одинаковых признаков. Для трендов используются заданные пары
короткого и длинного окна. Частоту `count / window_days` отдельно не генерируем:
при фиксированном окне это масштабированная копия count для того же дерева.""", "src/auto_feature_generator.py")
    section("11. Удаление бесполезных и повторяющихся колонок", """До дорогого обучения убираем признаки, которые почти всегда одинаковы,
почти полностью пропущены или дублируют другие колонки. Затем отсеиваем почти
полностью коррелированные варианты. Все эти решения принимаются только по train;
validation и test не подсказывают, какие колонки удалить.

Fixed и shared controls проходят этот шаг внутри собственных разрешенных окон.
Иначе all-history мог бы удалиться как копия 180d, а затем фильтр fixed окон удалил бы
и 180d. Контрольная модель потеряла бы доступный сигнал из-за порядка операций.

Такой отсев еще не выбирает лучший набор по качеству. Он лишь сокращает явную
избыточность. Оставшиеся наборы будут переобучены и сравнены по validation LogLoss.""", "src/feature_selection.py")
    section("12. Агент, предлагающий следующие признаки", """Первый список кандидатов ограничен сеткой окон. После обучения можно посмотреть,
какие семьи используются моделью, и попробовать соседние масштабы. Например,
вокруг 30 дней эвристика предложит окна примерно 20 и 42 дня.

Здесь «агент» — детерминированная функция, а не LLM. Она получает текущие specs,
gain importance и историю уже проверенных предложений. На выходе — новый небольшой
список `FeatureSpec`. Повторно проверять отклоненную колонку не нужно.

Дальше предложение проходит обычный цикл расчета и обучения. Оставляем его только
при улучшении search LogLoss не меньше порога. Есть необязательный интерфейс для
внешней LLM, но она может вернуть лишь проверяемые описания признаков; API по умолчанию
не вызывается, произвольный SQL не выполняется.""", "src/feature_agent.py")
    section("13. Дополнительная проверка: SVD-признаки", """Счетчики описывают историю явно. SVD проверяет другой сигнал: можно ли сжать
матрицу пользователь–item в небольшие векторы и использовать близость интересов.
Положительные и отрицательные реакции входят в матрицу с весами +1 и -1.

Пространство item-векторов обучаем только на ранних 10% событий и затем фиксируем.
Для более поздней строки профиль пользователя — средний вектор прошлых items
с учетом знака реакции. В бустинг добавляются dot product, cosine similarity и нормы векторов.
Item, неизвестный раннему basis, получает нулевой вектор.

Строки первых 10% нельзя использовать для обучения этой модели: basis уже знает
их последующие события внутри warmup. Поэтому они исключаются и из SVD-варианта,
и из его manual/auto controls. Такое сравнение отделяет вклад векторов от изменения
числа обучающих наблюдений.""", "src/representations.py")
    section("14. Как измеряется качество", """Основной критерий — LogLoss: он оценивает предсказанную вероятность и сильнее
штрафует уверенные ошибки. AUC показывает, насколько хорошо отделяются положительные
и отрицательные исходы, но сам по себе не оценивает калибровку вероятностей.

Для NDCG, Recall и MAP собираем группы одного пользователя с одинаковым timestamp,
минимум двумя разными items и обоими классами. Сравниваем порядок внутри такой группы.
Это вспомогательная проверка среди наблюдавшихся событий. Здесь нет candidate generation
по каталогу, а высокий Recall@10 может объясняться короткими группами.

Разницу LogLoss оцениваем парным bootstrap по пользователям. События одного человека
могут быть зависимы, поэтому отдельные строки не пересэмплируем независимо. Модели
в bootstrap фиксированы: интервал не включает неопределенность всего поиска признаков.""", "src/metrics.py")
    section("15. Обучение LightGBM и настройка параметров", """IDs и metadata передаются LightGBM как категориальные поля. Словарь значений
строится на обучающем префиксе; неизвестная категория кодируется как -1.
При настройке число деревьев выбирается early stopping по search LogLoss.

Optuna проверяет число листьев, минимальный размер листа, learning rate, долю колонок
и регуляризацию. Для скорости fit каждого trial ограничен 300 тысячами train-строк.
При этом признаки уже рассчитаны по полной предшествующей истории: выборка строк
не должна обеднять их прошлое. Лучшие параметры затем проверяются на полном train.

У manual и auto по 20 trials, хотя общий расход auto больше из-за перебора признаков.
SQLite хранит историю trials, отдельный checkpoint — состояние TPE sampler.
Одного seed недостаточно для точного продолжения прерванного поиска.""", "src/model.py")
    section("16. Что сохраняется после каждого обучения", """Каждая строка `experiments.csv` связана с конкретной моделью: у нее есть имя,
seed, список входов, число деревьев, метрики, время fit и измерение памяти процесса.
Рядом сохраняются booster и описание его признаков. Это позволяет проверить таблицу
результатов повторным inference, а не полагаться на вручную выписанные числа.

Большие матрицы удерживаются в памяти, поэтому RSS отражает весь процесс, а не только
модель. Нулевое время генерации в отдельной строке часто означает reuse ранее
рассчитанного блока. Это не делает создание признаков бесплатным.""", "src/experiments.py")
    md("""## 17. Проверки до большого обучения

На маленькой таблице можно вручную отобрать события раньше `T` и сравнить ответ
с движком. Такой медленный расчет используем как эталон в тестах всех агрегаций.
Здесь цикл по строкам уместен: важнее прозрачность проверки, чем скорость на семи строках.
Сам движок работает векторизованно.

Отдельно меняем текущие и будущие исходы и проверяем, что прошлые признаки не меняются.
Есть случаи с одинаковыми timestamps, пустой историей, неизвестными категориями,
ranking-метриками и warmup SVD. Тесты находятся прямо в notebook и запускаются ниже;
проектные test-модули импортировать не требуется.

Эти проверки ловят ошибки вычислений. Они не могут доказать, что исходный агрегат
действительно был доступен в production в указанную дату: для этого нужен `available_at`.
""")
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        code(implementation(str(path.relative_to(ROOT)), strip_decorators=True), test_source=path.name)
    code('''checked_cases = 0
test_fixture = events()
for test_keys in [(), ("user_id",), ("item_id",), ("user_id", "item_id")]:
    for test_event in ["all", "positive", "negative"]:
        for test_window in [None, 1, 3]:
            test_all_aggregations_against_slow_oracle(test_fixture, test_keys, test_event, test_window)
            checked_cases += 1
test_functions = [(name, function) for name, function in list(globals().items())
                  if name.startswith("test_") and inspect.isfunction(function)
                  and name != "test_all_aggregations_against_slow_oracle"]
for name, function in sorted(test_functions):
    if "events" in inspect.signature(function).parameters:
        function(test_fixture)
    else:
        function()
    checked_cases += 1
assert checked_cases >= 54
print(f"PASS: {checked_cases} checks; strict cutoff, ties, mutations, pruning, encoding, ranking, SVD and safe proposer")''')
    section("18. Как не пересчитывать одинаковые признаки", """В разных экспериментах часто нужны одни и те же колонки. Сохраняем рассчитанный
блок в parquet и используем повторно. Ключ cache зависит от данных, списка specs,
версии движка и структуры его кода. Если расчеты изменились, старую матрицу нельзя
выдать за новую только потому, что имена колонок совпали.

AST-подпись игнорирует комментарии и форматирование: перенос функций в notebook
не делает их другой реализацией. Подписи покрывают ключевые расчеты; прочие изменения
протокола по-прежнему требуют обновления версии и отдельного output.""", "scripts/run_study.py", {"cached_features", "psutil_rss", "selection_signature"})
    section("19. Запуск поиска и фиксация модели", """Здесь собираются определенные выше функции. Сначала настраиваем manual модель
и обучаем ее ablation stages. Затем сравниваем автоматические наборы: только
7d/30d/all окна, полную сетку окон и несколько бюджетов признаков. Gain предлагает
порядок колонок, а search LogLoss решает, какой набор оставить.

Shared control проверяет вариант с общими окнами для всех семейств. После выбора
структуры проводим auto HPO и две итерации предложений агента. Затем сохраняем
`locked_selection.json`: признаки и параметры больше не меняются по audit/test.
Для основных выбранных моделей повторяем обучение с seeds 42, 2024 и 7.

В `saved` эта ячейка только проверяет соответствие данных, настроек и реализации
сохраненному run. В `smoke` и `full` запускается реальное обучение. На этапе поиска
test scores не читаются.""", "scripts/run_study.py", {"develop"})
    code('''if MODE == "saved":
    manifest = json.loads((OUTPUT_ROOT / "run_manifest.json").read_text())
    assert manifest["dataset_fingerprint"] == dataset_fingerprint(data_df), "Saved results belong to different data"
    assert manifest["config"] == CONFIG and manifest["auto_config"] == AUTO_CONFIG, "Use a new run directory for a changed configuration"
    assert manifest["protocol_version"] == STUDY_PROTOCOL_VERSION
    assert manifest["selection_signature"] == selection_signature()
    assert manifest["engine_signature"] == feature_engine_signature()
    lock = json.loads((OUTPUT_ROOT / "locked_selection.json").read_text())
    print("Saved full study: model fitting is skipped; implementation above is executable in smoke/full mode.")
elif (OUTPUT_ROOT / "final_results.json").exists():
    manifest = json.loads((OUTPUT_ROOT / "run_manifest.json").read_text())
    assert manifest["dataset_fingerprint"] == dataset_fingerprint(data_df)
    assert manifest["config"] == CONFIG and manifest["auto_config"] == AUTO_CONFIG
    lock = json.loads((OUTPUT_ROOT / "locked_selection.json").read_text())
    print("Independent run is already complete; reuse its locked results.")
else:
    lock = develop(data_df, split, CONFIG, AUTO_CONFIG, OUTPUT_ROOT)
assert lock["locked_before_audit_and_test"]
print("Locked manual:", lock["manual_name"])
print("Locked automatic:", lock["auto_name"], "| historical specs:", len(lock["auto_features"]))''')
    audit_source = implementation("scripts/audit_controls.py", {"audit_window_controls"})
    audit_source = audit_source.replace('(pl.col("seed") == 42)', '(pl.col("seed") == manifest["config"]["seed"])')
    md("""## 20. Переносятся ли выбранные окна на следующий период

Лучший search score может быть удачным совпадением. Поэтому выбранные fixed, adaptive
и shared controls оцениваем на следующем участке времени — audit. Источник каждой
контрольной модели уже выбран по search; после чтения audit этот выбор не меняем.

Fixed control тоже автоматически выбирает колонки, но только среди привычных окон.
Сравнение с ним помогает отделить общую пользу фабрики признаков от пользы более
широкой сетки. Различия состава и числа колонок остаются, поэтому это не чистый
причинный эксперимент над одним окном. Controls обучены с одним seed.
""")
    code(audit_source, implementation_source="scripts/audit_controls.py")
    code('''if MODE != "saved" and not (OUTPUT_ROOT / "audit_controls.csv").exists():
    audit_window_controls(data_df, split, OUTPUT_ROOT)
if (OUTPUT_ROOT / "audit_controls.csv").exists():
    display(pl.read_csv(OUTPUT_ROOT / "audit_controls.csv").select("experiment_name", "n_features", "logloss", "auc"))''')
    section("21. Финальная оценка", """Признаки, параметры и число деревьев уже выбраны. Теперь переобучаем основные
модели на первых 85% событий и предсказываем последние 15%. Early stopping по test
не делаем: число деревьев берем из development обучения.

Признаки оцениваются в режиме online replay: более ранняя реакция внутри test может
стать историей для более позднего события. Это соответствует последовательной обработке
при условии, что исход уже доступен. Для delayed labels понадобилась бы дополнительная
граница `available_at`, которой в этих файлах нет.

Сохраняем probabilities и booster для каждого seed, чтобы финальные метрики можно
было проверить независимо. Повторная оценка после аудита остается оценкой на ранее
известном test, даже при корректно закрытом выборе внутри этого запуска.""", "scripts/run_study.py", {"finalize"})
    code('''if MODE != "saved":
    finalize(data_df, split, CONFIG, OUTPUT_ROOT)
final_results = json.loads((OUTPUT_ROOT / "final_results.json").read_text())
assert final_results["test_used_for_selection"] is False
experiments = pl.read_csv(OUTPUT_ROOT / "experiments.csv")
if (OUTPUT_ROOT / "audit_controls.csv").exists():
    experiments = pl.concat([experiments.filter(pl.col("split") != "validation_audit_control"),
                             pl.read_csv(OUTPUT_ROOT / "audit_controls.csv")], how="diagonal_relaxed")
    if MODE != "saved":
        experiments.write_csv(OUTPUT_ROOT / "experiments.csv")
assert experiments.height == experiments.unique(["experiment_name", "seed", "split"]).height
assert np.isfinite(experiments["auc"].to_numpy()).all()
assert np.isfinite(experiments["logloss"].to_numpy()).all()
print("Real experiment records:", experiments.height)
display(experiments.filter((pl.col("split") == "test") & (pl.col("seed") == CONFIG["seed"])).select(
    "experiment_name", "n_features", "logloss", "auc", "ndcg_3", "ndcg_5", "ndcg_10", "recall_10", "map_10"))''')
    md("""## 22. Какие группы признаков помогли

Читаем ablation сверху вниз: к базовым категориям добавляются user, item, user–item
и metadata histories, затем rates и trends. Разница между соседними строками показывает
вклад группы при уже добавленных предыдущих признаках. Она не равна эффекту этой
же группы в любом другом наборе: признаки могут заменять друг друга.

Рядом показаны автоматические бюджеты 32/64/96/128 и полный pool. По ним видно,
можно ли сократить число колонок без заметной потери. Все значения в этой таблице —
search validation. Baseline выбирается среди выполненных manual stages по этому
участку, а не после просмотра test. Полный перебор всех сочетаний families не проводился.

Ниже также журнал Optuna: число завершенных trials, их параметры и лучшие значения.
Это позволяет отличить реально выполненный tuning от просто заданного в config бюджета.
""")
    code('''display(experiments.filter((pl.col("split") == "validation_search") & (pl.col("seed") == CONFIG["seed"])).select(
    "experiment_name", "n_features", "logloss", "auc", "training_time", "feature_generation_time", "memory_usage"))
for arm in ["manual", "auto"]:
    trials = pl.read_csv(OUTPUT_ROOT / f"{arm}_tuning.csv")
    print(arm, "HPO completed / total attempts:", trials.filter(pl.col("state") == "COMPLETE").height, "/", trials.height)
display(pl.read_csv(OUTPUT_ROOT / "manual_tuning.csv").sort("value", nulls_last=True).head(5))
display(pl.read_csv(OUTPUT_ROOT / "auto_tuning.csv").sort("value", nulls_last=True).head(5))''')
    md("""## 23. Что осталось в выбранном наборе

`selected_features.csv` перечисляет фактические исторические колонки automatic модели.
Группировка по entity показывает, какие окна сохранились у пользователя, item и их
связей. Это результат поиска в заданной сетке, а не доказательство единственного
правильного срока интереса.

Журнал агента читается отдельно: `before_logloss`, `after_logloss` и `accepted`
показывают, прошла ли группа новых окон порог улучшения. Улучшение меньше порога
означает DROP, даже если loss немного снизился. В конце — причины удаления
кандидатов при train pruning.
""")
    code('''selected_features = pl.read_csv(OUTPUT_ROOT / "selected_features.csv")
display(selected_features.select("feature", "entity", "event", "aggregation", "window", "long_window_days"))
display(selected_features.group_by("entity").agg(pl.len().alias("selected_specs"), pl.col("window").unique().sort()))
print("Selected explicit trends:", selected_features.filter(pl.col("aggregation") == "trend").height)
if (OUTPUT_ROOT / "agent_history.json").exists():
    agent_trace = json.loads((OUTPUT_ROOT / "agent_history.json").read_text())
    display(pl.DataFrame([{k: v for k, v in row.items() if k != "proposed"} for row in agent_trace]))
display(pl.read_csv(OUTPUT_ROOT / "pruned_features.csv").head(20))''')
    md("""В KION период наблюдения короче 180 дней. Поэтому 180d и all-history могут
дать одинаковые значения, и pruning оставляет один из вариантов. По такому выбору
нельзя заключить, что именно 180 дней — лучший масштаб памяти.

В сохраненном полном KION run выбран весь оставшийся adaptive pool: 493 specs.
Компактный набор не уложился в заданный допуск по loss. В Yambda сохранились 32 specs,
в основном пользовательские истории и несколько связей. Значит, один и тот же поиск
может дать очень разную компактность. Другой run или smoke не обязан повторить эти числа.

Для Yambda сильный пользовательский сигнал во многом описывает склонность выражать
like/dislike. Высокий classification AUC сам по себе не доказывает хороший выбор
следующего трека.
""")
    section("24. Графики и их смысл", """На importance смотрим, какие колонки использовал выбранный booster. Это train gain:
коррелированные окна могут делить между собой важность, поэтому маленький gain
не доказывает бесполезность всей семьи. График качества по числу признаков помогает
оценить цену сокращения набора.

Распределение окон и heatmap показывают фактический состав selected specs. Суммарный
gain по entity — описание модели, а не причинный вклад этой entity. Последний график
показывает средний test LogLoss и разброс seeds; у categorical и warmup/SVD arms
только один seed, поэтому оцененного seed-разброса у них нет.""", "scripts/plot_results.py", {"plots"})
    code('''plots(OUTPUT_ROOT)
for plot_name in ["metric_by_experiment.png", "metric_vs_features.png", "top30_importance.png",
                  "window_distribution.png", "entity_window_heatmap.png", "feature_family_gain.png",
                  "feature_counts.png", "test_metrics.png"]:
    display(Markdown("### " + plot_name.removesuffix(".png").replace("_", " ")))
    display(Image(filename=str(OUTPUT_ROOT / "plots" / plot_name)))''')
    md("""## 25. Что можно заключить из сравнения

Сначала сравниваем средний LogLoss основных моделей по трем заранее заданным seeds.
Затем смотрим парный bootstrap для seed 42. Эти проверки отвечают на разные вопросы:
первая показывает разброс обучения при фиксированном наборе, вторая — неопределенность
разницы на пользователях test при уже обученных моделях.

Если automatic выигрывает, это поддерживает полезность выбранного набора на данном
периоде. Чтобы утверждать, что сам поиск устойчиво переносится во времени, понадобятся
несколько temporal splits. Для SVD сравниваем `auto_plus_svd` с `auto_warmup`, где
исключены те же ранние строки: иначе эффект векторов смешался бы с размером train.

На главный вопрос об автоматизации есть и отдельный ответ по коду: конкретные колонки
и окна действительно генерируются и выбираются программой. Человек по-прежнему
определяет задачу, допустимые связи, правила расчета и бюджет. Метрики нужны, чтобы
проверить, дает ли такая автоматизация практическую пользу.
""")
    code('''test_summary = experiments.filter(pl.col("split") == "test").group_by("experiment_name").agg(
    pl.len().alias("seeds"), pl.col("logloss").mean().alias("mean_logloss"),
    pl.col("logloss").std().alias("sd_logloss"), pl.col("auc").mean().alias("mean_auc")).sort("experiment_name")
display(test_summary)
print(json.dumps(final_results["comparisons"], indent=2))
summary_lookup = {row["experiment_name"]: row for row in test_summary.to_dicts()}
mean_difference = summary_lookup["automatic"]["mean_logloss"] - summary_lookup["manual"]["mean_logloss"]
comparison = final_results["comparisons"]["test_automatic_vs_manual"]
lower, upper = comparison["logloss_ci_95"]
print(f"Mean test LogLoss, automatic − manual: {mean_difference:+.6f}")
if upper < 0 and mean_difference < 0:
    print("Automatic model improves this test; confidence interval is conditional on fixed models.")
else:
    print("Stable superiority of automatic temporal features over manual baseline is NOT established.")
print("Feature generation/selection is automated; humans still define target, schema, grammar, validation and budget.")
if MODE == "smoke":
    print("SMOKE results verify execution only; they are not evidence for the full-data research conclusion.")''')
    md("""### Сводка по двум датасетам

Если оба полных run находятся рядом, выводим их результаты в одной таблице.
Сравнивать KION и Yambda по абсолютному LogLoss некорректно: targets и доли классов
разные. Сравниваем manual и automatic внутри каждого источника. Результаты smoke
в эту сводку не входят.
""")
    code('''dataset_summaries = []
for dataset_name in ["kion", "yambda"]:
    dataset_root = WORK_DIR / "artifacts" / dataset_name
    if not (dataset_root / "final_results.json").exists():
        continue
    dataset_experiments = pl.read_csv(dataset_root / "experiments.csv")
    dataset_result = dataset_experiments.filter(
        (pl.col("split") == "test") & pl.col("experiment_name").is_in(["manual", "automatic"])).group_by(
        "experiment_name").agg(pl.col("logloss").mean(), pl.col("auc").mean())
    for row in dataset_result.to_dicts():
        dataset_summaries.append({"dataset": dataset_name, **row})
if dataset_summaries:
    display(pl.DataFrame(dataset_summaries).sort(["dataset", "experiment_name"]))''')
    md("""## 26. Время, память и оставшиеся ограничения

Автоматический поиск экономит ручную работу, но требует дополнительных расчетов
и обучений. Ниже время создания блоков признаков, память процесса и время final fit.
Это разные измерения: короткий final fit еще не означает дешевый поиск в целом.
Сегменты известных/новых пользователей помогают заметить, если общий score скрывает
проблему cold start; после test по этим сегментам модель не перенастраиваем.
""")
    code('''generation_profile = json.loads((OUTPUT_ROOT / "generation_profile.json").read_text())
print(json.dumps(generation_profile, indent=2))
display(experiments.filter(pl.col("split") == "test").select("experiment_name", "seed", "n_features", "training_time", "memory_usage"))
if (OUTPUT_ROOT / "test_segments.csv").exists():
    display(pl.read_csv(OUTPUT_ROOT / "test_segments.csv").select("model", "segment", "rows", "logloss", "auc"))''')
    md("""RSS измеряет память всего процесса с удерживаемыми матрицами. Cache profile хранит
время первоначального создания блока. В повторном KION run часть блоков использована
заново при совпавших data/spec/engine hashes; эти времена нельзя выдавать за новое
измерение длительности расчета. Время обучения моделей измерялось отдельно.

Главное ограничение исследования — один временной split и многократное использование
search validation. Три seeds не заменяют проверку на нескольких будущих периодах.
Кроме того, здесь есть допущения о готовности labels и статичности metadata, а категории
и авторы сведены к одной связи. Для production их нужно проверять по реальным логам.

На миллиард событий прямое расширение этого расчета не подойдет: одна матрица
из 400 float32 колонок заняла бы около 1.6 TB без накладных расходов. Поиск пришлось бы
проводить на выборке, а историю считать по разделенным entity states и хранить так,
чтобы на запрос выдавались значения только из допустимого прошлого. Поздние события
потребовали бы пересчета затронутой истории. Производительность на 1B+ здесь не измерялась.

Следующие полезные проверки — несколько rolling-origin splits, полный контекст
прослушиваний и отмен реакций Yambda, реальные candidate sets и новый закрытый holdout.
После этого можно сравнивать более сложные модели и проводить online A/B test.
GNN и Transformers пока рассмотрены в обзоре, их результатов в эксперименте нет.

Модульная версия сохранена для сопровождения, но все нужные для расчета функции
уже находятся в этом notebook.
""")
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook.metadata.language_info = {"name": "python", "version": "3.13.5"}
    notebook.metadata.standalone = True
    notebook.metadata.execution_modes = ["saved", "smoke", "full"]
    notebook.metadata.description = "All implementations inline; no project module imports"
    nbf.validate(notebook)
    # Проверяем, что готовый notebook можно выполнить без проектных Python-модулей.
    for cell in notebook.cells:
        if cell.cell_type != "code" or cell.source.startswith("# Установка"):
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("src", "scripts", "tests"))
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(("src", "scripts", "tests")) for alias in node.names)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, DESTINATION)
    print(f"Created {DESTINATION}: {len(cells)} cells, {sum(c.cell_type == 'code' for c in cells)} code cells")


if __name__ == "__main__":
    build()
