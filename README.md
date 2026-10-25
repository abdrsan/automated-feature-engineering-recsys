# Сравнение методов Feature Engineering

Какие признаки дают лучший баланс качества, времени расчета и интерпретируемости
при одной модели LightGBM?

**Основной отчет и весь код:** [feature_engineering_comparison.ipynb](FINAL_SUBMISSION/feature_engineering_comparison.ipynb).
Краткое описание эксперимента: [REPORT.md](REPORT.md).

## Данные и методы

- [KION](https://github.com/irsafilo/KION_DATASET): 1 594 519 пригодных user–item агрегатов; target — доля просмотра ≥50%.
- [Yambda](https://huggingface.co/datasets/yandex/yambda): 989 232 likes/dislikes из flat/50m; target — like против dislike.

Оценивается binary classification среди наблюдаемых взаимодействий, а не CTR по показам
или выдача по всему каталогу. Raw data не включены в Git: notebook загружает
зафиксированные версии и проверяет SHA256. Условия использования данных указаны
в источниках; MIT LICENSE этого репозитория относится к коду.

Сравниваются шесть вариантов: raw IDs/metadata, manual temporal aggregates,
manual + trends, automated fixed windows, полный automated pool и отбор из него.
Внутри dataset одинаковые строки, target, split, seed=42 и параметры LightGBM.
Поиск использует только search validation; контрольная validation и test не меняют выбор.
Исторические признаки используют события строго раньше прогнозируемого timestamp.

## Основные результаты

Меньше LogLoss — лучше. Полная таблица: [results/full/comparison.csv](results/full/comparison.csv).

| Dataset | Manual LogLoss | Auto pool LogLoss | Auto selected LogLoss | Inputs pool → selected |
|---|---:|---:|---:|---:|
| KION | 0.577836 | 0.572512 | 0.572429 | 509 → 144 |
| Yambda | 0.161425 | 0.160768 | 0.160820 | 447 → 69 |

Автоматические признаки улучшили результат на обоих источниках. Отбор сильно сократил
набор при близком качестве; на Yambda полный pool остался немного лучше selected.
Добавление trends к manual помогло KION и слегка ухудшило Yambda.
Стоимость дополнительных обучений поиска учитывается отдельно от финальной модели.

![Качество и стоимость методов](results/full/comparison.png)

## Запуск

Проверено на Python 3.13, macOS arm64.

```bash
git clone https://github.com/abdrsan/automated-feature-engineering-recsys.git
cd automated-feature-engineering-recsys
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebooks.txt
```

Откройте основной notebook в Jupyter или VS Code, выберите это окружение и выполните
**Restart Kernel → Run All**. Первая code cell устанавливает недостающие зависимости
через `%pip` в текущий kernel. При необходимости установите frontend Jupyter отдельно.

- `MODE="full"` по умолчанию: оба dataset полностью в описанной выборке.
- `MODE="demo"`: весь pipeline на первых 100k событий; это проверка запуска, а не full results.
- `DATASETS`: позволяет оставить один источник.
- `FE_RAW_DIR`: необязательный путь к уже скачанным папкам `kion/` и `yambda/`.

Новый запуск сохраняется в `.runs/`: модели, predictions, таблицы, графики и параметры
выбора. Full может занимать десятки минут и требует больше памяти, чем demo.
Опубликованные результаты и predictions находятся в `results/full/`.

```bash
python -m pytest -q
python -m scripts.verify_published_results
```

Вторая команда пересчитывает метрики из predictions и проверяет SHA256 и выполнение
ячеек сохраненного notebook. Она не переобучает модели.

## Ограничения

Один временной split и один seed; test уже наблюдался ранее, поэтому это повторная
оценка на известном holdout. В данных нет `available_at` и исторических версий metadata:
доступность прошлых событий предполагается. KION имеет точность до дня и не содержит
повторных user–item пар; Yambda использует только явные реакции. Выводы не являются
проверкой full-catalog retrieval или production latency.
