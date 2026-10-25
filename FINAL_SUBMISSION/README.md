# Feature Engineering Methods Comparison

Основной отчет и весь код находятся в
[feature_engineering_comparison.ipynb](feature_engineering_comparison.ipynb).
Notebook сравнивает шесть наборов признаков для LightGBM на KION и Yambda.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Откройте notebook в Jupyter или VS Code и выполните **Restart Kernel → Run All**.
Первая code cell проверяет зависимости и устанавливает недостающие пакеты через `%pip`.
По умолчанию запускается `MODE="full"`; сохраненные outputs относятся к этому режиму.
Для быстрой проверки можно выбрать `MODE="demo"` — первые 100k событий каждого dataset.
Его результаты нельзя сравнивать с полным экспериментом.

Raw files загружаются автоматически по зафиксированным URLs и проверяются по SHA256.
Для готовых данных можно задать `FE_RAW_DIR` с папками `kion/` и `yambda/`.
Новый запуск сохраняет модели, predictions, таблицы и графики в отдельную папку `.runs/`.

Автоматические признаки улучшили LogLoss относительно manual на обоих источниках;
отбор уменьшил число входов до 144 для KION и 69 для Yambda при близком качестве.
Ограничения: один split и seed, известный holdout, предполагаемая доступность истории.
Задача оценивает наблюдаемые взаимодействия, а не рекомендации по всему каталогу.

Сводные результаты и инструкции проверки — в [README репозитория](../README.md).
