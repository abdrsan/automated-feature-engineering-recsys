# Feature Engineering Methods Comparison

Какие признаки дают лучший баланс качества, времени расчета и интерпретируемости
в одной LightGBM-модели? Основной отчет и весь код — в
[feature_engineering_comparison.ipynb](feature_engineering_comparison.ipynb).

Используются KION (доля просмотра ≥50% среди наблюдаемых агрегатов) и явные
likes/dislikes Yambda flat/50m. Сравниваются raw IDs/metadata, manual temporal
aggregates, manual + trends, automated fixed windows, широкий automated pool
и validation selection. SVD из исходного проекта показан отдельно.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Откройте notebook в Jupyter или VS Code и выполните Run All с этим окружением.
`MODE="full"` по умолчанию — полные данные; `MODE="demo"` — весь pipeline на первых 100k строк.
Можно выбрать один dataset через `DATASETS`. По умолчанию выполняются оба.
Full требует заметно больше времени и памяти; результаты demo не заменяют full.
Параметры общие внутри dataset, seed=42; нового HPO в notebook нет.

Raw files загружаются по pinned URLs и сверяются с SHA256. Можно заранее указать
`FE_RAW_DIR` на папку с `kion/` и `yambda/`. При запуске внутри исходного проекта
его raw files находятся автоматически. `.runs/` содержит реально обученные модели,
predictions, protocol lock, CSV, графики и checksums текущего запуска.

Исторические признаки используют только события `<T`; encoder/pruning fit-ятся
на train. Test уже был известен до аудита, отсутствуют production `available_at`,
а задачи не измеряют full-catalog retrieval. Один новый seed не доказывает
устойчивость всей процедуры поиска. Полные проверенные выводы находятся в notebook;
исходные experiments сохранены отдельно в родительском проекте.

## Проверенные результаты

Полный controlled run, seed42, общие параметры внутри dataset:

| dataset | method | inputs | logloss | auc |
|---|---|---|---|---|
| kion | raw | 16 | 0.619689 | 0.676487 |
| kion | manual_trends | 118 | 0.577493 | 0.727001 |
| kion | auto_selected | 144 | 0.572429 | 0.731613 |
| yambda | raw | 5 | 0.234020 | 0.833212 |
| yambda | manual_trends | 92 | 0.161747 | 0.936464 |
| yambda | auto_selected | 69 | 0.160820 | 0.936956 |

Сохраненные outputs notebook относятся к `full`; повторный Run All по умолчанию запускает `full`. Full и isolated demo проверены; метрики и predictions full независимо воспроизведены. Статус: **READY WITH LIMITATIONS**. Главные ограничения — известный test, один новый seed и availability assumptions.
