> Исторический документ исходного исследования. Текущее сравнение методов и статус публикации — в [README](README.md) и [основном notebook](FINAL_SUBMISSION/feature_engineering_comparison.ipynb). Старые и новые эксперименты используют разные правила выбора наборов и не подменяют друг друга.

# Соответствие заданию

Основной результат — [автономный notebook](notebooks/00_full_research.ipynb).
Он содержит реализации в ячейках и не импортирует проектные модули. Переключатель
`DATASET` выбирает KION или Yambda; `MODE` — saved, smoke или full.

| Требование | Реализация и проверяемый результат |
|---|---|
| Исследовать реальные данные до выбора модели | [Схемы, EDA и постановки](REPORT.md); фактические EDA: [KION](artifacts/kion/eda_full.json), [Yambda](artifacts/yambda/eda_full.json) |
| Исследовать альтернативы и сравнить подходы | [RESEARCH.md](RESEARCH.md): DFS/Featuretools, AutoFE/RL/AutoRecSys, boosting, MF, GNN, sequential и tabular Transformers; первичные источники и таблица trade-offs |
| Минимальный и настроенный manual baseline | [LightGBM и Optuna](src/model.py), [manual families и ablation](src/baseline_features.py); 20 COMPLETE HPO trials отдельно для manual и auto на каждом датасете |
| Декларативные признаки | [FeatureSpec](src/feature_spec.py): entities, events, windows, count/sum/mean/min/max/nunique, recency, rates/conversion/frequency, trends |
| Векторизованный temporal engine | [Polars backend](src/feature_engine.py), интервал `[T−W,T)`; SQL-пример в notebook; без Python loops по interaction rows в движке |
| Автоматические кандидаты и окна | [Генератор](src/auto_feature_generator.py), [train-only pruning](src/feature_selection.py), validation selection размеров и окон в [pipeline](scripts/run_study.py) |
| Closed-loop feature agent | [Heuristic proposer и безопасный optional LLM hook](src/feature_agent.py); реальные KEEP/DROP: [KION](artifacts/kion/agent_history.json), [Yambda](artifacts/yambda/agent_history.json) |
| Temporal validation, test вне search | Train/search/audit/test ≈70/7.5/7.5/15%; timestamp ties не разделяются; [KION lock](artifacts/kion/locked_selection.json), [Yambda lock](artifacts/yambda/locked_selection.json) |
| Leakage tests | [Jan 1/Jan 2, tied times, current/future mutations, prefix invariance](tests/test_no_leakage.py); [independent aggregation oracle](tests/test_feature_engine.py); [SVD cutoff](tests/test_representations.py) |
| Метрики и воспроизводимость | AUC, LogLoss и proxy NDCG/Recall/MAP@3/5/10; три заданных seeds для основных моделей; [KION experiments](artifacts/kion/experiments.csv), [Yambda experiments](artifacts/yambda/experiments.csv) |
| Ablation и диагностические controls | Base → user → item → user-item → user-category → user-author → rates → trends; fixed/adaptive pools, budgets и shared-window control; таблицы в [отчете](REPORT.md) |
| Выбранные features и windows | [KION features](artifacts/kion/selected_features.csv), [KION windows](artifacts/kion/selected_windows.csv); [Yambda features](artifacts/yambda/selected_features.csv), [Yambda windows](artifacts/yambda/selected_windows.csv) |
| Importance и графики | [KION importance](artifacts/kion/feature_importance.csv), [Yambda importance](artifacts/yambda/feature_importance.csv); по восемь графиков в каждом dataset-каталоге, также встроены в notebook |
| Learned representations | [SVD и temporal user profiles](src/representations.py), одинаковый warmup cutoff и training rows для matched controls; результат сравнения приведен в отчете |
| Runtime, память и масштабирование | Фактические generation/fit timings, process RSS; ограничения 1B+/100B и production architecture в [REPORT.md](REPORT.md) |
| README, отчет и воспроизводимый запуск | [README.md](README.md), [REPORT.md](REPORT.md), [зависимости notebook](requirements-notebooks.txt) |
| Автономность notebook и целостность результатов | [Проверка изолированного запуска](scripts/validate_full_notebook.py), [верификатор](scripts/verify_datasets.py), [результат проверки](artifacts/datasets_verification.json) |

## Проверенное состояние после аудита

Первичный независимый аудит: [AUDIT_REPORT.md](AUDIT_REPORT.md).
Исправления: [FIXES.md](FIXES.md). Итоговая повторная проверка и реальные новые
числа: [FINAL_AUDIT.md](FINAL_AUDIT.md). Прежние test результаты сохранены в artifacts/pre_audit.

Manual reference выбирается среди всех проверенных ablation stages по search validation.
Оба полных pipeline переобучили models и HPO с нуля. KION переиспользовал только
проверенный content-addressed feature cache; независимая регенерация совпала. 88 тестов и независимые
model/metric/cache проверки проходят; notebook проверен без проектных модулей.

Текущие итоги — post-audit re-evaluation уже известного test. Выводы относятся к KION
watch-percentage classification и Yambda like/dislike conditional scoring, а не CTR или
full-catalog retrieval. Строгий cutoff <T доказан для движка при оговоренных assumptions
о доступности labels и static metadata. Rolling-origin/new blind test остаются future work.
