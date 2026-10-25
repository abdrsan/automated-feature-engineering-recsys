> Исторический документ исходного исследования. Текущее сравнение методов и статус публикации — в [README](README.md) и [основном notebook](FINAL_SUBMISSION/feature_engineering_comparison.ipynb). Старые и новые эксперименты используют разные правила выбора наборов и не подменяют друг друга.

# Executive Summary

Независимый аудит исходного состояния, **до исправлений**, 16 октября 2026.
Исходное ТЗ прочитано из приложенного текста. README, REPORT и прежний checklist
использовались как проверяемые утверждения, а не как доказательства.

**Результаты исходных моделей воспроизводятся, но сравнение с manual reference и
воспроизводимость проекта имеют существенные недостатки. Исходный вердикт — C.**

Проверено кодом: оба полных набора данных, 18 final model/prediction pairs, 198 численных
сравнений метрик, 4 112 прямых проверок признаков на реальных строках, повторная генерация
346 236 248 feature cells. Значения из CSV совпали с независимыми формулами метрик;
predictions побитово совпали с повторным inference сохраненных моделей. Fake metrics
или placeholder основного алгоритма не обнаружены.

Дополнительно по сохраненному исходному состоянию переоценены **все 84 development/audit
checkpoints**, сверены CSV с SQLite trials и заново обучены четыре лучших HPO trial.
Их objective values совпали в точности. Доказательства:
[KION search replay](artifacts/audit/before/kion/search_replay_summary.json),
[Yambda search replay](artifacts/audit/before/yambda/search_replay_summary.json),
[KION HPO](artifacts/audit/before/kion/best_hpo_replayed.csv),
[Yambda HPO](artifacts/audit/before/yambda/best_hpo_replayed.csv).

Исходные модели и CSV сохранены в `artifacts/pre_audit/{kion,yambda}`,
исходный standalone notebook — `artifacts/pre_audit/00_full_research.ipynb`.

Исходные 56 тестов проходят. Собственные 23 проверки аудитора: **21 PASS, 2 FAIL**.
Падающие проверки воспроизводят реальные ошибки M1 и M3, не являются произвольными
проверками стиля. См. [лог](artifacts/audit/before/independent_tests.log).

# Final Score

| Категория | Баллы | Основание |
|---|---:|---|
| Problem coverage | 13/15 | Основные этапы выполнены; постановки conditional scoring уже catalog recommendation |
| Data & temporal correctness | 18/20 | Строгий cutoff подтвержден; availability агрегатов и metadata не доказана источником |
| Manual baseline quality | 4/10 | HPO есть, но более сильная manual ablation исключена из reference selection |
| Automated feature engineering | 15/20 | Реальная грамматика; fixed/shared controls ослаблены общим pruning (M5) |
| Evaluation correctness | 12/15 | Метрики верны; proxy queries и один temporal cutoff ограничивают вывод |
| Research depth | 8/10 | Содержательное сравнение подходов, SVD выполнен; большинство альтернатив только обсуждалось |
| Reproducibility | 1/5 | Clean CLI оставляет недостающий EDA artifact; cache и HPO resume не полностью воспроизводимы |
| Code quality | 4/5 | Простой читаемый pipeline, полезные тесты; lifecycle артефактов требует защиты |
| **TOTAL** | **75/100** | **C — Acceptable but significant issues remain** |

Первоначальная оценка была 77/100. Дополнительная проверка исходного кода обнаружила
M5; оценка автоматизации снижена на 2 балла. Это уточнение относится к исходному
состоянию, до исправления controls, и не меняет проверенные исходные метрики.

# Critical Issues

**Подтвержденных P0 не обнаружено.** Текущее событие или future events в проверенные
исторические признаки не попадают. Значения final metrics не сфабрикованы. Это не
доказательство production availability всех полей: соответствующие ограничения ниже.

# Major Issues

## M1 / P1 — manual reference исключает более сильную ручную ablation

`develop()` выбирает только между `manual` и `manual_plus_trend`. На Yambda
`plus_metadata_relations` имеет search LogLoss **0.172084298**, а выбранный
`manual_plus_trend` — **0.172373535**. Все arms реально обучены с одним набором tuned
параметров. Игнорирование лучшего доступного ручного варианта ослабляет главный контроль;
небольшой test выигрыш auto нельзя уверенно защищать как выигрыш над лучшим проверенным
manual набором. На KION данная ошибка выбора непосредственно не меняет исходный reference.

Доказательства: `scripts/run_study.py`, строки выбора `manual_name`; experiment CSV;
собственный тест `test_audit_manual_reference_cannot_ignore_a_better_validated_ablation`.

## M2 / P1 — документированный clean pipeline не создает обязательный EDA artifact

`scripts.write_dataset_report` читает `artifacts/{dataset}/eda_full.json`, но
`scripts.run_study` пишет только `eda.json`. Существующий `scripts.eda` поддерживает
только MovieLens. Свежий Yambda smoke dev/final выполнен в отдельном пустом output:
модели и final metrics появились, `eda_full.json` отсутствует. Значит, опубликованный
порядок команд не воспроизводит весь итоговый отчет с нуля без скрытого ручного шага.

Доказательства: [fresh CLI log](artifacts/audit/before/cli_smoke.log),
`scripts/eda.py`, `scripts/run_study.py`, `scripts/write_dataset_report.py`.

## M3 / P1 — изменение реализации не инвалидирует cache и model resume

Feature cache key содержит данные и specs, но не версию движка. Manifest и resume specs
не фиксируют версию протокола/исходников. После исправления feature engine существующие
matrices и модели могут молча переиспользоваться. Собственный тест меняет объявленную
версию и результат generator: второй вызов возвращает старый cache. Это риск недействующего
исправления и некорректного смешивания результатов; фактического такого смешивания в
исходных проверенных final predictions не найдено.

## M4 / P1 — HPO resume заново инициализирует TPE sampler

SQLite хранит trials, но состояние sampler не сохраняется. После перезапуска seed
снова начинает ту же случайную последовательность при уже существующих trials.
В KION trials 0 и 2 имеют одинаковые параметры и score после прерывания. Двадцать
COMPLETE trials действительно существуют, но исходная последовательность interrupted
run не равна непрерывному clean run. Заявление о точной воспроизводимости resume завышено.

## M5 / P1 — fixed/shared controls наследуют pruning более широкого adaptive pool

Исходный pipeline сначала удаляет duplicates/correlated columns во всем adaptive
пространстве, затем фильтрует retained features по разрешенным fixed windows. Если
all-history удален как duplicate 180d, а 180d запрещен в fixed control, оба aliases
исчезают из control. В исходном KION найдено 22 таких all-history → 180d удаления (21 exact duplicate, 1 correlation).
После независимого fixed pruning восстановлены 20 inputs; еще два имеют aliases внутри fixed pool. Это ослабляет исходный fixed baseline по причине порядка вычислений, а не качества окна.
Shared-window control также наследует отбор и gain широкого pool.

Независимый regression воспроизводит проблему: 1d/7d/all совпадают на train; после
adaptive pruning остается 1d, и последующий fixed filter теряет доступный 7d signal.
Правильный порядок — сначала определить допустимый pool, затем независимо prune и fit.
Доказательства: исходные pruned_features.csv и scripts/run_study.py; новый тест
test_audit_fixed_pool_keeps_fixed_alias_of_adaptive_only_duplicate. Числа primary
auto/manual не сфабрикованы, но вывод о пользе расширения окон требует новых controls.
Проверенный список: [lost fixed aliases](artifacts/audit/before/fixed_pool_lost_aliases.csv).

## M6 / ограничение, не устранимое изменением кода — test уже раскрыт

Исправления будут проверяться на тех же test rows, которые уже оценены в исходном
исследовании и в этом аудите. Такой повторный run необходимо называть **post-audit
re-evaluation**, а не новым слепым подтверждением. Новые параметры/структура должны
выбираться только по search validation. Восстановить неизвестность test невозможно;
для независимого подтверждения нужны новые данные или заранее закрытые временные folds.

# Minor Issues

- Дополнительный независимый SQLite test выявил расхождение illustrative SQL с Polars:
  SUM пустой истории дает NULL в SQL и 0 в движке. SQL не используется при training,
  поэтому сохраненные метрики это не затрагивает; reference SQL требует COALESCE.
  Дополнительный test обнаружил, что SQL equality не соответствует Polars nulls_equal=True
  для nullable keys: expected past count=1, SQL=0. Нужен null-safe join; training backend
  этим не затронут. Лог: artifacts/audit/before/reference_null_sql_test.log.

- Candidate CSV содержит 637/464 specs, тогда как generation profile сообщает 621/448:
  разница — 16 последующих agent proposals. Это не фальсификация, но стадии следует
  называть initial и total explored, чтобы counts читались однозначно.
- Gain importance — train proposal signal со смещением при коррелированных окнах;
  permutation/SHAP отсутствуют. Они не обязательны для выбранного варианта метода.
- API temporal_split предполагает уже отсортированные данные; явной проверки этого
  входного контракта нет. Основные loaders данные сортируют, поэтому текущие runs корректны.
- Cache hit выводится из нулевого generation_seconds в отдельных model rows: это не
  точное происхождение каждого feature block и не end-to-end runtime поиска.
- KION 180d эквивалентно all-history на фактическом периоде короче 180 дней.
- В Yambda сохраняются одинаковые raw reactions, возможны log duplicates. Они не
  пересекают temporal boundaries, но влияют на вес наблюдений и history counts.

# Requirement Matrix

| Requirement | Status | Evidence | Comment |
|---|---|---|---|
| Working ML pipeline | PASS | fresh CLI smoke, 18 model replays | Training/evaluation реально выполняются |
| Temporal split | PASS | src/split.py, независимый split test | Глобальный timestamp cutoff, ties не пересекают границы |
| Strong manual boosting baseline | PARTIAL | manual_tuning.csv, M1 | Tuning есть; reference selection неполон |
| User features | PASS | baseline_features, raw-history oracle | Counts, means, recency, rates |
| Item features | PASS | feature_engine, real-data spot checks | Популярность и outcomes только до T |
| User-item features | PASS | manual specs, oracle | В KION постоянны из-за отсутствия повторных пар |
| Additional entity relations | PASS | user-category, user-author, item-gender | Первичные metadata связи упрощены явно |
| Automated feature generation | PASS | generator, 621/448 initial specs | Комбинации создаются программно |
| Declarative FeatureSpec | PASS | реальные cache/model specs | Используется движком и proposer |
| Automatic selection/search | PASS | pruning, top budgets, refits, models | Gain proposals и validation выбор |
| Automatic temporal window search | PARTIAL | candidate specs, distinct window pools | Реальный дискретный поиск; fixed/shared controls требуют M5 |
| Leakage prevention | PASS | 4 112 spot checks, 346M regenerated cells | При оговоренных assumptions о доступности исходов |
| Leakage tests | PASS | 56 original tests, independent oracle | Current/future/ties проверены отдельно |
| Justified LightGBM model | PASS | model.py, replayed boosters | Conditional binary classification |
| Validation metrics | PASS | saved dev boosters/specs/CSV | Search используется многократно; audit отдельно |
| Final test metrics | PASS | 198 independent metric comparisons | Формулы и predictions подтверждены |
| Ablation study | PASS | 26 seed-42 search arms на dataset | Вклад групп условен на порядок добавления |
| Selected features | PASS | CSV == lock == final model names | Установлено фактическим replay |
| Selected windows | PASS | CSV == selected specs | Наличие window не доказывает causal necessity |
| Feature importance | PASS | gain == сохраненный source booster | Final refit importance отдельно не заявляется |
| Real experiments | PASS | models, predictions, logs, trials | Не только таблицы в REPORT |
| Plots | PASS | plot_results.py | Данные графиков происходят из artifacts |
| README | PARTIAL | M2, M3, M4 | Clean reporting/resume claims требуют исправления |
| REPORT | PARTIAL | M1, M5 и M6 | Основной вывод требует более сильных controls и раскрытия повторной оценки |
| Reproducibility from scratch | PARTIAL | fresh smoke; missing eda_full | Нужен полный повтор исправленного pipeline |
| Feature agent | PASS | 2×8 реально оцененных proposals | KEEP threshold не пройден, это честный отрицательный результат |
| Optional learned representations | PASS | frozen warmup SVD + matched controls | Future train edges в basis не используются |
| Full-catalog ranking | NOT APPLICABLE | Реальная scoring постановка | Нет impression/candidate benchmark; proxy явно ограничен |
| SHAP/permutation importance | NOT IMPLEMENTED | Код и artifacts отсутствуют | Не обязательны; не выдаются за выполненные |
| Full GNN/Transformer training | NOT IMPLEMENTED | Только related work | Optional, отсутствие не является P0/P1 |

# Repository Map

```text
src/
  data.py, datasets.py, dataset_sources.py  # raw ingestion, target, metadata, provenance
  split.py                                # temporal boundaries
  baseline_features.py                     # manual specs, staged ablation
  feature_spec.py, feature_engine.py       # validated grammar, Polars calculations
  auto_feature_generator.py                # candidate factory
  feature_selection.py, feature_agent.py   # train pruning, safe proposals
  model.py, experiments.py, metrics.py     # fitting/HPO, tracking, evaluation
  representations.py                      # prefix-frozen SVD
scripts/
  run_study.py                             # development, lock, final refit
  audit_controls.py, rescore_ranking.py     # post-lock diagnostics
  make_full_notebook.py                    # literal inline notebook build
  write_dataset_report.py, plot_results.py # artifact-driven reporting
configs/                                  # model budget and feature search grammar
notebooks/00_full_research.ipynb            # standalone implementation + outputs
tests/                                    # original tests
audit/                                    # independent reviewer checks
artifacts/{kion,yambda}/                    # CSV/JSON, models, predictions, logs, plots
```

`sequential_rl/` — отдельная работа в общей папке, не является частью этого проекта и
не изменяется аудитом. MovieLens результаты архивные и не считаются KION/Yambda runs.

# Dataset Audit

Числа заново получены из физических файлов и нормализованных строк, а не перенесены из
README. Полные cardinalities, nulls и quantiles взаимодействий:
[KION](artifacts/audit/before/kion/data_statistics.json),
[Yambda](artifacts/audit/before/yambda/data_statistics.json).

| Dataset | Raw reactions | Used | Users | Items | Duplicate user/item/time/label |
|---|---:|---:|---:|---:|---:|
| KION | 1 594 787 | 1 594 519 | 567 555 | 12 639 | 0 |
| Yambda likes/dislikes | 989 232 | 989 232 | 8 544 | 194 996 | 6 654 |

KION исключает 268 отсутствующих targets. Связи metadata join m:1, count сохранен.
Фактический архив KION отличается от рекламируемых 5.47M в
[официальном README](https://github.com/irsafilo/KION_DATASET); отчет правильно описывает
физический файл. Количество likes/dislikes совпадает с
[официальной карточкой Yambda](https://huggingface.co/datasets/yandex/yambda).
[Формула времени в статье](https://arxiv.org/html/2505.22238v1) подтверждает elapsed seconds,
округленные до 5s; дополнительного умножения в коде нет.

# Leakage Audit

**PASS для реализованных исторических расчетов.**

Finite windows используют `closed=left`, затем проверяют max source timestamp < T.
All-history объединяет ties в atomic timestamp blocks до cumulative/shift.
Recency не смотрит на строку T. Оба счетчика rates/conversions исключают T, trends имеют
одинаковую temporal границу. Encoder и pruning обучаются на train prefix. Текущие
label/event/history_value/watched_pct/total_dur не входят в base matrix. Target encoding
не используется. Fit scaler на full data отсутствует. Разрешенная online replay история
внутри test не равна использованию будущего: исходы более раннего времени доступны позже.

Ограничение: KION outcome aggregated по паре и last_watch_day; настоящий available_at
не дан. Static metadata, как и is_organic context, предполагаются наблюдаемыми.
Нельзя экстраполировать этот PASS на scoring до первого просмотра или на delayed outcomes.

# Feature Engine Audit

Синтетический Jan 1 click / Jan 2 click / Jan 5 purchase: user-item click counts
**expected [0,1,2], actual [0,1,2], PASS**. Дополнительные tests покрывают duplicate
timestamps, пустую историю, nullable values/keys, все primitive aggregations и ratios.
Независимые ожидаемые значения и фактические значения для каждой реальной проверенной
строки: [KION](artifacts/audit/before/kion/feature_spot_checks.csv),
[Yambda](artifacts/audit/before/yambda/feature_spot_checks.csv).

| Проверка | KION | Yambda | Status |
|---|---:|---:|---|
| Direct raw-history spot comparisons | 2 432 | 1 680 | PASS |
| Regenerated cached feature cells | 242 366 888 | 103 869 360 | PASS |
| Final booster/prediction replays | 9 | 9 | PASS |
| Infinite historical feature values | 0 | 0 | PASS |

NaN/null fractions сохранены отдельно в feature_missingness.csv каждого audit каталога.
Missing recency/mean при отсутствии истории ожидаемы; counts/rates конечны. Many-to-many
row explosion не обнаружено. Main pipeline сохраняет порядок canonical row_id.

# Metric Audit

Формулы аудитора не используют src.metrics: AUC через Mann–Whitney/rankdata, LogLoss
через явную формулу, ranking через отдельную сортировку каждой query и DCG/AP суммы.
Synthetic positive-at-rank-2 дает NDCG@3=1/log2(3), MAP@3=0.5, Recall@3=1.

198 строк Reported/Reproduced/Difference находятся в
[KION metric audit](artifacts/audit/before/kion/metrics_reproduced.csv) и
[Yambda metric audit](artifacts/audit/before/yambda/metrics_reproduced.csv).
Расхождения на уровне floating-point rounding (<1e-10). Probability arrays из сохраненных
models и parquet совпадают в пределах 1e-13; фактический max difference — 0.

Candidate protocol: observed user × exact timestamp; только distinct-item mixed-label
группы. Это не full catalog и не sampled-negative retrieval benchmark. Исключение
одноклассовых query делает proxy условным; высокий Recall@10 объясняется также малым
числом кандидатов. По нему нельзя заявлять production retrieval quality.

# Baseline vs Automated

Основной endpoint — mean test LogLoss между тремя заранее выбранными seeds:

| Dataset | Manual inputs | Auto inputs | Manual AUC | Auto AUC | Manual LogLoss | Auto LogLoss | Relative reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| KION | 118 | 80 | 0.726828 | 0.729059 | 0.577868 | 0.574244 | 0.627% |
| Yambda | 92 | 37 | 0.936545 | 0.936790 | 0.161658 | 0.161150 | 0.314% |

Эти значения корректно пересчитаны, но M1 ослабляет интерпретацию второго сравнения.
Полная таблица baseline0/manual/auto-all/selected/embedding со scope, runtime и metrics
находится в experiment CSV: baseline0 и auto-all имеют search metrics, не final test.
Seed SD и paired user bootstrap реально вычислены; CI seed42 не оценивает variability
всей feature search процедуры. Один global cutoff не проверяет устойчивость во времени.

# Automated Search Audit

Автоматизированы комбинации entity/event/aggregation/window, train pruning, gain proposals,
refit budgets, validation selection, proposals соседних окон и KEEP/DROP. Optuna здесь
настраивает модель, **не feature masks**. Человек определяет schema paths, events, grammar,
window grid, objective, budget и thresholds. Это structured automated search, не полностью
самостоятельный feature invention. FeatureSpec не декоративный: specs определяют реальные
matrices и имена booster inputs.

| Stage | KION | Yambda |
|---|---:|---:|
| Initial candidate specs | 621 | 448 |
| After train-only pruning | 493 | 442 |
| Total explored including agent | 637 | 464 |
| Final historical specs | 64 | 32 |

# Selected Windows Audit

Для каждой строки selected_windows подтверждено точное совпадение с selected feature
specs и final model columns. Для каждого выбранного окна найдены альтернативы в candidate
grid и реальные validation model refits top32/64/96/128/all. Подробный по-family список —
[KION CSV](artifacts/pre_audit/kion/selected_windows.csv), [Yambda CSV](artifacts/pre_audit/yambda/selected_windows.csv).
Shared-window control реально обучен и отдельно оценен; состав признаков тоже меняется,
поэтому он не доказывает чистый эффект entity-specific window при фиксированных primitives.

# Ablation Audit

Все перечисленные в experiment CSV seed-42 ablation arms имеют сохраненные booster/specs.
User-item в KION не добавляет сигнала: пары не повторяются, соответствующие historical
counts постоянны. Trends дают маленькое изменение; агент не проходит KEEP threshold.
На Yambda пользовательские rates доминируют над item features: target во многом отражает
склонность пользователя выражать like/dislike. Это не проверка next-track ranking.

# Reproducibility Audit

Отдельный свежий CLI smoke действительно переобучен с пустым output и SQLite; основной
fit/evaluation путь работает. End-to-end report из опубликованной последовательности
не воспроизводится без eda_full.json. M3/M4 мешают безопасному resume после изменений.
Абсолютных пользовательских путей в core training code не найдено. Notebook содержит
inline реализации; отсутствие imports из src само по себе не заменяет проверку clean Run All.
После исправлений она должна быть повторена в изолированной папке.

# Unsupported Claims

- «Сильнейший manual baseline» — не доказано среди уже проверенных ablation subsets (M1).
- «Полная воспроизводимость с нуля по README» — опровергается отсутствующим EDA (M2).
- «Resume полностью продолжает исходный поиск» — неверно для TPE sampler state (M4).
- «Разные окна необходимы для разных entities» — causal необходимость не доказана;
  подтвержден только выбор разных дискретных inputs ограниченным search.
- «Fixed/shared controls честно изолируют пользу adaptive окон» — нарушено порядком
  pruning и filtering (M5); эффект требует повторного эксперимента.
- «Улучшены рекомендации по всему каталогу» — данными/протоколом не проверяется.
  Основной отчет это ограничение уже оговаривает.

# Missing Requirements

Не завершены безопасная версионизация runs, clean artifact/report generation и честный
выбор лучшего проверенного manual reference. Полный GNN/Transformer и SHAP не обязательны;
их отсутствие нельзя считать нарушением исходного ТЗ. Для научной уверенности остаются
rolling-origin validation и независимый новый test, которых в исходном проекте нет.

# What Was Done Well

- Timestamp ties обработаны корректно, включая all-history и embedding profiles.
- Присутствуют настоящий candidate factory, pruning, model refits и работающий proposer.
- Models/predictions/metrics согласованы; нет признаков нарисованных результатов.
- Warmup SVD исключает свои training rows одинаково из matched arms.
- Отрицательные результаты agents/SVD сохраняются и не выдаются за улучшения.
- Отчет отделяет conditional scoring, proxy ranking и официальные retrieval benchmarks.

# Fix Plan

| Priority | Issue | Действие |
|---|---|---|
| P0 | Нет подтвержденных | Не менять метрики вручную |
| P1 | M1 | Выбирать reference среди всех реально обученных manual ablation stages по search; freeze до audit/test |
| P1 | M2 | Автоматически создавать full EDA artifact и проверять clean CLI/report lifecycle |
| P1 | M3 | Добавить engine/protocol versions в cache/manifest/model specs; повторить runs в пустых outputs |
| P1 | M4 | Сохранять TPE state и сверять его с trial history; тест непрерывного vs resumed поиска |
| P1 | M5 | Restrict pool до pruning; независимо prune/fit fixed и shared controls; повторить selection и final runs |
| P1 | M6 | Явно пометить повторную оценку post-audit; не объявлять тот же test новым blind confirmation |
| P2 | Statistics | Rolling-origin replication, group permutation, multiple metadata relations, new future holdout |

После фиксации этого отчета исходные artifacts должны быть сохранены отдельно. Все новые
метрики обязаны происходить из повторного fitting/inference. Этот файл сохраняет
**исходный** verdict; итоговое состояние оценивается в FINAL_AUDIT.md.

# Final Verdict

**C — Acceptable but significant issues remain, 75/100.** Проверенные вычисления и цифры
заслуживают доверия в заявленных условных задачах. Основной сравнительный вывод и clean
reproducibility пока требуют исправлений; уверенная защита без этих оговорок преждевременна.
