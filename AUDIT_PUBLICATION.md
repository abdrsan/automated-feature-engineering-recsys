# Итоговый аудит: Feature Engineering Methods Comparison

Дата: 25 октября 2026. Статус: **READY WITH LIMITATIONS**.
Неофициальная экспертная оценка — **87/100**. Это не официальная шкала задания:
сильная сторона — проверяемые расчеты и реальный controlled experiment; ограничения —
известный test, один split и неподтвержденная production availability.

## Что было в проекте

KION: 1 594 519 пригодных агрегированных строк, target watched_pct≥50.
Yambda: 989 232 явных likes/dislikes из flat/50m, target like против dislike.
Изначально реализованы FeatureSpec, Polars temporal engine, manual counts/means/recency/rates,
trends, candidate factory, train-only pruning, gain/top-K validation search, independent
fixed/shared-window controls, Optuna, deterministic feature proposer и warmup SVD.
GNN/Transformers/ALS обсуждались как альтернативы, их результаты не приписываются проекту.

Raw → temporal split → train encoder → past-only feature matrices → manual HPO/ablations →
factory/pruning/budget search → lock → audit → 85% refit → final test. По 54 experiment rows
на dataset, включая search/audit/test; основные manual/auto имеют три seeds, не все arms.
Original source specs, boosters, predictions и ledgers сохранены.

## Обнаруженные проблемы и минимальные изменения

1. Исходный исследовательский фокус — automated windows. В существующих final locks
   параметры manual/auto фактически совпали на обоих dataset: подмена модели не найдена.
   Для более широкой темы нужно отдельно показать остальные feature methods и стоимость
   дополнительного поиска, а SVD сравнивать на одинаковых warmup populations.
   Создан отдельный controlled comparison на тех же данных и
   существующих реализациях: общие manual-tuned параметры, один seed, одинаковые split/rows.
   SVD и старые трехseedовые итоги отделены от основной таблицы. Новых методов не добавлено.
2. Прежние generation profiles смешивали стоимость shared matrices и reuse cache.
   В новом сравнении generation каждого уникального финального набора измеряется заново.
   Search cost показан отдельно. Alias одной модели явно помечен `shared_with`.
3. При проверке новой краткой оболочки clean venv выявил зависимость `.to_pandas()` от
   отсутствующего pyarrow. Убрано ненужное преобразование; используются небольшие lists
   of dicts. Падение сохранено в `failed_missing_pyarrow.log`; успешный full run повторен
   после исправления. Это ошибка новой оболочки, а не выдуманный defect старого engine.
4. Известный test и отсутствие `available_at` невозможно исправить переобучением.
   Ограничения оставлены явно. Старые experiments не заменены новыми числами.

Подтвержденных новых P0/current-event/future-event leakage ошибок не обнаружено.
Все 313 исходных файлов snapshot сохранили SHA256. Добавлены только новый
submission и файлы текущего аудита. Другие проекты и корневой FINAL_SUBMISSION не изменялись.

## Независимые проверки

- **94 tests PASS**: original 88 + synthetic metric ties, exact reuse/standalone
  и обе ветки установки зависимостей в текущем kernel.
- Оба исходных final runs проверены заново: 20 final prediction replays,
  896 905 465 regenerated feature values, 9 504 raw-history spot checks.
- Current-event exclusion, equal timestamps, future mutations, joins, empty history,
  ratio smoothing, encoder train prefix, SVD cutoff и split boundaries проверены.
- Для нового полного сравнения: 36 независимых сравнений
  LogLoss/AUC/AP, maximum difference 2.22e-16;
  параметры, booster feature names, число деревьев и test population совпали.
- Все unique final boosters нового сравнения повторно предсказали сохраненные predictions;
  признаки для replay рассчитаны заново существующим engine.
- Clean venv установлен по submission requirements, pip check прошел. Notebook full
  выполнен top-to-bottom для обоих dataset; isolated demo работает без src/scripts.
  Исходные full HPO studies целиком заново не оптимизировались: новый experiment не делает HPO.
- Raw files сверены с pinned SHA256. Повторное скачивание всех raw файлов в этом аудите
  не выполнялось; внешняя загрузка из пустого raw directory заново не проверялась.

## Основное сравнение: новый full run, seed42

| dataset | method | inputs | logloss | auc | average_precision | delta_logloss_vs_raw | generation_seconds | fit_seconds | matrix_mib | shared_with |
|---|---|---|---|---|---|---|---|---|---|---|
| kion | raw | 16 | 0.619689 | 0.676487 | 0.513175 | 0.000000 | 0.014441 | 6.006823 | 97.321716 |  |
| kion | manual | 106 | 0.577836 | 0.726984 | 0.586076 | -0.041854 | 23.497935 | 11.967805 | 644.756371 |  |
| kion | manual_trends | 118 | 0.577493 | 0.727001 | 0.585852 | -0.042196 | 23.582419 | 15.452567 | 717.747658 |  |
| kion | auto_fixed | 80 | 0.573697 | 0.730223 | 0.590820 | -0.045993 | 13.984350 | 19.547052 | 486.608582 |  |
| kion | auto_pool | 509 | 0.572512 | 0.731281 | 0.591844 | -0.047177 | 28.714919 | 64.558537 | 3096.047100 |  |
| kion | auto_selected | 144 | 0.572429 | 0.731613 | 0.591841 | -0.047261 | 14.748581 | 29.766906 | 875.895447 |  |
| yambda | raw | 5 | 0.234020 | 0.833212 | 0.976265 | 0.000000 | 0.005182 | 7.330630 | 18.868103 |  |
| yambda | manual | 80 | 0.161425 | 0.936561 | 0.991667 | -0.072595 | 10.036127 | 4.048885 | 301.889648 |  |
| yambda | manual_trends | 92 | 0.161747 | 0.936464 | 0.991652 | -0.072273 | 10.412020 | 4.061994 | 347.173096 |  |
| yambda | auto_fixed | 37 | 0.161770 | 0.936351 | 0.991617 | -0.072250 | 5.603622 | 3.351379 | 139.623962 |  |
| yambda | auto_pool | 447 | 0.160768 | 0.937150 | 0.991775 | -0.073252 | 18.591662 | 13.831724 | 1686.808411 |  |
| yambda | auto_selected | 69 | 0.160820 | 0.936956 | 0.991717 | -0.073200 | 7.445228 | 5.089535 | 260.379822 |  |

Generation seconds относятся к historical features на всех строках. Общий encoder/metadata
preprocessing одинаков и выполняется вне этих измерений. Matrix MiB — nbytes float32 input,
не peak RAM. Fit seconds — только fit, без generation, search, prediction/metrics overhead.
CPU/RAM общего desktop не изолированы; результаты не являются controlled systems benchmark.
Если `shared_with` заполнен, строка показывает тот же measured run и не суммируется второй раз.

- **kion:** наименьший наблюдаемый test LogLoss у `auto_selected`: 0.572429. Auto-selected − manual+trends = -0.005065; 95% user-cluster CI [-0.005543, -0.004556]. Это условный интервал фиксированных моделей, один seed, не оценка всей search procedure.
- **yambda:** наименьший наблюдаемый test LogLoss у `auto_pool`: 0.160768. Auto-selected − manual+trends = -0.000927; 95% user-cluster CI [-0.001638, -0.000301]. Это условный интервал фиксированных моделей, один seed, не оценка всей search procedure.

## Что можно заключить

Фабрика и selection работают: candidate grammar действительно превращается в признаки,
каждый top-K refit оценивается на search, test в этот выбор не входит. Польза автоматизации
процесса не равна доказательству необходимости каждого окна. Train gain при коррелирующих
признаках не является causal importance. В KION 180d эквивалентно all-history, user-item
пары не повторяются; постоянные user-item признаки не должны трактоваться как полезные.

Практический выбор зависит от бюджета: raw проще, но теряет историю; manual легче
объяснять и сопровождать; factory требует больше offline fits. Небольшой selected набор
может облегчить расчет и serving, если сохраняет качество. Точный winner и цены приведены
в таблице, без подбора удобного seed после просмотра test.

## Финальная проверка submission

| Category | Status | Evidence / limitation |
|---|---|---|
| Methodological correctness | PARTIAL | Known holdout; one temporal split |
| Fairness of comparison | PASS | Same target/rows/model parameters/seed; same candidate rows |
| Feature engineering quality | PASS | 27 original definitions reused without AST changes |
| Data leakage prevention | PARTIAL | Engine tests PASS; label/metadata availability assumed |
| Experimental evidence | PASS | Original boosters and fresh full comparison independently replayed |
| Interpretation | PASS | No official benchmark, unique-window or unconditional-significance claims |
| Reproducibility | PASS | Full notebook in clean venv, isolated demo; current macOS only |
| Clarity | PASS | Four submission files; ten notebook sections |

```text
FINAL_SUBMISSION/
├── feature_engineering_comparison.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

После запуска notebook создает ignored `.runs/` с реальными моделями/predictions/CSV.
Данные находятся снаружи либо загружаются в ignored `.data/`. Эти generated files не
являются дополнительными исходными файлами submission.
Записанный notebook содержит outputs полного запуска; default повторного Run All — full.
Это явно обозначено в настройках, таблице и README; demo не называется full reproduction.

## Оставшиеся ограничения

Известный holdout; один temporal split и один seed нового контролируемого сравнения;
предыдущие три seeds проверяют фиксированные old selections, а не новый search.
Production availability labels/metadata не подтверждена. KION — conditional aggregate
classification, Yambda — reaction-conditioned classification без полного listen/unlike контекста.
Proxy ranking старого проекта не является full-catalog benchmark и не используется
в качестве основной метрики нового сравнения. Стабильность и причинность улучшений
нельзя обобщать за пределы этих условий. Перенос на чистую OS/container не проверен.

Подробные evidence: `results/audit/`.
Новые результаты: `results/full`. Полный исходный проект остается рядом с submission.


## Проверки публикации

Основной notebook: 21/21 code cells, full, обучение на обоих датасетах в чистом venv. Дополнительный: 47/47, saved mode для KION и Yambda; модельное обучение там не повторялось. В обоих есть выполняемая установка недостающих зависимостей. Статус проекта: **READY WITH LIMITATIONS**.
