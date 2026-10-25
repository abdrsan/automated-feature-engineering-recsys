> Исторический документ исходного исследования. Текущее сравнение методов и статус публикации — в [README](README.md) и [основном notebook](FINAL_SUBMISSION/feature_engineering_comparison.ipynb). Старые и новые эксперименты используют разные правила выбора наборов и не подменяют друг друга.

# Исправления по независимому аудиту

Исходные findings и score сохранены в [AUDIT_REPORT.md](AUDIT_REPORT.md).
Метрики не редактировались вручную: обе полные study переобучены с новыми SQLite и models.
KION переиспользовал только content-addressed matrices с теми же data/spec/engine hashes;
полная независимая регенерация затем проверила их значения. Yambda выполнен в чистом venv
с пустым output. Исходные результаты — artifacts/pre_audit. Частичный промежуточный
run v2 сохранен в artifacts/intermediate_v2 и не включен в итоговые таблицы.
Generation profiles сохраняют измерение при создании matrix, а не текущий elapsed
cache hit; model fit timings измерены заново. Сравнение стоимости не объявляется
контролируемым end-to-end benchmark.
Повторный test уже наблюдался: это post-audit re-evaluation, не новый blind confirmation.

## M1 — selection manual reference

- **Problem:** лучшая проверенная manual ablation могла не стать baseline.
- **Root cause:** hardcoded выбор только manual/manual_plus_trend.
- **Change:** минимум search LogLoss среди всех выполненных manual stages; tie-break по
  числу inputs и имени; pool и selected source зафиксированы в lock до audit/test.
- **Files:** scripts/run_study.py, scripts/verify_datasets.py, audit/test_independent.py.
- **Old result:** регрессионный тест FAIL; Yambda игнорировал более сильный search arm.
- **New result:** регрессионный тест PASS; фактический минимум всех stages проверен
  в новых artifacts. Ни один test score не участвует в выборе.
- **Impact:** главный контроль усилен; прежние выводы о величине выигрыша не переносятся
  автоматически на исправленный baseline.

| Dataset | Old manual source | New manual source | Old auto source | New auto source |
| --- | --- | --- | --- | --- |
| kion | manual_plus_trend | manual_plus_trend | auto_adaptive_top64 | auto_adaptive_all |
| yambda | manual_plus_trend | plus_metadata_relations | auto_adaptive_top32 | auto_adaptive_top32 |

## M2 — отсутствующий EDA artifact в clean run

- **Problem:** report generator требовал eda_full.json, который CLI не создавал.
- **Root cause:** прежний EDA script был привязан к MovieLens; дополнительные EDA были
  сделаны до опубликованного CLI lifecycle.
- **Change:** study_statistics и автоматическое сохранение eda_full после final evaluation.
- **Files:** src/data.py, scripts/run_study.py, scripts/make_full_notebook.py, README.md.
- **Old result:** свежий original smoke создал final_results, но eda_full отсутствовал.
- **New result:** clean standalone smoke и обе новые полные study создают EDA; report
  generator запускается по опубликованной последовательности без ручных файлов.
- **Impact:** устранен скрытый шаг воспроизводимости отчетов.

## M3 — invalidation cache/model resume

- **Problem:** изменение реализации не меняло cache identity.
- **Root cause:** учитывались только data/specs, параметры и train_start.
- **Change:** FEATURE_ENGINE_VERSION, protocol version и canonical AST signatures
  вычислений/поиска/fit включены в cache, manifest и model specs. Несовместимость
  вызывает ошибку, требуется новый output.
- **Files:** src/provenance.py, src/feature_engine.py, src/experiments.py,
  scripts/run_study.py, scripts/make_full_notebook.py.
- **Old result:** тест с изменением версии возвращал устаревшую matrix — FAIL.
- **New result:** повторное вычисление при смене версии — PASS; CLI и inline notebook
  используют согласованные signatures.
- **Impact:** исправление движка больше не маскируется старыми matrices/models.

## M4 — TPE resume state

- **Problem:** возобновление SQLite study заново запускало RNG sampler.
- **Root cause:** seed повторно задавался, но sampler object не сохранялся.
- **Change:** атомарный local sampler checkpoint с hash trial history, загрузка только
  собственного state; mismatch после аварии запрещает silent resume.
- **Files:** src/model.py, audit/test_independent.py, README.md.
- **Old result:** в исходном interrupted KION 20 COMPLETE, но есть повторные параметры;
  непрерывный и interrupted search не обязаны совпадать.
- **New result:** независимый тест 14 непрерывных trials против 6+8 resumed trials
  дает идентичные параметры и objective history. В новых полных runs — по 20 COMPLETE
  manual/auto trials на каждый dataset, то есть 80 фактических completed evaluations.
- **Impact:** корректный checkpoint resume воспроизводим. Hard kill со stale state
  требует нового run; старая sequence не объявляется воспроизведенной.

## M5 — независимость fixed/shared controls

- **Problem:** pruning широкого adaptive пространства выполнялся до ограничения окон
  control. Разрешенный fixed feature мог исчезнуть вместе с adaptive-only alias.
- **Root cause:** train-equivalent all-history/180d (и другие aliases) выбирались в
  общем pool; последующий window filter мог удалить единственного retained alias.
- **Change:** restrict fixed/shared pool FIRST, затем отдельные pruning и fit; shared
  gain ranking считается собственным booster. Profiles/reasons сохраняются отдельно.
  Протокол поднят до audit-v3; частичный v2 остановлен и сохранен отдельно.
- **Files:** src/feature_selection.py, scripts/run_study.py, src/provenance.py,
  audit/test_independent.py, scripts/verify_datasets.py, reports и standalone notebook.
- **Old result:** в KION 22 all-history aliases удалены в пользу 180d; старый
  fixed filter теряет обе колонки. Независимый toy test воспроизводит потерю 7d alias.
- **New result:** regression PASS; KION fixed pool восстановил 20 inputs; у controls собственные разрешенные и pruned pools,
  собственные модели и реальные validation scores. Итоговый выбор повторен с нуля.
- **Impact:** вывод о качестве window search больше не опирается на ослабленные controls.
  Положительный эффект индивидуального окна этим все равно не доказан.

## M6 — раскрытие повторного использования test

- **Problem:** исправления оцениваются на уже известном holdout.
- **Root cause:** независимый аудит и исходное исследование уже прочитали test scores.
- **Change:** evaluation_status явно указывает post-audit re-evaluation; README, REPORT,
  notebook и этот отчет не называют повторный запуск новым blind confirmation.
- **Files:** scripts/run_study.py, scripts/write_dataset_report.py,
  scripts/make_full_notebook.py, README.md, FINAL_AUDIT.md.
- **Old result:** исходные значения сохранены; их неизвестность восстановить нельзя.
- **New result:** все новые choices основаны на search, раскрытие test отражено явно.
- **Impact:** исправлена интерпретация; для независимого подтверждения нужен новый holdout.

## Дополнительные исправления

Illustrative SUM SQL исправлен через COALESCE: пустая история теперь дает 0, как Polars.
Nullable entity keys используют IS NOT DISTINCT FROM, как nulls_equal=True в движке.
Независимые SQLite regressions для empty SUM и null keys первоначально падали, затем прошли. На trained matrices
это не влияет: реальный backend всегда был Polars.
Temporal split теперь явно отклоняет unsorted/null/empty timestamps. Baseline categorical
получил отдельный final test arm. Независимые audit tests включены в default pytest.
README, REPORT, task checklist и notebook обновлены по реальным post-audit outputs.
Окружение установлено заново в отдельном venv; pip check прошел. Все семь исходных
файлов повторно скачаны из pinned URLs и совпали по SHA256.

## R1 — воспроизводимость независимого bootstrap

- **Problem:** при повторной проверке доверительный интервал немного менялся при том же seed.
- **Root cause:** Polars не гарантирует порядок групп после group_by. Random draw выбирал
  индексы групп, поэтому одинаковым индексам могли соответствовать разные пользователи.
- **Change:** перед bootstrap группы сортируются по user_id; добавлен независимый тест
  с расчетом ожидаемого интервала и обратным порядком исходных строк.
- **Files:** audit/run_checks.py, audit/test_recheck.py.
- **Old result:** KION independent CI менялся между повторениями; point estimate совпадал.
- **New result:** оба интервала пересчитаны из реальных predictions и воспроизводятся
  в отдельных процессах. Точные before/after значения — в [RECHECK.md](RECHECK.md).
- **Impact:** исправлен только диагностический bootstrap аудита. Primary metrics,
  основной bootstrap, model selection, обученные модели и predictions не менялись.

## Old result / New result

До исправлений, mean test между тремя seeds:

| Dataset | Model | Seeds | Inputs | AUC | LogLoss | Seed SD | Proxy NDCG@10 | Proxy Recall@10 | Mean fit s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kion | automatic | 3 | 80 | 0.729059 | 0.574244 | 0.000238 | 0.873740 | 0.999849 | 13.378899 |
| kion | manual | 3 | 118 | 0.726828 | 0.577868 | 0.000489 | 0.872576 | 0.999822 | 13.022304 |
| yambda | automatic | 3 | 37 | 0.936790 | 0.161150 | 0.000440 | 0.805388 | 0.969028 | 4.951705 |
| yambda | manual | 3 | 92 | 0.936545 | 0.161658 | 0.000150 | 0.805187 | 0.968729 | 6.995362 |

После исправлений; warmup/categorical arms имеют один seed:

| Dataset | Model | Seeds | Inputs | AUC | LogLoss | Seed SD | Proxy NDCG@10 | Proxy Recall@10 | Mean fit s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kion | auto_plus_svd | 1 | 513 | 0.730918 | 0.573067 | — | 0.873759 | 0.999849 | 77.321136 |
| kion | auto_warmup | 1 | 509 | 0.730356 | 0.572847 | — | 0.873770 | 0.999849 | 110.930937 |
| kion | automatic | 3 | 509 | 0.730904 | 0.572533 | 0.000238 | 0.874282 | 0.999858 | 139.102019 |
| kion | categorical | 1 | 16 | 0.676487 | 0.619689 | — | 0.867477 | 0.999876 | 12.757742 |
| kion | manual | 3 | 118 | 0.726552 | 0.577981 | 0.000423 | 0.872135 | 0.999831 | 25.336135 |
| kion | manual_warmup | 1 | 118 | 0.726888 | 0.578137 | — | 0.872542 | 0.999822 | 20.239038 |
| yambda | auto_plus_svd | 1 | 41 | 0.936640 | 0.161202 | — | 0.800673 | 0.968864 | 4.783509 |
| yambda | auto_warmup | 1 | 37 | 0.936878 | 0.160898 | — | 0.804188 | 0.968434 | 4.878745 |
| yambda | automatic | 3 | 37 | 0.936790 | 0.161150 | 0.000440 | 0.805388 | 0.969028 | 5.857563 |
| yambda | categorical | 1 | 5 | 0.833212 | 0.234020 | — | 0.779924 | 0.966758 | 12.399641 |
| yambda | manual | 3 | 65 | 0.936045 | 0.163072 | 0.000446 | 0.809052 | 0.968894 | 7.198762 |
| yambda | manual_warmup | 1 | 65 | 0.936390 | 0.163300 | — | 0.802994 | 0.969447 | 5.505751 |

Изменения чисел нельзя приписывать только M1: весь pipeline повторен с непрерывным
воспроизводимым HPO вместо прежнего interrupted KION search. Primary auto/manual
сравниваются внутри одного нового run; before/after — diagnostic, а не causal ablation.
Yambda manual, лучший на search среди проверенных stages, оказался хуже прежнего полного
manual на test. Это не основание менять reference после test: результат показывает
нестабильность validation выбора. Отдельные старые test значения остаются в таблице выше.
