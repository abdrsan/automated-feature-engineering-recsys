> Исторический документ исходного исследования. Текущее сравнение методов и статус публикации — в [README](README.md) и [основном notebook](FINAL_SUBMISSION/feature_engineering_comparison.ipynb). Старые и новые эксперименты используют разные правила выбора наборов и не подменяют друг друга.

# Final Audit

Повторный аудит после исправлений, 16 октября 2026. Исходный отчет и score сохранены;
данные, models и результаты до исправлений доступны в artifacts/pre_audit.

## Before / After score

| Категория | Before | After |
|---|---:|---:|
| Problem coverage | 13/15 | 14/15 |
| Data & temporal correctness | 18/20 | 18/20 |
| Manual baseline quality | 4/10 | 8/10 |
| Automated feature engineering | 15/20 | 18/20 |
| Evaluation correctness | 12/15 | 13/15 |
| Research depth | 8/10 | 8/10 |
| Reproducibility | 1/5 | 4/5 |
| Code quality | 4/5 | 4/5 |
| **TOTAL** | **75/100 — C** | **87/100 — B** |

Оценка — экспертная рубрика, а не статистическая метрика. Ее рост отражает устранение
methodological/reproducibility errors и проверенные evidence; положительный auto gain
не является условием получения хорошей исследовательской оценки.

## Fixed issues

M1: manual reference выбирается по search среди всех проверенных manual stages.
M2: clean CLI создает full EDA и полный report lifecycle.
M3: cache/manifest/model specs проверяют версии и AST signatures.
M4: TPE state сохраняется и сопоставляется с trials; resume regression проходит.
M5: fixed/shared controls независимо prune/fit внутри разрешенного pool.
M6: повторная оценка явно названа post-audit, новый blind confirmation не заявляется.
Полные изменения и causal ограничения сравнения before/after — в [FIXES.md](FIXES.md).

## Requirement recheck

Полная исходная matrix находится в [AUDIT_REPORT.md](AUDIT_REPORT.md). После исправлений
изменились следующие статусы; остальные computational PASS подтверждены повторными
checks ниже, optional NOT IMPLEMENTED остаются без изменения.

| Requirement | Before | After | Evidence / qualification |
|---|---|---|---|
| Strong manual baseline | PARTIAL | PASS | 20 HPO trials; reference min среди всех реально обученных stages |
| Automatic window search | PARTIAL | PASS | candidate specs, independent fixed/shared pools, реальные refits и scores |
| README / REPORT consistency | PARTIAL | PASS | artifact-driven report, точные команды и explicit limitations |
| Reproducibility | PARTIAL | PASS | fresh source downloads, clean venv, полный run, isolated notebook; та же OS |
| Test isolation of corrected choices | PARTIAL | PARTIAL | Код использует только search, но test был известен до исправлений |
| Production label/metadata availability | PARTIAL | PARTIAL | strict event-time engine проверен, available_at отсутствует |
| Full-catalog recommendation | NOT APPLICABLE | NOT APPLICABLE | реализовано conditional scoring и observed-query proxy |
| Full GNN/Transformer, SHAP/permutation | NOT IMPLEMENTED | NOT IMPLEMENTED | optional alternatives, performance не заявлен |

| Dataset | Old manual source | New manual source | Old auto source | New auto source |
| --- | --- | --- | --- | --- |
| kion | manual_plus_trend | manual_plus_trend | auto_adaptive_top64 | auto_adaptive_all |
| yambda | manual_plus_trend | plus_metadata_relations | auto_adaptive_top32 | auto_adaptive_top32 |

## Evidence of independent recheck

| Dataset | Rows | Direct feature checks | Regenerated cells | Final model replays | Dev/audit replays | Best HPO refits |
| --- | --- | --- | --- | --- | --- | --- |
| kion | 1594519 | 8176 | 814799209 | 10 | 44 | 2 |
| yambda | 989232 | 1328 | 82106256 | 10 | 44 | 2 |

Final predictions повторно вычислены из booster files; AUC/LogLoss/NDCG/Recall/MAP
пересчитаны независимыми формулами. Каждый development/audit checkpoint повторно оценен,
model feature names сверены с specs и hashes. CSV trials сверены с actual SQLite,
лучшие manual/auto HPO trials заново fit-нуты и objective values воспроизведены.
Raw dataset counts, missing values, categorical cardinalities, per-entity distributions,
time boundaries и NaN rates сохранены в artifacts/audit/after/{kion,yambda}.

**88 tests PASS** в рабочем и в чистом venv; pip check — PASS. Seven fresh source
downloads совпали с использованными SHA256. Самостоятельный notebook прошел Run All
в изолированной папке без src/scripts; saved-mode также проверяет signatures.
Точные ledgers, difference tables, logs и environment freeze — в artifacts/audit.
Clean venv использует текущую OS и ее системные библиотеки; перенос на чистый container
или другую платформу этим аудитом не проверен. Версии всех установленных Python dependencies
сохранены; AST signatures не заменяют управление версиями внешних библиотек/протокола.
Проверенная машина: macOS arm64, 24 GiB RAM, 4 threads на LightGBM; snapshot —
[hardware.json](artifacts/audit/hardware.json). Это общий desktop, CPU/RAM не изолированы;
неизвестная модель CPU не подставляется по предположению.

Дополнительная проверка 16 октября: [RECHECK.md](RECHECK.md). Повторены full feature
regeneration, model/metric replay, лучшие HPO trials и оба standalone smoke runs.
В независимом bootstrap зафиксирован порядок пользователей; основной pipeline не менялся.

## Final metrics

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

- **kion:** mean ΔLogLoss auto−manual=-0.005448, относительное снижение=0.943%. Независимый user-cluster bootstrap seed42: [-0.005613, -0.004795], 300 repeats; поддерживает меньший conditional LogLoss automatic. CI условен на fixed models, не на всю search procedure.
- **yambda:** mean ΔLogLoss auto−manual=-0.001921, относительное снижение=1.178%. Независимый user-cluster bootstrap seed42: [-0.004221, -0.001006], 300 repeats; поддерживает меньший conditional LogLoss automatic. CI условен на fixed models, не на всю search procedure.

## Selected windows audit

| Dataset | Family | Reported primary windows | Verified spec windows | Status |
| --- | --- | --- | --- | --- |
| kion | author | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 180d, all_history | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 180d, all_history | PASS |
| kion | category | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 180d, all_history | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 180d, all_history | PASS |
| kion | global | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 180d, all_history | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 180d, all_history | PASS |
| kion | item | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, all_history | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, all_history | PASS |
| kion | item_gender | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 180d, all_history | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 180d, all_history | PASS |
| kion | user | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, all_history | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, all_history | PASS |
| kion | user_author | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, all_history | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, all_history | PASS |
| kion | user_category | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, all_history | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, all_history | PASS |
| yambda | user | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 180d, all_history | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 180d, all_history | PASS |
| yambda | user_author | 60d, 90d, 180d, all_history | 60d, 90d, 180d, all_history | PASS |
| yambda | user_item | 180d, all_history | 180d, all_history | PASS |

Доказано, что окна являются настоящими кандидатами/inputs, а pool/feature budgets
оцениваются на validation. Это не доказательство уникальной оптимальности отдельных
окон: модель может сохранить широкий pool. Если selected=all-pruned pool, следует
говорить «сохранены все непостоянные кандидаты», а не «найден компактный оптимальный
набор». Original и corrected search могут выбирать разные наборы при близком loss.

## Window controls: factory versus adaptive windows

| Dataset | Source selected on search | Inputs | Search LogLoss | Audit LogLoss |
| --- | --- | --- | --- | --- |
| kion | auto_fixed_windows_top64 | 80 | 0.590014 | 0.590241 |
| kion | auto_adaptive_all | 509 | 0.589198 | 0.589466 |
| kion | shared_windows_control | 224 | 0.589959 | 0.591415 |
| yambda | auto_fixed_windows_top32 | 37 | 0.172227 | 0.171955 |
| yambda | auto_adaptive_top32 | 37 | 0.171809 | 0.170887 |
| yambda | shared_windows_control | 37 | 0.173403 | 0.171053 |

Fixed control уже автоматизирует relations, aggregations и feature budget при 7/30/all
окнах. Его улучшение относительно manual нельзя приписать адаптивным windows.
Сравнение adaptive/fixed/shared выше проверяет более узкую гипотезу об окнах; отрицательный
результат следует сохранять. Эти arms оценены на audit после lock, не используются для
смены final модели; их сравнение не изолирует одну колонку при прочих равных.
Все window controls имеют seed42. Состав и порядок колонок, subsampling и early stopping
также влияют на результат; отдельная устойчивость control comparisons по seeds не проверена.

## Leakage and scientific correctness

**PASS для point-in-time engine при declared availability assumptions.** Current events,
timestamp ties и future outcomes исключены; independent oracle и full cache regeneration
согласованы. Отсутствие production available_at не исправлено выдуманным timestamp.
Warmup SVD basis и user profiles проверены отдельно; warmup rows исключаются из всех
matched controls. Использование более ранних outcomes внутри test — online replay,
не future leakage. Нужны реальные label delays для production.

Manual и auto имеют одинаковый initial HPO budget: 20 trials; feature search auto
дополнительно стоит fitting rounds, что явно учитывается. Manual HPO сначала делается
на полном manual, а stages используют те же параметры; это reasonably tuned контроль,
но не global optimum всех feature/hyperparameter subsets.

## Remaining issues and missing evidence

- **P1, невозможно исправить кодом:** test уже был раскрыт. Новые сравнения не являются
  независимым blind replication; следующий защитный эксперимент требует нового holdout.
- **P2:** один global temporal split, нет rolling-origin stability. Три seeds фиксированного
  feature set не оценивают устойчивость всего structured search.
- **P2:** выбранный Yambda manual выиграл search у полного manual, но проиграл ему на test;
  новая величина auto gain частично зависит от этой нестабильности reference selection.
  Оба исходных числа сохранены, test не используется для возврата к удобному baseline.
- **P2:** causal необходимость индивидуальных windows не изолирована; shared control
  меняет состав features. Выбранные окна нельзя объявлять физическими сроками интереса.
- KION — агрегированная watch-percentage classification, Yambda — like/dislike среди
  explicit reactions без полного listen/unlike context. Это не impression CTR/retrieval.
- Static metadata assumed, multidimensional relations упрощены; нет production history
  доступности. Duplicate Yambda reactions оставлены как raw observations.
- Generation profiles KION относятся к созданию reused matrices. Фактические новые
  fit times доступны, но cache reuse и системная нагрузка ограничивают runtime comparison.
- Ranking лишь observed user×timestamp mixed-label proxy, без full-catalog candidates.
  Малые группы объясняют высокий Recall@10. Official benchmark не воспроизведен.
- SHAP/permutation, full GNN/Transformer и distributed 1B+ не выполнялись. Они optional;
  источники и scalability обсуждаются, performance claims не присваиваются.

## Can this work be defended?

Да, как исследование **structured temporal feature factory для conditional scoring**,
с проверенными расчетами и воспроизводимыми artifacts. Нельзя защищать его как
универсальное превосходство auto над любым manual baseline, доказательство оптимальности
окон или production/full-catalog recommender. Отрицательные/неубедительные результаты
нужно показывать вместе с положительными, не выбирать лучший seed после test.

## Final verdict

**B — Good solution, 87/100.** Все устранимые подтвержденные P1 исправлены и проверены
повторными runs. Подтвержденных P0 нет. Новая blind оценка, временная устойчивость и
production availability остаются явно незакрытыми исследовательскими вопросами.
