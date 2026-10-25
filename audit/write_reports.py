"""Generate post-fix reports from actual audit evidence and experiment ledgers."""
import json
from pathlib import Path

import polars as pl

from scripts.write_dataset_report import result_table


def table(rows, columns):
    return result_table(rows, [(key,label) for key,label in columns])


def main():
    old, new, summaries, comparison, windows, controls = {}, {}, [], [], [], []
    for name in ["kion","yambda"]:
        def load(parent, filename):
            return json.loads((Path(parent)/name/filename).read_text())
        old[name]=load("artifacts/pre_audit","locked_selection.json")
        new[name]=load("artifacts","locked_selection.json")
        checked=load("artifacts/audit/after","summary.json")
        search=load("artifacts/audit/after","search_replay_summary.json")
        assert checked["status"]==search["status"]=="PASS"
        summaries.append({"dataset":name,**checked,**search})
        control_rows=pl.read_csv(Path("artifacts")/name/"audit_controls.csv")
        development=pl.read_csv(Path("artifacts")/name/"experiments.csv").filter(
            (pl.col("split")=="validation_search") & (pl.col("seed")==42))
        for row in control_rows.to_dicts():
            search_row=development.filter(pl.col("experiment_name")==row["experiment_name"]).row(0,named=True)
            controls.append({"dataset":name,"model":row["experiment_name"],"inputs":row["n_features"],
                             "search_logloss":search_row["logloss"],"audit_logloss":row["logloss"]})
        for stage,parent in [("before","artifacts/pre_audit"),("after","artifacts")]:
            frame=pl.read_csv(Path(parent)/name/"experiments.csv").filter(pl.col("split")=="test")
            values=frame.group_by("experiment_name").agg(pl.len().alias("seeds"),pl.col("n_features").first(),
                pl.col("auc").mean(),pl.col("logloss").mean(),pl.col("logloss").std().alias("sd_logloss"),
                pl.col("ndcg_10").mean(),pl.col("recall_10").mean(),pl.col("training_time").mean().alias("fit_seconds")).sort("experiment_name")
            comparison.extend({"stage":stage,"dataset":name,**row} for row in values.to_dicts())
        selected=pl.read_csv(Path("artifacts")/name/"selected_features.csv")
        for entity in sorted(selected["entity"].unique()):
            subset=selected.filter(pl.col("entity")==entity)
            window_order=lambda w: 100000 if w=="all_history" else int(w[:-1])
            reported=sorted(subset["window"].unique().to_list(),key=window_order)
            verified=sorted({"all_history" if json.loads(r["spec_json"])["window_days"] is None else str(json.loads(r["spec_json"])["window_days"])+"d" for r in subset.to_dicts()},key=window_order)
            assert reported==verified
            windows.append({"dataset":name,"family":entity,"reported":", ".join(reported),"verified":", ".join(verified),"status":"PASS"})
    pl.DataFrame(comparison).write_csv("artifacts/audit/before_after_metrics.csv")
    pl.DataFrame(windows).write_csv("artifacts/audit/selected_windows_verified.csv")
    pl.DataFrame(controls).write_csv("artifacts/audit/window_controls_verified.csv")
    cols=[("dataset","Dataset"),("experiment_name","Model"),("seeds","Seeds"),("n_features","Inputs"),
          ("auc","AUC"),("logloss","LogLoss"),("sd_logloss","Seed SD"),("ndcg_10","Proxy NDCG@10"),
          ("recall_10","Proxy Recall@10"),("fit_seconds","Mean fit s")]
    before_table=table([r for r in comparison if r["stage"]=="before" and r["experiment_name"] in ["manual","automatic"]],cols)
    after_table=table([r for r in comparison if r["stage"]=="after"],cols)
    changes=table([{"dataset":n,"before":old[n]["manual_name"],"after":new[n]["manual_name"],
                    "before_auto":old[n]["auto_name"],"after_auto":new[n]["auto_name"]} for n in old],
        [("dataset","Dataset"),("before","Old manual source"),("after","New manual source"),("before_auto","Old auto source"),("after_auto","New auto source")])
    fixes=f"""# Исправления по независимому аудиту

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

{changes}

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

{before_table}

После исправлений; warmup/categorical arms имеют один seed:

{after_table}

Изменения чисел нельзя приписывать только M1: весь pipeline повторен с непрерывным
воспроизводимым HPO вместо прежнего interrupted KION search. Primary auto/manual
сравниваются внутри одного нового run; before/after — diagnostic, а не causal ablation.
Yambda manual, лучший на search среди проверенных stages, оказался хуже прежнего полного
manual на test. Это не основание менять reference после test: результат показывает
нестабильность validation выбора. Отдельные старые test значения остаются в таблице выше.
"""
    Path("FIXES.md").write_text(fixes)
    stats_table=table(summaries,[("dataset","Dataset"),("rows","Rows"),("spot_checks","Direct feature checks"),
        ("regenerated_feature_cells","Regenerated cells"),("model_prediction_replays","Final model replays"),
        ("checkpoint_evaluations","Dev/audit replays"),("best_hpo_trials_refitted","Best HPO refits")])
    verdicts=[]
    for n in old:
        rows={r["experiment_name"]:r for r in comparison if r["stage"]=="after" and r["dataset"]==n}
        delta=rows["automatic"]["logloss"]-rows["manual"]["logloss"]
        bootstrap=json.loads((Path("artifacts/audit/after")/n/"independent_bootstrap.json").read_text())
        lower,upper=bootstrap["ci_95"]
        text="поддерживает меньший conditional LogLoss automatic" if upper<0 else "не доказывает устойчивое превосходство automatic"
        verdicts.append(f"- **{n}:** mean ΔLogLoss auto−manual={delta:+.6f}, относительное снижение={-100*delta/rows['manual']['logloss']:.3f}%. Независимый user-cluster bootstrap seed42: [{lower:+.6f}, {upper:+.6f}], 300 repeats; {text}. CI условен на fixed models, не на всю search procedure.")
    final=f"""# Final Audit

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

{changes}

## Evidence of independent recheck

{stats_table}

Final predictions повторно вычислены из booster files; AUC/LogLoss/NDCG/Recall/MAP
пересчитаны независимыми формулами. Каждый development/audit checkpoint повторно оценен,
model feature names сверены с specs и hashes. CSV trials сверены с actual SQLite,
лучшие manual/auto HPO trials заново fit-нуты и objective values воспроизведены.
Raw dataset counts, missing values, categorical cardinalities, per-entity distributions,
time boundaries и NaN rates сохранены в artifacts/audit/after/{{kion,yambda}}.

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

{after_table}

""" + "\n".join(verdicts) + f"""

## Selected windows audit

{table(windows,[("dataset","Dataset"),("family","Family"),("reported","Reported primary windows"),("verified","Verified spec windows"),("status","Status")])}

Доказано, что окна являются настоящими кандидатами/inputs, а pool/feature budgets
оцениваются на validation. Это не доказательство уникальной оптимальности отдельных
окон: модель может сохранить широкий pool. Если selected=all-pruned pool, следует
говорить «сохранены все непостоянные кандидаты», а не «найден компактный оптимальный
набор». Original и corrected search могут выбирать разные наборы при близком loss.

## Window controls: factory versus adaptive windows

{table(controls,[("dataset","Dataset"),("model","Source selected on search"),("inputs","Inputs"),("search_logloss","Search LogLoss"),("audit_logloss","Audit LogLoss")])}

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
"""
    Path("FINAL_AUDIT.md").write_text(final)
    checklist=Path("TASK_CHECKLIST.md").read_text()
    start=next(checklist.index(header) for header in ["## Проверенное состояние после аудита",
        "## Что действительно показали эксперименты"] if header in checklist)
    checklist=checklist[:start]+"""## Проверенное состояние после аудита

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
"""
    Path("TASK_CHECKLIST.md").write_text(checklist)
    print("Wrote FIXES.md, FINAL_AUDIT.md and updated TASK_CHECKLIST.md from actual evidence")


if __name__=="__main__":
    main()
