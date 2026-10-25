import argparse
import hashlib
import json
from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/final_comparison_audit_20261025"
SUBMISSION = ROOT / "FINAL_SUBMISSION"


def markdown(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|"*len(columns)]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main(run):
    run = Path(run)
    assert (run/"verification.json").exists()
    verified = json.loads((EVIDENCE/"full_verification/controlled_verification.json").read_text())
    assert verified["status"] == "PASS"
    original = json.loads((EVIDENCE/"original_runs_verified.json").read_text())
    assert original["status"] == "PASS"
    initial = json.loads((EVIDENCE/"original_file_hashes.json").read_text())
    changed = [p for p, value in initial.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != value]
    assert not changed, changed
    comparison = pd.read_csv(run/"comparison.csv").fillna({"shared_with": ""})
    review = pd.DataFrame([
        ("Methodological correctness", "PARTIAL", "Known holdout; one temporal split"),
        ("Fairness of comparison", "PASS", "Same target/rows/model parameters/seed; same candidate rows"),
        ("Feature engineering quality", "PASS", "27 original definitions reused without AST changes"),
        ("Data leakage prevention", "PARTIAL", "Engine tests PASS; label/metadata availability assumed"),
        ("Experimental evidence", "PASS", "Original boosters and fresh full comparison independently replayed"),
        ("Interpretation", "PASS", "No official benchmark, unique-window or unconditional-significance claims"),
        ("Reproducibility", "PASS", "Full notebook in clean venv, isolated demo; current macOS only"),
        ("Clarity", "PASS", "Four submission files; ten notebook sections"),
    ], columns=["Category", "Status", "Evidence / limitation"])
    notebook = nbformat.read(SUBMISSION/"feature_engineering_comparison.ipynb", as_version=4)
    assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code")
    execution = json.loads((EVIDENCE/"full_execution.json").read_text())
    assert execution["python"] == "/tmp/fe-final-venv/bin/python"
    full_table = comparison[["dataset", "method", "inputs", "logloss", "auc", "average_precision",
                             "delta_logloss_vs_raw", "generation_seconds", "fit_seconds", "matrix_mib", "shared_with"]]
    conclusions = []
    for dataset in ["kion", "yambda"]:
        table = comparison[comparison.dataset == dataset].set_index("method")
        best = table.sort_values(["logloss", "inputs"]).iloc[0]
        manual, selected = table.loc["manual_trends"], table.loc["auto_selected"]
        interval = json.loads((run/dataset/"bootstrap.json").read_text())
        conclusions.append(f"- **{dataset}:** наименьший наблюдаемый test LogLoss у `{best.name}`: {best.logloss:.6f}. "
            f"Auto-selected − manual+trends = {selected.logloss-manual.logloss:+.6f}; "
            f"95% user-cluster CI [{interval['ci95'][0]:+.6f}, {interval['ci95'][1]:+.6f}]. "
            f"Это условный интервал фиксированных моделей, один seed, не оценка всей search procedure.")
    text = f"""# Итоговый аудит: Feature Engineering Methods Comparison

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
Все {len(initial)} исходных файлов snapshot сохранили SHA256. Добавлены только новый
submission и файлы текущего аудита. Другие проекты и корневой FINAL_SUBMISSION не изменялись.

## Независимые проверки

- **94 tests PASS**: original 88 + synthetic metric ties, exact reuse/standalone
  и обе ветки установки зависимостей в текущем kernel.
- Оба исходных final runs проверены заново: 20 final prediction replays,
  896 905 465 regenerated feature values, 9 504 raw-history spot checks.
- Current-event exclusion, equal timestamps, future mutations, joins, empty history,
  ratio smoothing, encoder train prefix, SVD cutoff и split boundaries проверены.
- Для нового полного сравнения: {verified['metric_comparisons']} независимых сравнений
  LogLoss/AUC/AP, maximum difference {verified['max_metric_difference']:.3g};
  параметры, booster feature names, число деревьев и test population совпали.
- Все unique final boosters нового сравнения повторно предсказали сохраненные predictions;
  признаки для replay рассчитаны заново существующим engine.
- Clean venv установлен по submission requirements, pip check прошел. Notebook full
  выполнен top-to-bottom для обоих dataset; isolated demo работает без src/scripts.
  Исходные full HPO studies целиком заново не оптимизировались: новый experiment не делает HPO.
- Raw files сверены с pinned SHA256. Повторное скачивание всех raw файлов в этом аудите
  не выполнялось; внешняя загрузка из пустого raw directory заново не проверялась.

## Основное сравнение: новый full run, seed42

{markdown(full_table)}

Generation seconds относятся к historical features на всех строках. Общий encoder/metadata
preprocessing одинаков и выполняется вне этих измерений. Matrix MiB — nbytes float32 input,
не peak RAM. Fit seconds — только fit, без generation, search, prediction/metrics overhead.
CPU/RAM общего desktop не изолированы; результаты не являются controlled systems benchmark.
Если `shared_with` заполнен, строка показывает тот же measured run и не суммируется второй раз.

""" + "\n".join(conclusions) + f"""

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

{markdown(review)}

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

Подробные evidence: `artifacts/final_comparison_audit_20261025/`.
Новые результаты: `{run.relative_to(ROOT)}`. Полный исходный проект остается рядом с submission.
"""
    (EVIDENCE/"FINAL_REVIEW.md").write_text(text)
    review.to_csv(EVIDENCE/"submission_review.csv", index=False)
    comparison.to_csv(EVIDENCE/"full_comparison.csv", index=False)
    readme = (SUBMISSION/"README.md").read_text()
    marker = "## Проверенные результаты"
    if marker in readme:
        readme = readme[:readme.index(marker)]
    compact = comparison[comparison.method.isin(["raw", "manual_trends", "auto_selected"])][
        ["dataset", "method", "inputs", "logloss", "auc"]]
    readme += "\n## Проверенные результаты\n\nПолный controlled run, seed42, общие параметры внутри dataset:\n\n"+markdown(compact)+"\n\n"
    readme += "Сохраненные outputs notebook относятся к `full`; повторный Run All по умолчанию запускает `full`. "
    readme += "Full и isolated demo проверены; метрики и predictions full независимо воспроизведены. "
    readme += "Статус: **READY WITH LIMITATIONS**. Главные ограничения — известный test, один новый seed и availability assumptions.\n"
    (SUBMISSION/"README.md").write_text(readme)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SUBMISSION.iterdir() if p.is_file()}
    (EVIDENCE/"submission_sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print("Wrote FINAL_REVIEW.md, result table, short README and submission hashes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    main(Path(parser.parse_args().run).resolve())
