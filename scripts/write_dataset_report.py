import json
import math
from pathlib import Path

import polars as pl


def result_table(rows, columns):
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if value is None or isinstance(value, float) and not math.isfinite(value):
                values.append("—")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def dataset_report(name):
    root = Path("artifacts") / name
    load = lambda filename: json.loads((root / filename).read_text())
    manifest, protocol, eda, lock, profile, final = [load(filename) for filename in
        ["run_manifest.json", "dataset_protocol.json", "eda_full.json", "locked_selection.json", "generation_profile.json", "final_results.json"]]
    experiments = pl.read_csv(root / "experiments.csv")
    if (root / "audit_controls.csv").exists():
        experiments = pl.concat([experiments.filter(pl.col("split") != "validation_audit_control"),
                                  pl.read_csv(root / "audit_controls.csv")], how="diagonal_relaxed")
        experiments.write_csv(root / "experiments.csv")
    dev = experiments.filter((pl.col("split") == "validation_search") & (pl.col("seed") == 42))
    audit = experiments.filter(pl.col("split").is_in(["validation_audit", "validation_audit_control"]) & (pl.col("seed") == 42))
    test = experiments.filter((pl.col("split") == "test") & (pl.col("seed") == 42))
    summary = experiments.filter(pl.col("split") == "test").group_by("experiment_name").agg(
        pl.len().alias("seeds"), pl.col("logloss").mean(), pl.col("logloss").std().alias("sd_logloss"),
        pl.col("auc").mean(), pl.col("auc").std().alias("sd_auc")).sort("experiment_name")
    by_model = {r["experiment_name"]: r for r in summary.to_dicts()}
    difference = by_model["automatic"]["logloss"] - by_model["manual"]["logloss"]
    comparison = final["comparisons"]["test_automatic_vs_manual"]
    lower, upper = comparison["logloss_ci_95"]
    if upper < 0 and difference < 0:
        verdict = "Автоматический набор улучшил LogLoss на этом test: среднее по seeds меньше, а paired CI для заранее выбранного seed 42 ниже нуля."
    elif difference < 0:
        verdict = "Средний LogLoss automatic меньше, но устойчивое превосходство не доказано: paired CI для seed 42 включает ноль или противоречит среднему seeds."
    else:
        verdict = "Автоматический набор не улучшил средний LogLoss manual baseline; положительное преимущество по основному критерию не заявляется."
    features = pl.read_csv(root / "selected_features.csv")
    importance = pl.read_csv(root / "feature_importance.csv")
    pruned = pl.read_csv(root / "pruned_features.csv")
    tuning_attempts = {arm: {"completed": pl.read_csv(root / f"{arm}_tuning.csv").filter(pl.col("state") == "COMPLETE").height,
                              "attempts": pl.read_csv(root / f"{arm}_tuning.csv").height} for arm in ["manual", "auto"]}
    agent = load("agent_history.json")
    split = manifest["split"]
    windows = []
    for row in features.group_by("entity").agg(pl.len().alias("n"), pl.col("window").unique(),
                                                pl.col("long_window_days").drop_nulls().unique()).sort("entity").to_dicts():
        values = set(row["window"]) | {f"{w}d" for w in row["long_window_days"]}
        windows.append({"entity": row["entity"], "n": row["n"], "windows": ", ".join(sorted(values, key=lambda v: 10000 if v == "all_history" else int(v[:-1])))})
    metrics = [("experiment_name", "Model/experiment"), ("n_features", "Inputs"),
               ("logloss", "LogLoss ↓"), ("auc", "AUC ↑"), ("ndcg_5", "proxy NDCG@5"),
               ("ndcg_10", "proxy NDCG@10"), ("recall_10", "proxy Recall@10"), ("ranking_groups", "Proxy groups")]
    lines = [f"## {protocol['name']}",
             "### Данные и target",
             f"Фактический размер после нормализации: **{eda['rows']:,} строк**, **{eda['users']:,} users**, **{eda['items']:,} items**. Дубликаты по user/item/time/label: {eda['duplicate_events']:,}; повторные user–item строки: {eda['repeat_user_item_rows']:,}.",
             f"Target: `{protocol['target']}`. Events: {protocol['events']}. Cohort: {protocol['cohort']}.",
             f"Время: {protocol['time']}. category_id: {protocol['category_id']}; author: {protocol['author']}.",
             result_table([{"column": k, "dtype": v, "missing": eda['nulls'][k]} for k, v in eda['schema'].items()],
                          [("column", "Column"), ("dtype", "Type"), ("missing", "Missing")]),
             "### Temporal split",
             result_table([{"split": k, **v} for k, v in eda["splits"].items()],
                          [("split", "Part"), ("rows", "Rows"), ("start", "Start coordinate"), ("end", "End coordinate"), ("positive_rate", "Positive rate (test hidden during EDA)")]),
             "### Baseline, ablation и поиск на validation",
             f"Manual: **{lock['manual_name']}**; automatic: **{lock['auto_name']}**. HPO: {manifest['config']['hyperparameter_trials']} trials для каждого. Candidate specs: {profile['n_candidates']}; после train pruning: {profile['n_after_pruning']}; selected: {features.height}. Base inputs не входят в selected count.",
             f"Completed/attempts HPO: {tuning_attempts}. Прерванные попытки сохраняются в tuning CSV; budget считает только COMPLETE, чтобы возобновление не сокращало число оцененных конфигураций.",
             result_table(dev.to_dicts(), metrics),
             "В последовательной ablation rates добавляются после metadata relations. Все ablation arms используют одни manual tuned параметры; вклад группы условен на уже добавленных предыдущих группах. Fixed/shared controls сначала ограничивают допустимый pool, затем независимо выполняют train pruning и fit; shared ranking использует gain собственного pool. Controls меняют состав и размер набора; это не чистый причинный эффект одного окна.",
             "### Audit и финальный test",
             result_table(audit.to_dicts(), metrics), result_table(test.to_dicts(), metrics),
             "Среднее/SD на test между заранее заданными seeds (SVD arms — один seed):",
             result_table(summary.to_dicts(), [("experiment_name", "Model"), ("seeds", "Seeds"), ("logloss", "Mean LogLoss"),
                                               ("sd_logloss", "SD LogLoss"), ("auc", "Mean AUC"), ("sd_auc", "SD AUC")]),
             f"Automatic − manual: mean ΔLogLoss **{difference:+.6f}**; paired ΔLogLoss seed 42 **{comparison['delta_logloss']:+.6f}**, 95% user-cluster CI **[{lower:+.6f}, {upper:+.6f}]**. ΔAUC seed 42 {comparison['delta_auc']:+.6f}. Bootstrap {comparison['repeats']} repeats, фиксированные модели. Неопределенность всей процедуры search этим CI не покрывается.",
             f"**{verdict}**",
             "### Selected temporal windows",
             result_table(windows, [("entity", "Entity"), ("n", "Selected specs"), ("windows", "Windows including trend dependencies")]),
             f"Explicit trend specs в выбранном наборе: {features.filter(pl.col('aggregation') == 'trend').height}. Наличие разных окон не доказывает их необходимость; сравнение shared_windows_control на audit важнее одного heatmap.",
             "### Feature agent и pruning",
             result_table([{k:v for k,v in r.items() if k != "proposed"} for r in agent],
                          [("round", "Round"), ("before_logloss", "Before"), ("after_logloss", "After"), ("accepted", "KEEP"), ("generation_seconds", "Generation s")]),
             f"Pruning log содержит {pruned.height} удалений. Все proposals и причины сохранены отдельно; rejected proposals не включаются в locked набор.",
             "### Feature importance",
             result_table(importance.head(20).to_dicts(), [("feature", "Feature"), ("importance_gain", "Train gain"), ("entity", "Family")]),
             "Gain — train-only proposal signal со смещением при корреляции; не причинная importance. SHAP и permutation importance не вычислялись, такие результаты не заявляются.",
             "### Learned representations",
             "SVD basis fit-ится на первых 10% событий. Их строки исключены одинаково в manual_warmup/auto_warmup/auto_plus_svd. User profile — mean signed item vectors по строгому прошлому. Unknown basis items имеют нулевой vector. Этот эксперимент не использует будущие train edges."]
    if name == "yambda" and features.filter(pl.col("entity") == "user").height > features.height / 2:
        lines.insert(-2, "Выбранный Yambda набор в основном состоит из user histories; item/category families в top32 не попали. Это согласуется с доминированием склонности пользователя выражать like/dislike в данной conditional задаче — интерпретация по составу features, а не доказанная причина. Высокий AUC поэтому нельзя трактовать как умение выбрать лучший следующий трек из каталога.")
    if name == "kion":
        lines.insert(-2, "В KION весь наблюдаемый период короче 180 дней. Поэтому 180d здесь эквивалентно all-history; pruning может оставить только один из этих aliases. Выбранное 180d не доказывает оптимальность именно 180 дней. User–item пары не повторяются, и их исторические счетчики постоянны; эта семья удаляется автоматически.")
    if "test_svd_vs_auto_warmup" in final["comparisons"]:
        c = final["comparisons"]["test_svd_vs_auto_warmup"]
        lines.append(f"SVD − auto_warmup ΔLogLoss **{c['delta_logloss']:+.6f}**, CI **[{c['logloss_ci_95'][0]:+.6f}, {c['logloss_ci_95'][1]:+.6f}]**; ΔAUC {c['delta_auc']:+.6f}. Один seed; conditional CI. Manual+SVD не обучался, весь возможный выигрыш hybrid нельзя приписать именно auto search.")
    dev_by_name = {row["experiment_name"]: row for row in dev.to_dicts()}
    audit_by_name = {row["experiment_name"]: row for row in audit.to_dicts()}
    trend_delta = dev_by_name["manual_plus_trend"]["logloss"] - dev_by_name["manual"]["logloss"]
    adaptive_choices = dev.filter(pl.col("experiment_name").str.starts_with("auto_adaptive"))
    adaptive_best = adaptive_choices.filter(pl.col("logloss") <= adaptive_choices["logloss"].min()
        + manifest["auto_config"]["keep_min_logloss_gain"]).sort("n_features").row(0, named=True)
    budget_delta = adaptive_best["logloss"] - dev_by_name["auto_adaptive_all"]["logloss"]
    expansion_delta = dev_by_name["auto_adaptive_all"]["logloss"] - dev_by_name["auto_fixed_windows_all"]["logloss"]
    shared_delta = audit_by_name[adaptive_best["experiment_name"]]["logloss"] - audit_by_name["shared_windows_control"]["logloss"]
    explored = pl.read_csv(root / "candidate_features.csv").height
    removed_share = 1 - features.height / explored
    generation_ratio = 1 + profile["additional_candidates"]["seconds"] / profile["manual"]["seconds"]
    hpo_fit_seconds = {arm: pl.read_csv(root / f"{arm}_tuning.csv")["user_attrs_training_seconds"].sum()
                       for arm in ["manual", "auto"]}
    manual_search_seconds = dev.filter(pl.col("experiment_name").is_in(
        ["base_default", *lock["manual_selection_pool"]]))["training_time"].sum() + hpo_fit_seconds["manual"]
    auto_search_seconds = dev.filter(pl.col("experiment_name").str.starts_with("auto_") &
        ~pl.col("experiment_name").is_in(["auto_warmup_control", "auto_plus_svd"]) |
        pl.col("experiment_name").str.starts_with("agent_") |
        pl.col("experiment_name").str.starts_with("shared_"))["training_time"].sum() + hpo_fit_seconds["auto"]
    lines.extend(["### Ответы на исследовательские вопросы",
        f"Manual multi-scale trends меняют search LogLoss на {trend_delta:+.6f}; это небольшой условный эффект, а не универсальная польза ratios. Итоговый automatic набор содержит {features.filter(pl.col('aggregation') == 'trend').height} explicit trend specs.",
        f"Начальный pruning: {profile['n_candidates']} → {profile['n_after_pruning']}. Всего исследовано {explored} specs с учетом agent proposals, в финале {features.height}, на {removed_share:.1%} меньше total explored. До HPO/agent выбранный adaptive budget против adaptive_all дает Δ search LogLoss {budget_delta:+.6f} при одинаковых initial parameters. Curve по budgets показывает фактическую зависимость качества от числа inputs; компактный оптимум заранее не предполагается.",
        f"Adaptive_all − fixed-window pool: Δ search LogLoss {expansion_delta:+.6f}; отрицательная разница означает преимущество adaptive в этом сравнении. На audit выбранный adaptive набор − shared-window control: Δ LogLoss {shared_delta:+.6f}. Знак результата сохранен; более гибкое пространство не обязано выигрывать. Состав features и процедура отбора тоже различаются; отдельный причинный эффект индивидуальных окон не изолирован.",
        f"Расчет manual плюс всех дополнительных кандидатов занимает примерно {generation_ratio:.2f}× manual generation. Это только feature computation, без дополнительных fitting/search rounds. Итоговый serving использует выбранные specs, а не весь исследованный pool.",
        f"Человек перестал выбирать каждую конкретную колонку и окно; он по-прежнему задает target, допустимые отношения, primitives, temporal protocol и budget. Agent принял {sum(bool(r['accepted']) for r in agent)} из {len(agent)} групп по search threshold. Автоматизация процесса и доказанный вклад каждой его стадии — разные утверждения."])
    lines.extend(["### Вычислительная стоимость",
        f"Измеренная сумма fit time стадии manual (seed42 ablations + HPO): **{manual_search_seconds:.1f} s**; дополнительной auto search стадии (pool/budget/agent/shared fits + HPO): **{auto_search_seconds:.1f} s**, **{auto_search_seconds/manual_search_seconds:.2f}×** manual стадии сверх нее. HPO отдельно: manual {hpo_fit_seconds['manual']:.1f} s, auto {hpo_fit_seconds['auto']:.1f} s. Здесь нет prediction/metrics overhead, feature generation, representations, дополнительных seeds и final refits; это не elapsed wall time.",
        f"Manual + manual trends generation: **{profile['manual']['seconds']:.3f} s**, matrix {profile['manual']['memory_mb']:.1f} MiB; дополнительные auto candidates: **{profile['additional_candidates']['seconds']:.3f} s**, matrix {profile['additional_candidates']['memory_mb']:.1f} MiB. Final selected features на всех строках: {final['final_feature_profile']['seconds']:.3f} s.",
        result_table(dev.select("experiment_name", "n_features", "training_time", "memory_usage").to_dicts(),
                     [("experiment_name", "Experiment"), ("n_features", "Inputs"), ("training_time", "Fit s"), ("memory_usage", "Peak process MiB")]),
        "RSS — peak процесса с удерживаемыми матрицами. Generation=0/cache_hit=True означает reuse, но cache_hit в model rows не является точной трассировкой происхождения каждого block. Профили cache сохраняют измеренное время создания matrix; в повторном KION оно относится к предыдущему вычислению с тем же engine/data/spec hash. HPO time хранится отдельно в tuning CSV; указанные fit times не являются end-to-end latency.",
        "### Графики"])
    segments = root / "test_segments.csv"
    if segments.exists():
        lines.insert(-1, "Post-lock diagnostics: known/unknown IDs и organic/recommendation-driven, где доступны. Они не используются для смены модели.\n\n" +
                     result_table(pl.read_csv(segments).to_dicts(), [("model", "Model"), ("segment", "Segment"), ("rows", "Rows"), ("logloss", "LogLoss"), ("auc", "AUC")]))
    for plot in ["metric_by_experiment", "metric_vs_features", "entity_window_heatmap", "top30_importance", "feature_counts", "window_distribution", "feature_family_gain", "test_metrics"]:
        lines.append(f"![{name} {plot}](artifacts/{name}/plots/{plot}.png)")
    return "\n\n".join(lines), {"dataset": name, "rows": eda["rows"], "manual_logloss": by_model['manual']['logloss'],
        "automatic_logloss": by_model['automatic']['logloss'], "delta_mean_logloss": difference,
        "manual_auc": by_model['manual']['auc'], "automatic_auc": by_model['automatic']['auc'],
        "selected_specs": features.height, "conclusion": verdict}


def main():
    sections, summaries = [], []
    for name in ["kion", "yambda"]:
        section, summary = dataset_report(name)
        sections.append(section)
        summaries.append(summary)
        local_report = section.replace(f"(artifacts/{name}/plots/", "(plots/")
        (Path("artifacts") / name / "REPORT.md").write_text(
            "Полный протокол и ограничения: [основной отчет](../../REPORT.md).\n\n" + local_report + "\n")
    intro = """# Automated Temporal Feature Search: KION и Yambda

## Problem, motivation и research question

Можно ли уменьшить ручной выбор temporal features, задав bounded grammar
`entity × event × aggregation × window`, и сохранить или улучшить сильный LightGBM baseline?
Проверяем это на двух реальных датасетах, указанных пользователем. MovieLens больше не
является основным результатом; прежний отчет сохранен в REPORT_MOVIELENS.md.
Текущее состояние — **post-audit re-evaluation**. Исходные runs и notebook сохранены
в artifacts/pre_audit. Test уже наблюдался до исправлений; новый run не является
слепым независимым подтверждением. Структуры и параметры повторно выбирались только
на search validation по заранее описанным исправленным правилам.

## Источники, постановка и границы выводов

- [KION official repository](https://github.com/irsafilo/KION_DATASET): скачанный interactions.csv.zip содержит **1 594 787** строк за 2021-03-13—2021-08-22, хотя README описывает больший размер и другой конечный год. 268 строк с отсутствующим watched_pct не получают label и исключены; основной run использует все **1 594 519** оставшихся строк без user sampling. Это агрегированные user–item записи с last_watch_dt; target — reported доля просмотра >=50%, условно на наблюдаемой паре и дате последнего просмотра. Это не оценка качества рекомендаций до первого просмотра и не воспроизведение официального KION contest.
- [Yambda official dataset](https://huggingface.co/datasets/yandex/yambda): берутся все likes/dislikes из flat/50m и artist/album mappings. **989 232** явных реакции, а не весь Yambda-50M и не Yambda-5B. Like/dislike — наблюдаемые положительные/отрицательные действия, но classifier оценивает знак реакции условно на ее наличии; negatives из ненаблюдаемых пар не придумываются. Listen/unlike/undislike контекст не включен; event counts не являются состоянием текущего like после отмены.
- [Yambda paper](https://arxiv.org/html/2505.22238v1): формула времени `floor((T−T0)/5)*5` означает относительные **секунды**, уже округленные до 5s. Умножение timestamp еще на пять ошибочно. Epoch 1970 — только внутренняя координата. Calendar hour/weekday отключены. Альбом и исполнитель сведены к minimum metadata ID; это детерминированное упрощение многозначной связи, исключающее размножение строк.

Два разных target не позволяют напрямую объявить один датасет «легче» по AUC или
сравнивать абсолютный LogLoss между ними. Сравниваются relative manual/auto изменения
внутри каждого dataset. Ranking ниже — proxy по user × timestamp с >=2 различными item
и mixed labels; группы с повторением одного item исключены целиком. Это не full-catalog
recall. Для KION это день, для Yambda — 5s bin. Итоговый ranking audit пересчитывает
только proxy metrics по сохраненным моделям; AUC, LogLoss и выбор моделей не меняются.
Если eligible groups отсутствуют, ranking metrics сохраняются как NaN/—, а не выдумываются.

## Related approaches и выбор метода

DFS/Featuretools дают relational primitives, generic AutoFE и RL/NFS исследуют generation,
AutoFIS/AutoField — interactions/fields, Optuna — параметрический search. Их отличие от
нашей постановки и стоимость GNN, MF, sequential Transformers, FT-Transformer/TabPFN
разобраны с первичными источниками и сравнительной таблицей в [RESEARCH.md](RESEARCH.md).
Здесь выбор остается hybrid temporal factory: она отвечает именно на вопрос windows,
интерпретируема и проверяема на CPU. Сложные модели не обучаются ради названия.

## Validation, leakage prevention и strong baseline

Global temporal split: около 70% train, 7.5% search validation, 7.5% audit validation,
15% test; равные timestamps не пересекают границы. Категориальный encoder и pruning
fit-ятся только на train. Search validation выбирает features/windows/HPO, audit/test
не меняют lock. Затем final модели refit-ятся на первых 85% с числом деревьев,
выбранным на search; manual/automatic проверяются на трех seeds [42,2024,7].

Feature engine использует только `[T−W,T)` или all-history `<T`. Для KION исключается
**весь текущий день**; для Yambda — весь текущий 5s bin. Текущие history_value/watched_pct/
total_dur/label/event не являются base inputs. Polars rolling closed=left; cumulative
агрегаты предварительно объединяют timestamp ties. Source timestamp audit и cardinality
join checks дополняются independent naive oracle, prefix invariance, current/future
mutation tests и тестами реальных dataset schemas и единиц времени.

Online replay допускает использование outcomes предыдущих времен внутри validation/test.
Для KION это явное offline допущение: reported aggregate доступен после last_watch_day;
доступность в production и timestamps сессий в файле не задокументированы достаточно,
поэтому такая схема не доказывает возможность scoring до начала просмотра. Static
metadata считаются заранее известными; нужны versioned joins для production.

Manual baseline содержит IDs и доступные static categories; user/item/user-item/user-category/
user-author counts, recency, positive-event rates, historical watched_pct means в KION;
окна 7/30 days/all history. После независимого аудита reference выбирается среди всех
выполненных manual ablation stages по search LogLoss, а не только двух полных наборов.
Pool и выбранный source фиксируются до audit/test. Это не exhaustive search всех
подмножеств ручных families. Параметры HPO первоначально настроены на полном manual,
после чего все manual stages сравниваются с одними и теми же параметрами.
Каждая manual и auto модель получает 20 Optuna trials (300k sampled fit rows с полной
историей, затем full-train re-fit). Никакого random train/test split нет.

## Automated generation, selection и feature agent

FeatureSpec валидирует allowlisted entities/events/aggregations/windows. Поддержаны count,
sum/mean/min/max, nunique, recency, event_rate/conversion/frequency и short/long trends.
Actual window grid: 1,3,7,14,21,30,60,90,180 days и all_history. Есть global, user, item,
user-item, category/album, user-category/album, author/director/artist, user-author,
item-gender где доступно. Fixed-window control использует только 7/30/all.

Train pruning удаляет constants, sparse/missing, duplicates и near-correlated features.
Gain предлагает top32/64/96/128 наборы; re-fit по validation выбирает размер и окна.
При разнице loss <=0.0001 предпочитается меньший набор. Shared-window control ограничивает
все families общими тремя окнами. Agent делает две bounded группы новых соседних окон;
KEEP при улучшении validation LogLoss >=0.0001. Отрицательные результаты сохраняются.
Optional LLM hook возвращает только validated FeatureSpecs; arbitrary SQL не исполняется.

## Сводка реальных final test результатов
"""
    intro += "\n" + result_table(summaries, [("dataset", "Dataset"), ("rows", "Rows"), ("manual_logloss", "Manual mean LogLoss"),
        ("automatic_logloss", "Auto mean LogLoss"), ("delta_mean_logloss", "Δ auto−manual"),
        ("manual_auc", "Manual mean AUC"), ("automatic_auc", "Auto mean AUC"), ("selected_specs", "Selected specs")])
    previous_yambda = Path("artifacts/pre_audit/yambda/experiments.csv")
    if previous_yambda.exists():
        previous_manual = pl.read_csv(previous_yambda).filter(
            (pl.col("split") == "test") & (pl.col("experiment_name") == "manual"))["logloss"].mean()
        current_manual = next(row["manual_logloss"] for row in summaries if row["dataset"] == "yambda")
        intro += (f"\n\nPost-audit stability check: Yambda manual, лучший среди stages на search, имеет test "
                  f"LogLoss {current_manual:.6f}; прежний полный manual — {previous_manual:.6f}. Это ухудшение "
                  "показывает нестабильность validation выбора; reference не менялся после test. Поэтому "
                  "новую величину auto gain нельзя трактовать как чистый эффект исправления. Исходные "
                  "результаты сохранены в artifacts/pre_audit; before/after comparison — в FIXES.md.")
    outro = """## Ограничения, production scalability и conclusions

Выбор отдельных features/windows автоматизирован, но target, availability assumptions,
schema relations, grammar и budget остаются решениями исследователя. Качество auto
оценивается отдельно на каждом dataset, отрицательные или неубедительные итоги сохраняются.
Небольшие изменения search validation сами по себе не доказывают перенос на test.
Агрегированный KION и action-conditioned Yambda не заменяют impression-log validation.
Нужны несколько rolling-origin splits и независимый production evaluation.

Все показанные значения вычислены реальными runs; timings не симулировались. Графики,
selected windows и agent trace сохранены отдельно по dataset. Learned SVD — дополнительный
controlled эксперимент, а не автоматическое доказательство преимущества feature search.

Локально обработаны 1.59M и 0.99M событий. 1B+ здесь не запускался: dense O(NF) matrix
не подходит для такого масштаба. Нужны entity-sharded state, streaming counters,
point-in-time feature store, partitioned offline ETL, shared primitive cache, bounded
search на выборке и serving только выбранных specs. Exact rolling distinct/min/max
имеют разные costs; late arrivals требуют watermark/retractions/backfill. GNN требуют
distributed graph sampling, Transformers — bounded sequences, distributed training,
embedding tables и serving. Для 100B events понадобится отдельная инфраструктура.

Future work: добавить полный listen/unlike/undislike context Yambda с отдельными
query/history tables, настоящий candidate-generation/ranking benchmark, versioned metadata,
label availability timestamps, cost-aware search, permutation importance, decayed counters,
несколько safe representation snapshots и rolling-origin evaluation. Official benchmark
MAP/Recall в этом отчете не присваиваются нашему scoring pipeline.
"""
    Path("REPORT.md").write_text(intro + "\n\n" + "\n\n".join(sections) + "\n\n" + outro)
    pl.DataFrame(summaries).write_csv("artifacts/dataset_comparison.csv")


if __name__ == "__main__":
    main()
