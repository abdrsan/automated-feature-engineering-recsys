import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def markdown_table(rows, columns):
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(root, destination):
    root = Path(root)
    load = lambda name: json.loads((root / name).read_text())
    manifest, lock, profile, final = [load(n) for n in ["run_manifest.json", "locked_selection.json",
                                                       "generation_profile.json", "final_results.json"]]
    eda = load("eda.json")
    experiments = pl.read_csv(root / "experiments.csv")
    if (root / "audit_controls.csv").exists():
        controls = pl.read_csv(root / "audit_controls.csv")
        experiments = pl.concat([experiments.filter(pl.col("split") != "validation_audit_control"),
                                  controls], how="diagonal_relaxed")
        experiments.write_csv(root / "experiments.csv")
    dev = experiments.filter((pl.col("split") == "validation_search") & (pl.col("seed") == 42))
    audit = experiments.filter(pl.col("split").is_in(["validation_audit", "validation_audit_control"]) & (pl.col("seed") == 42))
    test = experiments.filter((pl.col("split") == "test") & (pl.col("seed") == 42))
    comparison = final["comparisons"]["test_automatic_vs_manual"]
    audit_comparison = final["comparisons"]["audit_automatic_vs_manual"]
    selected = pl.read_csv(root / "selected_features.csv")
    importance = pl.read_csv(root / "feature_importance.csv")
    agents = load("agent_history.json") if (root / "agent_history.json").exists() else []
    pruned = pl.read_csv(root / "pruned_features.csv")
    full_eda = json.loads(Path("artifacts/eda_full.json").read_text()) if root == Path("artifacts") and Path("artifacts/eda_full.json").exists() else None
    n_train, search_end, dev_end, n_all = [manifest["split"][k] for k in
                                          ["train_end", "search_end", "validation_end", "n_rows"]]
    metrics_columns = [("experiment_name", "Experiment"), ("n_features", "Inputs"),
                       ("logloss", "LogLoss ↓"), ("auc", "AUC ↑"),
                       ("ndcg_5", "proxy NDCG@5"), ("ndcg_10", "proxy NDCG@10"),
                       ("recall_10", "proxy Recall@10")]
    summaries = experiments.filter(pl.col("split") == "test").group_by("experiment_name").agg(
        pl.len().alias("seeds"), pl.col("logloss").mean().alias("logloss_mean"),
        pl.col("logloss").std().fill_null(0).alias("logloss_sd"),
        pl.col("auc").mean().alias("auc_mean"), pl.col("auc").std().fill_null(0).alias("auc_sd")).sort("experiment_name")
    summary_lookup = {r["experiment_name"]: r for r in summaries.to_dicts()}
    mean_delta = summary_lookup["automatic"]["logloss_mean"] - summary_lookup["manual"]["logloss_mean"]
    dev_lookup = {r["experiment_name"]: r for r in dev.to_dicts()}
    ablation_rates = dev_lookup["manual"]["logloss"] - dev_lookup["plus_user_category"]["logloss"]
    ablation_trend = dev_lookup["manual_plus_trend"]["logloss"] - dev_lookup["manual"]["logloss"]
    full_auto = dev_lookup["auto_adaptive_all"]
    selected_auto = min((r for r in dev.to_dicts() if r["experiment_name"].startswith("auto_adaptive")), key=lambda r:r["logloss"])
    n_trends = selected.filter(pl.col("aggregation") == "trend").height
    control_rows = {r["experiment_name"]: r for r in audit.to_dicts() if r["split"] == "validation_audit_control"}
    diagnostics = f'''## Диагностика основного результата

Средний test LogLoss между тремя seeds: manual
**{summary_lookup['manual']['logloss_mean']:.6f}**, automatic
**{summary_lookup['automatic']['logloss_mean']:.6f}**; Δ = **{mean_delta:+.6f}**.
Эта проверка не поддерживает устойчивое превосходство automatic model, несмотря на
улучшение search validation и audit. Seed 42 нельзя выдавать за общий положительный результат.

Наблюдаемые факты: manual уже содержит сильные historical rating statistics;
user–item не повторяются; часть distinct features точно дублирует counts; временные окна
коррелируют; небольшой выигрыш search validation ослабевает на длинном test интервале.
Переобучение search validation и temporal shift — правдоподобные объяснения, но этим
экспериментом их причинный вклад не установлен. Test находится в периоде значительно
длиннее validation; нужны несколько rolling-origin cutoffs для проверки этой гипотезы.

В дополнение к ranking gain проверены matched-parameter controls на audit, без изменения
main lock. Их результаты ниже уже включены в audit table. Shared windows могут быть
достаточны; выбранные разные окна сами по себе не доказывают их необходимость.

### Ответы на исследовательские вопросы

| Вопрос | Что показал эксперимент |
|---|---|
| Auto generation улучшает сильный baseline? | На search и audit — да; на test устойчивое преимущество не доказано, среднее seeds хуже. |
| Какие окна выбраны? | Все конкретные FeatureSpecs и entity/window dependencies сохранены в selected_features.csv и selected_windows.csv. |
| Нужны ли отдельные окна по entities? | Не доказано: shared-window control конкурентоспособен на audit. |
| Помогают ли ratios/trends? | Manual rates: Δ validation LogLoss {ablation_rates:+.6f}; 7/30 trends: {ablation_trend:+.6f}. В финальном automatic наборе {n_trends} explicit trend specs. |
| Сколько можно удалить? | Из {profile['n_candidates']} initial candidates после pruning осталось {profile['n_after_pruning']}; selection использует {selected.height}. |
| Какова стоимость? | Manual generation {profile['manual']['seconds']:.3f}s; manual + дополнительные candidates {profile['manual']['seconds'] + profile['additional_candidates']['seconds']:.3f}s, плюс повторное обучение. |
| Что дает рост feature budget? | Adaptive full: {full_auto['n_features']} inputs / LogLoss {full_auto['logloss']:.6f}; лучшая budget arm: {selected_auto['n_features']} / {selected_auto['logloss']:.6f}. |
| Есть ли diminishing returns? | Полный набор не лучше лучшей меньшей budget arm; больше колонок не гарантирует качество. |
| Какие families полезны? | Historical item rating means, item–gender means, user rates/recency и user–category means имеют высокий training gain; ablations дают независимую проверку части families. |
| Меньше ли ручной работы? | Конкретные spec/window choices делает pipeline. Target, schema relations, grammar, budget и availability assumptions остаются решениями человека. |

### Дополнительная learned representation

После freeze SVD basis на warmup prefix user profile строится только из предыдущих signed
item vectors. Добавлены dot, cosine, profile norm и item norm. На final test сравнение
auto+SVD с auto_warmup имеет одинаковое начало fit, поэтому выигрыш не объясняется
разным количеством training rows. Это отдельно проверенный путь улучшения, а не
доказательство превосходства автоматических temporal features самих по себе.
'''
    if "test_svd_vs_auto_warmup" in final["comparisons"]:
        svd = final["comparisons"]["test_svd_vs_auto_warmup"]
        diagnostics += f'''\nSVD ΔLogLoss = **{svd['delta_logloss']:.6f}**, 95% paired user CI
**[{svd['logloss_ci_95'][0]:.6f}, {svd['logloss_ci_95'][1]:.6f}]**;
ΔAUC = {svd['delta_auc']:.6f}. SVD arm проверен на одном seed, его CI условен на fixed models.
Manual+SVD на test не сравнивался; нельзя приписать весь выигрыш hybrid именно auto search.
'''
    windows = []
    for entity in sorted(selected["entity"].unique().to_list()):
        entity_rows = selected.filter(pl.col("entity") == entity)
        values = set(entity_rows["window"].to_list())
        values.update(f"{w}d" for w in entity_rows["long_window_days"].drop_nulls().to_list())
        ordered = sorted(values, key=lambda w: 100000 if w == "all_history" else int(w[:-1]))
        windows.append({"entity": entity, "n": entity_rows.height, "windows": ", ".join(ordered)})
    if comparison["logloss_ci_95"][1] < 0:
        conclusion = "На этом temporal test автоматический набор улучшил LogLoss относительно сильнейшего выбранного manual baseline; paired user-bootstrap CI лежит ниже нуля."
    elif comparison["delta_logloss"] < 0:
        conclusion = "Устойчивое преимущество автоматических temporal features на test не доказано: небольшой выигрыш для seed 42 имеет CI, включающий ноль, а проверка нескольких seeds не подтверждает превосходство."
    else:
        conclusion = "На этом temporal test автоматический набор не улучшил LogLoss сильного manual baseline. Положительное преимущество по основному критерию не заявляется."
    tuning_seconds = 0.
    for name in ["manual_tuning.csv", "auto_tuning.csv"]:
        trials = pl.read_csv(root / name)
        if "user_attrs_training_seconds" in trials.columns:
            tuning_seconds += trials["user_attrs_training_seconds"].sum()
    training_seconds = experiments["training_time"].sum()
    manual_seconds = profile["manual"]["seconds"]
    candidate_seconds = profile["additional_candidates"]["seconds"]
    costs = dev.select("experiment_name", "n_features", "training_time", "memory_usage").to_dicts()
    accepted = sum(bool(r["accepted"]) for r in agents)
    text = f'''# Автоматизация temporal feature engineering для рекомендательного scoring

## 1. Problem

Предсказываем вероятность высокой оценки (`rating >= 4`) для наблюдаемой пары user–movie
в момент T. Главный объект исследования — построение причинно корректных исторических
признаков для градиентного бустинга. Это исследование preference scoring среди оцененных
фильмов; MovieLens не содержит impressions или истинных отрицательных показов.

## 2. Motivation

Ручные признаки фиксируют решения об entities, типах событий, агрегатах и временных окнах.
Для разных сущностей полезная память может отличаться. Цель — перенести ручное решение с
конкретного `user_positive_count_7d` на разрешенное пространство признаков, а затем измерить
полезность автоматически выбранного набора.

## 3. Research question

Можно ли получить сопоставимое или лучшее качество сильного manual boosting baseline,
задав ограниченную грамматику признаков и выбирая их состав и окна по temporal validation?
Дополнительно проверяем размер набора, trends, shared windows, computational cost и
learned representations. Основной критерий заранее фиксирован: LogLoss. AUC и ranking
proxy — дополнительные измерения, не основания для смены критерия после test.

## 4. Related approaches

Исследованы DFS/Featuretools, autofeat, RL/NFS, AutoFIS/AutoField, Optuna feature search,
boosting/LambdaRank, MF/embeddings, LightGCN/link prediction, SASRec, TabTransformer,
FT-Transformer и TabPFN-like подходы. Сравнение, ссылки на research papers, official docs,
author repositories и production/competition практики приведены в [RESEARCH.md](RESEARCH.md).

DFS дает relational grammar, AutoFIS/AutoField автоматизируют interactions/fields, но
не решают temporal aggregation автоматически. Sequential и graph models меняют objective
и инфраструктуру. В этом задании ограниченный search лучше изолирует исследовательский
вопрос и остается проверяемым на CPU. Полные GNN/Transformers не обучались.

## 5. Approach comparison

Основной trade-off: interpretable temporal search сохраняет привычный scoring backend,
но требует широкой матрицы кандидатов и нескольких fitting rounds. Уменьшаем пространство
до небольшого allowlist отношений и окон. Gain используется как дешевый proposal signal;
keep/drop проверяется реальным переобучением. Сравнительная инженерная таблица для 1M
и 1B+ событий — в RESEARCH.md; ее оценки не являются экспериментальными метриками.

## 6. Dataset

Использован [официальный MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).
Архив проверен по официальной MD5 `c4d9eecfca2ab87c1945afe126590906`.
Всего в основном запуске **{n_all:,} interactions**. EDA development prefix: {eda['rows']:,}
строк, {eda['users']:,} users, {eda['items']:,} items, диапазон {eda['start']} — {eda['end']} UTC.
В prefix найдено {eda['duplicate_events']} дубликатов полных событий и
{eda['repeat_user_item_rows']} повторных user–item строк;
{eda['same_timestamp_rows']:,} строк сверх числа уникальных timestamps.

Реальная схема: `user_id`, `item_id`, `rating`, `timestamp`, `gender`, `age`,
`occupation`, `zip`, `genres`, `category_id`, `release_year`, `label`, `event`, `row_id`.
`positive` означает rating >= 4; `negative` — rating <= 3. Эти события не называются
view/click/purchase. `rating` используется исключительно в исторических aggregates.
`zip` не входит в модель. Первый жанр — category_id, полная комбинация жанров — base
category; multilabel genre aggregation остается ограничением. Статическая доступность
metadata предполагается, история изменений в датасете отсутствует.

Цитирование данных: F. Maxwell Harper, Joseph A. Konstan (2015), *The MovieLens Datasets:
History and Context*, DOI 10.1145/2827872. Сырые данные исключены из Git по
[условиям GroupLens](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt).

## 7. Validation

Глобальный chronological split без random train_test_split:

| Раздел | Строки | Назначение |
|---|---:|---|
| Train | {n_train:,} | Fit и train-only preprocessing/pruning |
| Validation search | {search_end - n_train:,} | HPO, feature budget, окна, KEEP/DROP |
| Validation audit | {dev_end - search_end:,} | Проверка уже зафиксированных решений |
| Test | {n_all - dev_end:,} | Финальное сравнение заранее заявленных моделей |

Равные timestamps целиком принадлежат одному разделу. Категории кодируются только по
fit prefix; unseen values имеют code -1. После lock выполняются три seeds
{manifest['config']['seeds']}; результаты seeds не меняют selection. Final models refit
на первых {dev_end:,} строках с фиксированным числом деревьев, выбранным на search validation.
Audit scores принадлежат development models, test scores — refitted models; их нельзя
сравнивать как один и тот же trained checkpoint.

## 8. Leakage prevention

Для query T конечное окно — **[T − W, T)**, all-history — **timestamp < T**.
Polars rolling имеет `closed="left"`; all-history сначала группирует равные entity/time
блоки, затем cumulative и shift. Нельзя просто shift-нуть строки при совпадающих timestamps.
Векторизованный движок проверяет максимальный source timestamp и размер результатов joins.
Тесты сверяют все primitives с независимым naive oracle; проверяют Jan 1/2/5, timestamp
ties, boundaries, current/future target mutation, prefix invariance и representation cutoff.

Протокол — **online replay**: исход более раннего события считается известным до следующего
более позднего события, в том числе внутри validation/test. Это не frozen batch evaluation.
При задержке labels нужен `available_at`, а при batch scoring — общий query cutoff.
Эти дополнительные условия имеющимся датасетом проверить нельзя.

## 9. Manual baseline

Base: user/item IDs, demographics, primary genre, full genre set, release year, hour/weekday.
Manual: total/positive/negative counts, 7/30-day counts, recency, mean historical rating
для user, item, user–item, user–category, а также сглаженные positive/all rates.
Это dataset-specific версия click/purchase features. Trends 7/30 days проверяются отдельной
ablation. Сильнейший manual вариант выбран на validation search: **{lock['manual_name']}**.
Manual HPO: {manifest['config']['hyperparameter_trials']} Optuna trials, до
{manifest['config']['max_tuning_train_rows']:,} случайно выбранных **training rows** для
ускорения fitting, с полной историей признаков; победитель затем обучается на полном train.
Это sampling внутри fit, не случайный split данных.

## 10. Automated feature generation

FeatureSpec содержит keys, event, aggregation, window_days и безопасные параметры derived
features. Поддержаны global/user/item/user–item/category/user–category/item–gender,
а generic generator обнаруживает brand/author/merchant при наличии колонок. Реальные
candidate windows: {manifest['auto_config']['windows']} days, None = all-history.
Поддержаны count/sum/mean/min/max/nunique/recency/event_rate/conversion/frequency/trend.
В candidate space отсутствует frequency для фиксированного окна: это линейный дубликат count.
Finite global/category nunique не генерируется из-за дорогой exact distinct cardinality;
bounded entity distinct и all-history distinct сохраняются.

В исходном пространстве **{profile['n_candidates']} FeatureSpecs**, после train-only pruning
**{profile['n_after_pruning']}**, в выбранном автоматическом наборе **{selected.height}**
(base inputs считаются отдельно). Trends — сглаженные short/long count ratios, не rates
нормированные на длину окна. Conversion primitive означает отношение event counts;
на MovieLens он не трактуется как purchase funnel.

## 11. Feature search strategy

Hybrid pipeline: large bounded generation → constants/missing/duplicates/correlation pruning
только на train → train gain ranking → validation re-fit нескольких feature budgets →
адаптивные окна → {manifest['config']['hyperparameter_trials']} automatic HPO trials →
bounded feature-agent proposals. Fixed-window selection ограничена 7/30 days и all history.
Adaptive search расширяет окна; состав выбирается по loss, а не заранее по entity.
Shared-window control разрешает одни три окна всем entities с близким feature budget.
Этот control диагностический: он не полностью устраняет различия в корреляции и составе.

Агент выполнил {len(agents)} rounds, принято {accepted}. Он предлагает соседние окна
семейств с высоким gain. Группа принимается при улучшении LogLoss не менее
{manifest['auto_config']['keep_min_logloss_gain']}. LLM integration ограничена входным
JSON валидатором; оплачиваемого API и произвольного SQL execution нет. Подробный trace:
`artifacts/agent_history.json`. Выбранная модель: **{lock['auto_name']}**.

## 12. Experiments

Все строки ниже вычислены реальными запусками на search validation, seed 42.
Все HPO trials сохранены отдельно; это позволяет видеть полный budget.

{markdown_table(dev.to_dicts(), metrics_columns)}

## 13. Ablation study

Последовательная ablation добавляет user → item → user–item → user–category → rates →
trends. Во всех arms используются одни temporal splits и manual tuned parameters.
Следующие auto_fixed_windows / auto_adaptive rows показывают selection и window search.
Такая последовательная ablation измеряет условный вклад группы после предыдущих;
training gain не следует приравнивать к этому вкладу.

![Validation comparison](artifacts/plots/metric_by_experiment.png)
![Feature budget](artifacts/plots/metric_vs_features.png)

## 14. Results

Зафиксированные модели, audit validation, seed 42:

{markdown_table(audit.to_dicts(), metrics_columns)}

Final test, seed 42:

{markdown_table(test.to_dicts(), metrics_columns)}

Test mean и SD между seeds; warmup arms имеют один seed и меньший объем fit:

{markdown_table(summaries.to_dicts(), [('experiment_name','Model'),('seeds','Seeds'),('logloss_mean','Mean LogLoss'),('logloss_sd','SD'),('auc_mean','Mean AUC'),('auc_sd','SD')])}

Основное paired сравнение, automatic − manual:
ΔLogLoss = **{comparison['delta_logloss']:.6f}**, user-cluster bootstrap 95% CI
**[{comparison['logloss_ci_95'][0]:.6f}, {comparison['logloss_ci_95'][1]:.6f}]**;
ΔAUC = {comparison['delta_auc']:.6f}, 95% CI
[{comparison['auc_ci_95'][0]:.6f}, {comparison['auc_ci_95'][1]:.6f}].
На audit ΔLogLoss = {audit_comparison['delta_logloss']:.6f}.
Bootstrap {comparison['repeats']} repeats по {comparison['bootstrap_users']} users,
фиксированные модели; CI не включает неопределенность всей процедуры search/training.

{conclusion}

Средний test результат между seeds: automatic − manual LogLoss **{mean_delta:+.6f}**.
Главный вывод учитывает и это измерение, и CI; формально лучшего seed 42 недостаточно.

Ranking: query = user × exact timestamp; берутся только группы >=2 с positive и negative
ratings. AUC/LogLoss считаются по всем событиям. Это reranking proxy для одновременно
оцененных фильмов, без full-catalog negatives. CSV содержит NDCG@3/5/10, Recall@3/5/10,
MAP@3/5/10, число query groups и ranking rows. Нельзя заявлять catalog Recall по этим данным.

![Final test](artifacts/plots/test_metrics.png)

## 15. Selected temporal windows

Окна ниже получены из зафиксированных FeatureSpecs, включая long dependency у trends:

{markdown_table(windows, [('entity','Entity'),('n','Selected specs'),('windows','Windows')])}

Разные selected windows показывают, что search может задавать temporal scale отдельно
для разных families. Это не доказывает существование единственного оптимального окна:
окна коррелированы, search ограничен, gain rankings нестабильны. Shared-window control
и fixed-window comparisons дают прямую проверку полезности дополнительной свободы.

![Windows](artifacts/plots/window_distribution.png)
![Entity windows](artifacts/plots/entity_window_heatmap.png)

## 16. Feature importance

Training gain зафиксированной automatic model:

{markdown_table(importance.head(15).to_dicts(), [('feature','Feature'),('importance_gain','Gain'),('entity','Entity')])}

Gain чувствителен к корреляции и не является причинной importance. Re-fit budget curves
и ablations сильнее подтверждают полезность набора. SHAP не вычислялся; интерпретация
не опирается на несуществующие SHAP/permutation results.

![Importance](artifacts/plots/top30_importance.png)
![Family gain](artifacts/plots/feature_family_gain.png)
![Feature counts](artifacts/plots/feature_counts.png)

## 17. Computational cost

Manual features (+ trend ablation) на {dev_end:,} development rows: **{manual_seconds:.3f} s**,
feature matrix **{profile['manual']['memory_mb']:.1f} MiB**.
Дополнительные auto candidates: **{candidate_seconds:.3f} s**,
matrix **{profile['additional_candidates']['memory_mb']:.1f} MiB**.
Все финальные нужные features на {n_all:,} rows:
**{final['final_feature_profile']['seconds']:.3f} s**.
Сумма logged model fitting time: {training_seconds:.1f} s, HPO fitting дополнительно
{tuning_seconds:.1f} s. Эти суммы не включают весь ETL, prediction, bootstrap и disk I/O.

{markdown_table(costs, [('experiment_name','Experiment'),('n_features','Inputs'),('training_time','Fit seconds'),('memory_usage','Peak process MiB')])}

RSS — peak процесса с удерживаемыми матрицами; значения не следует трактовать как
изолированное потребление конкретной модели. В experiments.csv feature generation=0 и
cache_hit=True означает повторное использование, а не нулевую стоимость расчета.
Фабрика не использует Python loops по interaction rows; циклы идут по families/specs.
Naive row loops существуют только в маленьком тестовом oracle.

## 18. Limitations

Один explicit-feedback dataset, один глобальный temporal cutoff и два validation сегмента;
это не доказательство для CTR, CVR или всех recommender systems. Search многократно
использует search validation и может переобучиться. Audit/test ограничивают этот риск,
но не заменяют rolling-origin trials и внешние datasets. Выбранное HPO на sample train
может не быть оптимальным на полном train. Static metadata availability и instant labels
предполагаются. Primary genre — упрощение; impression logs отсутствуют. Exact user–item
counts бесполезны при отсутствии повторов и удаляются pruning. Sparse/correlated filters
могут удалять признаки, полезные лишь для редкого сегмента. Небольшой deterministic agent
не является полноценным autonomous AutoML и не доказывает преимущество LLM.

SVD basis фиксирован на первых 10%; это гарантирует cutoff, но ограничивает coverage и
адаптацию. Warmup comparisons имеют собственные controls и не смешиваются с main arms.
Отрицательный embedding результат не опровергает ALS/item2vec/GNN вообще.

## 19. Production scalability

Измерен локальный pipeline около 1M interactions. 1B+ не запускался, утверждений о
фактически достигнутой distributed performance нет. Широкая dense матрица требует
O(NF) памяти: 1B × 400 float32 ≈ 1.6 TB без overhead. Production: entity-sharded event
state, rolling counters/deques, point-in-time feature store, offline partitioned ETL,
cache aggregation primitives, bounded candidate search на выборке и serving только
выбранного набора. All-history nunique требует growing state или approximate sketches;
exact arbitrary rolling min/max/distinct имеют разные стоимости и не становятся дешевыми
только благодаря DSL. Для late arrivals нужны watermark/retractions и backfill.

Graph models требуют distributed graph/negative sampling; Transformers — sequence
truncation, distributed training, embedding tables, freshness и serving infrastructure.
100B events — отдельная инфраструктурная задача; прямой запуск этого local pipeline
на таком размере не предлагается.

## 20. Conclusions

**Можно автоматизировать генерацию и отбор temporal features**, сохранив контролируемую
семантику и интерпретируемый boosting. Реализован полный цикл от schema до KEEP/DROP,
получены реальные selected windows, pruning curves и честные test metrics.

**{conclusion}**

На главный вопрос отвечаем ограниченно: человек меньше выбирает отдельные features,
но по-прежнему задает target, допустимые relations/primitives/windows, протокол валидации
и budget. Автоматизация процесса сама по себе не гарантирует роста качества.

## 21. Future work

Повторить на impression/event данных с delayed labels; несколько rolling-origin folds
и внешний dataset; полноценные multilabel genre histories; group permutation importance;
cost-aware feature search и streaming compiled backend; time-decay kernels вместо только
hard windows; несколько safe embedding snapshots; matched-budget shared/entity windows;
LLM proposer с тем же allowlist без executable SQL; настоящее LGBMRanker evaluation
на реальных request candidate sets и затем online A/B test.
'''
    text = text.replace("## 15. Selected temporal windows", diagnostics + "\n## 15. Selected temporal windows")
    if full_eda:
        cold_rows = [{"split": k, "cold": v["cold_users_fraction"], "start": v["start"], "end": v["end"]}
                     for k, v in full_eda["splits"].items()]
        text = text.replace("## 8. Leakage prevention", "Доля строк с user ID вне исходного train, и временные границы:\n\n" +
                            markdown_table(cold_rows, [("split", "Split"), ("cold", "Cold-user fraction"),
                                                       ("start", "Start UTC"), ("end", "End UTC")]) +
                            "\n\n## 8. Leakage prevention")
    segment_path = root / "test_segments.csv"
    if segment_path.exists():
        segment_table = markdown_table(pl.read_csv(segment_path).to_dicts(), [("model","Model"), ("segment","User segment"),
                                    ("rows","Rows"), ("logloss","LogLoss"), ("auc","AUC")])
        text = text.replace("## 15. Selected temporal windows", "Test diagnostics по known/unknown user ID относительно final fit prefix, seed 42:\n\n" +
                            segment_table + "\n\nЭтот анализ выполнен после lock; он не меняет выбранные модели.\n\n## 15. Selected temporal windows")
    import os
    relative_plots = os.path.relpath(root / "plots", Path(destination).parent)
    text = text.replace("artifacts/plots/", relative_plots + "/")
    Path(destination).write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--report", default="REPORT.md")
    args = parser.parse_args()
    write_report(args.output, args.report)
