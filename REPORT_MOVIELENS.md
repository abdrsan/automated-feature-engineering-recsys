# Автоматизация temporal feature engineering для рекомендательного scoring

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
Всего в основном запуске **1,000,209 interactions**. EDA development prefix: 850,177
строк, 5,715 users, 3,669 items, диапазон 2000-04-25 23:05:32 — 2000-12-10 04:20:42 UTC.
В prefix найдено 0 дубликатов полных событий и
0 повторных user–item строк;
479,766 строк сверх числа уникальных timestamps.

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
| Train | 700,144 | Fit и train-only preprocessing/pruning |
| Validation search | 75,017 | HPO, feature budget, окна, KEEP/DROP |
| Validation audit | 75,016 | Проверка уже зафиксированных решений |
| Test | 150,032 | Финальное сравнение заранее заявленных моделей |

Равные timestamps целиком принадлежат одному разделу. Категории кодируются только по
fit prefix; unseen values имеют code -1. После lock выполняются три seeds
[42, 2024, 7]; результаты seeds не меняют selection. Final models refit
на первых 850,177 строках с фиксированным числом деревьев, выбранным на search validation.
Audit scores принадлежат development models, test scores — refitted models; их нельзя
сравнивать как один и тот же trained checkpoint.

Доля строк с user ID вне исходного train, и временные границы:

| Split | Cold-user fraction | Start UTC | End UTC |
| --- | --- | --- | --- |
| train | 0.000000 | 2000-04-25 23:05:32 | 2000-11-22 03:06:09 |
| validation_search | 0.720263 | 2000-11-22 03:06:26 | 2000-11-28 23:13:10 |
| validation_audit | 0.811280 | 2000-11-28 23:13:14 | 2000-12-10 04:20:42 |
| test | 0.493261 | 2000-12-10 04:20:54 | 2003-02-28 17:49:50 |

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
ablation. Сильнейший manual вариант выбран на validation search: **manual_plus_trend**.
Manual HPO: 20 Optuna trials, до
300,000 случайно выбранных **training rows** для
ускорения fitting, с полной историей признаков; победитель затем обучается на полном train.
Это sampling внутри fit, не случайный split данных.

## 10. Automated feature generation

FeatureSpec содержит keys, event, aggregation, window_days и безопасные параметры derived
features. Поддержаны global/user/item/user–item/category/user–category/item–gender,
а generic generator обнаруживает brand/author/merchant при наличии колонок. Реальные
candidate windows: [1, 3, 7, 14, 21, 30, 60, 90, 180, None] days, None = all-history.
Поддержаны count/sum/mean/min/max/nunique/recency/event_rate/conversion/frequency/trend.
В candidate space отсутствует frequency для фиксированного окна: это линейный дубликат count.
Finite global/category nunique не генерируется из-за дорогой exact distinct cardinality;
bounded entity distinct и all-history distinct сохраняются.

В исходном пространстве **482 FeatureSpecs**, после train-only pruning
**382**, в выбранном автоматическом наборе **128**
(base inputs считаются отдельно). Trends — сглаженные short/long count ratios, не rates
нормированные на длину окна. Conversion primitive означает отношение event counts;
на MovieLens он не трактуется как purchase funnel.

## 11. Feature search strategy

Hybrid pipeline: large bounded generation → constants/missing/duplicates/correlation pruning
только на train → train gain ranking → validation re-fit нескольких feature budgets →
адаптивные окна → 20 automatic HPO trials →
bounded feature-agent proposals. Fixed-window selection ограничена 7/30 days и all history.
Adaptive search расширяет окна; состав выбирается по loss, а не заранее по entity.
Shared-window control разрешает одни три окна всем entities с близким feature budget.
Этот control диагностический: он не полностью устраняет различия в корреляции и составе.

Агент выполнил 2 rounds, принято 0. Он предлагает соседние окна
семейств с высоким gain. Группа принимается при улучшении LogLoss не менее
0.0001. LLM integration ограничена входным
JSON валидатором; оплачиваемого API и произвольного SQL execution нет. Подробный trace:
`artifacts/agent_history.json`. Выбранная модель: **auto_tuned**.

## 12. Experiments

Все строки ниже вычислены реальными запусками на search validation, seed 42.
Все HPO trials сохранены отдельно; это позволяет видеть полный budget.

| Experiment | Inputs | LogLoss ↓ | AUC ↑ | proxy NDCG@5 | proxy NDCG@10 | proxy Recall@10 |
| --- | --- | --- | --- | --- | --- | --- |
| base_default | 10 | 0.583951 | 0.752600 | 0.833940 | 0.848634 | 0.998895 |
| base_categorical | 10 | 0.583428 | 0.753055 | 0.833756 | 0.848660 | 0.998938 |
| plus_user_temporal | 25 | 0.546814 | 0.790459 | 0.834230 | 0.849098 | 0.998757 |
| plus_item_temporal | 40 | 0.545893 | 0.791214 | 0.836418 | 0.851070 | 0.998891 |
| plus_user_item | 55 | 0.545893 | 0.791214 | 0.836418 | 0.851070 | 0.998891 |
| plus_user_category | 70 | 0.541376 | 0.794965 | 0.837553 | 0.851974 | 0.998943 |
| manual | 82 | 0.538769 | 0.797552 | 0.836814 | 0.851190 | 0.998893 |
| manual_plus_trend | 91 | 0.538529 | 0.797588 | 0.836763 | 0.850887 | 0.998949 |
| auto_fixed_windows_all | 146 | 0.538431 | 0.797934 | 0.837771 | 0.851897 | 0.998804 |
| auto_fixed_windows_top32 | 42 | 0.538446 | 0.797799 | 0.836992 | 0.851248 | 0.998808 |
| auto_fixed_windows_top64 | 74 | 0.538418 | 0.797749 | 0.837901 | 0.852100 | 0.998880 |
| auto_fixed_windows_top96 | 106 | 0.538161 | 0.798047 | 0.837873 | 0.851901 | 0.998888 |
| auto_fixed_windows_top128 | 138 | 0.538186 | 0.798165 | 0.837649 | 0.851613 | 0.998842 |
| auto_adaptive_all | 392 | 0.537486 | 0.798625 | 0.838106 | 0.852229 | 0.998836 |
| auto_adaptive_top32 | 42 | 0.538395 | 0.798013 | 0.837813 | 0.852218 | 0.998920 |
| auto_adaptive_top64 | 74 | 0.537422 | 0.798671 | 0.838591 | 0.852817 | 0.998831 |
| auto_adaptive_top96 | 106 | 0.537304 | 0.798612 | 0.838883 | 0.853140 | 0.998909 |
| auto_adaptive_top128 | 138 | 0.537103 | 0.798952 | 0.837599 | 0.851893 | 0.998820 |
| shared_windows_control | 138 | 0.537142 | 0.799023 | 0.837593 | 0.852001 | 0.998847 |
| auto_tuned | 138 | 0.536670 | 0.799271 | 0.838051 | 0.852529 | 0.998883 |
| agent_round_1 | 146 | 0.537274 | 0.798747 | 0.837262 | 0.851434 | 0.998825 |
| agent_round_2 | 146 | 0.537157 | 0.798937 | 0.838631 | 0.852558 | 0.998835 |
| manual_warmup_control | 91 | 0.539397 | 0.796902 | 0.836676 | 0.851003 | 0.998892 |
| auto_warmup_control | 138 | 0.537982 | 0.798033 | 0.838422 | 0.852633 | 0.998812 |
| auto_plus_svd | 142 | 0.532512 | 0.803644 | 0.844095 | 0.857967 | 0.998866 |

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

| Experiment | Inputs | LogLoss ↓ | AUC ↑ | proxy NDCG@5 | proxy NDCG@10 | proxy Recall@10 |
| --- | --- | --- | --- | --- | --- | --- |
| manual | 91 | 0.541128 | 0.791123 | 0.846546 | 0.855911 | 0.998546 |
| automatic | 138 | 0.539278 | 0.792951 | 0.848356 | 0.857652 | 0.998477 |
| manual_warmup | 91 | 0.541887 | 0.790491 | 0.846057 | 0.855319 | 0.998479 |
| auto_warmup | 138 | 0.540101 | 0.792119 | 0.847655 | 0.856769 | 0.998392 |
| auto_plus_svd | 142 | 0.533413 | 0.799253 | 0.851273 | 0.859972 | 0.998511 |
| auto_fixed_windows_top96 | 106 | 0.540400 | 0.791903 | 0.847286 | 0.856510 | 0.998522 |
| auto_adaptive_top128 | 138 | 0.539829 | 0.792432 | 0.848061 | 0.857321 | 0.998432 |
| shared_windows_control | 138 | 0.539616 | 0.792799 | 0.848504 | 0.857668 | 0.998410 |

Final test, seed 42:

| Experiment | Inputs | LogLoss ↓ | AUC ↑ | proxy NDCG@5 | proxy NDCG@10 | proxy Recall@10 |
| --- | --- | --- | --- | --- | --- | --- |
| manual | 91 | 0.543486 | 0.792902 | 0.845102 | 0.852519 | 0.998924 |
| automatic | 138 | 0.543318 | 0.792938 | 0.843825 | 0.851343 | 0.998873 |
| manual_warmup | 91 | 0.543670 | 0.792822 | 0.844022 | 0.851500 | 0.998922 |
| auto_warmup | 138 | 0.543461 | 0.792682 | 0.844305 | 0.851684 | 0.998911 |
| auto_plus_svd | 142 | 0.538216 | 0.798166 | 0.849780 | 0.857082 | 0.998915 |

Test mean и SD между seeds; warmup arms имеют один seed и меньший объем fit:

| Model | Seeds | Mean LogLoss | SD | Mean AUC | SD |
| --- | --- | --- | --- | --- | --- |
| auto_plus_svd | 1 | 0.538216 | 0.000000 | 0.798166 | 0.000000 |
| auto_warmup | 1 | 0.543461 | 0.000000 | 0.792682 | 0.000000 |
| automatic | 3 | 0.543922 | 0.000708 | 0.792412 | 0.000623 |
| manual | 3 | 0.543468 | 0.000056 | 0.792969 | 0.000079 |
| manual_warmup | 1 | 0.543670 | 0.000000 | 0.792822 | 0.000000 |

Основное paired сравнение, automatic − manual:
ΔLogLoss = **-0.000168**, user-cluster bootstrap 95% CI
**[-0.000855, 0.000506]**;
ΔAUC = 0.000037, 95% CI
[-0.000646, 0.000773].
На audit ΔLogLoss = -0.001850.
Bootstrap 300 repeats по 1512 users,
фиксированные модели; CI не включает неопределенность всей процедуры search/training.

Устойчивое преимущество автоматических temporal features на test не доказано: небольшой выигрыш для seed 42 имеет CI, включающий ноль, а проверка нескольких seeds не подтверждает превосходство.

Средний test результат между seeds: automatic − manual LogLoss **+0.000454**.
Главный вывод учитывает и это измерение, и CI; формально лучшего seed 42 недостаточно.

Ranking: query = user × exact timestamp; берутся только группы >=2 с positive и negative
ratings. AUC/LogLoss считаются по всем событиям. Это reranking proxy для одновременно
оцененных фильмов, без full-catalog negatives. CSV содержит NDCG@3/5/10, Recall@3/5/10,
MAP@3/5/10, число query groups и ranking rows. Нельзя заявлять catalog Recall по этим данным.

![Final test](artifacts/plots/test_metrics.png)

## Диагностика основного результата

Средний test LogLoss между тремя seeds: manual
**0.543468**, automatic
**0.543922**; Δ = **+0.000454**.
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
| Помогают ли ratios/trends? | Manual rates: Δ validation LogLoss -0.002607; 7/30 trends: -0.000240. В финальном automatic наборе 0 explicit trend specs. |
| Сколько можно удалить? | Из 482 initial candidates после pruning осталось 382; selection использует 128. |
| Какова стоимость? | Manual generation 4.882s; manual + дополнительные candidates 14.203s, плюс повторное обучение. |
| Что дает рост feature budget? | Adaptive full: 392 inputs / LogLoss 0.537486; лучшая budget arm: 138 / 0.537103. |
| Есть ли diminishing returns? | Полный набор не лучше лучшей меньшей budget arm; больше колонок не гарантирует качество. |
| Какие families полезны? | Historical item rating means, item–gender means, user rates/recency и user–category means имеют высокий training gain; ablations дают независимую проверку части families. |
| Меньше ли ручной работы? | Конкретные spec/window choices делает pipeline. Target, schema relations, grammar, budget и availability assumptions остаются решениями человека. |

### Дополнительная learned representation

После freeze SVD basis на warmup prefix user profile строится только из предыдущих signed
item vectors. Добавлены dot, cosine, profile norm и item norm. На final test сравнение
auto+SVD с auto_warmup имеет одинаковое начало fit, поэтому выигрыш не объясняется
разным количеством training rows. Это отдельно проверенный путь улучшения, а не
доказательство превосходства автоматических temporal features самих по себе.

SVD ΔLogLoss = **-0.005244**, 95% paired user CI
**[-0.006223, -0.004300]**;
ΔAUC = 0.005483. SVD arm проверен на одном seed, его CI условен на fixed models.
Manual+SVD на test не сравнивался; нельзя приписать весь выигрыш hybrid именно auto search.

Test diagnostics по known/unknown user ID относительно final fit prefix, seed 42:

| Model | User segment | Rows | LogLoss | AUC |
| --- | --- | --- | --- | --- |
| manual | known_user_id | 101667 | 0.541279 | 0.797473 |
| manual | unknown_user_id | 48365 | 0.548126 | 0.781876 |
| automatic | known_user_id | 101667 | 0.541463 | 0.797211 |
| automatic | unknown_user_id | 48365 | 0.547219 | 0.782570 |
| manual_warmup | known_user_id | 101667 | 0.541339 | 0.797461 |
| manual_warmup | unknown_user_id | 48365 | 0.548571 | 0.781437 |
| auto_warmup | known_user_id | 101667 | 0.541242 | 0.797306 |
| auto_warmup | unknown_user_id | 48365 | 0.548124 | 0.781652 |
| auto_plus_svd | known_user_id | 101667 | 0.536277 | 0.802199 |
| auto_plus_svd | unknown_user_id | 48365 | 0.542292 | 0.787888 |

Этот анализ выполнен после lock; он не меняет выбранные модели.

## 15. Selected temporal windows

Окна ниже получены из зафиксированных FeatureSpecs, включая long dependency у trends:

| Entity | Selected specs | Windows |
| --- | --- | --- |
| category | 18 | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 180d, all_history |
| global | 1 | 180d |
| item | 15 | 1d, 3d, 7d, 60d, 90d, 180d, all_history |
| item_gender | 30 | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, 180d, all_history |
| user | 36 | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, all_history |
| user_category | 28 | 1d, 3d, 7d, 14d, 21d, 30d, 60d, 90d, all_history |

Разные selected windows показывают, что search может задавать temporal scale отдельно
для разных families. Это не доказывает существование единственного оптимального окна:
окна коррелированы, search ограничен, gain rankings нестабильны. Shared-window control
и fixed-window comparisons дают прямую проверку полезности дополнительной свободы.

![Windows](artifacts/plots/window_distribution.png)
![Entity windows](artifacts/plots/entity_window_heatmap.png)

## 16. Feature importance

Training gain зафиксированной automatic model:

| Feature | Gain | Entity |
| --- | --- | --- |
| item__all__mean__all__rating | 264601.673096 | item |
| user_id | 149290.338699 | base |
| item__all__mean__180d__rating | 80835.313187 | item |
| item_gender__all__mean__all__rating | 62445.921127 | item_gender |
| item_gender__all__mean__180d__rating | 40218.022369 | item_gender |
| item_id | 39931.758808 | base |
| user__positive__event_rate__1d__all | 35225.020435 | user |
| item__positive__event_rate__all__all | 29944.453552 | item |
| user__positive__event_rate__7d__all | 28997.799713 | user |
| user__negative__recency__all | 28699.875622 | user |
| user_category__all__mean__all__rating | 25736.800537 | user_category |
| user__positive__event_rate__90d__all | 25431.967438 | user |
| user__positive__event_rate__60d__all | 19662.061966 | user |
| user_category__all__mean__3d__rating | 18452.218098 | user_category |
| user_category__positive__event_rate__90d__all | 18322.651310 | user_category |

Gain чувствителен к корреляции и не является причинной importance. Re-fit budget curves
и ablations сильнее подтверждают полезность набора. SHAP не вычислялся; интерпретация
не опирается на несуществующие SHAP/permutation results.

![Importance](artifacts/plots/top30_importance.png)
![Family gain](artifacts/plots/feature_family_gain.png)
![Feature counts](artifacts/plots/feature_counts.png)

## 17. Computational cost

Manual features (+ trend ablation) на 850,177 development rows: **4.882 s**,
feature matrix **268.4 MiB**.
Дополнительные auto candidates: **9.321 s**,
matrix **1312.0 MiB**.
Все финальные нужные features на 1,000,209 rows:
**10.090 s**.
Сумма logged model fitting time: 297.5 s, HPO fitting дополнительно
169.7 s. Эти суммы не включают весь ETL, prediction, bootstrap и disk I/O.

| Experiment | Inputs | Fit seconds | Peak process MiB |
| --- | --- | --- | --- |
| base_default | 10 | 6.911488 | 1306.265625 |
| base_categorical | 10 | 7.667072 | 1606.750000 |
| plus_user_temporal | 25 | 6.958766 | 1705.171875 |
| plus_item_temporal | 40 | 4.686538 | 1974.328125 |
| plus_user_item | 55 | 4.746820 | 2135.296875 |
| plus_user_category | 70 | 6.171977 | 2071.187500 |
| manual | 82 | 6.056401 | 2352.796875 |
| manual_plus_trend | 91 | 6.833088 | 2414.390625 |
| auto_fixed_windows_all | 146 | 12.058867 | 5005.968750 |
| auto_fixed_windows_top32 | 42 | 5.170321 | 4795.609375 |
| auto_fixed_windows_top64 | 74 | 7.416910 | 4826.015625 |
| auto_fixed_windows_top96 | 106 | 9.875985 | 4500.843750 |
| auto_fixed_windows_top128 | 138 | 10.629571 | 4548.890625 |
| auto_adaptive_all | 392 | 28.504428 | 5799.843750 |
| auto_adaptive_top32 | 42 | 4.897340 | 4484.468750 |
| auto_adaptive_top64 | 74 | 7.566524 | 4884.234375 |
| auto_adaptive_top96 | 106 | 10.145238 | 5567.875000 |
| auto_adaptive_top128 | 138 | 10.997004 | 5512.234375 |
| shared_windows_control | 138 | 12.040262 | 4393.984375 |
| auto_tuned | 138 | 8.173297 | 2790.671875 |
| agent_round_1 | 146 | 8.577918 | 3643.812500 |
| agent_round_2 | 146 | 8.439301 | 3818.265625 |
| manual_warmup_control | 91 | 5.414350 | 4173.546875 |
| auto_warmup_control | 138 | 6.630409 | 4119.984375 |
| auto_plus_svd | 142 | 6.625562 | 3987.281250 |

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

**Устойчивое преимущество автоматических temporal features на test не доказано: небольшой выигрыш для seed 42 имеет CI, включающий ноль, а проверка нескольких seeds не подтверждает превосходство.**

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
