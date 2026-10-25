# Техническое исследование

## Основные данные: KION и Yambda

По уточнению постановки основное исследование теперь использует
[KION](https://github.com/irsafilo/KION_DATASET) и
[Yambda](https://huggingface.co/datasets/yandex/yambda).
MovieLens остается архивным предварительным экспериментом.

KION дает богатые film/user metadata и агрегированные user–item watches с последней
датой. Actual public archive содержит 1,594,787 строк, после удаления 268 строк без
watched_pct — 1,594,519. README описывает больший dataset; для выводов используется
фактический скачанный файл. Target — watched_pct>=50 среди наблюдаемых агрегатов;
весь текущий день исключается. Это не impression-level CTR и не воспроизведение contest.

Yambda дает настоящие action types и повторные user–item события. Практичный локальный
эксперимент: все 989,232 likes/dislikes из flat/50m, плюс artist/album mappings.
Классификация like vs dislike относится к conditional explicit-feedback scoring.
Listens/unlikes/undislikes в этом run не включены: historical event counts нельзя
считать полным preference state. Полные аудиоэмбеддинги не загружаются; controlled SVD
из прошлого — независимая дешевая representation alternative.

[Статья Yambda](https://arxiv.org/html/2505.22238v1) задает относительное время в секундах,
округленное к 5s: floor((T−T0)/5)*5. Дополнительное умножение на пять недопустимо.
Calendar fields отключены; epoch datetime служит только внутренней координатой.

Выбор подхода сохраняется: ограниченная temporal factory изолирует исследуемые windows
и relations при одинаковом LightGBM backend. Graph/sequence модели были бы естественнее
для full-catalog retrieval, но меняли бы target и evaluation protocol. Их результаты
нельзя напрямую сравнивать с нашим conditional scoring. Детали доступности labels,
timestamp granularity и metadata assumptions входят в REPORT.md и standalone notebook.

Исследовательский вопрос: можно ли заменить ручной выбор конкретных temporal features
поиском в ограниченной грамматике `entity × event × aggregation × window`, сохранив
градиентный бустинг, строгий cutoff и интерпретируемость?

Ниже приведены инженерные оценки для этого задания, а не измеренные сравнительные
результаты перечисленных моделей. «Высокое качество» означает потенциал при подходящем
датасете. Полные GNN и Transformers в этом репозитории не обучаются.

| Подход | Потенциал качества | Сложность и стоимость | 1M | 1B+ | Leakage risk | Интерпретируемость | Как доказать результат |
|---|---|---|---|---|---|---|---|
| Manual LightGBM/CatBoost | Высокий для табличного scoring | Низкая/средняя; CPU | Реалистично | Distributed ETL, offline training, sampled fitting | Средний: агрегаты и target encoding | Высокая | Temporal holdout, ablation |
| DFS / Featuretools | Средний/высокий, зависит от primitives | Средняя; relational space быстро растет | Ограниченный depth и число features | Нужен специализированный агрегатор, не прямой pandas запуск | Cutoff поддерживается, границу нужно настроить | Высокая | Сравнить одинаковые признаки и cutoff |
| Large temporal factory + gain selection | Высокий потенциал | Средняя; широкая матрица дорогая | Несколько сотен features | Распределенные window states и поиск на выборке | Средний; единый движок облегчает аудит | Высокая | Фиксированные vs найденные окна, pruning curve |
| Greedy group search | Высокий при небольшом бюджете | Много последовательных fitting rounds | Реалистично для десятков групп | Offline выборка + переиспользование расчетов | Validation overfitting при многих шагах | Высокая | Лог KEEP/DROP и следующий временной holdout |
| Optuna feature masks | Высокий потенциал, не гарантирован | Экспоненциальная структура масок, TPE не устраняет сложность | Компактные family masks | Поиск на репрезентативной выборке | Многократный selection bias | Высокая | Равный budget и нетронутый test |
| RL / Neural Feature Search | Потенциал зависит от задачи | Высокая; controller и множество переобучений | Возможно, но не оправдано здесь | Специализированная инфраструктура | Cutoff и reward contamination | Средняя | Нужно отделить вклад controller от числа trials |
| AutoFIS / AutoField / AutoRecSys | Высокий для CTR при богатых полях | Средняя/высокая; differentiable field/interaction search | Реалистично с GPU | Distributed embeddings и обучение | Временной split, leakage в preprocessing | Средняя/низкая | Temporal reimplementation, разные search budgets |
| MF / ALS / item2vec / graph embeddings + GBM | Высокий для collaborative structure | Средняя | Реалистично | Sharding, incremental или periodic refresh | Высокий: train-row embedding нельзя fit на будущем train | Средняя | Prefix-frozen basis и одинаковые train rows |
| LightGCN / heterogeneous GNN link prediction | Высокий для каталогового retrieval | Высокая, граф и negative sampling | Возможно на GPU | Distributed graph sampling, freshness и cold start | Высокий: будущие edges в графе | Средняя/низкая | Retrieval protocol, одинаковые кандидаты |
| SASRec / sequential Transformer | Высокий для next-item prediction | Высокая; другая цель и candidate evaluation | Возможно | Distributed training, bounded sequences и serving | Высокий: sequence masks, split и target alignment | Низкая | Next-item benchmark с честным cutoff |
| TabTransformer / FT-Transformer | Потенциал на табличных данных | Выше GBM; сами историю не строят | Возможно, GPU полезен | Sampling / distributed training | Исторические features все равно нужны | Низкая | Равные inputs и training budget |
| TabPFN-like | Сильный small-data потенциал | Foundation-model inference; не temporal engine | Не основной инструмент для 1M | Прямой full-data запуск не рассматривается | Preprocessing и контекст labels | Низкая | Отдельный обоснованный small-data протокол |

## Что переносим из опубликованных подходов

**Deep Feature Synthesis.** Признак строится композицией relational paths и aggregation /
transformation primitives. Отсюда берем идею декларативной грамматики, но ограничиваем
глубину отношений и число окон. Полный DFS здесь добавил бы стоимость и усложнил аудит
без ответа на главный вопрос об автоматическом выборе временных масштабов.
Источники: [Kanter & Veeramachaneni, DSAA 2015](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf),
[исходный код Featuretools](https://github.com/alteryx/featuretools).

**Temporal cutoff.** Featuretools предоставляет cutoff_time и training_window. Его
`include_cutoff_time` по умолчанию равен True; для нашего контракта необходимо False.
Мы используем Polars `rolling(closed="left")`, а для all-history сначала объединяем
равные timestamps и лишь затем применяем cumulative/shift. SQL-подобная семантика
явно содержит `h.timestamp < q.timestamp`.
Источники: [Featuretools Handling Time](https://featuretools.alteryx.com/en/stable/getting_started/handling_time.html),
[Polars rolling API](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.rolling.html).

**Generic feature generation и RL.** autofeat генерирует нелинейные преобразования с
последующим отбором. NFS применяет recurrent controller и reinforcement learning к
цепочкам преобразований. Эти идеи полезны как generate→evaluate→select, но арифметические
преобразования уже готовых колонок не заменяют causal temporal aggregation. RL потребовал
бы дополнительных evaluation rounds и отдельного эксперимента о качестве controller.
Источники: [autofeat](https://arxiv.org/abs/1901.07329),
[NFS, ICDM 2019](https://www.microsoft.com/en-us/research/publication/neural-feature-search-a-neural-architecture-for-automated-feature-engineering/),
[AutoFS](https://arxiv.org/abs/2008.12001).

**AutoML в RecSys.** AutoFIS выбирает взаимодействия признаков в factorization-based CTR
моделях; AutoField выбирает поля differentiable controller-ом. Они автоматизируют другой
уровень пространства: наличие полей и взаимодействий уже представленных inputs. Ни один
из этих механизмов сам по себе не устанавливает, из какого исторического интервала считать
покупки или оценки. AutoRecSys — направление поиска архитектур, embeddings и interaction
blocks, а не гарантия замены исторической feature factory. Для нашей задачи структура
истории первична, neural interaction search оставляем альтернативой.
Источники: [AutoFIS](https://arxiv.org/abs/2003.11235),
[AutoField](https://arxiv.org/abs/2204.09078).

Из конкретных framework-ов [AutoRec / autorecsys](https://github.com/datamllab/AutoRec)
предлагает Keras-based search для rating prediction и CTR; его
[paper](https://arxiv.org/abs/2007.07224) описывает model search и HPO в TensorFlow ecosystem.
[Auto-Surprise](https://github.com/ISG-Siegen/Auto-Surprise) автоматизирует выбор
алгоритма и гиперпараметров Surprise с TPE, см.
[paper](https://arxiv.org/abs/2008.13532). Это альтернативы автоматизации model choice,
но они не заменяют требование строгой временной доступности inputs. Для данного
исследования замена одновременно feature engine и model family мешала бы изолировать
вклад temporal search. Обобщение search fields/embeddings/interactions/architectures
представлено в [AutoML for Deep Recommender Systems](https://arxiv.org/abs/2203.13922).

**Boosting и ranking.** LightGBM поддерживает бинарное scoring и LambdaRank с query groups.
Competition и production experience NVIDIA демонстрирует применение GBM совместно с
learned representations. Это поддерживает практическую применимость hybrid подхода, но
не доказывает преимущество нашей фабрики на KION или Yambda. Здесь оптимизируем logloss;
ranking по наблюдаемым взаимодействиям одного времени — вспомогательный proxy.
Источники: [LGBMRanker API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html),
[NVIDIA RecSys best practices](https://docs.nvidia.com/deeplearning/performance/recsys-best-practices/index.html),
[competition ETL и XGBoost notebook](https://nvidia-merlin.github.io/NVTabular/v1.0.0/examples/winning-solution-recsys2020-twitter/01-02-04-Download-Convert-ETL-with-NVTabular-Training-with-XGBoost.html).

**Representation learning.** LightGCN упрощает graph convolution до propagation по
user-item графу. SASRec использует attention по предыдущим items для next-item prediction.
В нашей постановке link prediction и next-item отличаются от conditional scoring
наблюдаемого просмотра или реакции. Поэтому реализуем небольшой SVD/co-occurrence hybrid: basis
обучается только на warmup prefix, а user profile обновляется строго по прошлому. Полный
граф будущих train edges недопустим даже при последующем temporal train/test split.
Источники: [LightGCN, SIGIR 2020](https://arxiv.org/abs/2002.02126),
[LightGCN implementation](https://github.com/gusye1234/LightGCN-PyTorch),
[SASRec, ICDM 2018](https://arxiv.org/abs/1808.09781),
[авторская реализация SASRec](https://github.com/kang205/SASRec).

**Tabular Transformers.** TabTransformer contextualizes categorical embeddings;
FT-Transformer is a general tabular backbone. Они не получают temporal history из
таблицы событий автоматически. В original TabPFN paper 2025 исследуется small-data regime
до 10k samples; это характеристика цитируемой работы, а не утверждение о всех будущих
версиях библиотеки. Для 1M interactions сравнение с ними отвлекло бы бюджет от temporal
feature search.
Источники: [TabTransformer](https://arxiv.org/abs/2012.06678),
[FT-Transformer, NeurIPS 2021](https://arxiv.org/abs/2106.11959),
[TabPFN, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6).

## Выбранная гипотеза и trade-off

Основной метод: bounded temporal factory → train-only pruning → train gain ranking →
validation selection по размеру набора и доступным окнам → небольшая Optuna настройка →
две bounded agent proposals. Fixed-window selection служит контрольным экспериментом.
Человек задает schema relations, event meaning и допустимую грамматику; отдельные окна
и итоговые признаки выбираются по temporal validation. Это перенос ручной работы на
уровень search space, а не полная автоматизация исследования.

Оптимизация гиперпараметров использует [Optuna / TPE](https://arxiv.org/abs/1907.10902).
Gain — дешевый, но смещенный proposal signal; он не доказывает причинную полезность.
Полезность итогового набора проверяется переобучением, audit validation и final test.
Попарно коррелированные окна и ограниченный budget не позволяют объявлять выбранное
окно единственным «правильным» временем памяти пользователя.

## Dataset provenance

MovieLens 1M предоставлен GroupLens для исследований. Рейтинги являются explicit feedback,
а не кликами или показами. Источники:
[официальный датасет](https://grouplens.org/datasets/movielens/1m/),
[официальный README и условия использования](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt),
[Harper & Konstan, 2015: History and Context](https://files.grouplens.org/papers/harper-tiis2015.pdf).
Сырые данные исключены из Git; скачивание воспроизводится отдельно.
