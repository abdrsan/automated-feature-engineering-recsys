import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml

from src.auto_feature_generator import generate_candidates, discover_entities
from src.baseline_features import ablation_groups, manual_features, manual_trends
from src.data import dataset_fingerprint, describe_data, load_interactions, load_movielens, save_json, study_statistics
from src.datasets import load_kion, load_yambda, dataset_protocol
from src.experiments import ExperimentRunner, MemoryTracker, candidate_metadata
from src.feature_agent import propose_features
from src.feature_engine import generate_features, FEATURE_ENGINE_VERSION, feature_engine_signature
from src.feature_selection import prune_candidates, prune_fixed_candidates
from src.feature_spec import FeatureSpec
from src.metrics import evaluate, paired_user_bootstrap, ranking_metrics
from src.model import base_matrix, fit_encoder, fit_model, tune_model, matrix_from_features
from src.representations import temporal_svd_features
from src.split import temporal_split
from src.provenance import STUDY_PROTOCOL_VERSION, implementation_signature


def selection_signature():
    return implementation_signature(develop, fit_model, tune_model, manual_features, manual_trends,
        ablation_groups, generate_candidates, discover_entities, prune_candidates, prune_fixed_candidates, propose_features,
        ExperimentRunner.run, evaluate, ranking_metrics)


def cached_features(df, specs, root, name):
    folder = root / "cache"
    folder.mkdir(exist_ok=True)
    key = hashlib.sha256((FEATURE_ENGINE_VERSION + feature_engine_signature() + dataset_fingerprint(df)
                         + json.dumps([s.to_dict() for s in specs], sort_keys=True)).encode()).hexdigest()
    path = folder / f"{name}_{key}.parquet"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        print(f"CACHE {name}", flush=True)
        return pl.read_parquet(path), json.loads(meta_path.read_text())
    with MemoryTracker() as memory:
        result, meta = generate_features(df, specs, progress=lambda r: print(f"FEATURES {name}: {r}", flush=True))
    meta["peak_process_mb"] = memory.peak
    meta["key"] = key
    result.write_parquet(path)
    save_json(meta_path, meta)
    return result, meta


def develop(df, split, config, auto, root, only_baseline=False):
    dev = df.head(split.validation_end)
    fingerprint = dataset_fingerprint(df)
    manifest = {"dataset_fingerprint": fingerprint, "config": config, "auto_config": auto,
                "split": vars(split), "dataset": config.get("dataset_name", "MovieLens 1M" if "rating" in df.columns else "custom"),
                "protocol_version": STUDY_PROTOCOL_VERSION, "selection_signature": selection_signature(),
                "engine_version": FEATURE_ENGINE_VERSION, "engine_signature": feature_engine_signature()}
    path = root / "run_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Artifact directory belongs to another dataset/configuration. Choose a new --output.")
    save_json(path, manifest)
    encoder = fit_encoder(dev.head(split.train_end))
    base, base_names, categorical = base_matrix(dev, encoder)
    save_json(root / "encoder_development.json", encoder)
    events = set(dev["event"].unique().to_list())
    manual = manual_features(dev.schema, events)
    trends = manual_trends(dev.schema, events)
    manual_with_trends = manual + trends
    features, manual_meta = cached_features(dev, manual_with_trends, root, "manual")
    runner = ExperimentRunner(dev, features, base, base_names, categorical, split, config, root)
    runner.run("base_default", [], {}, generation_seconds=0.)
    x_manual = runner.matrix([s.name for s in manual])
    params, trials = tune_model(x_manual, dev["label"].to_numpy(), split.train, split.search,
                                base_names + [s.name for s in manual], categorical, config,
                                f"sqlite:///{root.resolve() / 'tuning.sqlite3'}", "manual")
    trials.to_csv(root / "manual_tuning.csv", index=False)
    save_json(root / "manual_params.json", params)
    del x_manual
    gc.collect()
    rows = {}
    manual_groups = dict(ablation_groups(manual_with_trends))
    for name, specs in manual_groups.items():
        rows[name], _ = runner.run(name, [s.name for s in specs], params,
                                  generation_seconds=manual_meta["seconds"] if name == "manual" else 0.)
    manual_name = min(manual_groups, key=lambda n: (rows[n]["logloss"], len(manual_groups[n]), n))
    manual_names = [s.name for s in manual_groups[manual_name]]
    if only_baseline:
        save_json(root / "baseline_selection.json", {"name": manual_name, "features": manual_names,
                                                    "params": params, "selected_on": "validation_search"})
        return

    candidates = list(dict.fromkeys([*generate_candidates(dev.schema, events, auto["windows"], auto["max_features"]),
                                     *manual_with_trends]))
    extra = [s for s in candidates if s.name not in features.columns]
    extras, auto_meta = cached_features(dev, extra, root, "candidates")
    features = features.hstack(extras.drop("row_id"))
    runner.features = features
    keep, reasons = prune_candidates(features, split.train_end, auto, config["seed"])
    reasons.write_csv(root / "pruned_features.csv")
    save_json(root / "generation_profile.json", {"manual": manual_meta, "additional_candidates": auto_meta,
                                                "n_candidates": len(candidates), "n_after_pruning": len(keep)})
    spec_lookup = {s.name: s for s in candidates}

    # Gain помогает выбрать, какие наборы стоит попробовать. Сам по себе высокий gain
    # не доказывает пользу признака: каждый набор заново обучаем и сравниваем на search.
    def select_pool(prefix, names):
        full_row, gains = runner.run(prefix + "_all", names, params,
                                     generation_seconds=auto_meta["seconds"] if prefix == "auto_adaptive" else 0.)
        options = [(full_row, names, gains)]
        ranked = sorted(names, key=lambda n: (-gains[n], n))
        for size in auto["selection_sizes"]:
            subset = ranked[:size]
            if len(subset) == len(names):
                continue
            row, subset_gains = runner.run(f"{prefix}_top{size}", subset, params,
                                           details={"candidate_pool_size": len(names)})
            options.append((row, subset, subset_gains))
        # Если разница укладывается в заранее заданные 0.0001 LogLoss, берем меньший набор.
        # Небольшой возможный выигрыш не всегда оправдывает десятки лишних колонок.
        best_loss = min(option[0]["logloss"] for option in options)
        eligible = [option for option in options if option[0]["logloss"] <= best_loss + auto["keep_min_logloss_gain"]]
        return min(eligible, key=lambda option: option[0]["n_features"])

    fixed, fixed_reasons = prune_fixed_candidates(features, candidates, split.train_end, auto, config["seed"])
    fixed_reasons.write_csv(root / "fixed_pruned_features.csv")
    save_json(root / "fixed_pool_profile.json", {"pruning_scope": "fixed pool only, before cross-window comparisons",
        "allowed_windows": [7,30,None], "after_pruning": len(fixed), "removed": fixed_reasons.height})
    fixed_row, fixed_names, fixed_gains = select_pool("auto_fixed_windows", fixed)
    adaptive_row, selected, gains = select_pool("auto_adaptive", keep)
    # Проверяем, нужны ли разные окна для разных связей. В shared control всем
    # семействам разрешены одни и те же три окна, выбранные по train gain.
    window_gain = {}
    for name in keep:
        window = spec_lookup[name].window_days
        if window is not None:
            window_gain[window] = window_gain.get(window, 0.) + gains.get(name, 0.)
    shared_windows = sorted(window_gain, key=lambda w: (-window_gain[w], w))[:3]
    shared_candidates = [s.name for s in candidates if s.window_days is None or
                         (s.window_days in shared_windows and s.aggregation != "trend")]
    shared_keep, shared_reasons = prune_candidates(features.select("row_id", *shared_candidates),
                                                   split.train_end, auto, config["seed"])
    shared_reasons.write_csv(root / "shared_pruned_features.csv")
    _, shared_gains = runner.run("shared_windows_pool", shared_keep, params)
    shared = sorted(shared_keep, key=lambda n: (-shared_gains.get(n, 0.), n))[:len(selected)]
    runner.run("shared_windows_control", shared, params,
               details={"shared_windows": ",".join(map(str, sorted(shared_windows)))})
    # После выбора структуры даем auto те же 20 HPO trials, что получил manual.
    # Дополнительные fit для поиска признаков учитываются отдельно: бюджет всего поиска выше.
    x_auto = runner.matrix(selected)
    auto_params, auto_trials = tune_model(x_auto, dev["label"].to_numpy(), split.train, split.search,
                                          base_names + selected, categorical, config,
                                          f"sqlite:///{root.resolve() / 'tuning.sqlite3'}", "automatic")
    auto_trials.to_csv(root / "auto_tuning.csv", index=False)
    del x_auto
    gc.collect()
    tuned_row, tuned_gains = runner.run("auto_tuned", selected, auto_params)
    best_row, best_names, best_gains, best_params = adaptive_row, selected, gains, params
    if tuned_row["logloss"] < best_row["logloss"]:
        best_row, best_gains, best_params = tuned_row, tuned_gains, auto_params
    agent_history = []
    for iteration in range(auto["agent_rounds"]):
        proposals = propose_features(dev.schema, events, [spec_lookup[n] for n in best_names],
                                     agent_history, best_gains, auto["agent_proposals_per_round"])
        if not proposals:
            break
        proposed_features, proposal_meta = cached_features(dev, proposals, root, f"agent_{iteration}")
        unseen = [s.name for s in proposals if s.name not in runner.features.columns]
        runner.features = runner.features.hstack(proposed_features.select(unseen))
        for spec in proposals:
            spec_lookup[spec.name] = spec
            if spec not in candidates:
                candidates.append(spec)
        names = list(dict.fromkeys([*best_names, *[s.name for s in proposals]]))
        row, new_gains = runner.run(f"agent_round_{iteration + 1}", names, best_params,
                                    generation_seconds=proposal_meta["seconds"])
        accepted = best_row["logloss"] - row["logloss"] >= auto["keep_min_logloss_gain"]
        agent_history.append({"round": iteration + 1, "proposed": [s.name for s in proposals],
                              "before_logloss": best_row["logloss"], "after_logloss": row["logloss"],
                              "accepted": accepted, "generation_seconds": proposal_meta["seconds"]})
        save_json(root / "agent_history.json", agent_history)
        if accepted:
            best_row, best_names, best_gains = row, names, new_gains
    # Если поиск с привычными 7/30/all окнами оказался сильнее, оставляем его.
    # Расширенное пространство не обязано выигрывать только потому, что оно шире.
    if fixed_row["logloss"] < best_row["logloss"]:
        best_row, best_names, best_gains, best_params = fixed_row, fixed_names, fixed_gains, params

    selected_specs = [spec_lookup[n] for n in best_names]
    metadata = candidate_metadata(candidates, best_names)
    metadata.write_csv(root / "candidate_features.csv")
    metadata.filter(pl.col("selected")).write_csv(root / "selected_features.csv")
    metadata.filter(pl.col("selected")).select("entity", "event", "aggregation", "window", "window_days",
                                               "long_window_days").unique().sort(["entity", "window_days"]).write_csv(root / "selected_windows.csv")
    pl.DataFrame([{"feature": n, "importance_gain": g,
                   "entity": spec_lookup[n].entity if n in spec_lookup else "base"}
                  for n, g in best_gains.items()]).sort("importance_gain", descending=True).write_csv(root / "feature_importance.csv")
    lock = {"manual_name": manual_name, "manual_features": manual_names, "manual_params": params,
            "manual_selection_pool": list(manual_groups), "manual_selection_data": "validation_search",
            "auto_name": best_row["experiment_name"], "auto_features": best_names, "auto_params": best_params,
            "selected_specs": [s.to_dict() for s in selected_specs],
            "all_specs": [s.to_dict() for s in candidates], "base_names": base_names,
            "shared_windows": shared_windows, "warmup_end": None,
            "search_only": True, "locked_before_audit_and_test": True}
    # SVD сравниваем отдельно. Его warmup исключаем из обучения всех matched controls,
    # иначе часть разницы объяснялась бы разным числом обучающих строк.
    if "rating" in dev.columns or config.get("run_representations", False):
        times = dev["timestamp"].to_numpy()
        warmup_end = int(np.searchsorted(times, times[int(df.height * 0.1)], side="left"))
        embedding, embedding_meta = temporal_svd_features(dev, warmup_end, seed=config["seed"])
        runner.features = runner.features.hstack(embedding.drop("row_id"))
        embedding_names = embedding.drop("row_id").columns
        reduced_train = np.arange(warmup_end, split.train_end)
        runner.run("manual_warmup_control", manual_names, params, train=reduced_train)
        runner.run("auto_warmup_control", best_names, best_params, train=reduced_train)
        embedding_row, _ = runner.run("auto_plus_svd", best_names + embedding_names,
                                       best_params, train=reduced_train)
        save_json(root / "representations.json", embedding_meta)
        lock["warmup_end"] = warmup_end
        lock["embedding_features"] = embedding_names
        lock["embedding_validation_logloss"] = embedding_row["logloss"]
    save_json(root / "locked_selection.json", lock)
    # Меняем только seed, сохраняя уже выбранные признаки и параметры.
    # По результатам этих повторов выбор не пересматриваем.
    for seed in config["seeds"]:
        if seed == config["seed"]:
            continue
        runner.run(manual_name, manual_names, params, seed=seed)
        runner.run(best_row["experiment_name"], best_names, best_params, seed=seed)
    runner.flush()
    return lock


def finalize(df, split, config, root):
    manifest = json.loads((root / "run_manifest.json").read_text())
    if dataset_fingerprint(df) != manifest["dataset_fingerprint"] or vars(split) != manifest["split"]:
        raise ValueError("Final evaluation dataset/splits differ from the locked development run")
    if config != manifest["config"]:
        raise ValueError("Final evaluation configuration differs from the locked run")
    if (manifest.get("protocol_version") != STUDY_PROTOCOL_VERSION or
        manifest.get("selection_signature") != selection_signature() or
        manifest.get("engine_signature") != feature_engine_signature()):
        raise ValueError("Implementation differs from locked development; use a fresh output directory")
    lock = json.loads((root / "locked_selection.json").read_text())
    if (root / "final_results.json").exists():
        print("Final evaluation already exists; reusing frozen results", flush=True)
        return
    specs = [FeatureSpec.from_dict(s) for s in lock["all_specs"]]
    needed = set(lock["manual_features"] + lock["auto_features"])
    features, meta = cached_features(df, [s for s in specs if s.name in needed], root, "final_features")
    encoder_dev = json.loads((root / "encoder_development.json").read_text())
    base_dev, base_names, categorical = base_matrix(df, encoder_dev)
    encoder_final = fit_encoder(df.head(split.validation_end))
    base_final, _, _ = base_matrix(df, encoder_final)
    save_json(root / "encoder_final.json", encoder_final)
    if lock["warmup_end"] is not None:
        embedding, _ = temporal_svd_features(df, lock["warmup_end"], seed=config["seed"])
        features = features.hstack(embedding.drop("row_id"))
    predictions, rows, segments = {}, [], []
    development_rows = pl.read_csv(root / "experiments.csv").to_dicts()
    arms = {"categorical": ("base_categorical", [], lock["manual_params"], 0),
            "manual": (lock["manual_name"], lock["manual_features"], lock["manual_params"], 0),
            "automatic": (lock["auto_name"], lock["auto_features"], lock["auto_params"], 0)}
    if lock["warmup_end"] is not None:
        arms.update({"manual_warmup": ("manual_warmup_control", lock["manual_features"], lock["manual_params"], lock["warmup_end"]),
                     "auto_warmup": ("auto_warmup_control", lock["auto_features"], lock["auto_params"], lock["warmup_end"]),
                     "auto_plus_svd": ("auto_plus_svd", lock["auto_features"] + lock["embedding_features"], lock["auto_params"], lock["warmup_end"])})
    for arm, (source_name, names, params, train_start) in arms.items():
        for seed in (config["seeds"] if arm in ["manual", "automatic"] else [config["seed"]]):
            source_seed = seed
            source = next(r for r in development_rows if r["experiment_name"] == source_name and r["seed"] == source_seed)
            x_dev = matrix_from_features(base_dev, features, names)
            booster = lgb.Booster(model_file=str(root / "models" / f"{source_name}_seed{source_seed}.txt"))
            audit_pred = booster.predict(x_dev[split.audit], num_threads=config["threads"])
            rows.append({"experiment_name": arm, "source_experiment": source_name, "seed": seed,
                         "split": "validation_audit", "n_features": len(base_names) + len(names),
                         "training_time": 0., "feature_generation_time": 0., "cache_hit": True,
                         "memory_usage": psutil_rss(), "best_iteration": source["best_iteration"],
                         "n_train": source["n_train"], **evaluate(df[split.audit], audit_pred)})
            if seed == config["seed"]:
                predictions[arm + "_audit"] = audit_pred
            del x_dev, booster
            gc.collect()
            print(f"FINAL FIT {arm} seed={seed}", flush=True)
            with MemoryTracker() as memory:
                x = matrix_from_features(base_final, features, names)
                train = np.arange(train_start, split.validation_end)
                model, seconds = fit_model(x, df["label"].to_numpy(), train, split.search,
                                             base_names + names, categorical, params, config,
                                             seed, fixed_iterations=int(source["best_iteration"]))
                pred = model.booster_.predict(x[split.test], num_threads=config["threads"])
                model.booster_.save_model(str(root / "models" / f"final_{arm}_seed{seed}.txt"))
                del x, model
                gc.collect()
            rows.append({"experiment_name": arm, "source_experiment": source_name, "seed": seed,
                         "split": "test", "n_features": len(base_names) + len(names),
                         "training_time": seconds, "feature_generation_time": meta["seconds"] if not rows else 0.,
                         "cache_hit": True, "memory_usage": memory.peak,
                         "best_iteration": source["best_iteration"], "n_train": len(train), **evaluate(df[split.test], pred)})
            predictions[arm + f"_test_seed{seed}"] = pred
            if seed == config["seed"]:
                known_values = pl.Series(list(encoder_final["user_id"]), dtype=pl.String).implode()
                known = df[split.test]["user_id"].cast(pl.String).is_in(known_values).to_numpy()
                for segment_name, mask in [("known_user_id", known), ("unknown_user_id", ~known)]:
                    if np.sum(mask) and df[split.test][np.flatnonzero(mask)]["label"].n_unique() == 2:
                        segments.append({"model": arm, "segment": segment_name, "rows": int(np.sum(mask)),
                                         **evaluate(df[split.test][np.flatnonzero(mask)], pred[mask])})
                if "is_organic" in df.columns:
                    for organic, segment_name in [(0, "recommendation_driven"), (1, "organic")]:
                        mask = df[split.test]["is_organic"].to_numpy() == organic
                        if mask.any() and df[split.test][np.flatnonzero(mask)]["label"].n_unique() == 2:
                            segments.append({"model": arm, "segment": segment_name, "rows": int(mask.sum()),
                                             **evaluate(df[split.test][np.flatnonzero(mask)], pred[mask])})
            pl.DataFrame({"row_id": df[split.test]["row_id"], "prediction": pred}).write_parquet(
                root / "predictions" / f"{arm}_seed{seed}.parquet")
            print(f"FINAL {arm} seed={seed}: {rows[-1]['logloss']:.6f}, {rows[-1]['auc']:.6f}", flush=True)
    all_rows = development_rows + rows
    pl.DataFrame(all_rows, infer_schema_length=None).write_csv(root / "experiments.csv")
    if segments:
        pl.DataFrame(segments).write_csv(root / "test_segments.csv")
    comparisons = {}
    for scope, part in [("audit", df[split.audit]), ("test", df[split.test])]:
        suffix = "_audit" if scope == "audit" else f"_test_seed{config['seed']}"
        comparisons[scope + "_automatic_vs_manual"] = paired_user_bootstrap(
            part, predictions["manual" + suffix], predictions["automatic" + suffix],
            repeats=config["bootstrap_repeats"], seed=config["seed"])
    if lock["warmup_end"] is not None:
        comparisons["test_svd_vs_auto_warmup"] = paired_user_bootstrap(
            df[split.test], predictions[f"auto_warmup_test_seed{config['seed']}"],
            predictions[f"auto_plus_svd_test_seed{config['seed']}"],
            repeats=config["bootstrap_repeats"], seed=config["seed"])
    save_json(root / "final_results.json", {"rows": rows, "comparisons": comparisons,
                                           "final_feature_profile": meta,
                                           "test_used_for_selection": False,
                                           "protocol_version": STUDY_PROTOCOL_VERSION,
                                           "evaluation_status": "post-audit re-evaluation of previously observed test"})
    save_json(root / "eda_full.json", study_statistics(df, split))
    print(json.dumps(comparisons, indent=2), flush=True)


def psutil_rss():
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["baseline", "dev", "final", "all"], default="all")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--data", help="Optional CSV/parquet with explicit binary target")
    parser.add_argument("--dataset", choices=["movielens", "kion", "yambda"], default="movielens")
    parser.add_argument("--user-modulus", type=int, default=1, help="KION deterministic user cohort; default 1 keeps full archive")
    parser.add_argument("--positive-label", type=int)
    parser.add_argument("--quick", action="store_true", help="100k chronological rows, 2 HPO trials; separate smoke output")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    auto = yaml.safe_load(Path("configs/auto_features.yaml").read_text())
    if args.dataset != "movielens":
        config.update(dataset_name=args.dataset, run_representations=True)
        auto["max_features"] = 900
        if args.output == "artifacts":
            args.output = f"artifacts/{args.dataset}"
    if args.quick:
        config.update(hyperparameter_trials=2, max_estimators=80, early_stopping_rounds=10,
                       seeds=[42], bootstrap_repeats=20)
        auto.update(windows=[1, 7, 30, None], selection_sizes=[16, 32], agent_rounds=1)
        if args.output in ["artifacts", f"artifacts/{args.dataset}"]:
            args.output = f"artifacts/{args.dataset}_smoke"
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.data:
        df = load_interactions(args.data, positive_label=args.positive_label)
    elif args.dataset == "kion":
        df = load_kion(user_modulus=args.user_modulus)
    elif args.dataset == "yambda":
        df = load_yambda()
    else:
        df = load_movielens()
    if args.quick:
        df = df.head(100000)
    split = temporal_split(df, tuple(config["split_fractions"]))
    save_json(root / "eda.json", describe_data(df.head(split.validation_end)))
    if args.dataset != "movielens":
        save_json(root / "dataset_protocol.json", dataset_protocol(args.dataset, args.user_modulus))
        source_manifest = Path(f"data/raw/{args.dataset}/source_manifest.json")
        if source_manifest.exists():
            save_json(root / "source_manifest.json", json.loads(source_manifest.read_text()))
    if args.stage in ["baseline", "dev", "all"]:
        if args.stage != "baseline" and (root / "final_results.json").exists():
            print("Study is complete; preserving locked selection and final results", flush=True)
        else:
            develop(df, split, config, auto, root, only_baseline=args.stage == "baseline")
    if args.stage in ["final", "all"]:
        finalize(df, split, config, root)


if __name__ == "__main__":
    main()
