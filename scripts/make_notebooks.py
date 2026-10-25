from pathlib import Path
import nbformat as nbf


def make_notebooks():
    intro = '''import os, sys
from pathlib import Path
root = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
os.chdir(root)
sys.path.insert(0, str(root))
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/yandex-mpl")
import json
import polars as pl
from IPython.display import display, Image
'''
    notebooks = {
        "01_eda": [
            ("markdown", "# EDA и фактическая схема\nMovieLens 1M: preference scoring, target rating>=4. Метрики test не используются при исследовании признаков."),
            ("code", intro),
            ("code", "eda = json.loads(Path('artifacts/eda_full.json').read_text())\nprint({k: eda[k] for k in ['rows','users','items','start','end','duplicate_events','repeat_user_item_rows','same_timestamp_rows']})\ndisplay(pl.DataFrame([{'column': k, 'dtype': v} for k,v in eda['schema'].items()]))"),
            ("code", "display(pl.DataFrame([{'split': k, **v} for k,v in eda['splits'].items()]))"),
            ("markdown", "Repeat user-item rows = 0: соответствующие count features будут постоянными. Равные timestamps должны исключаться одним блоком. Cold users в validation показывают, почему user IDs недостаточно."),
        ],
        "02_baseline": [
            ("markdown", "# Сильный manual LightGBM baseline\nПолное обучение выполняется CLI: `python -m scripts.run_baseline`. Этот notebook рассматривает реальные сохраненные эксперименты."),
            ("code", intro),
            ("code", "experiments = pl.read_csv('artifacts/experiments.csv')\nnames = ['base_default','base_categorical','plus_user_temporal','plus_item_temporal','plus_user_item','plus_user_category','manual','manual_plus_trend']\ndisplay(experiments.filter((pl.col('split')=='validation_search') & (pl.col('seed')==42) & pl.col('experiment_name').is_in(names)).select('experiment_name','n_features','auc','logloss','ndcg_5','training_time'))"),
            ("code", "display(pl.read_csv('artifacts/manual_tuning.csv').sort('value').head(10))"),
            ("markdown", "Все aggregates вычисляются до T. Встроенная model categorical encoding fit-ится только на training prefix; unknown category=-1. HPO fit может использовать выборку train rows, но история признаков остается полной."),
        ],
        "03_auto_features": [
            ("markdown", "# Structured feature search и безопасный proposer\nCLI: `python -m scripts.run_auto_features`. Ниже фактический search trace, выбранные окна и примитив SQL."),
            ("code", intro),
            ("code", "from src.feature_spec import FeatureSpec, to_sql\nspec = FeatureSpec(('user_id','category_id'), 'positive', 'count', 14)\nprint(to_sql(spec))"),
            ("code", "display(pl.read_csv('artifacts/selected_windows.csv'))\ndisplay(pl.read_csv('artifacts/pruned_features.csv').group_by('reason').len().sort('len',descending=True).head(15))"),
            ("code", "trace = json.loads(Path('artifacts/agent_history.json').read_text())\ndisplay(pl.DataFrame([{k:v for k,v in r.items() if k!='proposed'} for r in trace]))\nprint(trace)"),
            ("code", "display(Image(filename='artifacts/plots/entity_window_heatmap.png'))\ndisplay(Image(filename='artifacts/plots/metric_vs_features.png'))"),
        ],
        "04_results": [
            ("markdown", "# Locked audit и final test\nВ этом notebook результаты только анализируются. Код не выбирает окна или гиперпараметры по test. Полный вывод — REPORT.md."),
            ("code", intro),
            ("code", "experiments = pl.read_csv('artifacts/experiments.csv')\ndisplay(experiments.filter(pl.col('split').is_in(['validation_audit','test'])).select('split','experiment_name','seed','n_features','auc','logloss','ndcg_3','ndcg_5','ndcg_10','recall_10','map_10'))"),
            ("code", "results = json.loads(Path('artifacts/final_results.json').read_text())\nprint(json.dumps(results['comparisons'],indent=2))"),
            ("code", "display(Image(filename='artifacts/plots/top30_importance.png'))\ndisplay(Image(filename='artifacts/plots/test_metrics.png'))"),
            ("markdown", "Bootstrap CI условен на зафиксированных моделях и кластеризован по пользователям. Ranking groups=user×exact timestamp, mixed relevance; это proxy, а не full-catalog retrieval. Warmup SVD имеет отдельные controls с одинаковым training start."),
        ],
    }
    Path("notebooks").mkdir(exist_ok=True)
    for name, cells in notebooks.items():
        notebook = nbf.v4.new_notebook()
        notebook.cells = [nbf.v4.new_markdown_cell(content) if kind == "markdown" else
                          nbf.v4.new_code_cell(content) for kind, content in cells]
        notebook.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(notebook, f"notebooks/{name}.ipynb")


if __name__ == "__main__":
    make_notebooks()
