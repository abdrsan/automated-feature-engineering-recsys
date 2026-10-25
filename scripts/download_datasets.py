import argparse
from src.dataset_sources import fetch_sources

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kion", "yambda", "both"], default="both")
    args = parser.parse_args()
    for dataset in (["kion", "yambda"] if args.dataset == "both" else [args.dataset]):
        manifest = fetch_sources(dataset)
        print(dataset, "revision", manifest["revision"], flush=True)
