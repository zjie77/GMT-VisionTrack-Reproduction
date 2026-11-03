"""Print the saved six-metric report without model, dataset, or GPU dependencies."""
import argparse
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path,
                        default=root / "results/vision_evaluation/metrics.json")
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text())
    names = ("HOTA", "AssA", "MOTA", "IDF1", "CVIDF1", "CVMA_TrackEval")
    print("source | scenes | " + " | ".join(names))
    for source in ("dataset", "evaluation-kit", "json"):
        result = report[source]
        scores = " | ".join(f"{result['metrics_percent'][name]:.2f}" for name in names)
        print(f"{source} | {len(result['evaluated_scenes'])} | {scores}")
    print("CVMA_TrackEval has not been validated as equivalent to the paper's CVMA.")


if __name__ == "__main__":
    main()
