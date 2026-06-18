import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_boston_csv
from src.preprocessing import split_data
from src.feature_selection import get_all_feature_rankings
from src.model_evaluation import evaluate_all_methods_stepwise, compute_best_frontier
from src.visualization import plot_feature_selection_allinone


def main():
    parser = argparse.ArgumentParser(
        description="Generate Boston Housing feature selection performance chart."
    )
    parser.add_argument(
        "--data",
        default="data/boston_housing.csv",
        help="Path to Boston Housing CSV file (default: data/boston_housing.csv)",
    )
    parser.add_argument(
        "--output",
        default="reports/figures/feature_selection_performance_allinone.png",
        help="Output path for the all-in-one chart (default: reports/figures/feature_selection_performance_allinone.png)",
    )
    parser.add_argument(
        "--sweet-k",
        type=int,
        default=3,
        help="Sweet spot K value (default: 3)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Also save feature_selection_results.csv and feature_rankings.csv",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, args.data) if not os.path.isabs(args.data) else args.data
    output_path = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output

    print("=" * 60)
    print("Boston Housing Feature Selection Chart Generator")
    print("=" * 60)
    print(f"Data:   {data_path}")
    print(f"Output: {output_path}")
    print(f"Sweet Spot: k={args.sweet_k}")
    print()

    print("[1/6] Loading dataset...")
    X, y, feature_names = load_boston_csv(data_path)
    print(f"      Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"      Features: {feature_names}")
    print(f"      Target: medv (min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f})")

    print("\n[2/6] Splitting data (test_size=0.2, random_state=42)...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"      Train: {X_train.shape[0]} samples")
    print(f"      Test:  {X_test.shape[0]} samples")

    print("\n[3/6] Running 9 feature selection algorithms...")
    rankings = get_all_feature_rankings(X_train, y_train)
    for method, feats in rankings.items():
        print(f"      {method:15s} -> Top 3: {feats[:3]}")

    print("\n[4/6] Stepwise evaluation (k=1 to 13)...")
    results = evaluate_all_methods_stepwise(X_train, X_test, y_train, y_test, rankings, max_k=13)
    frontier = compute_best_frontier(results)
    results["Best (Frontier)"] = {"k": frontier["k"], "r2": frontier["r2"], "mse": frontier["mse"], "features": [[] for _ in frontier["k"]]}

    best_method = None
    best_r2 = -999
    for method in results:
        if method == "Best (Frontier)":
            continue
        top_r2 = max(results[method]["r2"])
        if top_r2 > best_r2:
            best_r2 = top_r2
            best_method = method
    print(f"      Best method: {best_method} (max R2 = {best_r2:.4f})")

    best_frontier_r2 = max(frontier["r2"])
    best_frontier_k = frontier["k"][frontier["r2"].index(best_frontier_r2)]
    print(f"      Best frontier: R2 = {best_frontier_r2:.4f} at k = {best_frontier_k}")

    print("\n[5/6] Generating all-in-one chart...")
    plot_feature_selection_allinone(results, rankings, sweet_k=args.sweet_k, output_path=output_path)

    if args.save_results:
        reports_dir = os.path.join(project_root, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        results_rows = []
        for method in results:
            if method == "Best (Frontier)":
                continue
            for i, k in enumerate(results[method]["k"]):
                results_rows.append({
                    "method": method,
                    "k": k,
                    "features": ", ".join(results[method]["features"][i]),
                    "r2": results[method]["r2"][i],
                    "mse": results[method]["mse"][i],
                })
        df_results = pd.DataFrame(results_rows)
        csv_path = os.path.join(reports_dir, "feature_selection_results.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"      Saved: {csv_path}")

        ranking_rows = []
        for rank in range(1, 14):
            row = {"rank": rank}
            for method in rankings:
                row[method.lower().replace(" ", "_").replace("(", "").replace(")", "")] = rankings[method][rank - 1]
            ranking_rows.append(row)
        df_rankings = pd.DataFrame(ranking_rows)
        csv_path = os.path.join(reports_dir, "feature_rankings.csv")
        df_rankings.to_csv(csv_path, index=False)
        print(f"      Saved: {csv_path}")

    print("\n[6/6] Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
