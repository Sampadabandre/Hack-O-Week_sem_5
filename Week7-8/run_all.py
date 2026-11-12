"""
Pipeline script to run both Regression and Classification experiments.
Executes models, prints comprehensive summary tables, saves static plots,
and exports benchmark metrics to model_results.json for the web app visualizer.
"""

import os
import json
import numpy as np
import pandas as pd

from src.regression import run_regression_experiments
from src.classification import run_classification_experiments

def export_json_summary(reg_df, clf_df):
    """Export benchmark data to JSON file for frontend integration."""
    output = {
        "regression": reg_df.to_dict(orient="records"),
        "classification": clf_df.to_dict(orient="records")
    }
    json_path = os.path.join(os.path.dirname(__file__), "model_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[INFO] Benchmark results exported to: {json_path}")

def main():
    print("============================================================")
    print("  WEEK 7-8 MACHINE LEARNING BENCHMARK PIPELINE")
    print("============================================================")
    
    # 1. Run Regression Experiments
    reg_df = run_regression_experiments()
    
    # 2. Run Classification Experiments
    clf_df = run_classification_experiments()

    # 3. Export JSON Artifacts
    export_json_summary(reg_df, clf_df)

    print("\n============================================================")
    print("  PIPELINE EXECUTION COMPLETE! ALL PLOTS SAVED TO plots/")
    print("============================================================")

if __name__ == "__main__":
    main()
