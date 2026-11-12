"""
Utilities for Machine Learning Regression and Classification Tasks.
Includes dataset generators, metric formatting helpers, and plot save utilities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification, make_blobs, make_moons

# Ensure deterministic output
np.random.seed(42)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_regression_data(n_samples=150, noise=15.0, data_type="polynomial"):
    """
    Generate synthetic data for regression tasks.
    - 'linear': Multi-feature linear data with redundant features for Ridge/Lasso.
    - 'polynomial': Single-feature non-linear dataset ideal for Polynomial & Ridge/Lasso fitting.
    """
    if data_type == "polynomial":
        X = np.sort(np.random.uniform(-3, 3, size=n_samples))
        # y = 0.5 * X^3 - 0.8 * X^2 - 2 * X + 3 + noise
        y = 0.5 * (X ** 3) - 0.8 * (X ** 2) - 2 * X + 3 + np.random.normal(0, noise, size=n_samples)
        X = X.reshape(-1, 1)
        return X, y
    elif data_type == "multi_linear":
        X, y, true_coef = make_regression(
            n_samples=n_samples,
            n_features=10,
            n_informative=4,
            noise=noise,
            coef=True,
            random_state=42
        )
        return X, y, true_coef

def generate_classification_data(n_samples=200, data_type="moons"):
    """
    Generate synthetic datasets for classification tasks.
    - 'linearly_separable': 2D binary classification ideal for Logistic Regression.
    - 'non_linear': Concentric or moon-shaped 2D data ideal for KNN.
    """
    if data_type == "linearly_separable":
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.5, random_state=42)
        return X, y
    elif data_type == "non_linear":
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)
        return X, y

def save_fig(fig, filename):
    """Save matplotlib figure to the plots directory cleanly."""
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {filepath}")

def print_section_header(title):
    """Format console output headers."""
    print("\n" + "=" * 60)
    print(f"  {title.upper()}")
    print("=" * 60)
