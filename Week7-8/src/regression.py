"""
Regression Machine Learning Module
Covers Linear Regression, Polynomial Regression, Ridge (L2), and Lasso (L1) Regularization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from src.utils import generate_regression_data, save_fig, print_section_header
except ImportError:
    from utils import generate_regression_data, save_fig, print_section_header


def evaluate_regression_metrics(y_true, y_pred):
    """Calculate MSE, RMSE, MAE, and R2 score."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def run_regression_experiments():
    """Run full benchmark for Linear, Polynomial, Ridge, and Lasso Regression."""
    print_section_header("1. Regression Experiments")

    # -------------------------------------------------------------
    # Experiment 1: Polynomial Regression & Overfitting / Regularization
    # -------------------------------------------------------------
    X, y = generate_regression_data(n_samples=120, noise=4.0, data_type="polynomial")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_grid = np.linspace(-3.2, 3.2, 300).reshape(-1, 1)

    # Models to test on polynomial dataset
    models = {
        "Simple Linear Regression": make_pipeline(LinearRegression()),
        "Polynomial Reg (Degree 2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        "Polynomial Reg (Degree 3)": make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
        "Overfit Polynomial (Degree 10)": make_pipeline(PolynomialFeatures(degree=10), LinearRegression()),
        "Ridge Reg (Degree 10, alpha=10.0)": make_pipeline(PolynomialFeatures(degree=10), StandardScaler(), Ridge(alpha=10.0)),
        "Lasso Reg (Degree 10, alpha=0.5)": make_pipeline(PolynomialFeatures(degree=10), StandardScaler(), Lasso(alpha=0.5, max_iter=10000))
    }

    results = []

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(X_train, y_train, color='blue', alpha=0.6, label='Train Data', s=35)
    ax.scatter(X_test, y_test, color='red', alpha=0.6, label='Test Data', marker='x', s=45)

    colors = ['#4361ee', '#4cc9f0', '#2ec4b6', '#e63946', '#7209b7', '#f72585']

    for (name, model), color in zip(models.items(), colors):
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_metrics = evaluate_regression_metrics(y_train, y_pred_train)
        test_metrics = evaluate_regression_metrics(y_test, y_pred_test)

        results.append({
            "Model": name,
            "Train RMSE": round(train_metrics["RMSE"], 3),
            "Test RMSE": round(test_metrics["RMSE"], 3),
            "Train R2": round(train_metrics["R2"], 3),
            "Test R2": round(test_metrics["R2"], 3)
        })

        # Predict grid for smooth curve plot
        y_grid = model.predict(X_grid)
        ax.plot(X_grid, y_grid, label=name, color=color, linewidth=2)

    ax.set_ylim(min(y) - 10, max(y) + 10)
    ax.set_title("Regression Model Fitting Comparison (Linear vs Polynomial vs Ridge/Lasso)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Feature X")
    ax.set_ylabel("Target y")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    save_fig(fig, "regression_comparison_curves.png")

    results_df = pd.DataFrame(results)
    print("\n--- Regression Performance Benchmark ---")
    print(results_df.to_string(index=False))

    # -------------------------------------------------------------
    # Experiment 2: Feature Selection & Shrinkage (Multi-feature Ridge vs Lasso)
    # -------------------------------------------------------------
    X_m, y_m, true_coef = generate_regression_data(n_samples=200, noise=10.0, data_type="multi_linear")
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_m, y_m, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_m_train_scaled = scaler.fit_transform(X_m_train)
    X_m_test_scaled = scaler.transform(X_m_test)

    lin_reg = LinearRegression().fit(X_m_train_scaled, y_m_train)
    ridge_reg = Ridge(alpha=20.0).fit(X_m_train_scaled, y_m_train)
    lasso_reg = Lasso(alpha=5.0).fit(X_m_train_scaled, y_m_train)

    fig_coef, ax_coef = plt.subplots(figsize=(10, 5))
    features = [f"X{i+1}" for i in range(X_m.shape[1])]
    x_indices = np.arange(len(features))

    ax_coef.bar(x_indices - 0.25, lin_reg.coef_, width=0.25, label="Linear Reg", color="#4361ee")
    ax_coef.bar(x_indices, ridge_reg.coef_, width=0.25, label="Ridge (L2)", color="#7209b7")
    ax_coef.bar(x_indices + 0.25, lasso_reg.coef_, width=0.25, label="Lasso (L1)", color="#f72585")

    ax_coef.set_xticks(x_indices)
    ax_coef.set_xticklabels(features)
    ax_coef.set_title("Coefficient Comparison: Linear vs Ridge (L2 Shrinkage) vs Lasso (L1 Feature Selection)", fontsize=12, fontweight='bold')
    ax_coef.set_xlabel("Feature Index")
    ax_coef.set_ylabel("Coefficient Weight")
    ax_coef.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_coef.legend()
    ax_coef.grid(True, linestyle=":", alpha=0.6)
    save_fig(fig_coef, "ridge_lasso_coefficient_shrinkage.png")

    return results_df


if __name__ == "__main__":
    run_regression_experiments()
