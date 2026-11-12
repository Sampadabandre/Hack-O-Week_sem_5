"""
Classification Machine Learning Module
Covers Logistic Regression and K-Nearest Neighbors (KNN).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)

try:
    from src.utils import generate_classification_data, save_fig, print_section_header
except ImportError:
    from utils import generate_classification_data, save_fig, print_section_header


def evaluate_classification_metrics(y_true, y_pred, y_prob=None):
    """Calculate Accuracy, Precision, Recall, F1, and ROC-AUC."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    roc_auc = None
    if y_prob is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
        except Exception:
            pass

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }


def plot_decision_boundaries(X, y, models_dict, filename="classification_decision_boundaries.png"):
    """Plot 2D decision boundaries for multiple classifiers side-by-side."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    fig, axes = plt.subplots(1, len(models_dict), figsize=(5 * len(models_dict), 4.5))
    if len(models_dict) == 1:
        axes = [axes]

    cmap_light = ListedColormap(['#a2d2ff', '#ffb703'])
    cmap_bold = ListedColormap(['#0077b6', '#d90429'])

    for ax, (name, model) in zip(axes, models_dict.items()):
        model.fit(X_scaled, y)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=30)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel("Feature 1 (Scaled)")
        ax.set_ylabel("Feature 2 (Scaled)")

    fig.tight_layout()
    save_fig(fig, filename)


def run_classification_experiments():
    """Run full benchmark for Logistic Regression and KNN."""
    print_section_header("2. Classification Experiments")

    # -------------------------------------------------------------
    # Experiment 1: Non-Linear Data (Moons) - Logistic Reg vs KNN
    # -------------------------------------------------------------
    X, y = generate_classification_data(n_samples=250, data_type="non_linear")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "KNN (K=1, Overfit)": KNeighborsClassifier(n_neighbors=1),
        "KNN (K=5, Optimal)": KNeighborsClassifier(n_neighbors=5),
        "KNN (K=25, Smoothed)": KNeighborsClassifier(n_neighbors=25)
    }

    results = []

    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        y_prob = None
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        metrics = evaluate_classification_metrics(y_test, y_pred, y_prob)

        results.append({
            "Classifier": name,
            "Accuracy": round(metrics["Accuracy"], 3),
            "Precision": round(metrics["Precision"], 3),
            "Recall": round(metrics["Recall"], 3),
            "F1-Score": round(metrics["F1-Score"], 3),
            "ROC-AUC": round(metrics["ROC-AUC"], 3) if metrics["ROC-AUC"] else "N/A"
        })

    # Plot Decision Boundaries
    plot_decision_boundaries(X_train, y_train, classifiers, filename="classification_decision_boundaries.png")

    # -------------------------------------------------------------
    # Experiment 2: KNN Hyperparameter K Tuning (Elbow Plot)
    # -------------------------------------------------------------
    k_range = range(1, 31)
    train_accs = []
    test_accs = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        train_accs.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
        test_accs.append(accuracy_score(y_test, knn.predict(X_test_scaled)))

    fig_k, ax_k = plt.subplots(figsize=(9, 5))
    ax_k.plot(k_range, train_accs, label='Train Accuracy', color='#4361ee', marker='o', linewidth=2)
    ax_k.plot(k_range, test_accs, label='Test Accuracy', color='#e63946', marker='s', linewidth=2)
    ax_k.set_title("KNN Accuracy vs K-Value (Hyperparameter Tuning)", fontsize=12, fontweight='bold')
    ax_k.set_xlabel("K Value (Number of Neighbors)")
    ax_k.set_ylabel("Accuracy")
    ax_k.set_xticks(k_range[::2])
    ax_k.legend()
    ax_k.grid(True, linestyle="--", alpha=0.5)
    save_fig(fig_k, "knn_k_tuning_elbow.png")

    # -------------------------------------------------------------
    # Experiment 3: ROC Curve & Confusion Matrix for Logistic Reg vs KNN
    # -------------------------------------------------------------
    log_reg = LogisticRegression().fit(X_train_scaled, y_train)
    knn_opt = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)

    fig_roc, (ax_roc, ax_cm1, ax_cm2) = plt.subplots(1, 3, figsize=(16, 4.5))

    # ROC Plot
    for model, name, color in [(log_reg, "Logistic Reg", "#4361ee"), (knn_opt, "KNN (K=5)", "#f72585")]:
        probs = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc_val = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.3f})", color=color, linewidth=2)

    ax_roc.plot([0, 1], [0, 1], color='black', linestyle=':', label='Random Guess')
    ax_roc.set_title("ROC Curve Comparison", fontsize=11, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle="--", alpha=0.5)

    # Confusion Matrices
    cm_log = confusion_matrix(y_test, log_reg.predict(X_test_scaled))
    cm_knn = confusion_matrix(y_test, knn_opt.predict(X_test_scaled))

    ConfusionMatrixDisplay(cm_log, display_labels=["Class 0", "Class 1"]).plot(ax=ax_cm1, cmap="Blues", colorbar=False)
    ax_cm1.set_title("Logistic Reg Confusion Matrix", fontsize=10, fontweight='bold')

    ConfusionMatrixDisplay(cm_knn, display_labels=["Class 0", "Class 1"]).plot(ax=ax_cm2, cmap="Purples", colorbar=False)
    ax_cm2.set_title("KNN (K=5) Confusion Matrix", fontsize=10, fontweight='bold')

    fig_roc.tight_layout()
    save_fig(fig_roc, "roc_confusion_matrices.png")

    results_df = pd.DataFrame(results)
    print("\n--- Classification Performance Benchmark ---")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    run_classification_experiments()
