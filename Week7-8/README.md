# Machine Learning: Regression & Classification Suite

A comprehensive, production-ready machine learning framework and interactive visualizer covering **Regression** (*Linear, Polynomial, Ridge/Lasso*) and **Classification** (*Logistic Regression, K-Nearest Neighbors*).

---

## 📌 Project Overview

This repository contains the complete implementation for **Week 7-8 Coursework**:
1. **Modular Machine Learning Library (`src/`)**:
   - `src/regression.py`: Ordinary Least Squares Linear Regression, Polynomial Feature Transformations, Ridge ($L_2$) Shrinkage, and Lasso ($L_1$) Feature Selection.
   - `src/classification.py`: Logistic Regression (Sigmoid decision bounds, log-loss minimization) and K-Nearest Neighbors ($K$-NN distance classification, hyperparameter elbow tuning).
   - `src/utils.py`: Data synthesis, metrics calculation, and plot generation utilities.
2. **Jupyter Notebooks**:
   - `01_regression.ipynb`: Interactive notebook for regression analysis with mathematical derivations and diagnostic curves.
   - `02_classification.ipynb`: Interactive notebook for classification analysis with ROC-AUC, confusion matrices, and decision boundary plots.
3. **Interactive Web Dashboard (`index.html`, `styles.css`, `app.js`)**:
   - Modern dark-mode web visualizer allowing real-time parameter tuning (Polynomial degree 1-10, Ridge/Lasso penalty $\alpha$, KNN $K$-neighbors, and decision thresholds).

---

## 🧮 Theoretical Background & Formulations

### 1. Regression Models
- **Linear Regression Loss (MSE)**:
  $$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$
- **Polynomial Feature Transformation**:
  $$h_\theta(x) = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_d x^d$$
- **Ridge Regression ($L_2$ Regularization)**:
  $$J_{Ridge}(\theta) = J(\theta) + \alpha \sum_{j=1}^{n} \theta_j^2$$
  *Shrinks coefficient weights towards zero to mitigate multicollinearity and high variance.*
- **Lasso Regression ($L_1$ Regularization & Feature Selection)**:
  $$J_{Lasso}(\theta) = J(\theta) + \alpha \sum_{j=1}^{n} |\theta_j|$$
  *Drives redundant coefficient weights exactly to zero, performing automatic feature selection.*

### 2. Classification Models
- **Logistic Regression Hypothesis (Sigmoid)**:
  $$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$
- **Binary Cross-Entropy Loss (Log-Loss)**:
  $$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$
- **K-Nearest Neighbors ($K$-NN Distance)**:
  $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
  *Non-parametric instance-based learning; sensitive to feature scaling and choice of $K$.*

---

## 📊 Benchmark Results

### 1. Regression Model Comparison
| Model | Train RMSE | Test RMSE | Train $R^2$ | Test $R^2$ | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Simple Linear Regression** | 5.113 | 4.999 | 0.111 | 0.056 | High Bias (Underfitting) |
| **Polynomial Reg (Degree 2)** | 4.750 | 4.245 | 0.233 | 0.319 | Moderate Improvement |
| **Polynomial Reg (Degree 3)** | 3.908 | 3.831 | 0.481 | 0.446 | **Optimal Non-linear Fit** |
| **Overfit Polynomial (Degree 10)** | 3.705 | 4.054 | 0.533 | 0.379 | High Variance (Overfitting) |
| **Ridge Reg (Degree 10, $\alpha=10.0$)** | 3.788 | 3.838 | 0.512 | 0.444 | Smooth Regularization |
| **Lasso Reg (Degree 10, $\alpha=0.5$)** | 3.926 | 3.848 | 0.476 | 0.441 | Zeroed Redundant Terms |

### 2. Classification Model Comparison
| Classifier | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 86.7% | 85.4% | 89.7% | 87.5% | 0.946 | Linear Decision Boundary |
| **KNN ($K=1$, Overfit)** | 86.7% | 87.2% | 87.2% | 87.2% | 0.866 | Sensitive to Noise |
| **KNN ($K=5$, Optimal)** | 88.0% | 85.7% | 92.3% | 88.9% | 0.955 | Balanced Flexible Bound |
| **KNN ($K=25$, Smoothed)** | 90.7% | 88.1% | 94.9% | 91.4% | 0.971 | Smooth Decision Regions |

---

## 🚀 How to Run

### 1. Execute the ML Pipeline & Benchmark
Run the automated pipeline to train all models, generate summary reports, output `model_results.json`, and save diagnostic plots into `plots/`:
```bash
python run_all.py
```

### 2. Launch the Interactive Web Visualizer
Open `index.html` directly in your browser or run a simple local server:
```bash
python -m http.server 8000
```
Then navigate to `http://localhost:8000`.

### 3. Open Jupyter Notebooks
Launch Jupyter Notebook to inspect step-by-step code and mathematical derivations:
```bash
jupyter notebook 01_regression.ipynb
```

---

## 📁 Repository Structure
```
week7-8/
├── 01_regression.ipynb             # Jupyter Notebook for Regression analysis
├── 02_classification.ipynb         # Jupyter Notebook for Classification analysis
├── run_all.py                      # Main pipeline execution script
├── index.html                      # Interactive Dashboard HTML layout
├── styles.css                      # Modern dark-mode styling & glassmorphism
├── app.js                          # Web canvas ML engine & dynamic rendering
├── model_results.json              # Exported benchmark metric JSON
├── src/
│   ├── utils.py                    # Dataset synthesis & plotting helpers
│   ├── regression.py               # Linear, Polynomial, Ridge/Lasso logic
│   └── classification.py           # Logistic Regression & KNN logic
└── plots/                          # Saved diagnostic plots (PNG format)
    ├── regression_comparison_curves.png
    ├── ridge_lasso_coefficient_shrinkage.png
    ├── classification_decision_boundaries.png
    ├── knn_k_tuning_elbow.png
    └── roc_confusion_matrices.png
```
