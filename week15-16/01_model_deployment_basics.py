"""
=============================================================================
Task 1: Model Deployment Basics (Saving & Loading Models with Pickle & Joblib)
=============================================================================
This script demonstrates:
1. Training a baseline machine learning model on the MNIST dataset.
2. Serializing (saving) models to disk using Python's native `pickle`.
3. Serializing (saving) models to disk using `joblib` (optimized for NumPy arrays).
4. Deserializing (loading) both formats back into memory.
5. Verifying prediction parity (saved vs in-memory models give identical outputs).
6. Performance benchmarks (file size, save speed, load speed).
7. Simulating a production deployment inference pipeline.
=============================================================================
"""

import os
import time
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

PICKLE_PATH = os.path.join(MODELS_DIR, "model_pipeline.pkl")
JOBLIB_PATH = os.path.join(MODELS_DIR, "model_pipeline.joblib")


def load_and_preprocess_sample(n_samples=5000):
    """
    Loads a subset of MNIST for quick, reproducible training of a baseline model.
    """
    print(f"[*] Loading {n_samples} samples from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, nrows=n_samples)
    
    y = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32) / 255.0  # Normalize pixels [0, 1]
    
    # Train/Validation split (80/20)
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"[+] Training set: {X_train.shape}, Validation set: {X_val.shape}")
    return X_train, y_train, X_val, y_val


def train_baseline_model(X_train, y_train):
    """
    Trains a Logistic Regression pipeline with StandardScaler.
    """
    print("\n[*] Training baseline Logistic Regression model...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=150, solver="lbfgs", random_state=42))
    ])
    
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    duration = time.time() - t0
    print(f"[+] Model trained in {duration:.2f} seconds.")
    return pipeline


def test_pickle_deployment(model, sample_input):
    """
    Saves and loads model using standard Python pickle.
    """
    print("\n--- 1. Testing Deployment with `pickle` ---")
    
    # Save model
    t0 = time.time()
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_save_time = time.time() - t0
    pickle_size = os.path.getsize(PICKLE_PATH) / 1024  # KB
    print(f"[+] Model saved with pickle to: {PICKLE_PATH}")
    print(f"    - File Size: {pickle_size:.2f} KB")
    print(f"    - Save Time: {pickle_save_time*1000:.2f} ms")
    
    # Load model
    t0 = time.time()
    with open(PICKLE_PATH, "rb") as f:
        loaded_model = pickle.load(f)
    pickle_load_time = time.time() - t0
    print(f"[+] Model reloaded from pickle in {pickle_load_time*1000:.2f} ms")
    
    # Inference verification
    original_preds = model.predict(sample_input)
    loaded_preds = loaded_model.predict(sample_input)
    np.testing.assert_array_equal(original_preds, loaded_preds)
    print("[OK] Verification Passed: Pickle predictions match original model 100%!")
    
    return loaded_model, pickle_size, pickle_save_time, pickle_load_time


def test_joblib_deployment(model, sample_input):
    """
    Saves and loads model using `joblib` with compression.
    """
    print("\n--- 2. Testing Deployment with `joblib` ---")
    
    # Save model with zlib compression (level 3 for speed + space)
    t0 = time.time()
    joblib.dump(model, JOBLIB_PATH, compress=3)
    joblib_save_time = time.time() - t0
    joblib_size = os.path.getsize(JOBLIB_PATH) / 1024  # KB
    print(f"[+] Model saved with joblib to: {JOBLIB_PATH}")
    print(f"    - File Size: {joblib_size:.2f} KB (Compressed)")
    print(f"    - Save Time: {joblib_save_time*1000:.2f} ms")
    
    # Load model
    t0 = time.time()
    loaded_model = joblib.load(JOBLIB_PATH)
    joblib_load_time = time.time() - t0
    print(f"[+] Model reloaded from joblib in {joblib_load_time*1000:.2f} ms")
    
    # Inference verification
    original_preds = model.predict(sample_input)
    loaded_preds = loaded_model.predict(sample_input)
    np.testing.assert_array_equal(original_preds, loaded_preds)
    print("[OK] Verification Passed: Joblib predictions match original model 100%!")
    
    return loaded_model, joblib_size, joblib_save_time, joblib_load_time


def simulate_production_service(model_path, is_joblib=True):
    """
    Simulates a standalone production inference endpoint or microservice.
    """
    print("\n--- 3. Simulating Production Inference Service ---")
    # Load model into memory (as done at server startup in Flask/FastAPI)
    print(f"[*] Booting service... Loading model from {os.path.basename(model_path)}...")
    if is_joblib:
        deployed_model = joblib.load(model_path)
    else:
        with open(model_path, "rb") as f:
            deployed_model = pickle.load(f)
            
    def predict_endpoint(image_vector):
        """
        Input: 784-element normalized float array or batch of arrays.
        Output: Predicted class digit and probability distribution.
        """
        if len(image_vector.shape) == 1:
            image_vector = image_vector.reshape(1, -1)
        pred_class = deployed_model.predict(image_vector)[0]
        probabilities = deployed_model.predict_proba(image_vector)[0]
        confidence = probabilities[pred_class]
        return int(pred_class), float(confidence)

    return predict_endpoint


def main():
    print("=" * 70)
    print(" WEEK 15-16: MODEL DEPLOYMENT BASICS (PICKLE & JOBLIB)")
    print("=" * 70)
    
    # 1. Load data
    X_train, y_train, X_val, y_val = load_and_preprocess_sample(n_samples=5000)
    
    # 2. Train baseline
    model = train_baseline_model(X_train, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"[+] Baseline Validation Accuracy: {val_acc * 100:.2f}%")
    
    # 3. Test Pickle
    _, p_size, p_save, p_load = test_pickle_deployment(model, X_val[:20])
    
    # 4. Test Joblib
    _, j_size, j_save, j_load = test_joblib_deployment(model, X_val[:20])
    
    # 5. Comparison Summary Table
    print("\n" + "=" * 55)
    print("   COMPARISON: PICKLE vs JOBLIB DEPLOYMENT")
    print("=" * 55)
    print(f"{'Metric':<25} | {'pickle':<12} | {'joblib (comp)':<12}")
    print("-" * 55)
    print(f"{'File Size on Disk':<25} | {p_size:>8.2f} KB | {j_size:>8.2f} KB")
    print(f"{'Save Time':<25} | {p_save*1000:>8.2f} ms | {j_save*1000:>8.2f} ms")
    print(f"{'Load Time':<25} | {p_load*1000:>8.2f} ms | {j_load*1000:>8.2f} ms")
    print("=" * 55)
    print("[Insight] `joblib` is tailored for Python objects with large NumPy")
    print("arrays and matrices. It supports built-in compression (e.g. zlib),")
    print("making model weights significantly smaller on disk.")
    
    # 6. Production Inference Simulation
    inference_fn = simulate_production_service(JOBLIB_PATH, is_joblib=True)
    test_sample = X_val[0]
    true_label = y_val[0]
    pred_label, conf = inference_fn(test_sample)
    print(f"\n[+] Production Request Simulated:")
    print(f"    - True Label      : {true_label}")
    print(f"    - Predicted Label : {pred_label}")
    print(f"    - Confidence      : {conf*100:.2f}%")
    print(f"    - Result          : {'CORRECT [OK]' if pred_label == true_label else 'INCORRECT'}")
    print("\n[OK] Task 1 Completed Successfully!\n")


if __name__ == "__main__":
    main()
