"""
Automated Dataset Downloader for Week 15-16
Downloads the MNIST dataset and formats it exactly like Kaggle's 'Digit Recognizer' competition:
- data/train.csv: 42,000 samples with 'label', 'pixel0'...'pixel783'
- data/test.csv: 28,000 samples with 'pixel0'...'pixel783'
"""

import os
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
NPZ_PATH = os.path.join(DATA_DIR, "mnist.npz")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

def download_and_prepare_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV):
        print(f"[+] Datasets already exist in {DATA_DIR}.")
        return

    print(f"[*] Downloading MNIST dataset from {MNIST_URL}...")
    urllib.request.urlretrieve(MNIST_URL, NPZ_PATH)
    print(f"[+] Download complete: {os.path.getsize(NPZ_PATH) / (1024*1024):.2f} MB")

    print("[*] Loading and formatting into Kaggle Digit Recognizer format...")
    with np.load(NPZ_PATH) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    # Flatten 28x28 images to 784 pixels
    x_all = np.concatenate([x_train, x_test], axis=0).reshape(-1, 784)
    y_all = np.concatenate([y_train, y_test], axis=0)

    # Kaggle competition split: 42,000 train, 28,000 test
    train_x = x_all[:42000]
    train_y = y_all[:42000]
    test_x = x_all[42000:70000]

    pixel_cols = [f"pixel{i}" for i in range(784)]

    print("[*] Saving data/train.csv (42,000 samples)...")
    train_df = pd.DataFrame(train_x, columns=pixel_cols)
    train_df.insert(0, "label", train_y)
    train_df.to_csv(TRAIN_CSV, index=False)
    print(f"[+] train.csv created ({os.path.getsize(TRAIN_CSV) / (1024*1024):.1f} MB)")

    print("[*] Saving data/test.csv (28,000 samples)...")
    test_df = pd.DataFrame(test_x, columns=pixel_cols)
    test_df.to_csv(TEST_CSV, index=False)
    print(f"[+] test.csv created ({os.path.getsize(TEST_CSV) / (1024*1024):.1f} MB)")

    # Clean up temporary npz
    if os.path.exists(NPZ_PATH):
        os.remove(NPZ_PATH)

    print("[OK] All datasets downloaded and prepared successfully!")

if __name__ == "__main__":
    download_and_prepare_mnist()
