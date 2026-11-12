"""
=============================================================================
Task 2: Perceptrons, Activation Functions, Loss Functions & Propagation
=============================================================================
This module implements from scratch in pure NumPy:
1. Single-Layer Perceptron (demonstrates learning rule & the famous XOR limitation).
2. Activation Functions and their analytical derivatives:
   - Sigmoid
   - ReLU & Leaky ReLU
   - Tanh
   - Softmax (numerically stable)
3. Loss Functions and their analytical gradients:
   - Mean Squared Error (MSE)
   - Binary Cross-Entropy (BCE)
   - Categorical Cross-Entropy (CCE)
   - Softmax + Categorical Cross-Entropy combined gradient (dZ = Y_hat - Y)
4. Numerical Gradient Checking to mathematically verify analytical derivatives.
5. Generates comparison plots saved to `plots/activation_functions.png`.
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# =====================================================================
# 1. PERCEPTRON (Rosenblatt's Perceptron from Scratch)
# =====================================================================
class Perceptron:
    """
    Classic Rosenblatt Perceptron with step activation function.
    Update rule: w = w + lr * (y - y_hat) * x
                 b = b + lr * (y - y_hat)
    """
    def __init__(self, n_inputs, lr=0.1, n_epochs=50):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = np.zeros(n_inputs)
        self.bias = 0.0

    def step_function(self, z):
        return np.where(z >= 0, 1, 0)

    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.step_function(z)

    def fit(self, X, y):
        n_samples = X.shape[0]
        for epoch in range(self.n_epochs):
            errors = 0
            for i in range(n_samples):
                y_hat = self.step_function(np.dot(X[i], self.weights) + self.bias)
                update = self.lr * (y[i] - y_hat)
                if update != 0.0:
                    self.weights += update * X[i]
                    self.bias += update
                    errors += 1
            if errors == 0:
                print(f"    [+] Perceptron converged at epoch {epoch + 1}!")
                break


def demonstrate_perceptron():
    print("\n--- 1. Perceptron Demo: Linearly Separable (AND) vs Non-Linear (XOR) ---")
    
    # 1. AND Gate (Linearly Separable)
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    
    p_and = Perceptron(n_inputs=2, lr=0.1, n_epochs=20)
    print("[*] Training Perceptron on AND gate...")
    p_and.fit(X_and, y_and)
    preds_and = p_and.forward(X_and)
    print(f"    - Predictions: {preds_and} (Target: {y_and}) -> Accuracy: {np.mean(preds_and == y_and)*100:.0f}%")
    
    # 2. XOR Gate (Non-Linearly Separable -> Single Perceptron FAILS!)
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    p_xor = Perceptron(n_inputs=2, lr=0.1, n_epochs=20)
    print("[*] Training Perceptron on XOR gate...")
    p_xor.fit(X_xor, y_xor)
    preds_xor = p_xor.forward(X_xor)
    print(f"    - Predictions: {preds_xor} (Target: {y_xor}) -> Accuracy: {np.mean(preds_xor == y_xor)*100:.0f}%")
    print("    [!] Minsky & Papert (1969) Proof: A single-layer perceptron CANNOT solve XOR.")
    print("        This proves the mathematical necessity of Multi-Layer Perceptrons (MLPs) & non-linear activations!")


# =====================================================================
# 2. ACTIVATION FUNCTIONS & ANALYTICAL DERIVATIVES
# =====================================================================
class Activations:
    @staticmethod
    def sigmoid(z):
        # Clip z to avoid numerical overflow in exp(-z)
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def sigmoid_derivative(z):
        s = Activations.sigmoid(z)
        return s * (1.0 - s)

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(np.float32)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        dz = np.ones_like(z, dtype=np.float32)
        dz[z <= 0] = alpha
        return dz

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        return 1.0 - np.tanh(z) ** 2

    @staticmethod
    def softmax(z):
        # Numerically stable softmax: subtract max along axis to prevent exp overflow
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


# =====================================================================
# 3. LOSS FUNCTIONS & GRADIENTS
# =====================================================================
class LossFunctions:
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error: L = 1/(2m) * sum((y_pred - y_true)^2)"""
        return 0.5 * np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        m = y_true.shape[0]
        return (y_pred - y_true) / m

    @staticmethod
    def binary_cross_entropy(y_true, y_pred, eps=1e-15):
        """Binary Cross-Entropy with clipping to avoid log(0)"""
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)) / m

    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred, eps=1e-15):
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        m = y_true.shape[0]
        return (- (y_true / y_pred) + (1.0 - y_true) / (1.0 - y_pred)) / m

    @staticmethod
    def categorical_cross_entropy(y_true_one_hot, y_pred, eps=1e-15):
        """
        Categorical Cross-Entropy:
        L = - 1/m * sum_i sum_k y_ik * log(y_pred_ik)
        """
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        m = y_true_one_hot.shape[0]
        return -np.sum(y_true_one_hot * np.log(y_pred)) / m

    @staticmethod
    def softmax_cce_gradient(y_true_one_hot, y_pred_softmax):
        """
        Combined analytical gradient for Softmax + Categorical Cross-Entropy:
        dL / dZ = (y_pred - y_true) / m
        This elegant cancellation avoids numerical instability and gradient explosion!
        """
        m = y_true_one_hot.shape[0]
        return (y_pred_softmax - y_true_one_hot) / m


# =====================================================================
# 4. NUMERICAL GRADIENT CHECKING
# =====================================================================
def check_gradient(func, grad_func, z, h=1e-5):
    """
    Approximates derivative using two-sided finite difference:
    f'(z) ~ (f(z + h) - f(z - h)) / (2 * h)
    """
    analytical = grad_func(z)
    numerical = (func(z + h) - func(z - h)) / (2.0 * h)
    relative_diff = np.abs(analytical - numerical) / (np.maximum(1e-8, np.abs(analytical) + np.abs(numerical)))
    return np.max(relative_diff)


def run_gradient_checks():
    print("\n--- 2. Numerical Gradient Checking of Analytical Derivatives ---")
    test_z = np.array([-2.5, -1.0, -0.1, 0.2, 1.5, 3.0], dtype=np.float64)
    
    # 1. Sigmoid
    err_sig = check_gradient(Activations.sigmoid, Activations.sigmoid_derivative, test_z)
    print(f"    - Sigmoid Gradient Error    : {err_sig:.2e} -> {'[PASS]' if err_sig < 1e-4 else '[FAIL]'}")
    
    # 2. Tanh
    err_tanh = check_gradient(Activations.tanh, Activations.tanh_derivative, test_z)
    print(f"    - Tanh Gradient Error       : {err_tanh:.2e} -> {'[PASS]' if err_tanh < 1e-4 else '[FAIL]'}")
    
    # 3. Leaky ReLU
    err_lrelu = check_gradient(Activations.leaky_relu, Activations.leaky_relu_derivative, test_z)
    print(f"    - Leaky ReLU Gradient Error : {err_lrelu:.2e} -> {'[PASS]' if err_lrelu < 1e-4 else '[FAIL]'}")
    
    # 4. Softmax + Cross Entropy check
    y_true = np.array([[0, 1, 0]], dtype=np.float64)
    z_logits = np.array([[1.2, 2.8, -0.5]], dtype=np.float64)
    y_pred = Activations.softmax(z_logits)
    analytical_grad = LossFunctions.softmax_cce_gradient(y_true, y_pred)
    
    # Numerical grad for Softmax+CCE
    num_grad = np.zeros_like(z_logits)
    h = 1e-5
    for i in range(z_logits.shape[1]):
        z_plus = z_logits.copy()
        z_plus[0, i] += h
        loss_plus = LossFunctions.categorical_cross_entropy(y_true, Activations.softmax(z_plus))
        
        z_minus = z_logits.copy()
        z_minus[0, i] -= h
        loss_minus = LossFunctions.categorical_cross_entropy(y_true, Activations.softmax(z_minus))
        
        num_grad[0, i] = (loss_plus - loss_minus) / (2 * h)
        
    diff = np.max(np.abs(analytical_grad - num_grad))
    print(f"    - Softmax + CCE Gradient Err: {diff:.2e} -> {'[PASS]' if diff < 1e-4 else '[FAIL]'}")


# =====================================================================
# 5. VISUALIZATION PLOT
# =====================================================================
def plot_activation_functions():
    print("\n--- 3. Plotting Activation Functions & Derivatives ---")
    z = np.linspace(-5, 5, 500)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Activation Functions and their Derivatives (NumPy Implementation)", fontsize=14, fontweight="bold")
    
    # Sigmoid
    axes[0, 0].plot(z, Activations.sigmoid(z), label="Sigmoid $\\sigma(z)$", color="#1f77b4", lw=2)
    axes[0, 0].plot(z, Activations.sigmoid_derivative(z), label="$\\sigma'(z)$", color="#aec7e8", ls="--", lw=2)
    axes[0, 0].set_title("Sigmoid")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Tanh
    axes[0, 1].plot(z, Activations.tanh(z), label="Tanh $\\tanh(z)$", color="#ff7f0e", lw=2)
    axes[0, 1].plot(z, Activations.tanh_derivative(z), label="$\\tanh'(z)$", color="#ffbb78", ls="--", lw=2)
    axes[0, 1].set_title("Tanh")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # ReLU
    axes[0, 2].plot(z, Activations.relu(z), label="ReLU $\\max(0, z)$", color="#2ca02c", lw=2)
    axes[0, 2].plot(z, Activations.relu_derivative(z), label="ReLU' $(z > 0)$", color="#98df8a", ls="--", lw=2)
    axes[0, 2].set_title("ReLU")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Leaky ReLU
    axes[1, 0].plot(z, Activations.leaky_relu(z, alpha=0.1), label="Leaky ReLU ($\\alpha=0.1$)", color="#d62728", lw=2)
    axes[1, 0].plot(z, Activations.leaky_relu_derivative(z, alpha=0.1), label="Derivative", color="#ff9896", ls="--", lw=2)
    axes[1, 0].set_title("Leaky ReLU")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Softmax Example
    logits = np.array([[-2.0, -1.0, 0.0, 1.0, 2.5, 4.0]])
    probs = Activations.softmax(logits)[0]
    axes[1, 1].bar([f"z={x}" for x in logits[0]], probs, color="#9467bd", alpha=0.8)
    axes[1, 1].set_title("Softmax Output Distribution")
    axes[1, 1].set_ylabel("Probability $\\sum = 1.0$")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    
    # Comparison Summary
    axes[1, 2].text(0.05, 0.90, "Summary of Roles:", fontsize=11, fontweight="bold")
    axes[1, 2].text(0.05, 0.70, "* Sigmoid: [0, 1], Binary classification output", fontsize=9.5)
    axes[1, 2].text(0.05, 0.55, "* Tanh: [-1, 1], Zero-centered hidden layers", fontsize=9.5)
    axes[1, 2].text(0.05, 0.40, "* ReLU: [0, inf), Default hidden activation (fast, solves vanishing gradient)", fontsize=9.5)
    axes[1, 2].text(0.05, 0.25, "* Leaky ReLU: Solves 'Dying ReLU' problem", fontsize=9.5)
    axes[1, 2].text(0.05, 0.10, "* Softmax: Multi-class classification output", fontsize=9.5)
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "activation_functions.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Activation plot saved to: {out_path}")


def main():
    print("=" * 70)
    print(" WEEK 15-16: PERCEPTRONS, ACTIVATIONS & LOSS FUNCTIONS (NUMPY)")
    print("=" * 70)
    demonstrate_perceptron()
    run_gradient_checks()
    plot_activation_functions()
    print("\n[OK] Task 2 Completed Successfully!\n")


if __name__ == "__main__":
    main()
