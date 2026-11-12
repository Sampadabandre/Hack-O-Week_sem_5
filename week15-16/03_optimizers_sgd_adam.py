"""
=============================================================================
Task 3: Optimizers from Scratch in NumPy (SGD, Momentum, Adam)
=============================================================================
This module provides pure NumPy implementations of:
1. Vanilla Stochastic Gradient Descent (SGD)
2. SGD with Momentum (Polyak Momentum)
3. Adam Optimizer (Adaptive Moment Estimation with bias corrections)
4. Comprehensive comparative benchmark on an anisotropic ill-conditioned landscape
   f(x, y) = 0.1*x^2 + 2.0*y^2
   Illustrating how SGD oscillates, Momentum dampens oscillations, and Adam
   dynamically adapts learning rates per parameter.
5. Saves comparative trajectories and convergence plots to:
   `plots/optimizers_comparison.png`
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# =====================================================================
# 1. OPTIMIZER CLASSES (NUMPY IMPLEMENTATIONS)
# =====================================================================
class VanillaSGD:
    """
    Standard Gradient Descent:
    theta = theta - lr * grad
    """
    def __init__(self, lr=0.05):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]
        return params


class MomentumSGD:
    """
    SGD with Momentum:
    v = beta * v + (1 - beta) * grad  (or standard velocity formulation: v = beta * v + lr * grad)
    theta = theta - v
    """
    def __init__(self, lr=0.05, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}

    def update(self, params, grads):
        if not self.v:
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params:
            self.v[key] = self.beta * self.v[key] + self.lr * grads[key]
            params[key] -= self.v[key]
        return params


class Adam:
    """
    Adam (Adaptive Moment Estimation) Optimizer:
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          (1st moment vector: Mean)
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        (2nd moment vector: Variance)
    m_hat = m_t / (1 - beta1^t)                        (Bias correction)
    v_hat = v_t / (1 - beta2^t)                        (Bias correction)
    theta = theta - lr * m_hat / (sqrt(v_hat) + eps)   (Parameter update)
    """
    def __init__(self, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0  # Timestep counter for bias correction

    def update(self, params, grads):
        self.t += 1
        if not self.m:
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        for key in params:
            g = grads[key]
            # Update biased 1st moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * g
            # Update biased 2nd raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (g ** 2)
            
            # Compute bias-corrected 1st and 2nd moment estimates
            m_hat = self.m[key] / (1.0 - (self.beta1 ** self.t))
            v_hat = self.v[key] / (1.0 - (self.beta2 ** self.t))
            
            # Apply parameter update
            params[key] -= (self.lr * m_hat) / (np.sqrt(v_hat) + self.eps)

        return params


# =====================================================================
# 2. BENCHMARK LOSS SURFACE
# =====================================================================
def anisotropic_loss(x, y):
    """
    Anisotropic quadratic bowl:
    f(x, y) = 0.1 * x^2 + 2.0 * y^2
    Global minimum is at (0, 0) with f(0, 0) = 0.
    The gradient in y is 20x steeper than in x, causing SGD to oscillate heavily.
    """
    return 0.1 * (x ** 2) + 2.0 * (y ** 2)

def anisotropic_grad(x, y):
    df_dx = 0.2 * x
    df_dy = 4.0 * y
    return df_dx, df_dy


def run_optimizer_simulation(optimizer_cls, opt_kwargs, start_pos=(-4.5, 3.5), steps=60):
    """
    Simulates optimization trajectory from a starting point.
    """
    opt = optimizer_cls(**opt_kwargs)
    params = {"x": float(start_pos[0]), "y": float(start_pos[1])}
    
    trajectory = [(params["x"], params["y"])]
    losses = [anisotropic_loss(params["x"], params["y"])]
    
    for _ in range(steps):
        gx, gy = anisotropic_grad(params["x"], params["y"])
        grads = {"x": gx, "y": gy}
        params = opt.update(params, grads)
        trajectory.append((params["x"], params["y"]))
        losses.append(anisotropic_loss(params["x"], params["y"]))
        
    return np.array(trajectory), np.array(losses)


# =====================================================================
# 3. COMPARISON & VISUALIZATION
# =====================================================================
def compare_and_plot_optimizers():
    print("\n[*] Running optimizer benchmark on anisotropic bowl landscape...")
    steps = 70
    start_point = (-4.0, 3.0)
    
    # 1. Vanilla SGD
    traj_sgd, loss_sgd = run_optimizer_simulation(VanillaSGD, {"lr": 0.2}, start_point, steps)
    # 2. Momentum SGD
    traj_mom, loss_mom = run_optimizer_simulation(MomentumSGD, {"lr": 0.08, "beta": 0.85}, start_point, steps)
    # 3. Adam
    traj_adam, loss_adam = run_optimizer_simulation(Adam, {"lr": 0.4, "beta1": 0.9, "beta2": 0.999}, start_point, steps)

    print(f"    - Vanilla SGD Final Loss   : {loss_sgd[-1]:.6f} (Final Pos: x={traj_sgd[-1, 0]:.3f}, y={traj_sgd[-1, 1]:.3f})")
    print(f"    - Momentum SGD Final Loss  : {loss_mom[-1]:.6f} (Final Pos: x={traj_mom[-1, 0]:.3f}, y={traj_mom[-1, 1]:.3f})")
    print(f"    - Adam Optimizer Final Loss: {loss_adam[-1]:.6f} (Final Pos: x={traj_adam[-1, 0]:.3f}, y={traj_adam[-1, 1]:.3f})")

    # Plotting
    fig = plt.figure(figsize=(15, 6))
    
    # Left subplot: 2D Contour Trajectories
    ax1 = fig.add_subplot(1, 2, 1)
    X = np.linspace(-5.0, 5.0, 400)
    Y = np.linspace(-4.0, 4.0, 400)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z_grid = anisotropic_loss(X_grid, Y_grid)
    
    contours = ax1.contour(X_grid, Y_grid, Z_grid, levels=np.logspace(-1, 2.5, 20), cmap="viridis", alpha=0.6)
    ax1.plot(traj_sgd[:, 0], traj_sgd[:, 1], "r.-", label="Vanilla SGD (Oscillates in steep valley)", lw=1.5, ms=5)
    ax1.plot(traj_mom[:, 0], traj_mom[:, 1], "b.-", label="Momentum SGD (Dampened oscillations)", lw=1.5, ms=5)
    ax1.plot(traj_adam[:, 0], traj_adam[:, 1], "g.-", label="Adam (Fast, adaptive direct path)", lw=2.0, ms=6)
    
    ax1.plot(start_point[0], start_point[1], "ko", ms=9, label="Start (-4, 3)")
    ax1.plot(0, 0, "k*", ms=15, label="Global Minimum (0, 0)")
    ax1.set_title("Optimization Trajectories in Loss Landscape", fontsize=12, fontweight="bold")
    ax1.set_xlabel("x (gentle gradient direction)")
    ax1.set_ylabel("y (steep gradient direction)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right subplot: Loss Convergence Curves
    ax2 = fig.add_subplot(1, 2, 2)
    iters = np.arange(len(loss_sgd))
    ax2.semilogy(iters, loss_sgd, "r-", label="Vanilla SGD", lw=2)
    ax2.semilogy(iters, loss_mom, "b-", label="Momentum SGD", lw=2)
    ax2.semilogy(iters, loss_adam, "g-", label="Adam Optimizer", lw=2.5)
    
    ax2.set_title("Loss Convergence (Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Iteration Step")
    ax2.set_ylabel("Loss (Log Scale)")
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "optimizers_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Optimizers comparison plot saved to: {out_path}")


def main():
    print("=" * 70)
    print(" WEEK 15-16: OPTIMIZERS FROM SCRATCH (SGD, MOMENTUM, ADAM)")
    print("=" * 70)
    compare_and_plot_optimizers()
    print("\n[OK] Task 3 Completed Successfully!\n")


if __name__ == "__main__":
    main()
