import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from collections import deque

warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(25)
np.random.seed(25)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


class BaseFDTD:

    def __init__(self, nx=2000, dx=0.01, c=343, cfl=1.0 / np.sqrt(3)):
        self.nx = nx
        self.dx = dx
        self.c = c
        self.cfl = cfl
        self.dt = 0.9 * cfl * dx / c
        self.x = np.linspace(0, (nx - 1) * dx, nx)
        self.p = None
        self.p_prev = None
        self.history = []

    def reset(self):
        self.p = np.zeros(self.nx)
        self.p_prev = np.zeros(self.nx)
        self.history = []

    def source(self, t, freq=500, amplitude=10.0):
        if t < 0.01:
            window = 0.5 * (1 - np.cos(2 * np.pi * t / 0.01))
            return amplitude * window * np.sin(2 * np.pi * freq * t)
        return 0.0

    def update_interior(self):
        p_new = np.zeros(self.nx)
        for i in range(1, self.nx - 1):
            p_new[i] = (2 * self.p[i] - self.p_prev[i] +
                        self.cfl ** 2 * (self.p[i + 1] - 2 * self.p[i] + self.p[i - 1]))
        return p_new

    def boundary_condition(self, p_new, step):
        # Left boundary
        p_new[0] = 0
        # Right boundary
        p_new[-1] = 0
        return p_new

    def simulate(self, n_steps=1000, src_idx=None, record_every=10):
        self.reset()
        if src_idx is None:
            src_idx = self.nx // 2

        for step in range(n_steps):
            t = step * self.dt
            src_val = self.source(t)
            self.p[src_idx] += src_val
            p_new = self.update_interior()
            p_new = self.boundary_condition(p_new, step)

            if step % record_every == 0:
                self.history.append({
                    'step': step,
                    't': t,
                    'p': self.p.copy(),
                    'p_boundary_right': p_new[-1]
                })

            self.p_prev = self.p.copy()
            self.p = p_new

        return self.history


class RealPhysicalBoundary:
    @staticmethod
    def apply(p_neighbor, p_neighbor_prev, freq=800):
        reflection = 0.85 / (1.0 + freq / 1500)
        nonlinear = 1.0 / (1.0 + 0.3 * np.abs(p_neighbor))
        return reflection * nonlinear * p_neighbor


class SimplifiedBoundary:
    @staticmethod
    def apply(p_neighbor, p_neighbor_prev):
        return 0.6 * p_neighbor


class ConstrainedBoundaryML(nn.Module):

    def __init__(self, input_dim=5, hidden_dims=[64, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [batch, 5] - [p_neighbor, p_prev_neighbor, gradient, t, freq_norm]
        """
        # ML predicts correction factor (range [-1, 1])
        correction = torch.tanh(self.network(x))

        # Physical constraint: boundary pressure must be within physically reasonable range
        p_neighbor = x[:, 0:1]

        # Method: predict reflection coefficient (between 0-1)
        reflection_coeff = 0.3 + 0.5 * correction  # Range [0.3, 0.8]

        # Output boundary pressure
        return reflection_coeff * p_neighbor


class TrainingDataGenerator:
    def __init__(self, nx=200, dx=0.01, c=343, cfl=0.5):
        self.nx = nx
        self.dx = dx
        self.c = c
        self.cfl = cfl
        self.dt = cfl * dx / c

    def generate(self, n_samples=200, n_steps=300):
        X_list = []
        y_list = []

        for sample_idx in range(n_samples):
            freq = np.random.uniform(500, 1000)
            amplitude = np.random.uniform(6.0, 10.0)

            fdtd = BaseFDTD(self.nx, self.dx, self.c, self.cfl)

            def boundary_with_physics(p_new, step):
                p_neighbor = fdtd.p[-2]
                p_prev_neighbor = fdtd.p_prev[-2]
                p_new[-1] = RealPhysicalBoundary.apply(p_neighbor, p_prev_neighbor, freq)
                p_new[0] = 0
                return p_new

            fdtd.boundary_condition = boundary_with_physics

            def source_with_params(t):
                if t < 0.01:
                    window = 0.5 * (1 - np.cos(2 * np.pi * t / 0.01))
                    return amplitude * window * np.sin(2 * np.pi * freq * t)
                return 0.0

            fdtd.source = source_with_params

            history = fdtd.simulate(n_steps=n_steps, record_every=2)

            for h in history:
                if h['step'] > 20:
                    # Normalize input features
                    features = np.array([
                        np.clip(h['p'][-2] / 5.0, -1, 1),  # Normalized pressure
                        np.clip(h['p'][-3] / 5.0, -1, 1) if len(h['p']) > 3 else 0,
                        np.clip((h['p'][-2] - h['p'][-3]) / 5.0, -1, 1) if len(h['p']) > 3 else 0,
                        np.clip(h['t'] / 0.02, 0, 1),  # Normalized time
                        (freq - 500) / 500,  # Normalized frequency
                    ])
                    X_list.append(features)

                    # Target output (normalized)
                    y_list.append([np.clip(h['p_boundary_right'] / 5.0, -1, 1)])

            if (sample_idx + 1) % 50 == 0:
                print(f"  Generating sample {sample_idx + 1}/{n_samples}")

        X = torch.FloatTensor(np.array(X_list))
        y = torch.FloatTensor(np.array(y_list))

        print(f"Generation complete: {len(X)} training samples")
        return X, y


class StableHybridSolver:

    def __init__(self, nx=200, dx=0.01, c=343, cfl=0.5, ml_model=None, use_ml=True):
        self.nx = nx
        self.dx = dx
        self.c = c
        self.cfl = cfl
        self.dt = cfl * dx / c

        self.ml_model = ml_model
        self.use_ml = use_ml

        if ml_model is not None:
            self.ml_model.eval()

        self.x = np.linspace(0, (nx - 1) * dx, nx)
        self.p = None
        self.p_prev = None
        self.history = []

    def reset(self):
        self.p = np.zeros(self.nx)
        self.p_prev = np.zeros(self.nx)
        self.history = []

    def source(self, t, freq=500, amplitude=10.0):
        if t < 0.01:
            window = 0.5 * (1 - np.cos(2 * np.pi * t / 0.01))
            return amplitude * window * np.sin(2 * np.pi * freq * t)
        return 0.0

    def update_interior(self):
        p_new = np.zeros(self.nx)
        for i in range(1, self.nx - 1):
            p_new[i] = (2 * self.p[i] - self.p_prev[i] +
                        self.cfl ** 2 * (self.p[i + 1] - 2 * self.p[i] + self.p[i - 1]))
        return p_new

    def ml_boundary(self, p_neighbor, p_prev_neighbor, t, freq):
        if self.ml_model is None or not self.use_ml:
            return 0.6 * p_neighbor

        # Feature extraction and normalization
        features = torch.FloatTensor([[
            np.clip(p_neighbor / 5.0, -1, 1),
            np.clip(p_prev_neighbor / 5.0, -1, 1),
            np.clip((p_neighbor - p_prev_neighbor) / 5.0, -1, 1),
            np.clip(t / 0.02, 0, 1),
            (freq - 500) / 500
        ]])

        if next(self.ml_model.parameters()).is_cuda:
            features = features.cuda()

        with torch.no_grad():
            pred_normalized = self.ml_model(features)
            pred = pred_normalized.item() * 5.0  # Denormalize

        # Strong physical constraints
        # 1. Boundary pressure cannot exceed the absolute value of incident pressure
        pred = np.clip(pred, -np.abs(p_neighbor), np.abs(p_neighbor))
        # 2. Boundary pressure sign must match incident pressure (absorption boundary characteristic)
        if p_neighbor * pred < 0:
            pred = 0.3 * p_neighbor
        # 3. Energy decay constraint
        if np.abs(pred) > 0.9 * np.abs(p_neighbor):
            pred = 0.85 * p_neighbor

        return pred

    def simulate(self, n_steps=600, src_freq=800, src_amplitude=8.0, record_every=10):
        self.reset()
        src_idx = self.nx // 2

        for step in range(n_steps):
            t = step * self.dt
            src_val = self.source(t, src_freq, src_amplitude)
            self.p[src_idx] += src_val

            p_new = self.update_interior()
            p_new[0] = 0

            # Use ML boundary
            p_neighbor = self.p[-2]
            p_prev_neighbor = self.p_prev[-2]
            p_new[-1] = self.ml_boundary(p_neighbor, p_prev_neighbor, t, src_freq)

            if step % record_every == 0:
                self.history.append({
                    'step': step,
                    't': t,
                    'p': self.p.copy(),
                    'p_boundary_right': p_new[-1]
                })

            self.p_prev = self.p.copy()
            self.p = p_new

        return self.history


# ==================== 6. Training function ====================

def train_model():
    print("\n" + "=" * 60)
    print("Training ML boundary model with physical constraints")
    print("=" * 60)

    print("Generating training data...")
    generator = TrainingDataGenerator(nx=200, dx=0.01, c=343, cfl=0.5)
    X, y = generator.generate(n_samples=150, n_steps=250)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")

    model = ConstrainedBoundaryML(input_dim=5, hidden_dims=[64, 128, 64])
    model = model.to(device)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    train_losses = []
    val_losses = []

    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}/200: Train Loss={loss.item():.9f}, Val Loss={val_loss.item():.9f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('training_history.png', dpi=150)
    plt.close()

    model.load_state_dict(torch.load('best_model.pth'))
    print(f"\nBest model saved (validation loss: {best_val_loss:.9f})")

    return model


# ==================== 7. Comparison experiment ====================

def comparison_experiment(ml_model):
    print("\n" + "=" * 60)
    print("Comparison experiment")
    print("=" * 60)

    test_freq = 800
    test_amplitude = 8.0

    # Real physical boundary
    print("\nRunning real physical boundary...")
    fdtd_true = BaseFDTD(nx=200, dx=0.01, c=343, cfl=0.5)

    def true_boundary(p_new, step):
        p_neighbor = fdtd_true.p[-2]
        p_new[-1] = RealPhysicalBoundary.apply(p_neighbor, fdtd_true.p_prev[-2], test_freq)
        p_new[0] = 0
        return p_new

    fdtd_true.boundary_condition = true_boundary
    history_true = fdtd_true.simulate(n_steps=600, record_every=10)

    # Simplified physical boundary
    print("Running simplified physical boundary...")
    fdtd_simple = BaseFDTD(nx=200, dx=0.01, c=343, cfl=0.5)

    def simple_boundary(p_new, step):
        p_neighbor = fdtd_simple.p[-2]
        p_new[-1] = SimplifiedBoundary.apply(p_neighbor, fdtd_simple.p_prev[-2])
        p_new[0] = 0
        return p_new

    fdtd_simple.boundary_condition = simple_boundary
    history_simple = fdtd_simple.simulate(n_steps=600, record_every=10)

    # ML boundary
    print("Running ML boundary solver...")
    hybrid_solver = StableHybridSolver(nx=200, dx=0.01, c=343, cfl=0.5,
                                       ml_model=ml_model, use_ml=True)
    history_ml = hybrid_solver.simulate(n_steps=600, src_freq=test_freq,
                                        src_amplitude=test_amplitude, record_every=10)

    # Calculate errors
    min_len = min(len(history_true), len(history_simple), len(history_ml))

    true_boundary_vals = [h['p_boundary_right'] for h in history_true[:min_len]]
    simple_boundary_vals = [h['p_boundary_right'] for h in history_simple[:min_len]]
    ml_boundary_vals = [h['p_boundary_right'] for h in history_ml[:min_len]]

    mae_simple = np.mean(np.abs(np.array(true_boundary_vals) - np.array(simple_boundary_vals)))
    mae_ml = np.mean(np.abs(np.array(true_boundary_vals) - np.array(ml_boundary_vals)))

    print(f"\nBoundary Condition MAE:")
    print(f"  Simplified physical boundary: {mae_simple:.6f}")
    print(f"  ML boundary:                 {mae_ml:.6f}")
    print(f"  Improvement:                 {(mae_simple - mae_ml) / mae_simple * 100:.1f}%")

    # Visualization - Only 2x1 layout (Boundary Pressure and Boundary Error)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    times = [h['t'] for h in history_true[:min_len]]

    # Plot 1: Boundary Pressure Comparison
    axes[0].plot(times, true_boundary_vals, 'k-', label='True', linewidth=2)
    axes[0].plot(times, simple_boundary_vals, 'r--', label='Simplified', alpha=0.7)
    axes[0].plot(times, ml_boundary_vals, 'b:', label='ML', alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Pressure')
    axes[0].set_title('Boundary Pressure')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Boundary Error Comparison
    axes[1].plot(times, np.abs(np.array(true_boundary_vals) - np.array(simple_boundary_vals)),
                 'r-', label='Simplified Error', alpha=0.7)
    axes[1].plot(times, np.abs(np.array(true_boundary_vals) - np.array(ml_boundary_vals)),
                 'b-', label='ML Error', alpha=0.7)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title('Boundary Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150)
    plt.close()

    print("\nComparison plot saved: comparison_results.png")


# ==================== 8. Main program ====================

def main():
    print("=" * 60)
    print("Acoustic FDTD Hybrid ML Solver (with physical constraints)")
    print("=" * 60)

    ml_model = train_model()
    comparison_experiment(ml_model)

    print("\nThe experiment is complete")


if __name__ == "__main__":
    main()