import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from collections import deque

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


class BaseFDTD:
    def __init__(self, nx=2000, dx=0.01, c=343, cfl=0.5):
        self.nx = nx
        self.dx = dx
        self.c = c
        self.cfl = cfl
        self.dt = cfl * dx / c
        self.x = np.linspace(0, (nx - 1) * dx, nx)

    def solve(self, freq=800, amplitude=8.0, n_steps=600, reflection_coeff=0.6, save_boundary=True):
        p = np.zeros(self.nx)
        p_prev = np.zeros(self.nx)
        history = []
        src_idx = self.nx // 2

        for step in range(n_steps):
            t = step * self.dt

            if t < 0.01:
                window = 0.5 * (1 - np.cos(2 * np.pi * t / 0.01))
                src_val = amplitude * window * np.sin(2 * np.pi * freq * t)
                p[src_idx] += src_val

            p_new = np.zeros(self.nx)
            for i in range(1, self.nx - 1):
                p_new[i] = (2 * p[i] - p_prev[i] +
                            self.cfl ** 2 * (p[i + 1] - 2 * p[i] + p[i - 1]))

            p_new[0] = 0
            p_new[-1] = reflection_coeff * p[-2]

            if step % 10 == 0:
                record = {'step': step, 't': t, 'p': p.copy()}
                if save_boundary:
                    record['p_boundary'] = p_new[-1]
                history.append(record)

            p_prev, p = p, p_new

        return history


class RealBoundary:
    @staticmethod
    def get_reflection_coeff(p_neighbor, freq):
        reflection = 0.85 / (1.0 + freq / 1500)
        nonlinear = 1.0 / (1.0 + 0.3 * np.abs(p_neighbor))
        return reflection * nonlinear


class UltraResidualBlock(nn.Module):
    """Ultra-large residual block"""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = self.fc1(x)
        x = nn.GELU()(x)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [batch, dim]
        residual = x
        batch_size = x.shape[0]

        # Add sequence dimension
        x = x.unsqueeze(1)  # [batch, 1, dim]

        qkv = self.qkv(x).reshape(batch_size, 1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, 1, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, 1, -1)
        x = self.proj(x)
        x = x.squeeze(1)

        return self.ln(x + residual)


class FeedForward(nn.Module):
    """Feed-forward network"""

    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.fc1(x)
        x = nn.GELU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class TransformerBlock(nn.Module):
    """Transformer block (Attention + FFN)"""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, dim * 4, dropout)

    def forward(self, x):
        x = self.attention(x)
        x = self.ffn(x)
        return x


class UltraLargeReflectionML(nn.Module):
    """
    Ultra-large reflection coefficient prediction model
    Parameters: ~2-3 million
    Architecture: Input projection + feature expansion + multi-layer Transformer + residual network + output
    """

    def __init__(self,
                 input_dim=4,
                 hidden_dim=512,
                 num_transformer_layers=6,
                 num_residual_layers=8,
                 num_heads=8,
                 use_history=True,
                 history_len=30,
                 dropout=0.1):
        super().__init__()

        self.use_history = use_history
        self.history_len = history_len
        self.hidden_dim = hidden_dim

        # Input projection (expand to high dimension)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # History information processing (expand to same dimension)
        if use_history:
            self.history_proj = nn.Sequential(
                nn.Linear(input_dim * history_len, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Additional feature projection
        self.extra_proj = nn.Sequential(
            nn.Linear(8, hidden_dim),  # 8 additional features
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer layers (global dependencies)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])

        # Residual blocks (local features)
        self.residual_blocks = nn.ModuleList([
            UltraResidualBlock(hidden_dim, dropout)
            for _ in range(num_residual_layers)
        ])

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output layer (multi-layer dimension reduction)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Learnable physical priors
        self.base_reflection = nn.Parameter(torch.tensor(0.7))
        self.reflection_range = nn.Parameter(torch.tensor(0.25))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, p_neighbor, p_prev, t, freq, history_buffer=None):
        """
        p_neighbor: [batch]
        p_prev: [batch]
        t: [batch]
        freq: [batch]
        history_buffer: [batch, history_len, input_dim]
        """
        # Ensure 1D
        if p_neighbor.dim() > 1:
            p_neighbor = p_neighbor.squeeze()
        if p_prev.dim() > 1:
            p_prev = p_prev.squeeze()
        if t.dim() > 1:
            t = t.squeeze()
        if freq.dim() > 1:
            freq = freq.squeeze()

        batch_size = p_neighbor.shape[0]

        # Normalize inputs
        p_norm = torch.clamp(p_neighbor / 5.0, -1, 1)
        p_prev_norm = torch.clamp(p_prev / 5.0, -1, 1)
        t_norm = torch.clamp(t / 0.02, 0, 1)
        freq_norm = (freq - 500) / 500

        # Base features
        base_features = torch.stack([p_norm, p_prev_norm, t_norm, freq_norm], dim=1)

        # Input projection
        x = self.input_proj(base_features)

        # History information processing
        if self.use_history and history_buffer is not None:
            hist_flat = history_buffer.view(batch_size, -1)
            hist_features = self.history_proj(hist_flat)
            x = x + 0.3 * hist_features

        # Calculate additional features
        dp_norm = p_norm - p_prev_norm
        d2p_norm = dp_norm - (p_prev_norm - torch.roll(p_prev_norm, shifts=1, dims=0) if batch_size > 1 else dp_norm)
        phase = torch.sin(2 * np.pi * freq_norm * t_norm)
        phase2 = torch.cos(4 * np.pi * freq_norm * t_norm)
        energy = torch.abs(p_norm)
        energy_ratio = energy / (torch.abs(p_prev_norm) + 1e-6)
        sign = torch.sign(p_norm * p_prev_norm)

        extra_features = torch.stack([
            dp_norm, d2p_norm, phase, phase2, energy, energy_ratio, sign, t_norm * freq_norm
        ], dim=1)

        extra_x = self.extra_proj(extra_features)
        x = x + 0.2 * extra_x

        # Transformer layers (capture global dependencies)
        for transformer in self.transformer_layers:
            x = transformer(x)

        # Residual layers (capture local features)
        residual_x = x
        for residual_block in self.residual_blocks:
            residual_x = residual_block(residual_x)

        # Feature fusion
        x = torch.cat([x, residual_x], dim=-1)
        x = self.fusion(x)

        # Output
        ml_coeff = self.output_layer(x).squeeze()

        # Temperature scaling
        ml_coeff = (ml_coeff - 0.5) * torch.sigmoid(self.temperature) + 0.5

        # Physical constraints
        coeff = torch.sigmoid(self.base_reflection) + \
                torch.sigmoid(self.reflection_range) * (ml_coeff - 0.5)
        coeff = torch.clamp(coeff, 0.5, 0.95)

        return coeff


class TrainingDataGenerator:
    def __init__(self, nx=200, dx=0.01, c=343, cfl=0.5, history_len=30):
        self.nx = nx
        self.dx = dx
        self.c = c
        self.cfl = cfl
        self.dt = cfl * dx / c
        self.history_len = history_len

    def generate(self, n_samples=80, n_steps=200):
        X_p = []
        X_p_prev = []
        X_t = []
        X_freq = []
        X_history = []
        y_coeff = []

        for sample_idx in range(n_samples):
            freq = np.random.uniform(500, 1000)
            amplitude = np.random.uniform(6.0, 10.0)

            p = np.zeros(self.nx)
            p_prev = np.zeros(self.nx)
            src_idx = self.nx // 2

            history_buffer = deque(maxlen=self.history_len)

            for step in range(n_steps):
                t = step * self.dt

                if t < 0.01:
                    window = 0.5 * (1 - np.cos(2 * np.pi * t / 0.01))
                    src_val = amplitude * window * np.sin(2 * np.pi * freq * t)
                    p[src_idx] += src_val

                p_new = np.zeros(self.nx)
                for i in range(1, self.nx - 1):
                    p_new[i] = (2 * p[i] - p_prev[i] +
                                self.cfl ** 2 * (p[i + 1] - 2 * p[i] + p[i - 1]))

                p_new[0] = 0
                true_reflection = RealBoundary.get_reflection_coeff(p[-2], freq)
                p_new[-1] = true_reflection * p[-2]

                current_features = [p[-2] / 5.0, p_prev[-2] / 5.0, t / 0.02, (freq - 500) / 500]
                history_buffer.append(current_features)

                if step > 20 and step % 5 == 0 and len(history_buffer) == self.history_len:
                    X_p.append(p[-2])
                    X_p_prev.append(p_prev[-2])
                    X_t.append(t)
                    X_freq.append(freq)
                    X_history.append(list(history_buffer))
                    y_coeff.append(true_reflection)

                p_prev, p = p, p_new

            if (sample_idx + 1) % 20 == 0:
                print(f"  Generating sample {sample_idx + 1}/{n_samples}")

        X_p = torch.FloatTensor(X_p)
        X_p_prev = torch.FloatTensor(X_p_prev)
        X_t = torch.FloatTensor(X_t)
        X_freq = torch.FloatTensor(X_freq)
        X_history = torch.FloatTensor(X_history)
        y_coeff = torch.FloatTensor(y_coeff)

        print(f"Generation complete: {len(X_p)} training samples")
        print(f"History window shape: {X_history.shape}")
        print(f"Reflection coefficient range: [{y_coeff.min():.4f}, {y_coeff.max():.4f}]")

        return (X_p, X_p_prev, X_t, X_freq, X_history), y_coeff


class HybridFDTDSolver:
    def __init__(self, nx=200, dx=0.01, c=343, cfl=0.5, ml_model=None, use_ml=True, history_len=30):
        self.nx = nx
        self.dx = dx
        self.c = c
        self.cfl = cfl
        self.dt = cfl * dx / c
        self.x = np.linspace(0, (nx - 1) * dx, nx)
        self.history_len = history_len

        self.ml_model = ml_model
        self.use_ml = use_ml

        if ml_model is not None:
            self.ml_model.eval()

    def solve(self, freq=800, amplitude=8.0, n_steps=600):
        p = np.zeros(self.nx)
        p_prev = np.zeros(self.nx)
        history = []
        reflection_history = []
        src_idx = self.nx // 2

        history_buffer = deque(maxlen=self.history_len)

        for step in range(n_steps):
            t = step * self.dt

            if t < 0.01:
                window = 0.5 * (1 - np.cos(2 * np.pi * t / 0.01))
                src_val = amplitude * window * np.sin(2 * np.pi * freq * t)
                p[src_idx] += src_val

            p_new = np.zeros(self.nx)
            for i in range(1, self.nx - 1):
                p_new[i] = (2 * p[i] - p_prev[i] +
                            self.cfl ** 2 * (p[i + 1] - 2 * p[i] + p[i - 1]))

            p_new[0] = 0

            if self.ml_model is not None and self.use_ml:
                with torch.no_grad():
                    p_neighbor_tensor = torch.FloatTensor([p[-2]]).to(device)
                    p_prev_tensor = torch.FloatTensor([p_prev[-2]]).to(device)
                    t_tensor = torch.FloatTensor([t]).to(device)
                    freq_tensor = torch.FloatTensor([freq]).to(device)

                    history_tensor = None
                    if len(history_buffer) == self.history_len:
                        history_list = list(history_buffer)
                        history_tensor = torch.FloatTensor(history_list).unsqueeze(0).to(device)

                    reflection = self.ml_model(p_neighbor_tensor, p_prev_tensor, t_tensor, freq_tensor, history_tensor)
                    reflection = reflection.item()
                    reflection = np.clip(reflection, 0.5, 0.95)

                p_new[-1] = reflection * p[-2]
                reflection_history.append(reflection)
            else:
                p_new[-1] = 0.6 * p[-2]
                reflection_history.append(0.6)

            current_features = [p[-2] / 5.0, p_prev[-2] / 5.0, t / 0.02, (freq - 500) / 500]
            history_buffer.append(current_features)

            if step % 10 == 0:
                history.append({
                    'step': step,
                    't': t,
                    'p': p.copy(),
                    'p_boundary': p_new[-1],
                    'reflection': reflection_history[-1]
                })

            p_prev, p = p, p_new

        return history


def train_model():
    print("\n" + "=" * 70)
    print("Training Ultra-large Reflection Coefficient Prediction Model")
    print("=" * 70)

    print("Generating training data...")
    generator = TrainingDataGenerator(nx=200, dx=0.01, c=343, cfl=0.5, history_len=30)
    (X_p, X_p_prev, X_t, X_freq, X_history), y_coeff = generator.generate(n_samples=60, n_steps=200)

    n_total = len(X_p)
    n_train = int(0.8 * n_total)

    X_p_train, X_p_val = X_p[:n_train], X_p[n_train:]
    X_p_prev_train, X_p_prev_val = X_p_prev[:n_train], X_p_prev[n_train:]
    X_t_train, X_t_val = X_t[:n_train], X_t[n_train:]
    X_freq_train, X_freq_val = X_freq[:n_train], X_freq[n_train:]
    X_history_train, X_history_val = X_history[:n_train], X_history[n_train:]
    y_train, y_val = y_coeff[:n_train], y_coeff[n_train:]

    print(f"Training set: {n_train}, Validation set: {n_total - n_train}")

    # Create ultra-large model
    model = UltraLargeReflectionML(
        input_dim=4,
        hidden_dim=512,
        num_transformer_layers=6,
        num_residual_layers=8,
        num_heads=8,
        use_history=True,
        history_len=30,
        dropout=0.1
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    X_p_train = X_p_train.to(device)
    X_p_prev_train = X_p_prev_train.to(device)
    X_t_train = X_t_train.to(device)
    X_freq_train = X_freq_train.to(device)
    X_history_train = X_history_train.to(device)
    y_train = y_train.to(device)

    X_p_val = X_p_val.to(device)
    X_p_prev_val = X_p_prev_val.to(device)
    X_t_val = X_t_val.to(device)
    X_freq_val = X_freq_val.to(device)
    X_history_val = X_history_val.to(device)
    y_val = y_val.to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Use AdamW + learning rate warmup
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    train_losses = []
    val_losses = []

    print("Starting training...")
    best_val_loss = float('inf')

    epochs = 150
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(X_p_train, X_p_prev_train, X_t_train, X_freq_train, X_history_train)
        loss = criterion(output, y_train)

        # Physical constraint loss
        if epoch > 50:
            # Smoothness loss
            smooth_loss = torch.mean(torch.abs(output[1:] - output[:-1]))
            # Range constraint loss
            range_loss = torch.mean(torch.relu(output - 0.95) + torch.relu(0.5 - output))
            loss = loss + 0.01 * smooth_loss + 0.005 * range_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_p_val, X_p_prev_val, X_t_val, X_freq_val, X_history_val)
            val_loss = criterion(val_output, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_ultra_large_model.pth')

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs}: Train Loss={loss.item():.9f}, Val Loss={val_loss.item():.9f}")

    # Training curves`
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    plt.plot(train_losses[50:], label='Train Loss (after 50)')
    plt.plot(val_losses[50:], label='Val Loss (after 50)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History (Zoomed)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ultra_large_training.png', dpi=150)
    plt.close()

    model.load_state_dict(torch.load('best_ultra_large_model.pth'))
    print(f"\nBest model saved (validation loss: {best_val_loss:.9f})")

    return model


def comparison_experiment(ml_model):
    print("\n" + "=" * 70)
    print("Comparison Experiment (Ultra-large Model)")
    print("=" * 70)

    test_freq = 800
    test_amplitude = 8.0

    # Real physical FDTD
    print("\nRunning real physical FDTD...")
    p = np.zeros(200)
    p_prev = np.zeros(200)
    history_real = []
    src_idx = 100
    dt = 0.5 * 0.01 / 343
    cfl_sq = 0.5 ** 2

    for step in range(600):
        t = step * dt
        if t < 0.01:
            window = 0.5 * (1 - np.cos(2 * np.pi * t / 0.01))
            src_val = test_amplitude * window * np.sin(2 * np.pi * test_freq * t)
            p[src_idx] += src_val

        p_new = np.zeros(200)
        for i in range(1, 199):
            p_new[i] = (2 * p[i] - p_prev[i] + cfl_sq * (p[i + 1] - 2 * p[i] + p[i - 1]))

        p_new[0] = 0
        true_reflection = RealBoundary.get_reflection_coeff(p[-2], test_freq)
        p_new[-1] = true_reflection * p[-2]

        if step % 10 == 0:
            history_real.append({'t': t, 'p': p.copy(), 'p_boundary': p_new[-1]})

        p_prev, p = p, p_new

    # Simplified physical FDTD
    print("Running simplified physical FDTD...")
    fdtd_simple = BaseFDTD(nx=200, dx=0.01, c=343, cfl=0.5)
    history_simple = fdtd_simple.solve(freq=test_freq, amplitude=test_amplitude,
                                       n_steps=600, reflection_coeff=0.6, save_boundary=True)

    # Hybrid FDTD
    print("Running ultra-large hybrid FDTD...")
    hybrid_solver = HybridFDTDSolver(nx=200, dx=0.01, c=343, cfl=0.5,
                                     ml_model=ml_model, use_ml=True, history_len=30)
    history_hybrid = hybrid_solver.solve(freq=test_freq, amplitude=test_amplitude, n_steps=600)

    min_len = min(len(history_real), len(history_simple), len(history_hybrid))

    # Calculate field errors only (no boundary errors)
    field_error_simple = 0
    field_error_hybrid = 0

    for i in range(min_len):
        real_p = history_real[i]['p']
        simple_p = history_simple[i]['p']
        hybrid_p = history_hybrid[i]['p']

        field_error_simple += np.mean(np.abs(real_p - simple_p))
        field_error_hybrid += np.mean(np.abs(real_p - hybrid_p))

    field_error_simple /= min_len
    field_error_hybrid /= min_len

    improvement = (field_error_simple - field_error_hybrid) / field_error_simple * 100

    print(f"\n{'=' * 50}")
    print(f"Full Field MAE:")
    print(f"  Simplified physical boundary: {field_error_simple:.6f}")
    print(f"  Hybrid solver:               {field_error_hybrid:.6f}")
    print(f"  Improvement:                 {improvement:.1f}%")
    print(f"{'=' * 50}")

    # Visualization - Only 2x2 layout with field comparison and summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    times = [h['t'] for h in history_real[:min_len]]

    # Plot 1: Pressure field comparison at time t1
    t_idx1 = len(history_real) // 4
    x = np.linspace(0, 1.99, 200)
    axes[0, 0].plot(x, history_real[t_idx1]['p'], 'k-', label='Real', linewidth=2)
    axes[0, 0].plot(x, history_simple[t_idx1]['p'], 'r--', label='Simplified', alpha=0.7)
    axes[0, 0].plot(x, history_hybrid[t_idx1]['p'], 'b:', label='Ultra-Large Hybrid', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Position (m)')
    axes[0, 0].set_ylabel('Pressure')
    axes[0, 0].set_title(f'Field at t={history_real[t_idx1]["t"]:.4f}s (t1)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Pressure field comparison at time t2
    t_idx2 = len(history_real) // 2
    axes[0, 1].plot(x, history_real[t_idx2]['p'], 'k-', label='Real', linewidth=2)
    axes[0, 1].plot(x, history_simple[t_idx2]['p'], 'r--', label='Simplified', alpha=0.7)
    axes[0, 1].plot(x, history_hybrid[t_idx2]['p'], 'b:', label='Ultra-Large Hybrid', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Position (m)')
    axes[0, 1].set_ylabel('Pressure')
    axes[0, 1].set_title(f'Field at t={history_real[t_idx2]["t"]:.4f}s (t2)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Field error evolution over time
    field_errors_simple = []
    field_errors_hybrid = []
    for i in range(min_len):
        field_errors_simple.append(np.mean(np.abs(history_real[i]['p'] - history_simple[i]['p'])))
        field_errors_hybrid.append(np.mean(np.abs(history_real[i]['p'] - history_hybrid[i]['p'])))

    axes[1, 0].plot(times, field_errors_simple, 'r-', label='Simplified Error', alpha=0.7)
    axes[1, 0].plot(times, field_errors_hybrid, 'b-', label='Hybrid Error', alpha=0.7)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Field MAE')
    axes[1, 0].set_title('Field Error Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Plot 4: Summary bar chart (Field only)
    methods = ['Field Error']
    simple_errors = [field_error_simple]
    hybrid_errors = [field_error_hybrid]
    x_pos = np.arange(len(methods))
    width = 0.35
    axes[1, 1].bar(x_pos - width / 2, simple_errors, width, label='Simplified', color='red', alpha=0.7)
    axes[1, 1].bar(x_pos + width / 2, hybrid_errors, width, label='Ultra-Large Hybrid', color='blue', alpha=0.7)
    axes[1, 1].set_ylabel('Mean Absolute Error')
    axes[1, 1].set_title(f'Full Field Error Summary (Improvement: {improvement:.1f}%)')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('ultra_large_comparison.png', dpi=150)
    plt.close()

    print(f"\nFull field improvement: {improvement:.1f}%")
    print("Comparison plot saved: ultra_large_comparison.png")

    return improvement


def main():
    print("=" * 70)
    print("Ultra-large Hybrid FDTD Solver")
    print("=" * 70)
    print("Network Architecture:")
    print("  - Hidden dimension: 512")
    print("  - Transformer layers: 6 layers (8-head attention)")
    print("  - Residual blocks: 8 layers")
    print("  - History window: 30 steps")
    print("  - Parameter count: ~2.5 million")
    print("=" * 70)

    ml_model = train_model()
    improvement = comparison_experiment(ml_model)

    print("\n" + "=" * 70)
    print(f"Final full field improvement: {improvement:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
