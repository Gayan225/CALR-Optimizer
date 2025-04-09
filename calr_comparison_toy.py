#calr_comparison_toy.py
import torch
import torch.nn as nn
from torch.optim import Optimizer
import matplotlib.pyplot as plt

# For reproducibility
torch.manual_seed(42)

# Generate toy nonlinear dataset
def generate_data(n=100):
    X = torch.rand(n, 2) * 2 - 1
    y = ((X[:, 0]**2 + X[:, 1]**2) > 0.5).float().unsqueeze(1)
    return X, y

# Simple nonlinear model
class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

# CALR optimizer with gradient-based curvature
class CALRGradient:
    def __init__(self, params, lr=1e-2, alpha=0.1, epsilon=1e-8):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g**2
            m_hat = self.m[i] / (1 - 0.9 ** self.t)
            v_hat = self.v[i] / (1 - 0.999 ** self.t)
            h = g.abs() + self.alpha
            update = self.lr / (h.sqrt() + self.epsilon) * m_hat
            p.data -= update

# CALR optimizer with diagonal Hessian approximation
class CALRHessian(Optimizer):
    def __init__(self, params, model, loss_fn, inputs, targets, lr=1e-2, delta=0.1, epsilon=1e-8):
        defaults = dict(lr=lr, delta=delta, epsilon=epsilon)
        super().__init__(params, defaults)
        self.model = model
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.targets = targets

    def step(self):
        self.model.zero_grad()
        output = self.model(self.inputs)
        loss = self.loss_fn(output, self.targets)
        loss.backward(create_graph=True)

        for group in self.param_groups:
            lr, delta, epsilon = group['lr'], group['delta'], group['epsilon']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                diag_h = torch.zeros_like(p)
                for i in range(p.numel()):
                    grad_i = g.flatten()[i]
                    if grad_i.requires_grad:
                        hess = torch.autograd.grad(grad_i, p, retain_graph=True, allow_unused=True)[0]
                        if hess is not None:
                            diag_h.view(-1)[i] = hess.view(-1)[i]
                p.data -= lr / (diag_h.abs() + delta + epsilon) * g
        return loss

# Training loop
def train(model, optimizer, X, y, loss_fn, epochs=500):
    losses = []
    for _ in range(epochs):
        model.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# Prepare data and loss function
X, y = generate_data()
loss_fn = nn.BCEWithLogitsLoss()

# Models
model_g = ToyNet()
model_h = ToyNet()

# Optimizers
opt_g = CALRGradient(model_g.parameters())
opt_h = CALRHessian(model_h.parameters(), model_h, loss_fn, X, y)

# Train
loss_g = train(model_g, opt_g, X, y, loss_fn)
loss_h = train(model_h, opt_h, X, y, loss_fn)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(loss_g, label="CALR (Gradient-based)", linestyle="-", marker="o", markersize=2)
plt.plot(loss_h, label="CALR (Diagonal Hessian)", linestyle="--", marker="s", markersize=2)
plt.title("Loss Curve Comparison on Toy Problem")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("toy_comparison_loss_plot.png")
plt.show()
