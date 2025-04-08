import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer

# Himmelblau's Function
def himmelblau_function(coords):
    x, y = coords
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Gradient of Himmelblau's Function
def himmelblau_gradient(coords):
    x, y = coords
    grad_x = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
    grad_y = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
    return np.array([grad_x, grad_y])

# CALR Default Optimizer
class CALRDefault(Optimizer):
    def __init__(self, params, lr=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8, alpha=0.1):
        defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon, 'alpha': alpha}
        super(CALRDefault, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2, epsilon, alpha = group['beta1'], group['beta2'], group['epsilon'], group['alpha']

                state['step'] += 1
                step = state['step']

                m.mul_(beta1).add_(grad, alpha=(1 - beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)

                p.data.add_(-group['lr'] / (v_hat.sqrt() + epsilon) * m_hat)

        return loss


# CALR with Gradient Clipping
class CALRGradClip(CALRDefault):
    def __init__(self, params, lr=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8, alpha=0.1, grad_clip_value=0.1):
        super().__init__(params, lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, alpha=alpha)
        self.grad_clip_value = grad_clip_value

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data = torch.clamp(p.grad.data, -self.grad_clip_value, self.grad_clip_value)
        return super().step(closure)


# CALR with Cosine Annealing Learning Rate
class CALRCosineLR(CALRDefault):
    def __init__(self, params, lr=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8, alpha=0.1, T_max=300):
        super().__init__(params, lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, alpha=alpha)
        self.T_max = T_max

    def step(self, closure=None):
        for group in self.param_groups:
            step = group.setdefault('step', 0)
            group['step'] = step + 1
            group['lr'] = group['lr'] * 0.5 * (1 + np.cos(np.pi * step / self.T_max))
        return super().step(closure)


# CALR with Gradient Clipping and Cosine LR
class CALRGradClipCosineLR(CALRGradClip):
    def __init__(self, params, lr=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8, alpha=0.1, grad_clip_value=0.1, T_max=300):
        super().__init__(params, lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, alpha=alpha, grad_clip_value=grad_clip_value)
        self.T_max = T_max

    def step(self, closure=None):
        for group in self.param_groups:
            step = group.setdefault('step', 0)
            group['step'] = step + 1
            group['lr'] = group['lr'] * 0.5 * (1 + np.cos(np.pi * step / self.T_max))
        return super().step(closure)


# Optimization Function
def optimize_himmelblau(optimizer_type, optimizer_params, steps=300):
    # Initialize parameters
    params = torch.tensor([0.0, 0.0], requires_grad=True)
    optimizer = optimizer_type([params], **optimizer_params)

    history_params, history_loss, history_grad_norm = [], [], []

    for step in range(steps):
        optimizer.zero_grad()
        loss = himmelblau_function(params.detach().numpy())
        grad = torch.tensor(himmelblau_gradient(params.detach().numpy()), dtype=torch.float32)
        grad_norm = np.linalg.norm(grad.numpy())

        history_params.append(params.detach().numpy())
        history_loss.append(loss)
        history_grad_norm.append(grad_norm)

        params.grad = grad
        optimizer.step()

    return np.array(history_params), np.array(history_loss), np.array(history_grad_norm)

# Visualization Functions
# Corrected Trajectories Plot
def plot_trajectory(histories, labels, save_path="himmelblau_trajectories_full.png"):
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau_function((X, Y))

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=50, cmap="viridis")
    for history, label in zip(histories, labels):
        params_hist, _, _ = history
        plt.plot(params_hist[:, 0], params_hist[:, 1], label=label)
        plt.scatter(params_hist[-1, 0], params_hist[-1, 1], marker='o', label=f"{label} (End)")
    plt.title("Optimizer Trajectories in x-y Plane")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Trajectories plot saved to {save_path}")
    plt.close()

def plot_loss(histories, labels, save_path="himmelblau_loss.png"):
    plt.figure(figsize=(8, 6))
    for history, label in zip(histories, labels):
        loss_hist = history[1]
        # Use a dashed line for Adam to make it more visible
        linestyle = "dashed" if label == "Adam" else "solid"
        plt.plot(range(len(loss_hist)), loss_hist, label=label,linestyle=linestyle)
    plt.title("Loss vs Steps")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")
    plt.close()

def plot_gradient_norm(histories, labels, save_path="himmelblau_gradient_norm.png"):
    plt.figure(figsize=(8, 6))
    for history, label in zip(histories, labels):
        grad_norm_hist = history[2]
        plt.plot(range(len(grad_norm_hist)), grad_norm_hist, label=label)
    plt.title("Gradient Norm vs Steps")
    plt.xlabel("Steps")
    plt.ylabel("Gradient Norm")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Gradient Norm plot saved to {save_path}")
    plt.close()

# Main Script
if __name__ == "__main__":
    steps = 350

    # Optimizer settings
    optimizers = {
        "Adam": (torch.optim.Adam, {"lr": 1e-2}),
        "RMSProp": (torch.optim.RMSprop, {"lr": 1e-2}),
        "Nadam": (torch.optim.NAdam, {"lr": 1e-2}),
        "CALR Default": (CALRDefault, {"lr": 1e-1, "alpha": 0.1}),
        "CALR Grad Clip": (CALRGradClip, {"lr": 1e-2, "alpha": 0.1, "grad_clip_value": 0.1}),
        #"CALR Cosine LR": (CALRCosineLR, {"lr": 1e-3, "alpha": 0.1, "T_max": steps}),
        #"CALR Grad Clip + Cosine LR": (CALRGradClipCosineLR, {"lr": 1e-2, "alpha": 0.1, "grad_clip_value": 0.1, "T_max": steps}),
    }

    histories, labels = [], []
    for name, (opt_type, opt_params) in optimizers.items():
        print(f"Running {name}...")
        params_hist, loss_hist, grad_norm_hist = optimize_himmelblau(opt_type, opt_params, steps=steps)
        histories.append((params_hist, loss_hist, grad_norm_hist))
        labels.append(name)

    # Generate Individual Plots
    plot_trajectory(histories, labels, save_path="himmelblau_trajectories14.png")
    plot_loss(histories, labels, save_path="himmelblau_loss13.png")
    plot_gradient_norm(histories, labels, save_path="himmelblau_gradient_norm14.png")