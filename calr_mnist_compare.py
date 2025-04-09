#calr_mnist_compare.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import Optimizer
import matplotlib.pyplot as plt

# Tiny NN
class TinyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    def forward(self, x): return self.net(x)

# Gradient-based CALR
class CALRGrad:
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
            if p.grad is None: continue
            g = p.grad
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g ** 2
            m_hat = self.m[i] / (1 - 0.9 ** self.t)
            h = g.abs() + self.alpha
            p.data -= self.lr / (h.sqrt() + self.epsilon) * m_hat

# Diagonal Hessian CALR
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
        out = self.model(self.inputs)
        loss = self.loss_fn(out, self.targets)
        loss.backward(create_graph=True)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                diag_h = torch.zeros_like(p)
                flat_g = g.view(-1)
                flat_diag = diag_h.view(-1)
                for i in range(flat_g.shape[0]):
                    if flat_g[i].requires_grad:
                        second = torch.autograd.grad(flat_g[i], p, retain_graph=True, allow_unused=True)[0]
                        if second is not None:
                            flat_diag[i] = second.view(-1)[i]
                p.data -= group['lr'] / (diag_h.abs() + group['delta'] + group['epsilon']) * g
        return loss

# Train
def train(model, opt, X, y, loss_fn, epochs=10):
    loss_hist = []
    for _ in range(epochs):
        model.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())
    return loss_hist

if __name__ == "__main__":
    torch.manual_seed(0)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    subset, _ = torch.utils.data.random_split(mnist, [16, len(mnist)-16])
    loader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=True)
    X, y = next(iter(loader))
    X = X.view(X.size(0), -1)
    y_onehot = nn.functional.one_hot(y, num_classes=10).float()
    loss_fn = nn.BCEWithLogitsLoss()

    model_g = TinyNN()
    model_h = TinyNN()
    opt_g = CALRGrad(model_g.parameters())
    opt_h = CALRHessian(model_h.parameters(), model_h, loss_fn, X, y_onehot)

    loss_g = train(model_g, opt_g, X, y_onehot, loss_fn)
    loss_h = train(model_h, opt_h, X, y_onehot, loss_fn)

    plt.figure(figsize=(10, 5))
    plt.plot(loss_g, label="CALR (Gradient-based)", marker='o')
    plt.plot(loss_h, label="CALR (Diagonal Hessian)", linestyle='--', marker='s')
    plt.title("Loss Comparison on Tiny MNIST Subset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mnist_toy_calr_comparison.png")
    plt.show()
