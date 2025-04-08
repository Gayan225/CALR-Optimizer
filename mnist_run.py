import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Seed for Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Define CALR Optimizer Variants
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
                    state['H'] = torch.ones_like(p.data)

                state['step'] += 1
                m, v, H = state['m'], state['v'], state['H']
                beta1, beta2, epsilon, lr, alpha = group['beta1'], group['beta2'], group['epsilon'], group['lr'], group['alpha']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                H = grad.abs().add_(alpha)

                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])
                adaptive_lr = lr / (H.sqrt() + epsilon)
                p.data.add_(-adaptive_lr * m_hat)

        return loss


# Neural Network Definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)


# Train and Evaluate Function
def train_and_evaluate_with_cost(model, optimizer, train_loader, test_loader, criterion, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []
    cost_curve = []  # To track the training cost across batches

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            cost_curve.append(loss.item())  # Track the cost curve
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss.append(running_loss / len(train_loader))
        train_accuracy.append(100.0 * correct / total)

        # Evaluate
        model.eval()
        test_running_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)

        test_loss.append(test_running_loss / len(test_loader))
        test_accuracy.append(100.0 * test_correct / test_total)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss[-1]:.4f} - Train Acc: {train_accuracy[-1]:.2f}%"
              f" - Test Loss: {test_loss[-1]:.4f} - Test Acc: {test_accuracy[-1]:.2f}%")

    return train_loss, train_accuracy, test_loss, test_accuracy, cost_curve


# Plot Results
def plot_results(histories, labels, metric, save_path):
    plt.figure(figsize=(10, 6))
    for history, label in zip(histories, labels):
        plt.plot(history, label=label)
    plt.title(f"{metric} vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_cost_curve(cost_curves, labels, save_path):
    plt.figure(figsize=(10, 6))
    for cost_curve, label in zip(cost_curves, labels):
        plt.plot(cost_curve, label=label)
    plt.title("Cost Curve (Training Cost per Batch)")
    plt.xlabel("Batch")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_cost_curve2(cost_curves, labels, save_path):
    plt.figure(figsize=(10, 6))

    for cost_curve, label in zip(cost_curves, labels):
        # Smooth the curve using a moving average
        smoothed_curve = np.convolve(cost_curve, np.ones(50) / 50, mode='valid')
        plt.plot(smoothed_curve, label=label)

    plt.title("Cost Curve (Training Cost per Batch)")
    plt.xlabel("Batch")
    plt.ylabel("Cost (Smoothed)")
    plt.yscale("log")  # Logarithmic scale for better visibility
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Main Code
if __name__ == "__main__":
    # Load Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST("./data", train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST("./data", train=False, transform=transform), batch_size=1000, shuffle=False)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer Configurations
    optimizers = {
        "Adam": (optim.Adam, {"lr": 1e-2}),
        "Nadam": (optim.NAdam, {"lr": 1e-2}),
        "RMSProp": (optim.RMSprop, {"lr": 1e-2}),
        "SGD": (optim.SGD, {"lr": 1e-2, "momentum": 0.9}),
        "CALR Default": (CALRDefault, {"lr": 1e-2, "alpha": 0.1}),
    }

    train_histories = {"loss": [], "accuracy": []}
    test_histories = {"loss": [], "accuracy": []}
    cost_curves = []
    labels = []

    for name, (opt_type, opt_params) in optimizers.items():
        print(f"Running {name}...")
        model = SimpleNN()
        optimizer = opt_type(model.parameters(), **opt_params)
        train_loss, train_acc, test_loss, test_acc, cost_curve = train_and_evaluate_with_cost(model, optimizer, train_loader, test_loader, criterion, epochs=50)

        train_histories["loss"].append(train_loss)
        train_histories["accuracy"].append(train_acc)
        test_histories["loss"].append(test_loss)
        test_histories["accuracy"].append(test_acc)
        cost_curves.append(cost_curve)
        labels.append(name)

    # Plot Results
    plot_results(train_histories["loss"], labels, "Training Loss", "mnist_training_loss_R6.png")
    plot_results(test_histories["loss"], labels, "Testing Loss", "mnist_testing_loss_R6.png")
    plot_results(train_histories["accuracy"], labels, "Training Accuracy", "mnist_training_accuracy_R6.png")
    plot_results(test_histories["accuracy"], labels, "Testing Accuracy", "mnist_testing_accuracy_R6.png")
    plot_cost_curve(cost_curves, labels, "mnist_cost_curve_R6.png")
    plot_cost_curve2(cost_curves, labels, "mnist_cost_curve2_R6.png")

    # Create a Table Summary
    summary = []
    for i, label in enumerate(labels):
        summary.append({
            "Optimizer": label,
            "Best Train Loss": min(train_histories["loss"][i]),
            "Best Train Acc (%)": max(train_histories["accuracy"][i]),
            "Best Test Loss": min(test_histories["loss"][i]),
            "Best Test Acc (%)": max(test_histories["accuracy"][i]),
        })

    summary_df = pd.DataFrame(summary)
    print("\nSummary of Results:")
    print(summary_df)
    summary_df.to_csv("mnist_optimizer_comparison_summary_R6.csv", index=False)



















