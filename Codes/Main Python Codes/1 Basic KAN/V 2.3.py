import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from deepkan import DeepKAN

EPOCHS = 150
LEARNING_RATE = 0.008
GRIDS = 50
SPLINE = 3


def print_comparison_table(t_eval_np, y_pred, y_exact_eval, example_number):
    print(f"\nComparison Table for Example {example_number}:")
    print(f"{'t':<10}{'Network Output':<20}{'Exact Solution':<20}")
    print("-" * 50)
    for t, pred, exact in zip(t_eval_np.flatten(), y_pred.flatten(), y_exact_eval.flatten()):
        print(f"{t:<10.4f}{pred:<20.6f}{exact:<20.6f}")


def train_model_with_physics(model, epochs, optimizer, t_train, ode_loss, patience=50):
    model.train()
    best_loss = float('inf')
    counter = 0
    losses = []

    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            z_pred = model(t_train)
            total_loss = ode_loss(t_train, z_pred)
            total_loss.backward()
            return total_loss

        loss = optimizer.step(closure)

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
        losses.append(loss.item())

    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.show()

    return model


def solve_example(architecture, example_number):
    # Example-specific parameters
    examples = [
        {"A": 1, "a": 0, "t_range": (0, 1), "exact": lambda t: (np.exp(-t**2 / 2) + t**5 + t**3 + t**2) / (t**3 + t + 1)},
        {"A": 3, "a": 0, "t_range": (0, 3), "exact": lambda t: (np.sin(4 * t) / 5 + np.cos(4 * t) / 10 + 2.9 * np.exp(-2 * t))},
        {"A": 0.5, "a": 0, "t_range": (0, 1), "exact": lambda t: (t + 1)**2 - 0.5 * np.exp(t)}
    ]

    params = examples[example_number - 1]
    A, a, t_range, exact = params.values()

    model = DeepKAN(
        input_dim=1,
        hidden_layers=architecture,
        spline_order=SPLINE,
    )

    def ode_loss(t, z):
        y = A + (t - a) * z
        dz_dt = torch.autograd.grad(z, t, grad_outputs=torch.ones_like(z), create_graph=True)[0]
        dy_dt = z + (t - a) * dz_dt

        if example_number == 1:
            f_term = (t + ((1 + 3 * t**2) / (1 + t + t**3))) * y
            g_term = 2 * t + t**3 + t**2 * ((1 + 3 * t**2) / (1 + t + t**3))
            residual = g_term - f_term
        elif example_number == 2:
            f_term = 2 * y
            g_term = torch.cos(4 * t)
            residual = g_term - f_term
        elif example_number == 3:
            f_term = y - t**2 + 1
            residual = f_term
        else:
            raise ValueError("Invalid example number!")

        return torch.mean(((dy_dt - residual) ** 2)) / 2

    t_train = torch.linspace(*t_range, 100).reshape(-1, 1).requires_grad_()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=LEARNING_RATE)

    model = train_model_with_physics(model, EPOCHS, optimizer, t_train, ode_loss)

    t_eval = torch.linspace(*t_range, 21).reshape(-1, 1).requires_grad_()
    y_pred = A + (t_eval - a) * model(t_eval).detach()
    y_pred = y_pred.detach().numpy()
    t_eval_np = t_eval.detach().numpy()
    y_exact_eval = exact(t_eval_np)

    print(f"MSE for example {example_number}: {np.mean((y_exact_eval - y_pred) ** 2)}")
    print(f"MAE for example {example_number}: {np.mean(np.abs(y_exact_eval - y_pred))}")
    print_comparison_table(t_eval_np, y_pred, y_exact_eval, example_number)

    return t_eval_np, y_pred, y_exact_eval


def plot_results(t1, y1, y1_exact, t2, y2, y2_exact, t3, y3, y3_exact, architecture):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(t1, y1, 'b-', label='KAN Solution')
    plt.plot(t1, y1_exact, 'g--', label='Exact Solution')
    plt.title('Example 1: Solution')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(t2, y2, 'r-', label='KAN Solution')
    plt.plot(t2, y2_exact, 'g--', label='Exact Solution')
    plt.title('Example 2: Solution')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(t3, y3, 'm-', label='KAN Solution')
    plt.plot(t3, y3_exact, 'c--', label='Exact Solution')
    plt.title('Example 3: Solution')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def main():
    architectures = [
        [50, 20],
    ]

    for architecture in architectures:
        print("Solving Example 1...")
        t1, y1, y1_exact = solve_example(architecture, 1)

        print("Solving Example 2...")
        t2, y2, y2_exact = solve_example(architecture, 2)

        print("Solving Example 3...")
        t3, y3, y3_exact = solve_example(architecture, 3)

        print("Plotting results...")
        plot_results(t1, y1, y1_exact, t2, y2, y2_exact, t3, y3, y3_exact, architecture)


if __name__ == "__main__":
    main()