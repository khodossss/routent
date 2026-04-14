"""Generate training visualization plots."""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def moving_average(data, window=100):
    if len(data) < window:
        window = max(1, len(data))
    cumsum = np.cumsum(data)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return cumsum[window - 1:] / window


def plot_reward_curve(rewards, output_dir):
    ma = moving_average(rewards)
    plt.figure(figsize=(10, 5))
    plt.plot(ma, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Reward (MA-100)")
    plt.title("Training Reward Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_curve.png"), dpi=150)
    plt.close()


def plot_accuracy_curve(corrects, output_dir):
    ma = moving_average([float(c) for c in corrects])
    plt.figure(figsize=(10, 5))
    plt.plot(ma, linewidth=1.5, color="green")
    plt.xlabel("Step")
    plt.ylabel("Accuracy (MA-100)")
    plt.title("Training Accuracy Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=150)
    plt.close()


def plot_cost_curve(costs, output_dir):
    ma = moving_average(costs)
    plt.figure(figsize=(10, 5))
    plt.plot(ma, linewidth=1.5, color="red")
    plt.xlabel("Step")
    plt.ylabel("Cost per Call (MA-100)")
    plt.title("Training Cost Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_curve.png"), dpi=150)
    plt.close()


def plot_model_distribution(models_used, output_dir, model_names=None, window=100):
    n = len(models_used)
    num_models = max(models_used) + 1 if models_used else 2
    if n < window:
        window = max(1, n)

    if model_names is None:
        model_names = [f"model_{i}" for i in range(num_models)]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]

    fractions = np.zeros((n - window + 1, num_models))
    for i in range(n - window + 1):
        chunk = models_used[i: i + window]
        for m in range(num_models):
            fractions[i, m] = chunk.count(m) / window

    x = np.arange(window - 1, n)
    plt.figure(figsize=(12, 6))
    plt.stackplot(
        x,
        [fractions[:, m] for m in range(num_models)],
        labels=[model_names[m] if m < len(model_names) else f"model_{m}" for m in range(num_models)],
        colors=colors[:num_models],
        alpha=0.8,
    )
    plt.xlabel("Step")
    plt.ylabel("Selection Fraction")
    plt.title("Model Selection Distribution Over Time")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_distribution.png"), dpi=150)
    plt.close()


def plot_loss_curves(updates, output_dir):
    if not updates:
        return

    policy_losses = [u.get("policy_loss", 0) for u in updates]
    value_losses = [u.get("value_loss", 0) for u in updates]
    entropies = [u.get("entropy", 0) for u in updates]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(policy_losses, linewidth=1.2)
    axes[0].set_title("Policy Loss")
    axes[0].set_xlabel("Update")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(value_losses, linewidth=1.2, color="orange")
    axes[1].set_title("Value Loss")
    axes[1].set_xlabel("Update")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(entropies, linewidth=1.2, color="green")
    axes[2].set_title("Entropy")
    axes[2].set_xlabel("Update")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--log", type=str, default="llm_router_rl/results/training_log.json")
    parser.add_argument("--output", type=str, default="llm_router_rl/results/plots")
    args = parser.parse_args()

    with open(args.log, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(args.output, exist_ok=True)

    steps = data["steps"]

    print("Generating plots...")
    plot_reward_curve(steps["rewards"], args.output)
    print("  reward_curve.png")

    plot_accuracy_curve(steps["corrects"], args.output)
    print("  accuracy_curve.png")

    plot_cost_curve(steps["costs"], args.output)
    print("  cost_curve.png")

    plot_model_distribution(steps["models_used"], args.output)
    print("  model_distribution.png")

    plot_loss_curves(data.get("updates", []), args.output)
    print("  loss_curves.png")

    print(f"\nAll plots saved to {args.output}/")


if __name__ == "__main__":
    main()
