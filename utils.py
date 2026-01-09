"""
Utility Functions for Visualization and Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd


def plot_training_curves(
    rl_rewards: List[float],
    rarl_rewards: List[float],
    save_path: str = "training_curves.png"
):
    """Plot training reward curves for both agents"""
    plt.figure(figsize=(12, 6))
    
    # Smooth curves using moving average
    window = max(1, len(rl_rewards) // 50)
    rl_smooth = pd.Series(rl_rewards).rolling(window=window, center=True).mean()
    rarl_smooth = pd.Series(rarl_rewards).rolling(window=window, center=True).mean()
    
    plt.plot(rl_smooth, label="Baseline RL", alpha=0.7, linewidth=2)
    plt.plot(rarl_smooth, label="RARL", alpha=0.7, linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Training Progress: Baseline RL vs RARL")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_price_comparison(
    rl_prices: List[List[float]],
    rarl_prices: List[List[float]],
    save_path: str = "price_comparison.png"
):
    """Plot price trajectories comparison"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot sample episodes
    num_samples = min(5, len(rl_prices))
    sample_indices = np.linspace(0, len(rl_prices) - 1, num_samples, dtype=int)
    
    for idx in sample_indices:
        axes[0].plot(rl_prices[idx], alpha=0.5, label=f"RL Episode {idx}" if idx == sample_indices[0] else "")
        axes[1].plot(rarl_prices[idx], alpha=0.5, label=f"RARL Episode {idx}" if idx == sample_indices[0] else "")
    
    axes[0].set_title("Baseline RL: Price Trajectories")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("RARL: Price Trajectories")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Price")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved price comparison to {save_path}")


def plot_metrics_comparison(
    comparison_results: Dict,
    save_path: str = "metrics_comparison.png"
):
    """Plot comparison of key metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    rl_metrics = comparison_results["rl_metrics"]
    rarl_metrics = comparison_results["rarl_metrics"]
    
    # Cumulative Reward
    axes[0, 0].bar(
        ["Baseline RL", "RARL"],
        [rl_metrics["cumulative_reward"], rarl_metrics["cumulative_reward"]],
        color=["#3498db", "#e74c3c"]
    )
    axes[0, 0].set_title("Cumulative Reward")
    axes[0, 0].set_ylabel("Total Profit")
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    
    # Price Stability
    axes[0, 1].bar(
        ["Baseline RL", "RARL"],
        [rl_metrics["price_stability"], rarl_metrics["price_stability"]],
        color=["#3498db", "#e74c3c"]
    )
    axes[0, 1].set_title("Price Stability (Higher is Better)")
    axes[0, 1].set_ylabel("Stability Score")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    
    # Recovery Time
    rl_recovery = rl_metrics["avg_recovery_time"] if rl_metrics["avg_recovery_time"] != np.inf else 0
    rarl_recovery = rarl_metrics["avg_recovery_time"] if rarl_metrics["avg_recovery_time"] != np.inf else 0
    
    axes[1, 0].bar(
        ["Baseline RL", "RARL"],
        [rl_recovery, rarl_recovery],
        color=["#3498db", "#e74c3c"]
    )
    axes[1, 0].set_title("Average Recovery Time (Lower is Better)")
    axes[1, 0].set_ylabel("Time Steps")
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    
    # Profit Stability
    axes[1, 1].bar(
        ["Baseline RL", "RARL"],
        [rl_metrics["avg_profit_stability"], rarl_metrics["avg_profit_stability"]],
        color=["#3498db", "#e74c3c"]
    )
    axes[1, 1].set_title("Profit Stability (Higher is Better)")
    axes[1, 1].set_ylabel("Stability Score")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved metrics comparison to {save_path}")


def plot_improvement_summary(
    comparison_results: Dict,
    save_path: str = "improvement_summary.png"
):
    """Plot percentage improvements"""
    improvements = {
        "Reward": comparison_results["reward_improvement_pct"],
        "Price Stability": comparison_results["stability_improvement_pct"],
        "Recovery Time": comparison_results["recovery_improvement_pct"]
    }
    
    colors = ["green" if v > 0 else "red" for v in improvements.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(improvements.keys(), improvements.values(), color=colors, alpha=0.7)
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.title("RARL Improvement over Baseline RL (%)")
    plt.ylabel("Improvement (%)")
    plt.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar, value in zip(bars, improvements.values()):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + (1 if height > 0 else -3),
            f"{value:.2f}%",
            ha="center",
            va="bottom" if height > 0 else "top"
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved improvement summary to {save_path}")


def print_comparison_summary(comparison_results: Dict):
    """Print formatted comparison summary"""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS: Baseline RL vs RARL")
    print("=" * 60)
    
    rl_metrics = comparison_results["rl_metrics"]
    rarl_metrics = comparison_results["rarl_metrics"]
    
    print("\nCUMULATIVE REWARD:")
    print(f"  Baseline RL: {rl_metrics['cumulative_reward']:.2f}")
    print(f"  RARL:        {rarl_metrics['cumulative_reward']:.2f}")
    print(f"  Improvement: {comparison_results['reward_improvement_pct']:.2f}%")
    
    print("\nPRICE STABILITY:")
    print(f"  Baseline RL: {rl_metrics['price_stability']:.4f}")
    print(f"  RARL:        {rarl_metrics['price_stability']:.4f}")
    print(f"  Improvement: {comparison_results['stability_improvement_pct']:.2f}%")
    
    print("\nRECOVERY TIME (after shocks):")
    rl_recovery = rl_metrics["avg_recovery_time"]
    rarl_recovery = rarl_metrics["avg_recovery_time"]
    if rl_recovery != np.inf:
        print(f"  Baseline RL: {rl_recovery:.2f} time steps")
        print(f"  RARL:        {rarl_recovery:.2f} time steps")
        print(f"  Improvement: {comparison_results['recovery_improvement_pct']:.2f}%")
    else:
        print(f"  Baseline RL: No recovery detected")
        print(f"  RARL:        {rarl_recovery:.2f} time steps")
    
    print("\nPROFIT STABILITY:")
    print(f"  Baseline RL: {rl_metrics['avg_profit_stability']:.4f}")
    print(f"  RARL:        {rarl_metrics['avg_profit_stability']:.4f}")
    
    print("\n" + "=" * 60)


def save_results_to_csv(comparison_results: Dict, save_path: str = "results.csv"):
    """Save results to CSV file"""
    rl_metrics = comparison_results["rl_metrics"]
    rarl_metrics = comparison_results["rarl_metrics"]
    
    data = {
        "Metric": [
            "Cumulative Reward",
            "Avg Episode Reward",
            "Price Stability",
            "Avg Recovery Time",
            "Profit Stability"
        ],
        "Baseline RL": [
            rl_metrics["cumulative_reward"],
            rl_metrics["avg_episode_reward"],
            rl_metrics["price_stability"],
            rl_metrics["avg_recovery_time"] if rl_metrics["avg_recovery_time"] != np.inf else None,
            rl_metrics["avg_profit_stability"]
        ],
        "RARL": [
            rarl_metrics["cumulative_reward"],
            rarl_metrics["avg_episode_reward"],
            rarl_metrics["price_stability"],
            rarl_metrics["avg_recovery_time"] if rarl_metrics["avg_recovery_time"] != np.inf else None,
            rarl_metrics["avg_profit_stability"]
        ],
        "Improvement (%)": [
            comparison_results["reward_improvement_pct"],
            None,
            comparison_results["stability_improvement_pct"],
            comparison_results["recovery_improvement_pct"],
            None
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")
