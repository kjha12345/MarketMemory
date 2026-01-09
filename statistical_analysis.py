"""
Statistical Analysis for Experimental Results

Provides statistical significance testing and confidence intervals
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import pandas as pd


def t_test_comparison(
    rl_metrics: List[float],
    rarl_metrics: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Perform paired t-test comparing RL vs RARL metrics.
    
    Args:
        rl_metrics: List of metric values for RL agent
        rarl_metrics: List of metric values for RARL agent
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    rl_array = np.array(rl_metrics)
    rarl_array = np.array(rarl_metrics)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(rl_array, rarl_array)
    
    # Calculate effect size (Cohen's d)
    differences = rarl_array - rl_array
    cohens_d = np.mean(differences) / (np.std(differences) + 1e-8)
    
    # Calculate confidence interval for mean difference
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se = std_diff / np.sqrt(n)
    t_critical = stats.t.ppf(1 - alpha/2, n - 1)
    ci_lower = mean_diff - t_critical * se
    ci_upper = mean_diff + t_critical * se
    
    is_significant = p_value < alpha
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "is_significant": is_significant,
        "mean_difference": float(mean_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "cohens_d": float(cohens_d),
        "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
    }


def analyze_multiple_runs(
    all_rl_metrics: List[Dict],
    all_rarl_metrics: List[Dict]
) -> Dict:
    """
    Analyze results across multiple independent runs.
    
    Args:
        all_rl_metrics: List of metric dictionaries for RL (one per run)
        all_rarl_metrics: List of metric dictionaries for RARL (one per run)
    
    Returns:
        Statistical analysis results
    """
    # Extract key metrics
    rl_rewards = [m["cumulative_reward"] for m in all_rl_metrics]
    rarl_rewards = [m["cumulative_reward"] for m in all_rarl_metrics]
    
    rl_stability = [m["price_stability"] for m in all_rl_metrics]
    rarl_stability = [m["price_stability"] for m in all_rarl_metrics]
    
    rl_recovery = [m["avg_recovery_time"] if m["avg_recovery_time"] != np.inf else np.nan 
                   for m in all_rl_metrics]
    rarl_recovery = [m["avg_recovery_time"] if m["avg_recovery_time"] != np.inf else np.nan 
                     for m in all_rarl_metrics]
    
    # Remove NaN values for recovery time
    rl_recovery_clean = [x for x in rl_recovery if not np.isnan(x)]
    rarl_recovery_clean = [x for x in rarl_recovery if not np.isnan(x)]
    
    # Statistical tests
    reward_test = t_test_comparison(rl_rewards, rarl_rewards)
    stability_test = t_test_comparison(rl_stability, rarl_stability)
    
    recovery_test = None
    if len(rl_recovery_clean) > 0 and len(rarl_recovery_clean) > 0:
        recovery_test = t_test_comparison(rl_recovery_clean, rarl_recovery_clean)
    
    # Summary statistics
    summary = {
        "reward": {
            "rl_mean": np.mean(rl_rewards),
            "rl_std": np.std(rl_rewards),
            "rarl_mean": np.mean(rarl_rewards),
            "rarl_std": np.std(rarl_rewards),
            "test": reward_test
        },
        "stability": {
            "rl_mean": np.mean(rl_stability),
            "rl_std": np.std(rl_stability),
            "rarl_mean": np.mean(rarl_stability),
            "rarl_std": np.std(rarl_stability),
            "test": stability_test
        }
    }
    
    if recovery_test:
        summary["recovery"] = {
            "rl_mean": np.mean(rl_recovery_clean),
            "rl_std": np.std(rl_recovery_clean),
            "rarl_mean": np.mean(rarl_recovery_clean),
            "rarl_std": np.std(rarl_recovery_clean),
            "test": recovery_test
        }
    
    return summary


def print_statistical_summary(analysis_results: Dict):
    """Print formatted statistical analysis results"""
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    
    for metric_name, results in analysis_results.items():
        if metric_name == "recovery" and results is None:
            continue
            
        print(f"\n{metric_name.upper()}:")
        print(f"  Baseline RL: {results['rl_mean']:.4f} ± {results['rl_std']:.4f}")
        print(f"  RARL:        {results['rarl_mean']:.4f} ± {results['rarl_std']:.4f}")
        
        test = results["test"]
        print(f"\n  Statistical Test:")
        print(f"    t-statistic: {test['t_statistic']:.4f}")
        print(f"    p-value:     {test['p_value']:.6f}")
        print(f"    Significant: {'Yes' if test['is_significant'] else 'No'} (α=0.05)")
        print(f"    Mean diff:   {test['mean_difference']:.4f}")
        print(f"    95% CI:      [{test['ci_lower']:.4f}, {test['ci_upper']:.4f}]")
        print(f"    Effect size: {test['effect_size']} (Cohen's d = {test['cohens_d']:.4f})")
    
    print("\n" + "=" * 60)
