"""
Evaluation Metrics and Analysis

Metrics:
- Cumulative reward (over many episodes)
- Stability (price variance)
- Recovery time (time to stabilize after shocks)
- Additional metrics
"""

import numpy as np
from typing import Dict, List, Tuple
from market_env import MarketEnvironment, MarketRegime
from agents import BaselineRLAgent, RARLAgent


class Evaluator:
    """Evaluates agent performance on various metrics"""
    
    def __init__(self, env: MarketEnvironment):
        self.env = env
    
    def evaluate_episode(
        self,
        agent,
        use_retrieval: bool = True,
        render: bool = False
    ) -> Dict:
        """
        Evaluate agent on a single episode.
        
        Returns:
            Dictionary with episode metrics
        """
        state, info = self.env.reset()
        done = False
        
        prices = []
        profits = []
        regimes = [info.get("regime", "Normal")]
        retrieval_info_list = []
        
        while not done:
            # Get action
            if isinstance(agent, RARLAgent):
                action, retrieval_info = agent.predict(state, use_retrieval=use_retrieval)
                if retrieval_info:
                    retrieval_info_list.append(retrieval_info)
            else:
                action, _ = agent.predict(state)
                retrieval_info = None
            
            # Step environment
            state, reward, terminated, truncated, info = self.env.step(action)
            
            prices.append(action[0])
            profits.append(reward)
            regimes.append(info.get("regime", "Normal"))
            
            done = terminated or truncated
        
        # Calculate metrics
        total_profit = sum(profits)
        avg_price = np.mean(prices)
        price_std = np.std(prices)
        price_variance = np.var(prices)
        
        # Calculate recovery time after shocks
        recovery_times = self._calculate_recovery_times(profits, regimes)
        
        # Calculate profit stability (coefficient of variation)
        profit_cv = np.std(profits) / (np.abs(np.mean(profits)) + 1e-8)
        
        return {
            "total_profit": total_profit,
            "avg_profit": np.mean(profits),
            "avg_price": avg_price,
            "price_std": price_std,
            "price_variance": price_variance,
            "price_range": np.max(prices) - np.min(prices),
            "recovery_times": recovery_times,
            "avg_recovery_time": np.mean(recovery_times) if recovery_times else 0,
            "profit_stability": 1.0 / (1.0 + profit_cv),  # Higher is better
            "regimes": regimes,
            "prices": prices,
            "profits": profits,
            "retrieval_info": retrieval_info_list
        }
    
    def _calculate_recovery_times(
        self,
        profits: List[float],
        regimes: List[str]
    ) -> List[int]:
        """
        Calculate time to recover after market shocks.
        Recovery is defined as returning to baseline profit level.
        """
        recovery_times = []
        baseline_profit = np.mean(profits[:10]) if len(profits) > 10 else profits[0]
        threshold = 0.1 * abs(baseline_profit)  # 10% of baseline
        
        shock_start = None
        for i, regime in enumerate(regimes):
            # Detect shock start
            if regime != "Normal" and shock_start is None:
                shock_start = i
            
            # Detect recovery
            if shock_start is not None and regime == "Normal":
                # Check if profit has recovered
                if i > shock_start:
                    recent_profits = profits[shock_start:i]
                    if len(recent_profits) > 0:
                        avg_recent = np.mean(recent_profits)
                        if abs(avg_recent - baseline_profit) < threshold:
                            recovery_time = i - shock_start
                            recovery_times.append(recovery_time)
                            shock_start = None
        
        return recovery_times
    
    def evaluate_multiple_episodes(
        self,
        agent,
        num_episodes: int = 50,
        use_retrieval: bool = True
    ) -> Dict:
        """
        Evaluate agent over multiple episodes.
        
        Returns:
            Aggregated metrics across episodes
        """
        all_metrics = []
        
        for episode in range(num_episodes):
            metrics = self.evaluate_episode(agent, use_retrieval=use_retrieval)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        total_profits = [m["total_profit"] for m in all_metrics]
        avg_prices = [m["avg_price"] for m in all_metrics]
        price_stds = [m["price_std"] for m in all_metrics]
        recovery_times = []
        for m in all_metrics:
            recovery_times.extend(m["recovery_times"])
        profit_stabilities = [m["profit_stability"] for m in all_metrics]
        
        return {
            "num_episodes": num_episodes,
            "cumulative_reward": np.sum(total_profits),
            "avg_episode_reward": np.mean(total_profits),
            "std_episode_reward": np.std(total_profits),
            "avg_price": np.mean(avg_prices),
            "price_stability": 1.0 / (1.0 + np.mean(price_stds)),  # Higher is better
            "avg_price_std": np.mean(price_stds),
            "avg_recovery_time": np.mean(recovery_times) if recovery_times else np.inf,
            "recovery_time_std": np.std(recovery_times) if recovery_times else 0,
            "avg_profit_stability": np.mean(profit_stabilities),
            "episode_metrics": all_metrics
        }
    
    def compare_agents(
        self,
        rl_agent: BaselineRLAgent,
        rarl_agent: RARLAgent,
        num_episodes: int = 50
    ) -> Dict:
        """
        Compare baseline RL and RARL agents.
        
        Returns:
            Comparison metrics
        """
        print("Evaluating Baseline RL Agent...")
        rl_metrics = self.evaluate_multiple_episodes(
            rl_agent,
            num_episodes=num_episodes,
            use_retrieval=False
        )
        
        print("Evaluating RARL Agent...")
        rarl_metrics = self.evaluate_multiple_episodes(
            rarl_agent,
            num_episodes=num_episodes,
            use_retrieval=True
        )
        
        # Calculate improvements
        reward_improvement = (
            (rarl_metrics["cumulative_reward"] - rl_metrics["cumulative_reward"]) /
            abs(rl_metrics["cumulative_reward"]) * 100
        )
        
        stability_improvement = (
            (rarl_metrics["price_stability"] - rl_metrics["price_stability"]) /
            rl_metrics["price_stability"] * 100
        )
        
        recovery_improvement = 0
        if rl_metrics["avg_recovery_time"] != np.inf:
            recovery_improvement = (
                (rl_metrics["avg_recovery_time"] - rarl_metrics["avg_recovery_time"]) /
                rl_metrics["avg_recovery_time"] * 100
            )
        
        return {
            "rl_metrics": rl_metrics,
            "rarl_metrics": rarl_metrics,
            "reward_improvement_pct": reward_improvement,
            "stability_improvement_pct": stability_improvement,
            "recovery_improvement_pct": recovery_improvement,
            "summary": {
                "RARL improves cumulative reward by": f"{reward_improvement:.2f}%",
                "RARL improves price stability by": f"{stability_improvement:.2f}%",
                "RARL improves recovery time by": f"{recovery_improvement:.2f}%"
            }
        }


# Fix: import env properly
def create_evaluator(env_config: Dict) -> Evaluator:
    """Helper function to create evaluator"""
    from market_env import MarketEnvironment
    env = MarketEnvironment(**env_config)
    return Evaluator(env)
