"""
RL and RARL Agent Implementations

Baseline RL Agent: Standard deep RL (PPO)
RARL Agent: RL + retrieval from memory bank
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from market_env import MarketEnvironment
from memory_bank import MemoryBank


class BaselineRLAgent:
    """
    Baseline RL Agent using PPO.
    Learns from experience but doesn't have access to historical memory.
    """
    
    def __init__(
        self,
        env: MarketEnvironment,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        seed: Optional[int] = None
    ):
        self.env = env
        self.agent_name = "BaselineRL"
        
        # Create vectorized environment
        def make_env():
            return MarketEnvironment(
                episode_length=env.episode_length,
                base_demand=env.base_demand,
                base_cost=env.base_cost,
                max_price=env.max_price,
                min_price=env.min_price,
                supply_capacity=env.supply_capacity,
                fixed_costs=env.fixed_costs,
                price_elasticity=env.price_elasticity,
                shock_probability=env.shock_probability,
                shock_magnitude=env.shock_magnitude,
                seed=seed
            )
        
        vec_env = DummyVecEnv([make_env])
        
        # Initialize PPO agent
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=0,
            seed=seed,
            device="cpu"
        )
    
    def predict(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict action given observation"""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action, None
    
    def learn(self, total_timesteps: int, callback: Optional[BaseCallback] = None):
        """Train the agent"""
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def save(self, filepath: str):
        """Save agent"""
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load agent"""
        self.model = PPO.load(filepath, env=self.model.get_env())


class RARLAgent:
    """
    Retrieval-Augmented RL Agent.
    Combines PPO with retrieval from memory bank.
    """
    
    def __init__(
        self,
        env: MarketEnvironment,
        memory_bank: MemoryBank,
        retrieval_weight: float = 0.3,
        k_retrieval: int = 5,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        seed: Optional[int] = None
    ):
        self.env = env
        self.memory_bank = memory_bank
        self.retrieval_weight = retrieval_weight
        self.k_retrieval = k_retrieval
        self.agent_name = "RARL"
        
        # Create vectorized environment
        def make_env():
            return MarketEnvironment(
                episode_length=env.episode_length,
                base_demand=env.base_demand,
                base_cost=env.base_cost,
                max_price=env.max_price,
                min_price=env.min_price,
                supply_capacity=env.supply_capacity,
                fixed_costs=env.fixed_costs,
                price_elasticity=env.price_elasticity,
                shock_probability=env.shock_probability,
                shock_magnitude=env.shock_magnitude,
                seed=seed
            )
        
        vec_env = DummyVecEnv([make_env])
        
        # Initialize PPO agent (same as baseline)
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=0,
            seed=seed,
            device="cpu"
        )
    
    def _retrieve_similar_experiences(self, state: np.ndarray) -> List[Dict]:
        """Retrieve similar experiences from memory bank"""
        return self.memory_bank.retrieve(state, k=self.k_retrieval)
    
    def _compute_retrieved_action(self, retrieved: List[Dict]) -> float:
        """
        Compute action based on retrieved experiences.
        Weighted average of past actions, weighted by similarity and reward.
        """
        if len(retrieved) == 0:
            return None
        
        weights = []
        actions = []
        
        for exp in retrieved:
            # Weight by similarity and reward (higher reward = more weight)
            similarity = exp["similarity"]
            reward = exp["reward"]
            # Normalize reward to [0, 1] range for weighting
            # Use a more robust normalization that handles any reward range
            reward_min, reward_max = -100, 100  # Conservative bounds
            reward_norm = max(0, min(1, (reward - reward_min) / (reward_max - reward_min)))
            
            weight = similarity * (1 + reward_norm)
            weights.append(weight)
            actions.append(exp["action"])
        
        # Normalize weights and handle edge cases
        weights = np.array(weights)
        weights_sum = weights.sum()
        
        # If weights sum to zero or are all negative, use uniform weighting
        if weights_sum <= 1e-8 or np.all(weights <= 0):
            weights = np.ones_like(weights) / len(weights)
        else:
            # Ensure all weights are non-negative
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
        
        # Weighted average action
        retrieved_action = np.average(actions, weights=weights)
        return retrieved_action
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
        use_retrieval: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Predict action given observation.
        Combines RL policy with retrieved experiences.
        """
        # Get RL policy action
        rl_action, _ = self.model.predict(observation, deterministic=deterministic)
        
        if use_retrieval and len(self.memory_bank.experiences) > 0:
            # Retrieve similar experiences
            retrieved = self._retrieve_similar_experiences(observation)
            
            if len(retrieved) > 0:
                # Compute action from retrieved experiences
                retrieved_action = self._compute_retrieved_action(retrieved)
                
                if retrieved_action is not None:
                    # Combine RL action with retrieved action
                    combined_action = (
                        (1 - self.retrieval_weight) * rl_action[0] +
                        self.retrieval_weight * retrieved_action
                    )
                    # Clip to valid range
                    combined_action = np.clip(
                        combined_action,
                        self.env.min_price,
                        self.env.max_price
                    )
                    
                    info = {
                        "rl_action": rl_action[0],
                        "retrieved_action": retrieved_action,
                        "combined_action": combined_action,
                        "retrieved_count": len(retrieved),
                        "retrieved_regimes": [r["regime_tag"] for r in retrieved]
                    }
                    
                    return np.array([combined_action]), info
        
        return rl_action, None
    
    def learn(self, total_timesteps: int, callback: Optional[BaseCallback] = None):
        """Train the agent"""
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def save(self, filepath: str):
        """Save agent"""
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load agent"""
        self.model = PPO.load(filepath, env=self.model.get_env())
