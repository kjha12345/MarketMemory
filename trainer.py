"""
Training Procedure with Phased Training

Phase 1: Stable training (normal market conditions)
Phase 2: Mixed environments with market shocks
"""

import numpy as np
from typing import Dict, List, Optional
from stable_baselines3.common.callbacks import BaseCallback
from market_env import MarketEnvironment
from agents import BaselineRLAgent, RARLAgent
from memory_bank import MemoryBank


class EpisodeCallback(BaseCallback):
    """Callback to collect episode data during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        # Get reward from info if available
        if "episode" in self.locals.get("infos", [{}])[0]:
            episode_info = self.locals["infos"][0]["episode"]
            self.episode_rewards.append(episode_info["r"])
            self.episode_lengths.append(episode_info["l"])
        return True


class Trainer:
    """Manages training procedure for both agents"""
    
    def __init__(
        self,
        env_config: Dict,
        rl_config: Dict,
        rarl_config: Dict,
        seed: Optional[int] = None
    ):
        """
        Initialize trainer.
        
        Args:
            env_config: Configuration for market environment
            rl_config: Configuration for baseline RL agent
            rarl_config: Configuration for RARL agent (includes memory_bank config)
            seed: Random seed
        """
        self.env_config = env_config
        self.rl_config = rl_config
        self.rarl_config = rarl_config
        self.seed = seed
        
        # Create environments
        self.env_rl = MarketEnvironment(**env_config, seed=seed)
        self.env_rarl = MarketEnvironment(**env_config, seed=seed)
        
        # Create memory bank for RARL
        memory_bank = MemoryBank(
            embedding_dim=rarl_config.get("embedding_dim", 384),
            use_gpu=rarl_config.get("use_gpu", False),
            similarity_metric=rarl_config.get("similarity_metric", "cosine")
        )
        
        # Create agents
        self.rl_agent = BaselineRLAgent(
            env=self.env_rl,
            **rl_config,
            seed=seed
        )
        
        self.rarl_agent = RARLAgent(
            env=self.env_rarl,
            memory_bank=memory_bank,
            retrieval_weight=rarl_config.get("retrieval_weight", 0.3),
            k_retrieval=rarl_config.get("k_retrieval", 5),
            **rl_config,
            seed=seed
        )
        
        self.memory_bank = memory_bank
    
    def train_phase1_stable(self, timesteps: int) -> Dict:
        """
        Phase 1: Train in stable market conditions (no shocks).
        
        Args:
            timesteps: Number of training timesteps
        
        Returns:
            Training statistics
        """
        print("=" * 60)
        print("PHASE 1: Stable Training (No Market Shocks)")
        print("=" * 60)
        
        # Create stable environments (no shocks)
        stable_config = self.env_config.copy()
        stable_config["shock_probability"] = 0.0
        
        env_rl_stable = MarketEnvironment(**stable_config, seed=self.seed)
        env_rarl_stable = MarketEnvironment(**stable_config, seed=self.seed)
        
        # Update agent environments
        self.rl_agent.env = env_rl_stable
        self.rarl_agent.env = env_rarl_stable
        
        # Train baseline RL
        print("\nTraining Baseline RL Agent...")
        callback_rl = EpisodeCallback()
        self.rl_agent.learn(timesteps // 2, callback=callback_rl)
        
        # Train RARL and collect experiences for memory bank
        print("\nTraining RARL Agent...")
        callback_rarl = EpisodeCallback()
        self.rarl_agent.learn(timesteps // 2, callback=callback_rarl)
        
        # Collect experiences from RARL training for memory bank
        self._collect_training_experiences(env_rarl_stable, num_episodes=50)
        
        return {
            "rl_episode_rewards": callback_rl.episode_rewards,
            "rarl_episode_rewards": callback_rarl.episode_rewards,
            "memory_bank_size": len(self.memory_bank.experiences)
        }
    
    def train_phase2_mixed(self, timesteps: int) -> Dict:
        """
        Phase 2: Train in mixed environments with market shocks.
        Continue adding to memory bank as episodes accumulate.
        
        Args:
            timesteps: Number of training timesteps
        
        Returns:
            Training statistics
        """
        print("=" * 60)
        print("PHASE 2: Mixed Training (With Market Shocks)")
        print("=" * 60)
        
        # Use original environment config with shocks
        self.rl_agent.env = self.env_rl
        self.rarl_agent.env = self.env_rarl
        
        # Train both agents
        print("\nTraining Baseline RL Agent...")
        callback_rl = EpisodeCallback()
        self.rl_agent.learn(timesteps // 2, callback=callback_rl)
        
        print("\nTraining RARL Agent...")
        callback_rarl = EpisodeCallback()
        self.rarl_agent.learn(timesteps // 2, callback=callback_rarl)
        
        # Collect experiences from RARL training (with shocks)
        self._collect_training_experiences(self.env_rarl, num_episodes=100)
        
        return {
            "rl_episode_rewards": callback_rl.episode_rewards,
            "rarl_episode_rewards": callback_rarl.episode_rewards,
            "memory_bank_size": len(self.memory_bank.experiences)
        }
    
    def _collect_training_experiences(
        self,
        env: MarketEnvironment,
        num_episodes: int
    ):
        """
        Collect experiences from running episodes and add to memory bank.
        This simulates the agent learning from its experiences.
        """
        print(f"\nCollecting {num_episodes} episodes for memory bank...")
        
        episode_id = len(self.memory_bank.experiences) // env.episode_length
        
        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []
            regime_tags = []
            
            state, info = env.reset()
            done = False
            
            while not done:
                # Use RARL agent to predict action
                action, _ = self.rarl_agent.predict(state, use_retrieval=True)
                
                states.append(state)
                actions.append(action[0])
                regime_tags.append(info.get("regime", "Normal"))
                
                state, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                
                done = terminated or truncated
            
            # Add episode to memory bank
            self.memory_bank.add_episode(
                states=states,
                actions=actions,
                rewards=rewards,
                regime_tags=regime_tags,
                episode_id=episode_id + episode
            )
        
        print(f"Memory bank now contains {len(self.memory_bank.experiences)} experiences")
    
    def get_agents(self):
        """Get trained agents"""
        return self.rl_agent, self.rarl_agent
    
    def get_memory_bank(self):
        """Get memory bank"""
        return self.memory_bank
