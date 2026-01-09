"""
Configuration file for experiment parameters
"""

# Environment Configuration
ENV_CONFIG = {
    "episode_length": 100,
    "base_demand": 1.0,
    "base_cost": 5.0,
    "max_price": 50.0,
    "min_price": 1.0,
    "supply_capacity": 100.0,
    "fixed_costs": 10.0,
    "price_elasticity": -2.0,
    "shock_probability": 0.1,  # 10% chance per timestep
    "shock_magnitude": 0.5,  # 50% magnitude
}

# RL Agent Configuration
RL_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
}

# RARL Agent Configuration
RARL_CONFIG = {
    "embedding_dim": 384,
    "use_gpu": False,
    "similarity_metric": "cosine",
    "retrieval_weight": 0.3,  # Weight of retrieved actions (0-1)
    "k_retrieval": 5,  # Number of similar experiences to retrieve
}

# Training Configuration
TRAINING_CONFIG = {
    "phase1_timesteps": 50000,  # Stable training
    "phase2_timesteps": 100000,  # Mixed training with shocks
    "seed": 42,
}

# Evaluation Configuration
EVAL_CONFIG = {
    "num_episodes": 50,
    "num_runs": 5,  # Number of independent runs for statistical significance
}
