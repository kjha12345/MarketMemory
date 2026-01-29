"""Simple test to isolate issues"""

import numpy as np
print("Testing imports...")
from market_env import MarketEnvironment
print("✓ Market environment imported")

# Test just the environment
env = MarketEnvironment(episode_length=10, seed=42)
state, info = env.reset()
print(f"✓ Environment reset: state shape = {state.shape}")

for i in range(5):
    action = np.array([25.0])
    state, reward, terminated, truncated, info = env.step(action)
    print(f"  Step {i+1}: reward = {reward:.2f}")
    if terminated or truncated:
        break

print("✓ Basic environment test passed!")
