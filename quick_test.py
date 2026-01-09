"""
Quick test script to verify the implementation works
"""

import numpy as np
from market_env import MarketEnvironment, MarketRegime
from memory_bank import MemoryBank
from agents import BaselineRLAgent, RARLAgent


def test_market_environment():
    """Test market environment"""
    print("Testing Market Environment...")
    env = MarketEnvironment(
        episode_length=20,
        shock_probability=0.2,
        seed=42
    )
    
    state, info = env.reset()
    print(f"Initial state: {state}")
    print(f"Initial regime: {info['regime']}")
    
    total_reward = 0
    for _ in range(10):
        action = np.array([25.0])  # Set price to 25
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step: price={action[0]:.2f}, reward={reward:.2f}, regime={info['regime']}")
        
        if terminated or truncated:
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print("✓ Market environment test passed\n")


def test_memory_bank():
    """Test memory bank"""
    print("Testing Memory Bank...")
    memory_bank = MemoryBank(embedding_dim=384)
    
    # Add some experiences
    for i in range(10):
        state = np.random.rand(5).astype(np.float32)
        action = float(20 + np.random.rand() * 10)
        reward = float(np.random.rand() * 50 - 25)
        regime = "Normal" if i < 5 else "Demand_Spike"
        
        memory_bank.add_experience(
            state=state,
            action=action,
            reward=reward,
            regime_tag=regime,
            episode_id=i // 5,
            time_step=i % 5
        )
    
    print(f"Added {len(memory_bank.experiences)} experiences")
    
    # Test retrieval
    query_state = np.random.rand(5).astype(np.float32)
    retrieved = memory_bank.retrieve(query_state, k=3)
    print(f"Retrieved {len(retrieved)} experiences")
    for i, exp in enumerate(retrieved):
        print(f"  {i+1}. Similarity: {exp['similarity']:.4f}, Action: {exp['action']:.2f}, Regime: {exp['regime_tag']}")
    
    stats = memory_bank.get_statistics()
    print(f"Statistics: {stats}")
    print("✓ Memory bank test passed\n")


def test_agents():
    """Test agents (quick initialization test)"""
    print("Testing Agents...")
    env = MarketEnvironment(episode_length=20, seed=42)
    memory_bank = MemoryBank()
    
    # Test RL agent initialization
    rl_agent = BaselineRLAgent(env=env, seed=42)
    print("✓ Baseline RL agent initialized")
    
    # Test RARL agent initialization
    rarl_agent = RARLAgent(env=env, memory_bank=memory_bank, seed=42)
    print("✓ RARL agent initialized")
    
    # Test prediction
    state = np.array([1.0, 5.0, 0.5, 100.0, 25.0], dtype=np.float32)
    rl_action, _ = rl_agent.predict(state)
    print(f"RL agent action: {rl_action[0]:.2f}")
    
    rarl_action, info = rarl_agent.predict(state, use_retrieval=False)
    print(f"RARL agent action (no retrieval): {rarl_action[0]:.2f}")
    
    print("✓ Agents test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("QUICK TEST SUITE")
    print("=" * 60)
    print()
    
    try:
        test_market_environment()
        test_memory_bank()
        test_agents()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
