# Retrieval-Augmented Reinforcement Learning for Economic Decision-Making

## Research Question
To what extent does incorporating retrieval-augmented reinforcement learning (RARL) improve economic decision-making efficiency—such as pricing, resource allocation, and market forecasting—compared to deep learning agents that rely solely on learned internal representations?

## Overview
This experiment compares two types of agents:
- **RL Agent**: Standard deep reinforcement learning agent (like a person who studied but threw away their notes)
- **RARL Agent**: Retrieval-augmented RL agent that can query historical experiences (like a person who kept all their notes in a searchable database)

The agent acts as a **shop owner or firm manager** making pricing decisions, while the market represents external conditions affecting the business.

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start

### Run a quick test
```bash
python quick_test.py
```

### Run the full experiment
```bash
python run_experiment.py
```

### Run with custom parameters
```bash
python run_experiment.py --phase1_timesteps 100000 --phase2_timesteps 200000 --eval_episodes 100 --num_runs 5
```

## Project Structure
- `market_env.py`: Market simulation environment with shocks and dynamics
- `memory_bank.py`: Vector database (FAISS) for storing and retrieving historical experiences
- `agents.py`: Baseline RL (PPO) and RARL agent implementations
- `trainer.py`: Training procedure with phased training (stable → mixed)
- `evaluator.py`: Evaluation metrics (reward, stability, recovery time)
- `statistical_analysis.py`: Statistical significance testing and confidence intervals
- `run_experiment.py`: Main experiment script
- `utils.py`: Visualization and utility functions
- `config.py`: Configuration parameters
- `quick_test.py`: Quick verification script

## Key Features

### Market Environment
- **State**: [demand_level, cost, time_normalized, supply_available, current_price]
- **Action**: Price to set (continuous)
- **Reward**: Profit = (price - cost) × sales - fixed_costs
- **Market Shocks**:
  - Demand spike (sudden increase in willingness to pay)
  - Demand crash (sudden decrease in willingness to pay)
  - Supply shock (production costs increase)
  - Policy change (pricing range restrictions)

### Memory Bank (RARL)
- Stores historical state-action-reward tuples with regime tags
- Uses FAISS for efficient similarity search
- Embeddings enable semantic similarity matching
- Retrieves k most similar past experiences for decision-making

### Training Procedure
1. **Phase 1**: Stable training (no market shocks)
   - Agents learn basic pricing strategies
   - RARL builds initial memory bank
2. **Phase 2**: Mixed training (with market shocks)
   - Agents adapt to changing conditions
   - RARL continues to expand memory bank
   - Tests robustness to disruptions

### Evaluation Metrics
- **Cumulative Reward**: Total profit over episodes
- **Price Stability**: Variance in pricing decisions (lower is better)
- **Recovery Time**: Time to stabilize after market shocks (lower is better)
- **Profit Stability**: Consistency of profits over time

### Statistical Analysis
- Paired t-tests for significance testing
- Confidence intervals for mean differences
- Effect size (Cohen's d) calculation
- Multiple independent runs for robustness

## Experimental Design

### RL Components
- **STATE**: Current market conditions (demand, cost, time, supply, price)
- **ACTION**: Price setting decision
- **REWARD**: Profit from sales

### Comparison
- **RL Agent**: Learns from experience but cannot access past memories
- **RARL Agent**: Learns from experience AND retrieves similar past situations

### Hypothesis
RARL should outperform baseline RL, especially:
- After market shocks (faster recovery)
- In novel but similar situations (better generalization)
- With more stable pricing decisions

## Results Output

The experiment generates:
- `results.csv`: Quantitative comparison metrics
- `training_curves.png`: Learning progress over time
- `price_comparison.png`: Price trajectory comparisons
- `metrics_comparison.png`: Side-by-side metric comparisons
- `improvement_summary.png`: Percentage improvements
- `statistical_analysis.json`: Statistical test results (if multiple runs)

## Configuration

Edit `config.py` to customize:
- Environment parameters (demand, costs, shocks)
- Agent hyperparameters (learning rate, batch size)
- RARL parameters (retrieval weight, k neighbors)
- Training and evaluation settings

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rarl_economic_decision_making,
  title = {Retrieval-Augmented Reinforcement Learning for Economic Decision-Making},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/MarketMemory}
}
```

## License
MIT License - see LICENSE file for details
