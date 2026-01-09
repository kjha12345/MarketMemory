# Experiment Design Document

## Research Question
To what extent does incorporating retrieval-augmented reinforcement learning (RARL) improve economic decision-making efficiency—such as pricing, resource allocation, and market forecasting—compared to deep learning agents that rely solely on learned internal representations?

## Experimental Setup

### Agent Perspective
The agent represents a **shop owner or firm manager** making pricing decisions. The market represents external conditions affecting the business.

### Market Environment

#### State Space
- `demand_level`: Current market demand (0-10)
- `cost`: Production cost per unit (0-20)
- `time_normalized`: Progress through episode (0-1)
- `supply_available`: Available inventory/capacity (0-100)
- `current_price`: Last set price (1-50)

#### Action Space
- Continuous price setting: `[min_price, max_price]` (default: [1, 50])

#### Reward Function
```
Profit = (price - cost) × min(demand(price), supply) - fixed_costs
```

Where `demand(price)` follows price elasticity:
```
demand = base_demand × (price/avg_price)^elasticity
```

#### Market Dynamics
- **Baseline**: Stable, predictable demand-price relationship
- **Supply Constraint**: Limited capacity per time step
- **Time Limit**: Episodes run for fixed number of steps

#### Market Shocks
1. **Demand Spike**: Sudden increase in willingness to pay (50-100% increase)
2. **Demand Crash**: Sudden decrease in willingness to pay (50% decrease)
3. **Supply Shock**: Production costs increase, capacity decreases
4. **Policy Change**: Pricing range restrictions imposed

### Memory Bank (RARL)

#### Storage Format
Each experience stored as:
- `state`: Market state vector
- `action`: Price chosen
- `reward`: Profit received
- `regime_tag`: Market condition type (Normal, Demand_Spike, etc.)
- `episode_id`: Episode identifier
- `time_step`: Time step within episode

#### Retrieval Mechanism
1. Embed current state using learned/normalized representation
2. Query FAISS vector database for k most similar states
3. Retrieve associated actions, rewards, and regime tags
4. Weight retrieved actions by similarity and reward quality
5. Combine with RL policy action

### Agents

#### Baseline RL Agent
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Architecture**: Multi-layer perceptron policy
- **Learning**: Updates neural network weights from experience
- **Memory**: No access to historical experiences

#### RARL Agent
- **Algorithm**: PPO + Retrieval-Augmented Generation
- **Architecture**: Same as baseline RL
- **Learning**: Updates neural network weights + queries memory bank
- **Memory**: Vector database of historical experiences
- **Action Combination**: Weighted average of RL action and retrieved actions

### Training Procedure

#### Phase 1: Stable Training
- **Duration**: 50,000 timesteps (default)
- **Conditions**: No market shocks (shock_probability = 0)
- **Purpose**: Learn basic pricing strategies
- **RARL**: Builds initial memory bank from stable episodes

#### Phase 2: Mixed Training
- **Duration**: 100,000 timesteps (default)
- **Conditions**: Market shocks enabled (shock_probability = 0.1)
- **Purpose**: Adapt to changing conditions, test robustness
- **RARL**: Continues expanding memory bank with diverse experiences

### Evaluation Metrics

#### Primary Metrics
1. **Cumulative Reward**: Sum of profits across all evaluation episodes
2. **Price Stability**: Inverse of price variance (higher = more stable)
3. **Recovery Time**: Average time steps to stabilize after market shocks

#### Secondary Metrics
- Average episode reward
- Price range (max - min)
- Profit stability (coefficient of variation)
- Regime-specific performance

### Statistical Analysis

#### Multiple Runs
- Run experiment N times with different random seeds
- Collect metrics for each run
- Perform statistical tests on differences

#### Tests
- **Paired t-test**: Compare RL vs RARL metrics
- **Confidence Intervals**: 95% CI for mean differences
- **Effect Size**: Cohen's d for practical significance

### Expected Outcomes

#### Hypotheses
1. **H1**: RARL achieves higher cumulative reward than baseline RL
2. **H2**: RARL shows better price stability (lower variance)
3. **H3**: RARL recovers faster from market shocks

#### Mechanisms
- **Generalization**: RARL can leverage similar past situations
- **Adaptation**: Faster response to novel but similar conditions
- **Robustness**: Better handling of regime changes

### Implementation Details

#### Technologies
- **RL Framework**: Stable-Baselines3 (PPO)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Environment**: Gymnasium (OpenAI Gym successor)
- **Visualization**: Matplotlib, Seaborn

#### Hyperparameters
- Learning rate: 3e-4
- Batch size: 64
- PPO epochs: 10
- Gamma (discount): 0.99
- Retrieval weight: 0.3 (30% retrieved, 70% RL)
- k neighbors: 5

### Validation

#### Internal Validity
- Controlled environment (same market dynamics)
- Same hyperparameters for both agents
- Same random seeds for reproducibility

#### External Validity
- Realistic market dynamics (price elasticity, shocks)
- Diverse market conditions (multiple shock types)
- Long-term evaluation (multiple episodes)

### Limitations

1. **Simplified Market**: Single product, no competition
2. **Deterministic Shocks**: Shocks follow predefined patterns
3. **Embedding Quality**: Simple state normalization (could use learned encoder)
4. **Retrieval Strategy**: Fixed weighting scheme (could be learned)

### Future Extensions

1. **Multi-product Markets**: Multiple products with interactions
2. **Competitive Environments**: Multiple agents competing
3. **Learned Embeddings**: Train encoder for better similarity
4. **Adaptive Retrieval**: Learn when/how much to use retrieval
5. **Temporal Patterns**: Consider sequence of past experiences
6. **Uncertainty Quantification**: Confidence intervals for actions
