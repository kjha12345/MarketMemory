"""
Market Environment for Economic Decision-Making Simulation

The agent (shop owner/firm manager) sets prices and receives rewards based on profit.
Market conditions can change with various shocks.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
from enum import Enum


class MarketRegime(Enum):
    """Types of market conditions"""
    NORMAL = "Normal"
    DEMAND_SPIKE = "Demand_Spike"
    DEMAND_CRASH = "Demand_Crash"
    SUPPLY_SHOCK = "Supply_Shock"
    POLICY_CHANGE = "Policy_Change"


class MarketEnvironment(gym.Env):
    """
    Market environment where an agent sets prices and receives profit-based rewards.
    
    State: [demand_level, cost, time_step_normalized, supply_available, current_price]
    Action: Price to set (continuous)
    Reward: Profit = (price - cost) * min(demand, supply) - fixed_costs
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        episode_length: int = 100,
        base_demand: float = 1.0,
        base_cost: float = 5.0,
        max_price: float = 50.0,
        min_price: float = 1.0,
        supply_capacity: float = 100.0,
        fixed_costs: float = 10.0,
        price_elasticity: float = -2.0,  # Negative: higher price = lower demand
        shock_probability: float = 0.1,
        shock_magnitude: float = 0.5,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.episode_length = episode_length
        self.base_demand = base_demand
        self.base_cost = base_cost
        self.max_price = max_price
        self.min_price = min_price
        self.supply_capacity = supply_capacity
        self.fixed_costs = fixed_costs
        self.price_elasticity = price_elasticity
        self.shock_probability = shock_probability
        self.shock_magnitude = shock_magnitude
        
        # State space: [demand_level, cost, time_normalized, supply_available, current_price]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, min_price]),
            high=np.array([10.0, 20.0, 1.0, supply_capacity, max_price]),
            dtype=np.float32
        )
        
        # Action space: price to set
        self.action_space = spaces.Box(
            low=np.array([min_price]),
            high=np.array([max_price]),
            dtype=np.float32
        )
        
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.time_step = 0
        self.current_regime = MarketRegime.NORMAL
        
        # Initialize state variables
        self.demand_level = self.base_demand
        self.cost = self.base_cost
        self.supply_available = self.supply_capacity
        self.current_price = (self.min_price + self.max_price) / 2
        
        # Track regime history for analysis
        self.regime_history = [self.current_regime]
        self.price_history = [self.current_price]
        self.profit_history = []
        
        # Shock timing (when shocks occur)
        self.shock_times = []
        self._generate_shock_schedule()
        
        state = self._get_state()
        info = {"regime": self.current_regime.value}
        return state, info
    
    def _generate_shock_schedule(self):
        """Generate random schedule of market shocks"""
        self.shock_times = []
        for t in range(1, self.episode_length):
            if self.rng.random() < self.shock_probability:
                shock_type = self.rng.choice([
                    MarketRegime.DEMAND_SPIKE,
                    MarketRegime.DEMAND_CRASH,
                    MarketRegime.SUPPLY_SHOCK,
                    MarketRegime.POLICY_CHANGE
                ])
                self.shock_times.append((t, shock_type))
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        time_normalized = self.time_step / self.episode_length
        
        # Clip state values to observation space bounds to prevent overflow
        demand_clipped = np.clip(self.demand_level, 0.0, 10.0)
        cost_clipped = np.clip(self.cost, 0.0, 20.0)
        time_clipped = np.clip(time_normalized, 0.0, 1.0)
        supply_clipped = np.clip(self.supply_available, 0.0, self.supply_capacity)
        price_clipped = np.clip(self.current_price, self.min_price, self.max_price)
        
        state = np.array([
            demand_clipped,
            cost_clipped,
            time_clipped,
            supply_clipped,
            price_clipped
        ], dtype=np.float32)
        
        # Additional check for NaN or Inf values
        if np.any(~np.isfinite(state)):
            # Replace NaN/Inf with safe defaults
            state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=0.0)
            # Re-clip after NaN replacement
            state[0] = np.clip(state[0], 0.0, 10.0)
            state[1] = np.clip(state[1], 0.0, 20.0)
            state[2] = np.clip(state[2], 0.0, 1.0)
            state[3] = np.clip(state[3], 0.0, self.supply_capacity)
            state[4] = np.clip(state[4], self.min_price, self.max_price)
        
        return state
    
    def _apply_market_shock(self, shock_type: MarketRegime):
        """Apply a market shock based on type"""
        self.current_regime = shock_type
        
        if shock_type == MarketRegime.DEMAND_SPIKE:
            # Sudden increase in willingness to pay
            self.demand_level = self.base_demand * (1 + self.shock_magnitude * 2)
            # Clip to observation space bounds
            self.demand_level = np.clip(self.demand_level, 0.0, 10.0)
        
        elif shock_type == MarketRegime.DEMAND_CRASH:
            # Sudden decrease in willingness to pay
            self.demand_level = self.base_demand * (1 - self.shock_magnitude)
            # Clip to observation space bounds
            self.demand_level = np.clip(self.demand_level, 0.0, 10.0)
        
        elif shock_type == MarketRegime.SUPPLY_SHOCK:
            # Production costs increase
            self.cost = self.base_cost * (1 + self.shock_magnitude)
            self.supply_available = self.supply_capacity * (1 - self.shock_magnitude * 0.5)
            # Clip to observation space bounds
            self.cost = np.clip(self.cost, 0.0, 20.0)
            self.supply_available = np.clip(self.supply_available, 0.0, self.supply_capacity)
        
        elif shock_type == MarketRegime.POLICY_CHANGE:
            # Policy restricts pricing range
            new_max_price = min(self.max_price * 0.8, self.max_price - 5)
            new_min_price = max(self.min_price * 1.2, self.min_price + 2)
            # Ensure min < max and both are within reasonable bounds
            if new_min_price < new_max_price:
                self.max_price = np.clip(new_max_price, 1.0, 50.0)
                self.min_price = np.clip(new_min_price, 1.0, self.max_price)
            # Also clip current price to new range
            self.current_price = np.clip(self.current_price, self.min_price, self.max_price)
    
    def _recover_from_shock(self):
        """Gradually recover from shock towards normal conditions"""
        recovery_rate = 0.05  # 5% recovery per step
        
        if self.current_regime != MarketRegime.NORMAL:
            # Gradually return to normal
            self.demand_level = self.demand_level * (1 - recovery_rate) + self.base_demand * recovery_rate
            self.cost = self.cost * (1 - recovery_rate) + self.base_cost * recovery_rate
            self.supply_available = self.supply_available * (1 - recovery_rate) + self.supply_capacity * recovery_rate
            
            # Clip to observation space bounds to prevent overflow
            self.demand_level = np.clip(self.demand_level, 0.0, 10.0)
            self.cost = np.clip(self.cost, 0.0, 20.0)
            self.supply_available = np.clip(self.supply_available, 0.0, self.supply_capacity)
            
            # Check if recovered
            if (abs(self.demand_level - self.base_demand) < 0.1 and
                abs(self.cost - self.base_cost) < 0.1):
                self.current_regime = MarketRegime.NORMAL
    
    def _calculate_demand(self, price: float) -> float:
        """
        Calculate demand based on price using price elasticity.
        Higher price = lower demand (negative elasticity)
        """
        # Ensure price is positive and within bounds
        price = max(0.01, min(price, self.max_price))
        avg_price = (self.min_price + self.max_price) / 2
        avg_price = max(0.01, avg_price)  # Prevent division by zero
        
        # Base demand adjusted by price elasticity
        price_ratio = price / avg_price
        # Clip price_ratio to prevent extreme values
        price_ratio = np.clip(price_ratio, 0.01, 100.0)
        
        # Calculate demand with elasticity
        try:
            demand = self.demand_level * (price_ratio ** self.price_elasticity)
        except (OverflowError, ValueError):
            # Fallback for extreme values
            if price_ratio > 1:
                demand = self.demand_level / (price_ratio ** abs(self.price_elasticity))
            else:
                demand = self.demand_level * (price_ratio ** abs(self.price_elasticity))
        
        # Add some noise to make it more realistic
        noise = self.rng.normal(1.0, 0.1)
        noise = np.clip(noise, 0.1, 2.0)  # Clip noise to prevent extreme values
        demand = max(0, demand * noise)
        
        # Ensure demand is finite
        if not np.isfinite(demand):
            demand = 0.0
        
        return demand
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Price to set [price]
        
        Returns:
            observation: New state
            reward: Profit from this step
            terminated: Episode ended
            truncated: Episode truncated
            info: Additional information
        """
        # Clip action to valid price range
        price = np.clip(action[0], self.min_price, self.max_price)
        self.current_price = price
        
        # Check for market shocks
        for shock_time, shock_type in self.shock_times:
            if self.time_step == shock_time:
                self._apply_market_shock(shock_type)
        
        # Recover from previous shocks
        if self.time_step > 0:
            self._recover_from_shock()
        
        # Calculate demand at this price
        demand = self._calculate_demand(price)
        
        # Sales are limited by supply
        sales = min(demand, self.supply_available)
        sales = max(0, sales)  # Ensure non-negative
        
        # Calculate profit with safeguards
        revenue = price * sales
        variable_costs = self.cost * sales
        profit = revenue - variable_costs - self.fixed_costs
        
        # Ensure profit is finite
        if not np.isfinite(profit):
            profit = -self.fixed_costs  # Fallback to worst case
        
        # Clip profit to reasonable range to prevent extreme values
        profit = np.clip(profit, -1000.0, 10000.0)
        
        # Update supply (replenish each step)
        self.supply_available = self.supply_capacity
        
        # Update state
        self.time_step += 1
        state = self._get_state()
        
        # Track history
        self.regime_history.append(self.current_regime)
        self.price_history.append(price)
        self.profit_history.append(profit)
        
        # Check if episode is done
        terminated = self.time_step >= self.episode_length
        truncated = False
        
        info = {
            "regime": self.current_regime.value,
            "demand": demand,
            "sales": sales,
            "profit": profit,
            "price": price
        }
        
        return state, profit, terminated, truncated, info
    
    def get_episode_summary(self) -> Dict:
        """Get summary of the completed episode"""
        return {
            "regime_history": [r.value for r in self.regime_history],
            "price_history": self.price_history,
            "profit_history": self.profit_history,
            "total_profit": sum(self.profit_history),
            "avg_price": np.mean(self.price_history),
            "price_std": np.std(self.price_history),
            "shock_times": self.shock_times
        }
