#!/usr/bin/env python3
"""
rarl_econ_singlefile.py

One-file implementation of:
- simulated market environment with shocks
- PPO baseline agent
- Retrieval-Augmented Reinforcement Learning (RARL) agent
- simple vector store (FAISS if available, fallback to sklearn)
- training & evaluation loops
- metrics and plotting

Usage:
    python rarl_econ_singlefile.py

Main configurable parameters are at the top under CONFIG. The script runs two experiments:
1) baseline PPO
2) RARL (PPO core + retrieval)

Notes:
- This is research/experimental code, intended as a starting point.
- For production experiments, split into modules, add better logging,
  hyperparameter sweeps, distributed training, and robust saving.
- If you have `faiss` and `sentence_transformers` installed, the retrieval will
  use them; otherwise uses an in-memory embedding (MLP) + sklearn NearestNeighbors.

Dependencies (install if missing):
    pip install numpy pandas matplotlib torch gymnasium scikit-learn tqdm

Optional (recommended):
    pip install faiss-cpu sentence-transformers

"""

import os
import math
import time
import random
import pickle
from collections import deque, namedtuple, defaultdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from tqdm import trange

# Attempt optional libraries
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---------------------------
# CONFIG
# ---------------------------
CONFIG = {
    # Environment
    "episode_length": 200,
    "initial_price": 10.0,
    "initial_demand": 100.0,
    "cost": 2.0,
    "capacity": 200.0,

    # Shock params
    "shock_schedule": [50, 120],   # time steps where shocks begin
    "shock_types": ["demand_drop", "inflation_jump"],  # match schedule length
    "demand_drop": {"magnitude": 0.5, "duration": 30},  # scale demand by magnitude
    "demand_surge": {"magnitude": 1.5, "duration": 30},
    "inflation_jump": {"magnitude": 1.2, "duration": 20},  # increases cost multiplicatively
    "competitor_entry": {"magnitude": 0.8, "duration": 40},  # decreases effective price

    # RL
    "seed": 1,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ppo_clip": 0.2,
    "ppo_epochs": 4,
    "ppo_minibatch_size": 64,
    "lr": 3e-4,
    "entropy_coef": 1e-3,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,

    # training
    "train_episodes": 400,
    "eval_episodes": 40,
    "rollout_timesteps": 2048,  # total steps per update (approx)
    "batch_size": 64,

    # retrieval
    "use_faiss": _HAS_FAISS,
    "embed_dim": 128,
    "retrieval_k": 5,
    "retrieval_threshold_entropy": 0.9,  # above this entropy, trigger retrieval
    "vector_store_max_size": 5000,

    # misc
    "save_dir": "./rarl_results",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Market Environment
# ---------------------------
class MarketEnv(gym.Env):
    """
    Very simple price-setting market environment.

    State (observation):
        - recent_price (scalar)
        - recent_demand (scalar)
        - capacity_remaining (scalar)
        - time_to_next_shock (scalar)
        - macro_inflation (scalar)
        - last_k_price_history (optional; included in observation vector length)

    Action:
        - continuous scalar: price change delta in [-max_delta, +max_delta]
          resulting price = max(0.1, price + delta)

    Reward:
        profit for the step = (price - cost) * quantity_sold
        with quantity determined by demand curve: demand = base_demand * f(price)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.max_price_delta = 2.0
        self.price = config["initial_price"]
        self.base_demand = config["initial_demand"]
        self.cost = config["cost"]
        self.capacity = config["capacity"]
        self.time = 0
        self.episode_length = config["episode_length"]
        self.price_history_len = 5

        # shock handling
        self.shock_schedule = list(config.get("shock_schedule", []))
        self.shock_types = list(config.get("shock_types", []))
        self.active_shocks = []  # list of dicts with type, remaining duration, magnitude

        # observation space: price, demand, capacity_util, time_frac, inflation, last_k_prices
        obs_dim = 2 + 1 + 1 + 1 + self.price_history_len
        high = np.array([1e4] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # action: continuous delta in [-max, +max]
        self.action_space = spaces.Box(low=np.array([-self.max_price_delta], dtype=np.float32),
                                       high=np.array([self.max_price_delta], dtype=np.float32),
                                       dtype=np.float32)

        self.price_history = deque([self.price] * self.price_history_len, maxlen=self.price_history_len)
        self.inflation = 1.0
        self.reset()

    def _apply_shocks_now(self):
        """Check schedule and activate new shocks."""
        for idx, t in enumerate(self.shock_schedule):
            if self.time == t:
                stype = self.shock_types[idx] if idx < len(self.shock_types) else None
                if stype == "demand_drop":
                    info = dict(type="demand", effect="drop", magnitude=self.cfg["demand_drop"]["magnitude"],
                                remaining=self.cfg["demand_drop"]["duration"])
                elif stype == "demand_surge":
                    info = dict(type="demand", effect="surge", magnitude=self.cfg["demand_surge"]["magnitude"],
                                remaining=self.cfg["demand_surge"]["duration"])
                elif stype == "inflation_jump":
                    info = dict(type="inflation", magnitude=self.cfg["inflation_jump"]["magnitude"],
                                remaining=self.cfg["inflation_jump"]["duration"])
                elif stype == "competitor_entry":
                    info = dict(type="competitor", magnitude=self.cfg["competitor_entry"]["magnitude"],
                                remaining=self.cfg["competitor_entry"]["duration"])
                else:
                    info = None
                if info:
                    self.active_shocks.append(info)

    def _decay_shocks(self):
        new_shocks = []
        for s in self.active_shocks:
            s["remaining"] -= 1
            if s["remaining"] > 0:
                new_shocks.append(s)
            else:
                pass  # end of shock
        self.active_shocks = new_shocks

    def _apply_shock_effects(self):
        """Return multiplicative factors for demand and cost/price"""
        demand_mult = 1.0
        inflation_mult = 1.0
        competitor_mult = 1.0
        for s in self.active_shocks:
            if s["type"] == "demand":
                if s["effect"] == "drop":
                    demand_mult *= s["magnitude"]
                elif s["effect"] == "surge":
                    demand_mult *= s["magnitude"]
            elif s["type"] == "inflation":
                inflation_mult *= s["magnitude"]
            elif s["type"] == "competitor":
                competitor_mult *= s["magnitude"]
        return demand_mult, inflation_mult, competitor_mult

    def demand_function(self, price: float, demand_mult: float):
        """
        Simple inverse demand curve: base_demand * exp(-alpha * price)
        alpha controls elasticity. We'll pick alpha so that price 10 leads to demand near base.
        """
        alpha = 0.03
        d = self.base_demand * np.exp(-alpha * price)
        return d * demand_mult

    def step(self, action):
        """
        action: array-like scalar delta
        """
        delta = float(np.clip(action, -self.max_price_delta, self.max_price_delta))
        self.price = max(0.1, self.price + delta)
        self.time += 1

        # Activate scheduled shocks
        self._apply_shocks_now()

        # Shock effects
        demand_mult, inflation_mult, competitor_mult = self._apply_shock_effects()

        # Effective cost (inflation)
        eff_cost = self.cost * inflation_mult

        # Quantity demanded
        demand = self.demand_function(self.price * competitor_mult, demand_mult)

        # Quantity sold limited by capacity
        sold = float(min(demand, self.capacity))

        # Profit
        profit = (self.price - eff_cost) * sold

        # reward = profit (can be scaled)
        reward = profit

        # update price history
        self.price_history.append(self.price)

        # decay shocks (reduce remaining)
        self._decay_shocks()

        done = self.time >= self.episode_length

        obs = self._get_obs()

        info = {
            "price": self.price,
            "demand": demand,
            "sold": sold,
            "profit": profit,
            "inflation": inflation_mult,
            "active_shocks": list(self.active_shocks)
        }
        return obs, reward, done, False, info

    def _get_obs(self):
        # price normalized by something; keep raw floats
        time_frac = (self.episode_length - self.time) / max(1, self.episode_length)
        capacity_util = 1.0  # placeholder (could track cumulative capacity)
        # Get current inflation multiplier from active shocks
        _, inflation_mult, _ = self._apply_shock_effects()
        obs = np.concatenate([
            np.array([self.price, self.base_demand], dtype=np.float32),
            np.array([capacity_util], dtype=np.float32),
            np.array([time_frac], dtype=np.float32),
            np.array([inflation_mult], dtype=np.float32),  # inflation multiplier
            np.array(list(self.price_history), dtype=np.float32),
        ]).astype(np.float32)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.price = self.cfg["initial_price"]
        self.base_demand = self.cfg["initial_demand"]
        self.time = 0
        self.active_shocks = []
        self.price_history = deque([self.price] * self.price_history_len, maxlen=self.price_history_len)
        self.inflation = 1.0
        return self._get_obs(), {}

    def render(self, mode="human"):
        print(f"[t={self.time}] price={self.price:.2f}, shocks={self.active_shocks}")


# ---------------------------
# Embedding & Vector Store (retrieval)
# ---------------------------
class SimpleEmbedder:
    """Fallback embedder: a small MLP that maps numeric arrays to a vector."""
    def __init__(self, input_dim: int, embed_dim: int = 128, device: str = "cpu"):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(input_dim, max(64, embed_dim)),
            nn.ReLU(),
            nn.Linear(max(64, embed_dim), embed_dim),
        ).to(self.device)
        # simple standardizer
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray):
        # X shape: (n, input_dim)
        Xs = self.scaler.fit_transform(X)
        with torch.no_grad():
            t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            out = self.net(t).cpu().numpy()
        return out

    def transform(self, X: np.ndarray):
        Xs = self.scaler.transform(X)
        with torch.no_grad():
            t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            out = self.net(t).cpu().numpy()
        return out


class VectorStore:
    """Hybrid vector store with FAISS if available, otherwise sklearn NearestNeighbors plus brute-force storage."""
    def __init__(self, embed_dim: int, max_size: int = 5000):
        self.embed_dim = embed_dim
        self.max_size = max_size
        self._n = 0
        self._vectors = None
        self._metas = []
        if CONFIG["use_faiss"]:
            try:
                self.index = faiss.IndexFlatL2(embed_dim)
                self._has_faiss = True
            except Exception:
                print("faiss import failed at runtime; falling back to sklearn neighbors")
                self._has_faiss = False
                self.index = None
        else:
            self._has_faiss = False
            self.index = None
        self._nn = None  # sklearn neighbor model lazily built

    def add(self, vectors: np.ndarray, metas: List[Any]):
        # vectors: (n, d)
        if self._vectors is None:
            self._vectors = vectors.copy()
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._metas.extend(metas)
        self._n = len(self._metas)
        # keep size bounded
        if self._n > self.max_size:
            # drop oldest
            drop = self._n - self.max_size
            self._vectors = self._vectors[drop:]
            self._metas = self._metas[drop:]
            self._n = len(self._metas)
        # rebuild sklearn index
        if not self._has_faiss:
            self._nn = NearestNeighbors(n_neighbors=min(self.n_neighbors(), max(1, min(10, self._vectors.shape[0]))),
                                        algorithm="auto").fit(self._vectors)
        else:
            try:
                self.index.reset()
                self.index.add(self._vectors.astype(np.float32))
            except Exception:
                self._has_faiss = False
                self._nn = NearestNeighbors(n_neighbors=min(self.n_neighbors(), max(1, min(10, self._vectors.shape[0]))),
                                            algorithm="auto").fit(self._vectors)

    def n_neighbors(self):
        return min(CONFIG["retrieval_k"], max(1, int(np.ceil(np.sqrt(max(1, self._n))))))

    def query(self, qvec: np.ndarray, topk: int = None):
        if topk is None:
            topk = CONFIG["retrieval_k"]
        if self._vectors is None or self._vectors.shape[0] == 0:
            return [], []
        if self._has_faiss and self.index is not None:
            D, I = self.index.search(qvec.astype(np.float32), topk)
            idxs = I[0].tolist()
        else:
            # sklearn
            k = min(topk, self._vectors.shape[0])
            distances, idxs = self._nn.kneighbors(qvec, n_neighbors=k, return_distance=True)
            idxs = idxs[0].tolist()
        results = [self._metas[i] for i in idxs]
        vectors = [self._vectors[i] for i in idxs]
        return vectors, results


# ---------------------------
# PPO Core Networks
# ---------------------------
def combined_shape(length, shape):
    if shape is None:
        return (length,)
    return (length, shape) if isinstance(shape, int) else (length, *shape)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        # actor
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.actor_net = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(last, action_dim)
        # log std parameter (for continuous actions)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # critic
        layers_v = []
        last = obs_dim
        for h in hidden_sizes:
            layers_v.append(nn.Linear(last, h))
            layers_v.append(nn.ReLU())
            last = h
        self.critic_net = nn.Sequential(*layers_v)
        self.value_head = nn.Linear(last, 1)

    def forward(self, obs):
        raise NotImplementedError

    def step(self, obs: torch.Tensor):
        # returns action (np), value (np), logprob (np), entropy (np)
        mu = self.actor_net(obs)
        mu = self.actor_mean(mu)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.value_head(self.critic_net(obs)).squeeze(-1)
        return action.detach().cpu().numpy(), value.detach().cpu().numpy(), logp.detach().cpu().numpy(), entropy.detach().cpu().numpy()

    def act(self, obs: torch.Tensor):
        mu = self.actor_net(obs)
        mu = self.actor_mean(mu)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)
        return action.detach().cpu().numpy(), logp.detach().cpu().numpy()

    def get_value(self, obs: torch.Tensor):
        with torch.no_grad():
            return self.value_head(self.critic_net(obs)).squeeze(-1).cpu().numpy()

    def get_logprob_and_entropy(self, obs: torch.Tensor, actions: torch.Tensor):
        mu = self.actor_net(obs)
        mu = self.actor_mean(mu)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return logp, entropy


# ---------------------------
# PPO Trainer & Buffer
# ---------------------------
Transition = namedtuple('Transition', ['obs', 'act', 'rew', 'done', 'value', 'logp'])

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device="cpu"):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.val_buf = []
        self.logp_buf = []
        self.gamma = gamma
        self.lam = lam
        self.device = device

    def store(self, obs, act, rew, done, value, logp):
        self.obs_buf.append(np.asarray(obs, dtype=np.float32))
        self.act_buf.append(np.asarray(act, dtype=np.float32))
        self.rew_buf.append(float(rew))
        self.done_buf.append(bool(done))
        self.val_buf.append(float(value))
        self.logp_buf.append(float(logp))

    def finish_path(self, last_val=0.0):
        """
        Compute GAE advantages and returns. Returns arrays for advantages and returns.
        """
        # compute advantages
        rewards = np.array(self.rew_buf, dtype=np.float32)
        vals = np.array(self.val_buf + [last_val], dtype=np.float32)
        dones = np.array(self.done_buf, dtype=np.float32)
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * vals[t + 1] * nonterminal - vals[t]
            adv[t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
        returns = adv + vals[:-1]
        # convert buffers to arrays
        obs = np.array(self.obs_buf, dtype=np.float32)
        acts = np.array(self.act_buf, dtype=np.float32)
        logps = np.array(self.logp_buf, dtype=np.float32)
        vals = np.array(self.val_buf, dtype=np.float32)
        # clear buffers
        self.obs_buf, self.act_buf, self.rew_buf, self.done_buf, self.val_buf, self.logp_buf = [], [], [], [], [], []
        return obs, acts, adv, returns, logps


class PPOTrainer:
    def __init__(self, obs_dim, act_dim, device="cpu", lr=3e-4, clip=0.2, epochs=10, minibatch_size=64,
                 value_loss_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.device = device
        self.ac = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.clip = clip
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def update(self, obs, acts, advs, returns, old_logps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        acts_t = torch.tensor(acts, dtype=torch.float32, device=self.device)
        advs_t = torch.tensor(advs, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_logps_t = torch.tensor(old_logps, dtype=torch.float32, device=self.device)

        # normalize advantages
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        dataset_size = obs_t.shape[0]
        for _ in range(self.epochs):
            # minibatch sampling
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                mb_obs = obs_t[mb_idx]
                mb_acts = acts_t[mb_idx]
                mb_advs = advs_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_oldlogp = old_logps_t[mb_idx]

                # new logprobs and entropy
                new_logp, entropy = self.ac.get_logprob_and_entropy(mb_obs, mb_acts)
                value = self.ac.value_head(self.ac.critic_net(mb_obs)).squeeze(-1)

                ratio = torch.exp(new_logp - mb_oldlogp)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, mb_returns)

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def save(self, path):
        torch.save(self.ac.state_dict(), path)

    def load(self, path):
        self.ac.load_state_dict(torch.load(path))


# ---------------------------
# Agents: Baseline PPO and RARL
# ---------------------------
class BaseAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: dict):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg
        self.device = cfg["device"]
        self.trainer = PPOTrainer(obs_dim=obs_dim, act_dim=act_dim, device=self.device,
                                  lr=cfg["lr"], clip=cfg["ppo_clip"], epochs=cfg["ppo_epochs"],
                                  minibatch_size=cfg["ppo_minibatch_size"],
                                  value_loss_coef=cfg["value_loss_coef"], entropy_coef=cfg["entropy_coef"],
                                  max_grad_norm=cfg["max_grad_norm"])

    def select_action(self, obs: np.ndarray):
        # Verify observation dimension matches expected
        if len(obs) != self.obs_dim:
            raise ValueError(f"Observation dimension mismatch: got {len(obs)}, expected {self.obs_dim}")
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, logp = self.trainer.ac.act(obs_t)
        return action[0], logp[0]

    def value(self, obs: np.ndarray):
        # Verify observation dimension matches expected
        if len(obs) != self.obs_dim:
            raise ValueError(f"Observation dimension mismatch: got {len(obs)}, expected {self.obs_dim}")
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.trainer.ac.get_value(obs_t)[0]


class RARLAgent(BaseAgent):
    """
    RARL agent wraps the same PPO core but concatenates retrieved context to observations when retrieval fires.
    Retrieval trigger: high policy entropy.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict, embedder: SimpleEmbedder, vstore: VectorStore, query_embedder: SimpleEmbedder):
        # augmented observation dimension: obs_dim + embed_dim
        self.base_obs_dim = obs_dim
        self.act_dim = act_dim
        self.embedder = embedder  # for episode summaries
        self.query_embedder = query_embedder  # for query vectors (obs or obs+action)
        self.vstore = vstore
        self.embed_dim = cfg["embed_dim"]
        # create trainer with augmented obs dim
        super().__init__(obs_dim + self.embed_dim, act_dim, cfg)
        self.retrieval_threshold_entropy = cfg["retrieval_threshold_entropy"]

    def _should_retrieve(self, obs: np.ndarray):
        # measure policy entropy on current obs (augmented with zero context for consistency)
        zero_ctx = np.zeros(self.embed_dim, dtype=np.float32)
        aug_obs = np.concatenate([obs, zero_ctx])
        obs_t = torch.tensor(aug_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # get entropy via a forward pass
        _, _, _, entropy = self.trainer.ac.step(obs_t)
        ent = float(entropy[0])
        return ent > self.retrieval_threshold_entropy

    def _form_retrieval_query(self, obs: np.ndarray, last_action: Optional[np.ndarray] = None):
        # build a numeric context vector (obs + last_action)
        # Always include action dimension, padding with zeros if last_action is None
        if last_action is None:
            action_vec = np.zeros(self.act_dim, dtype=np.float32)
        else:
            action_vec = np.asarray(last_action, dtype=np.float32).flatten()
            # Ensure action_vec has the right dimension
            if len(action_vec) != self.act_dim:
                action_vec = np.zeros(self.act_dim, dtype=np.float32)
        vec = np.concatenate([obs, action_vec])
        return vec.reshape(1, -1)

    def _retrieve_context(self, obs: np.ndarray, last_action: Optional[np.ndarray] = None):
        q = self._form_retrieval_query(obs, last_action)
        q_emb = self.query_embedder.transform(q)  # (1, embed_dim)
        vecs, metas = self.vstore.query(q_emb, topk=CONFIG["retrieval_k"])
        if len(vecs) == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)
        # aggregate retrieved vectors (mean)
        agg = np.mean(np.array(vecs, dtype=np.float32), axis=0)
        return agg.astype(np.float32)

    def select_action(self, obs: np.ndarray, last_action: Optional[np.ndarray] = None):
        # Verify base observation dimension
        if len(obs) != self.base_obs_dim:
            raise ValueError(f"Base observation dimension mismatch: got {len(obs)}, expected {self.base_obs_dim}")
        # decide whether to retrieve
        if self._should_retrieve(obs):
            ctx = self._retrieve_context(obs, last_action)
        else:
            ctx = np.zeros(self.embed_dim, dtype=np.float32)
        aug_obs = np.concatenate([obs, ctx])
        # Verify augmented observation dimension
        if len(aug_obs) != self.obs_dim:
            raise ValueError(f"Augmented observation dimension mismatch: got {len(aug_obs)}, expected {self.obs_dim}")
        obs_t = torch.tensor(aug_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, logp = self.trainer.ac.act(obs_t)
        return action[0], logp[0], ctx

    def value(self, obs: np.ndarray, last_action: Optional[np.ndarray] = None):
        # Verify base observation dimension
        if len(obs) != self.base_obs_dim:
            raise ValueError(f"Base observation dimension mismatch: got {len(obs)}, expected {self.base_obs_dim}")
        # For RARL, we need to augment observation with context (or zero context)
        # Use zero context for value estimation during training rollouts
        zero_ctx = np.zeros(self.embed_dim, dtype=np.float32)
        aug_obs = np.concatenate([obs, zero_ctx])
        # Verify augmented observation dimension
        if len(aug_obs) != self.obs_dim:
            raise ValueError(f"Augmented observation dimension mismatch: got {len(aug_obs)}, expected {self.obs_dim}")
        obs_t = torch.tensor(aug_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.trainer.ac.get_value(obs_t)[0]


# ---------------------------
# Runner / Experiment Utilities
# ---------------------------
def rollout_episode(env: MarketEnv, agent: BaseAgent, use_rarl: bool = False, vstore=None, embedder=None):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    info_history = []
    last_action = None

    while not done:
        if use_rarl:
            # RARLAgent select_action expects extra return
            action, logp, ctx = agent.select_action(obs, last_action)
        else:
            action, logp = agent.select_action(obs)
            ctx = None
        # action is array-like (scalar)
        obs2, reward, done, trunc, info = env.step(action)
        total_reward += reward
        last_action = action
        obs = obs2
        steps += 1
        info_history.append(info)
    return total_reward, steps, info_history


def gather_episodes_for_training(env: MarketEnv, agent: BaseAgent, cfg: dict, use_rarl: bool = False,
                                  embedder: Optional[SimpleEmbedder] = None, vstore: Optional[VectorStore] = None):
    """
    Collect transitions until we reach rollout_timesteps or a number of episodes.
    Returns arrays ready for PPO update.
    Also populates vector store with episodes (for RARL).
    """
    buf = PPOBuffer(obs_dim=None, act_dim=None, size=None, gamma=cfg["gamma"], lam=cfg["gae_lambda"], device=cfg["device"])
    collected_steps = 0
    observations_all = []
    actions_all = []
    rewards_all = []
    values_all = []
    logps_all = []
    done_all = []

    # We'll collect by episodes
    while collected_steps < cfg["rollout_timesteps"]:
        obs, _ = env.reset()
        done = False
        last_action = None
        # to store episode-specific data to add to vector store as an embedding
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_infos = []
        while not done and collected_steps < cfg["rollout_timesteps"]:
            if use_rarl:
                # retrieval-aware selection (but we don't use retrieval during training policy rollout to avoid leakage)
                # Instead: use current policy without retrieval (or optionally use retrieval - configurable)
                # We'll call select_action that returns (action, logp, ctx)
                a, logp, ctx = agent.select_action(obs, last_action)
                # For RARL, augment observation with context for storage
                aug_obs = np.concatenate([obs, ctx])
                val = agent.value(obs, last_action)
            else:
                a, logp = agent.select_action(obs)
                aug_obs = obs  # no augmentation for baseline
                val = agent.value(obs)
            obs_next, rew, done, trunc, info = env.step(a)
            buf.store(aug_obs, a, rew, done, val, logp)
            # accumulate episode arrays
            episode_obs.append(obs)
            episode_actions.append(a)
            episode_rewards.append(rew)
            episode_infos.append(info)
            # step
            obs = obs_next
            last_action = a
            collected_steps += 1
        # finish path: last value (bootstrap) = 0 (episodic)
        obs_arr, acts_arr, advs, returns, old_logps = buf.finish_path(last_val=0.0)
        # Save episode to vector store: we'll build a simple vector describing the episode: aggregated obs+act+reward features
        if vstore is not None and embedder is not None:
            # build a single vector summarizing the episode
            # features: mean(obs), std(obs), mean(action), total_reward
            ep_obs = np.array(episode_obs, dtype=np.float32)
            feat = np.concatenate([
                ep_obs.mean(axis=0),
                ep_obs.std(axis=0),
                np.array([np.mean(np.array(episode_actions, dtype=np.float32))], dtype=np.float32),
                np.array([np.sum(np.array(episode_rewards, dtype=np.float32))], dtype=np.float32)
            ]).reshape(1, -1)
            emb = embedder.fit_transform(feat) if vstore._n == 0 else embedder.transform(feat)
            metas = [{"episode_reward": float(np.sum(episode_rewards)), "time": time.time()}]
            vstore.add(emb, metas)

        # For training update, we can return arrays in a flattened manner
        if len(observations_all) == 0:
            observations_all = obs_arr
            actions_all = acts_arr
            rewards_all = advs  # temporarily store advs
            values_all = returns
            logps_all = old_logps
        else:
            observations_all = np.vstack([observations_all, obs_arr])
            actions_all = np.vstack([actions_all, acts_arr])
            rewards_all = np.concatenate([rewards_all, advs])
            values_all = np.concatenate([values_all, returns])
            logps_all = np.concatenate([logps_all, old_logps])
    # final arrays
    return observations_all, actions_all, rewards_all, values_all, logps_all


# ---------------------------
# Metrics & Evaluation
# ---------------------------
def evaluate_agent(env: MarketEnv, agent: BaseAgent, cfg: dict, episodes: int = 20, use_rarl: bool = False):
    rewards = []
    steps = []
    infos = []
    for _ in range(episodes):
        tot, s, info_hist = rollout_episode(env, agent, use_rarl=use_rarl)
        rewards.append(tot)
        steps.append(s)
        infos.append(info_hist)
    return {
        "rewards": np.array(rewards),
        "steps": np.array(steps),
        "infos": infos
    }


def compute_metrics(eval_results, oracle_reward: Optional[float] = None):
    # cumulative reward mean/std
    rewards = eval_results["rewards"]
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    # episodes-to-optimal: approximate by number of episodes achieving >= 95% of oracle_reward if provided
    episodes_to_optimal = None
    if oracle_reward is not None:
        target = 0.95 * oracle_reward
        episodes_to_optimal = float(np.sum(rewards >= target))
    # action variance (not tracked easily here)
    metrics = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "episodes_to_optimal": episodes_to_optimal
    }
    return metrics


def compute_recovery_time(info_histories: List[List[Dict[str, Any]]], shock_time: int, baseline_profit: float, threshold_frac: float = 0.9):
    """
    For each episode's info_history (list of per-step info dicts), compute time to recover
    to a fraction of baseline_profit after shock_time. Returns average recovery time steps.
    """
    times = []
    for info_hist in info_histories:
        # find index of shock_time if possible
        if shock_time >= len(info_hist):
            continue
        # compute profit time series
        profits = np.array([info.get("profit", 0.0) for info in info_hist])
        # find first time after shock_time where profit >= threshold_frac * baseline_profit
        target = threshold_frac * baseline_profit
        recovered = None
        for t in range(shock_time, len(profits)):
            if profits[t] >= target:
                recovered = t - shock_time
                break
        if recovered is not None:
            times.append(recovered)
    if len(times) == 0:
        return None
    return float(np.mean(times))


# ---------------------------
# Main Experiment Flow
# ---------------------------
def run_experiment(cfg: dict, mode: str = "baseline"):
    """
    mode: "baseline" or "rarl"
    Returns logs and metrics
    """
    set_seed(cfg["seed"])
    ensure_dir(cfg["save_dir"])

    # create env
    env = MarketEnv(cfg)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # vector store & embedder for RARL
    embedder = None
    query_embedder = None
    vstore = None
    if mode == "rarl":
        # choose embedder: sentence-transformer if available, otherwise SimpleEmbedder on numeric summary features
        # Here the embedder's expected input dims depend on episode summary features; we will set dynamically.
        # We'll create a numeric embedder that accepts (mean(obs), std(obs), mean(action), total_reward) => size = obs_dim*2 + 2
        embed_input_dim = obs_dim * 2 + 2
        if _HAS_SBERT:
            # create sentence transformer wrapper (but our inputs are numeric; this is unusual; skip SBERT)
            embedder = SimpleEmbedder(embed_input_dim, cfg["embed_dim"], device=cfg["device"])
        else:
            embedder = SimpleEmbedder(embed_input_dim, cfg["embed_dim"], device=cfg["device"])
        # Query embedder: accepts obs_dim (or obs_dim + act_dim if last_action is included)
        # We'll use obs_dim + act_dim to handle both cases
        query_input_dim = obs_dim + act_dim
        query_embedder = SimpleEmbedder(query_input_dim, cfg["embed_dim"], device=cfg["device"])
        vstore = VectorStore(embed_dim=cfg["embed_dim"], max_size=cfg["vector_store_max_size"])

    # create agents
    if mode == "baseline":
        agent = BaseAgent(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg)
    else:
        agent = RARLAgent(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg, embedder=embedder, vstore=vstore, query_embedder=query_embedder)

    # training loop
    print(f"Starting training mode={mode}, device={cfg['device']}")
    training_rewards = []
    eval_mean_rewards = []
    start_time = time.time()
    for ep in trange(cfg["train_episodes"], desc=f"Train ({mode})"):
        # collect training batch and populate vector store (if rarl)
        obs_arr, acts_arr, advs, returns, old_logps = gather_episodes_for_training(env, agent, cfg,
                                                                                  use_rarl=(mode == "rarl"),
                                                                                  embedder=embedder, vstore=vstore)
        # update policy
        try:
            agent.trainer.update(obs_arr, acts_arr, advs, returns, old_logps)
        except Exception as e:
            print("Warning: trainer update failed:", e)
        # do periodic evaluation
        if (ep + 1) % max(1, cfg["train_episodes"] // 10) == 0:
            eval_res = evaluate_agent(env, agent, cfg, episodes=cfg["eval_episodes"], use_rarl=(mode == "rarl"))
            metrics = compute_metrics(eval_res)
            eval_mean_rewards.append(metrics["mean_reward"])
            print(f"[{mode}] Ep {ep+1}: eval_mean_reward={metrics['mean_reward']:.2f}, std={metrics['std_reward']:.2f}")

    total_time = time.time() - start_time
    print(f"Training complete for mode={mode}. Time elapsed: {total_time:.1f}s")

    # final evaluation
    final_eval = evaluate_agent(env, agent, cfg, episodes=cfg["eval_episodes"], use_rarl=(mode == "rarl"))
    metrics = compute_metrics(final_eval)
    # compute recovery time if a shock schedule exists: use first shock
    shock_time = cfg["shock_schedule"][0] if len(cfg["shock_schedule"]) > 0 else None
    recovery_time = None
    if shock_time is not None:
        # baseline_profit approximated as mean profit prior to shock in baseline eval
        # We will estimate baseline_profit from the same eval info histories: mean of profits before shock_time
        profits_pre = []
        for info_hist in final_eval["infos"]:
            if shock_time < len(info_hist):
                profits_pre.append(np.mean([info_hist[t].get("profit", 0.0) for t in range(max(0, shock_time-20), shock_time)]))
        baseline_profit = np.mean(profits_pre) if len(profits_pre) > 0 else 1.0
        recovery_time = compute_recovery_time(final_eval["infos"], shock_time, baseline_profit, threshold_frac=0.9)

    # collect action variance estimate (rough): variance of actions during evaluation (if stored)
    # here we didn't store per-step actions in evaluate_agent; could extend, but approximate as zero
    result = {
        "mode": mode,
        "metrics": metrics,
        "final_eval": final_eval,
        "eval_history": eval_mean_rewards,
        "recovery_time": recovery_time
    }
    # save checkpoint
    chk_path = os.path.join(cfg["save_dir"], f"{mode}_ac.pt")
    agent.trainer.save(chk_path)
    print(f"Saved checkpoint to {chk_path}")

    return result


def plot_results(baseline_res, rarl_res, cfg):
    ensure_dir(cfg["save_dir"])
    # plot eval_history if available
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    if len(baseline_res["eval_history"]) > 0:
        ax[0].plot(baseline_res["eval_history"], label="baseline_eval_mean")
    if len(rarl_res["eval_history"]) > 0:
        ax[0].plot(rarl_res["eval_history"], label="rarl_eval_mean")
    ax[0].set_title("Evaluation Mean Reward Over Training (checkpoints)")
    ax[0].set_xlabel("checkpoint")
    ax[0].set_ylabel("mean reward")
    ax[0].legend()

    # final distribution of rewards
    b_rewards = baseline_res["final_eval"]["rewards"]
    r_rewards = rarl_res["final_eval"]["rewards"]
    ax[1].hist(b_rewards, bins=15, alpha=0.6, label="baseline")
    ax[1].hist(r_rewards, bins=15, alpha=0.6, label="rarl")
    ax[1].set_title("Final Evaluation Reward Distribution")
    ax[1].legend()

    plt.tight_layout()
    out_path = os.path.join(cfg["save_dir"], "results.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def main():
    cfg = CONFIG.copy()
    ensure_dir(cfg["save_dir"])
    # run baseline
    baseline_res = run_experiment(cfg, mode="baseline")
    # run RARL
    rarl_res = run_experiment(cfg, mode="rarl")

    # print summary
    print("\n=== Summary ===")
    for res in [baseline_res, rarl_res]:
        print(f"Mode: {res['mode']}")
        print("Mean reward: {:.2f} ± {:.2f}".format(res["metrics"]["mean_reward"], res["metrics"]["std_reward"]))
        print("Episodes-to-optimal:", res["metrics"]["episodes_to_optimal"])
        print("Recovery time (est):", res["recovery_time"])
        print("-" * 20)

    plot_results(baseline_res, rarl_res, cfg)

    # Save summary CSV
    summary = {
        "mode": [baseline_res["mode"], rarl_res["mode"]],
        "mean_reward": [baseline_res["metrics"]["mean_reward"], rarl_res["metrics"]["mean_reward"]],
        "std_reward": [baseline_res["metrics"]["std_reward"], rarl_res["metrics"]["std_reward"]],
        "episodes_to_optimal": [baseline_res["metrics"]["episodes_to_optimal"], rarl_res["metrics"]["episodes_to_optimal"]],
        "recovery_time": [baseline_res["recovery_time"], rarl_res["recovery_time"]]
    }
    df = pd.DataFrame(summary)
    csv_path = os.path.join(cfg["save_dir"], "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")


if __name__ == "__main__":
    main()

