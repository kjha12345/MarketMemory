"""
Memory Bank for RARL: Vector Database for Storing and Retrieving Historical Experiences

Stores state-action-reward tuples with embeddings for similarity search.
"""

import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import pickle
import os


@dataclass
class Experience:
    """A single experience tuple"""
    state: np.ndarray
    action: float
    reward: float
    regime_tag: str
    episode_id: int
    time_step: int


class MemoryBank:
    """
    Vector database for storing and retrieving historical market experiences.
    Uses embeddings to find similar past states.
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        use_gpu: bool = False,
        similarity_metric: str = "cosine"
    ):
        """
        Initialize memory bank.
        
        Args:
            embedding_dim: Dimension of state embeddings
            use_gpu: Whether to use GPU for FAISS
            similarity_metric: "cosine" or "l2"
        """
        self.embedding_dim = embedding_dim
        self.experiences: List[Experience] = []
        
        # Initialize FAISS index
        if similarity_metric == "cosine":
            # Normalize embeddings for cosine similarity
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for normalized vectors
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        
        self.similarity_metric = similarity_metric
        
        # Initialize embedding model (using a lightweight model)
        # We'll create embeddings from state vectors
        self.embedding_model = None  # We'll use a simple MLP instead
    
    def _state_to_embedding(self, state: np.ndarray) -> np.ndarray:
        """
        Convert state vector to embedding.
        For simplicity, we use the state directly with normalization,
        but in practice you could use a learned encoder.
        """
        # Normalize state to unit vector for cosine similarity
        state_norm = state / (np.linalg.norm(state) + 1e-8)
        
        # If state dim < embedding_dim, pad with zeros
        # If state dim > embedding_dim, truncate
        if len(state_norm) < self.embedding_dim:
            embedding = np.pad(state_norm, (0, self.embedding_dim - len(state_norm)))
        else:
            embedding = state_norm[:self.embedding_dim]
        
        return embedding.astype(np.float32)
    
    def add_experience(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        regime_tag: str,
        episode_id: int,
        time_step: int
    ):
        """Add a new experience to the memory bank"""
        experience = Experience(
            state=state.copy(),
            action=action,
            reward=reward,
            regime_tag=regime_tag,
            episode_id=episode_id,
            time_step=time_step
        )
        
        self.experiences.append(experience)
        
        # Create embedding and add to FAISS index
        embedding = self._state_to_embedding(state)
        self.index.add(embedding.reshape(1, -1))
    
    def add_episode(
        self,
        states: List[np.ndarray],
        actions: List[float],
        rewards: List[float],
        regime_tags: List[str],
        episode_id: int
    ):
        """Add an entire episode to the memory bank"""
        for t, (state, action, reward, regime_tag) in enumerate(
            zip(states, actions, rewards, regime_tags)
        ):
            self.add_experience(state, action, reward, regime_tag, episode_id, t)
    
    def retrieve(
        self,
        query_state: np.ndarray,
        k: int = 5
    ) -> List[Dict]:
        """
        Retrieve k most similar experiences to query state.
        
        Returns:
            List of dictionaries with keys: state, action, reward, regime_tag, similarity
        """
        if len(self.experiences) == 0:
            return []
        
        # Create query embedding
        query_embedding = self._state_to_embedding(query_state)
        
        # Normalize for cosine similarity if needed
        if self.similarity_metric == "cosine":
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Search FAISS index
        k = min(k, len(self.experiences))
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Retrieve experiences
        retrieved = []
        for idx, dist in zip(indices[0], distances[0]):
            exp = self.experiences[idx]
            similarity = 1.0 - dist if self.similarity_metric == "cosine" else -dist
            retrieved.append({
                "state": exp.state,
                "action": exp.action,
                "reward": exp.reward,
                "regime_tag": exp.regime_tag,
                "similarity": float(similarity),
                "episode_id": exp.episode_id,
                "time_step": exp.time_step
            })
        
        return retrieved
    
    def get_statistics(self) -> Dict:
        """Get statistics about the memory bank"""
        if len(self.experiences) == 0:
            return {
                "num_experiences": 0,
                "num_episodes": 0,
                "regime_distribution": {}
            }
        
        regime_counts = {}
        episode_ids = set()
        for exp in self.experiences:
            regime_counts[exp.regime_tag] = regime_counts.get(exp.regime_tag, 0) + 1
            episode_ids.add(exp.episode_id)
        
        return {
            "num_experiences": len(self.experiences),
            "num_episodes": len(episode_ids),
            "regime_distribution": regime_counts
        }
    
    def save(self, filepath: str):
        """Save memory bank to disk"""
        data = {
            "experiences": self.experiences,
            "embedding_dim": self.embedding_dim,
            "similarity_metric": self.similarity_metric
        }
        
        # Save experiences
        with open(filepath + "_experiences.pkl", "wb") as f:
            pickle.dump(data, f)
        
        # Save FAISS index
        faiss.write_index(self.index, filepath + "_index.faiss")
    
    def load(self, filepath: str):
        """Load memory bank from disk"""
        # Load experiences
        with open(filepath + "_experiences.pkl", "rb") as f:
            data = pickle.load(f)
        
        self.experiences = data["experiences"]
        self.embedding_dim = data["embedding_dim"]
        self.similarity_metric = data["similarity_metric"]
        
        # Load FAISS index
        self.index = faiss.read_index(filepath + "_index.faiss")
