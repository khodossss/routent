"""Rollout buffer for PPO in a contextual bandit setting."""

from typing import Generator, Tuple

import torch
import numpy as np


class RolloutBuffer:
    """Stores transitions collected during rollout for PPO updates.

    Designed for contextual bandit (horizon=1): no done flags, no discounting.
    Returns = rewards directly. Advantages = rewards - values.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.pos = 0

        self.observations = torch.zeros(capacity, obs_dim, dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.values = torch.zeros(capacity, dtype=torch.float32)

        self.returns = torch.zeros(capacity, dtype=torch.float32)
        self.advantages = torch.zeros(capacity, dtype=torch.float32)

    @property
    def size(self) -> int:
        return self.pos

    def add(
        self,
        obs: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
    ) -> None:
        """Add a single transition."""
        idx = self.pos
        self.observations[idx] = obs
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.pos += 1

    def compute_returns_and_advantages(self) -> None:
        """Compute returns and advantages for contextual bandit.

        Since horizon=1: returns = rewards, advantages = rewards - values.
        Advantages are then normalized to zero mean, unit variance.
        """
        n = self.pos
        self.returns[:n] = self.rewards[:n]
        self.advantages[:n] = self.rewards[:n] - self.values[:n]

        # Normalize advantages
        adv = self.advantages[:n]
        std = adv.std()
        if std > 1e-8:
            self.advantages[:n] = (adv - adv.mean()) / std

    def get_batches(
        self, batch_size: int
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """Yield random mini-batches from the buffer.

        Yields:
            Tuples of (obs, actions, old_log_probs, returns, advantages).
        """
        n = self.pos
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            yield (
                self.observations[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.returns[batch_idx],
                self.advantages[batch_idx],
            )

    def clear(self) -> None:
        """Reset the buffer position."""
        self.pos = 0
