"""Actor-Critic MLP policy network for LLM routing."""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    Shared backbone + separate actor head (policy) and critic head (value).

    Architecture:
        Input (feature_dim)
            -> Linear(feature_dim, 128) -> ReLU -> LayerNorm
            -> Linear(128, 64) -> ReLU -> LayerNorm
            |                              |
        Actor Head                    Critic Head
        Linear(64, num_actions)       Linear(64, 1)
        -> Categorical dist           -> Value scalar
    """

    def __init__(self, feature_dim: int, num_actions: int) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.num_actions = num_actions

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )

        # Actor head (policy): outputs logits over actions
        self.actor_head = nn.Linear(64, num_actions)

        # Critic head (value): outputs scalar state value
        self.critic_head = nn.Linear(64, 1)

        # Apply orthogonal initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization for all linear layers, bias set to 0."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            obs: Observation tensor of shape (feature_dim,) or (batch, feature_dim).

        Returns:
            action_logits: Tensor of shape (num_actions,) or (batch, num_actions).
            value: Tensor of shape (1,) or (batch, 1).
        """
        features = self.backbone(obs)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        return action_logits, value

    def get_action(
        self, obs: torch.Tensor
    ) -> Tuple[int, float, float]:
        """
        Sample an action from the policy for a single observation.

        Args:
            obs: Observation tensor of shape (feature_dim,) or (1, feature_dim).

        Returns:
            action: Selected action as an int.
            log_prob: Log probability of the selected action as a float.
            value: Estimated state value as a float.
        """
        with torch.no_grad():
            action_logits, value = self.forward(obs)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions under the current policy (used during PPO update).

        Args:
            obs: Batch of observations, shape (batch, feature_dim).
            actions: Batch of actions taken, shape (batch,).

        Returns:
            log_probs: Log probabilities of the actions, shape (batch,).
            entropy: Entropy of the action distribution, shape (batch,).
            values: Estimated state values, shape (batch,).
        """
        action_logits, values = self.forward(obs)
        dist = Categorical(logits=action_logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values.squeeze(-1)
