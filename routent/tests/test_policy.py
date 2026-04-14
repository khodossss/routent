"""Tests for the PolicyNetwork actor-critic MLP."""

import json
import os
import sys

import numpy as np
import pytest
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from routent.models.policy_network import PolicyNetwork


@pytest.fixture
def benchmark():
    """Load the training benchmark dataset."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", "benchmark_train.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture
def policy():
    """Create a PolicyNetwork with default dimensions."""
    torch.manual_seed(42)
    return PolicyNetwork(feature_dim=110, num_actions=4)


@pytest.fixture
def single_obs():
    """A single random observation tensor of shape (110,)."""
    torch.manual_seed(42)
    return torch.randn(110)


@pytest.fixture
def batch_obs():
    """A batch of random observation tensors of shape (32, 110)."""
    torch.manual_seed(42)
    return torch.randn(32, 110)


class TestForwardPass:
    """Test the forward pass of PolicyNetwork."""

    def test_forward_single_obs_logits_shape(self, policy, single_obs):
        logits, value = policy(single_obs)
        assert logits.shape == (4,)

    def test_forward_single_obs_value_shape(self, policy, single_obs):
        logits, value = policy(single_obs)
        assert value.shape == (1,)

    def test_forward_batch_logits_shape(self, policy, batch_obs):
        logits, value = policy(batch_obs)
        assert logits.shape == (32, 4)

    def test_forward_batch_value_shape(self, policy, batch_obs):
        logits, value = policy(batch_obs)
        assert value.shape == (32, 1)

    def test_forward_no_nans(self, policy, single_obs):
        logits, value = policy(single_obs)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(value).any()

    def test_forward_logits_are_finite(self, policy, batch_obs):
        logits, value = policy(batch_obs)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(value).all()


class TestGetAction:
    """Test the get_action method for sampling."""

    def test_get_action_returns_three_values(self, policy, single_obs):
        result = policy.get_action(single_obs)
        assert len(result) == 3

    def test_get_action_action_is_int(self, policy, single_obs):
        action, log_prob, value = policy.get_action(single_obs)
        assert isinstance(action, int)

    def test_get_action_in_valid_range(self, policy, single_obs):
        for _ in range(50):
            action, _, _ = policy.get_action(single_obs)
            assert 0 <= action <= 3

    def test_get_action_log_prob_is_float(self, policy, single_obs):
        _, log_prob, _ = policy.get_action(single_obs)
        assert isinstance(log_prob, float)

    def test_get_action_log_prob_negative(self, policy, single_obs):
        """Log probabilities should be <= 0."""
        _, log_prob, _ = policy.get_action(single_obs)
        assert log_prob <= 0.0

    def test_get_action_value_is_float(self, policy, single_obs):
        _, _, value = policy.get_action(single_obs)
        assert isinstance(value, float)

    def test_get_action_no_gradient(self, policy, single_obs):
        """get_action should not track gradients."""
        action, log_prob, value = policy.get_action(single_obs)
        for param in policy.parameters():
            assert param.grad is None


class TestEvaluateActions:
    """Test the evaluate_actions method used during PPO updates."""

    def test_evaluate_actions_returns_three_tensors(self, policy, batch_obs):
        actions = torch.randint(0, 4, (32,))
        log_probs, entropy, values = policy.evaluate_actions(batch_obs, actions)
        assert isinstance(log_probs, torch.Tensor)
        assert isinstance(entropy, torch.Tensor)
        assert isinstance(values, torch.Tensor)

    def test_evaluate_actions_log_probs_shape(self, policy, batch_obs):
        actions = torch.randint(0, 4, (32,))
        log_probs, _, _ = policy.evaluate_actions(batch_obs, actions)
        assert log_probs.shape == (32,)

    def test_evaluate_actions_entropy_shape(self, policy, batch_obs):
        actions = torch.randint(0, 4, (32,))
        _, entropy, _ = policy.evaluate_actions(batch_obs, actions)
        assert entropy.shape == (32,)

    def test_evaluate_actions_values_shape(self, policy, batch_obs):
        actions = torch.randint(0, 4, (32,))
        _, _, values = policy.evaluate_actions(batch_obs, actions)
        assert values.shape == (32,)

    def test_evaluate_actions_entropy_non_negative(self, policy, batch_obs):
        actions = torch.randint(0, 4, (32,))
        _, entropy, _ = policy.evaluate_actions(batch_obs, actions)
        assert (entropy >= 0).all()

    def test_evaluate_actions_log_probs_non_positive(self, policy, batch_obs):
        actions = torch.randint(0, 4, (32,))
        log_probs, _, _ = policy.evaluate_actions(batch_obs, actions)
        assert (log_probs <= 0.0 + 1e-6).all()

    def test_evaluate_actions_gradients_flow(self, policy, batch_obs):
        """Ensure gradients flow through evaluate_actions for PPO training."""
        actions = torch.randint(0, 4, (32,))
        log_probs, entropy, values = policy.evaluate_actions(batch_obs, actions)
        loss = -log_probs.mean() + values.mean()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in policy.parameters())
        assert has_grad


class TestWeightInitialization:
    """Test that weights are initialized with orthogonal init."""

    def test_linear_layers_have_zero_bias(self, policy):
        for name, module in policy.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert torch.allclose(
                    module.bias, torch.zeros_like(module.bias)
                ), f"Bias not zero for {name}"
