"""Tests for PPO training components."""

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from routent.config import Config
from routent.env.feature_extractor import TfidfFeatureExtractor
from routent.env.router_env import LLMRouterEnv
from routent.models.llm_pool import BaseLLM
from routent.models.policy_network import PolicyNetwork
from routent.training.buffer import RolloutBuffer
from routent.training.ppo import PPOTrainer


class StubLLM(BaseLLM):
    def __init__(self, name, correct=True):
        self.name = name
        self._correct = correct
    def generate(self, prompt, answer, difficulty, category):
        return (answer if self._correct else "wrong"), 100.0, 0.01


class StubPool:
    def __init__(self):
        self.models = [StubLLM("a"), StubLLM("b")]
    def get_model(self, idx):
        return self.models[idx]
    @property
    def num_models(self):
        return len(self.models)


BENCHMARK = [
    {"id": i, "question": q, "answer": a, "category": "math"}
    for i, (q, a) in enumerate([
        ("What is 2+2?", "4"), ("Capital of France?", "Paris"),
        ("Solve 3*5", "15"), ("What is 10-3?", "7"), ("Square root of 16?", "4"),
    ], 1)
]


@pytest.fixture
def setup():
    fe = TfidfFeatureExtractor(tfidf_max_features=100)
    fe.fit([item["question"] for item in BENCHMARK])
    dim = fe.feature_dim

    config = Config()
    config.total_feature_dim = dim
    config.num_models = 2
    config.rollout_steps = 16
    config.batch_size = 8
    config.ppo_epochs = 2

    env = LLMRouterEnv(
        benchmark=BENCHMARK, feature_extractor=fe, llm_pool=StubPool(), seed=42,
    )
    return config, env, dim


class TestRolloutBuffer:
    def test_buffer_creation(self):
        buf = RolloutBuffer(capacity=64, obs_dim=20)
        assert buf.size == 0

    def test_buffer_add(self):
        buf = RolloutBuffer(capacity=64, obs_dim=20)
        buf.add(torch.zeros(20), 0, -1.0, 0.5, 0.1)
        assert buf.size == 1

    def test_buffer_clear(self):
        buf = RolloutBuffer(capacity=64, obs_dim=20)
        buf.add(torch.zeros(20), 0, -1.0, 0.5, 0.1)
        buf.clear()
        assert buf.size == 0

    def test_get_batches_shapes(self):
        buf = RolloutBuffer(capacity=32, obs_dim=20)
        for _ in range(32):
            buf.add(torch.randn(20), 0, -1.0, 0.5, 0.1)
        buf.compute_returns_and_advantages()
        for obs, actions, lp, ret, adv in buf.get_batches(16):
            assert obs.shape[1] == 20
            assert actions.shape[0] == obs.shape[0]


class TestAdvantageComputation:
    def test_zero_mean(self):
        buf = RolloutBuffer(capacity=64, obs_dim=20)
        for _ in range(64):
            buf.add(torch.randn(20), 0, -1.0, np.random.rand(), np.random.rand())
        buf.compute_returns_and_advantages()
        assert abs(buf.advantages[:64].mean().item()) < 1e-5

    def test_returns_equal_rewards(self):
        buf = RolloutBuffer(capacity=32, obs_dim=20)
        rewards = [np.random.rand() for _ in range(32)]
        for r in rewards:
            buf.add(torch.randn(20), 0, -1.0, r, 0.0)
        buf.compute_returns_and_advantages()
        for i, r in enumerate(rewards):
            assert abs(buf.returns[i].item() - r) < 1e-5


class TestPPOTrainer:
    def test_collect_rollout(self, setup):
        config, env, dim = setup
        policy = PolicyNetwork(dim, 2)
        trainer = PPOTrainer(policy, config)
        trainer.collect_rollout(env)
        assert trainer.buffer.size == config.rollout_steps

    def test_update(self, setup):
        config, env, dim = setup
        policy = PolicyNetwork(dim, 2)
        trainer = PPOTrainer(policy, config)
        trainer.collect_rollout(env)
        stats = trainer.update()
        assert "policy_loss" in stats
        assert "value_loss" in stats

    def test_params_change(self, setup):
        config, env, dim = setup
        policy = PolicyNetwork(dim, 2)
        trainer = PPOTrainer(policy, config)
        params_before = [p.clone() for p in policy.parameters()]
        trainer.collect_rollout(env)
        trainer.update()
        changed = any(not torch.equal(a, b) for a, b in zip(params_before, policy.parameters()))
        assert changed
