"""Tests for DisjointLinUCB agent."""

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from routent.config import Config
from routent.env.router_env import LLMRouterEnv
from routent.models.base import BaseLLM
from routent.training.linucb import DisjointLinUCB, LinUCBTrainer


class _FixedFeatureExtractor:
    """Lightweight mock — returns a zero vector of fixed dim, no model downloads."""
    DIM = 16

    def fit(self, corpus): pass

    def transform(self, prompt: str) -> torch.Tensor:
        return torch.zeros(self.DIM, dtype=torch.float32)

    @property
    def feature_dim(self) -> int:
        return self.DIM


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

DIM = 16
N_ARMS = 3


@pytest.fixture
def agent():
    return DisjointLinUCB(num_actions=N_ARMS, feature_dim=DIM, alpha=1.0)


@pytest.fixture
def env_setup(tmp_path):
    fe = _FixedFeatureExtractor()
    fe.fit([item["question"] for item in BENCHMARK])
    config = Config()
    config.total_feature_dim = fe.feature_dim
    config.num_models = 2
    config.rollout_steps = 8
    config.model_names = ["a", "b"]
    config.results_dir = str(tmp_path)
    env = LLMRouterEnv(
        benchmark=BENCHMARK, feature_extractor=fe, llm_pool=StubPool(), seed=42,
    )
    return config, env


class TestDisjointLinUCBInit:
    def test_shapes(self, agent):
        assert len(agent.A_inv) == N_ARMS
        assert len(agent.b) == N_ARMS
        assert agent.A_inv[0].shape == (DIM, DIM)
        assert agent.b[0].shape == (DIM,)

    def test_A_inv_is_identity(self, agent):
        for a in range(N_ARMS):
            assert np.allclose(agent.A_inv[a], np.eye(DIM))

    def test_b_is_zeros(self, agent):
        for a in range(N_ARMS):
            assert np.allclose(agent.b[a], 0.0)


class TestSelectAction:
    def test_returns_valid_arm(self, agent):
        x = np.random.randn(DIM)
        action = agent.select_action(x)
        assert 0 <= action < N_ARMS

    def test_accepts_torch_tensor_via_get_action(self, agent):
        obs = torch.randn(DIM)
        action, _, _ = agent.get_action(obs)
        assert 0 <= action < N_ARMS

    def test_cold_start_visits_all_arms(self, agent):
        """With all arms equal, random tie-breaking should spread actions."""
        x = np.ones(DIM)
        counts = {a: 0 for a in range(N_ARMS)}
        for _ in range(300):
            counts[agent.select_action(x)] += 1
        # Every arm should be chosen at least once
        assert all(c > 0 for c in counts.values())


class TestUpdate:
    def test_A_inv_changes_after_update(self, agent):
        x = np.random.randn(DIM)
        A_inv_before = agent.A_inv[0].copy()
        agent.update(0, x, reward=1.0)
        assert not np.allclose(agent.A_inv[0], A_inv_before)

    def test_b_changes_after_update(self, agent):
        x = np.random.randn(DIM)
        agent.update(0, x, reward=0.5)
        assert not np.allclose(agent.b[0], 0.0)

    def test_only_chosen_arm_updates(self, agent):
        x = np.random.randn(DIM)
        b_before = [agent.b[a].copy() for a in range(N_ARMS)]
        agent.update(1, x, reward=1.0)
        assert np.allclose(agent.b[0], b_before[0])
        assert not np.allclose(agent.b[1], b_before[1])
        assert np.allclose(agent.b[2], b_before[2])

    def test_sherman_morrison_consistency(self, agent):
        """A_inv maintained via S-M should equal explicit inverse."""
        x = np.random.randn(DIM)
        A = np.eye(DIM) + np.outer(x, x)
        agent.update(0, x, reward=0.0)
        assert np.allclose(agent.A_inv[0], np.linalg.inv(A), atol=1e-10)


class TestGreedyAction:
    def test_greedy_exploits_after_learning(self, agent):
        """After many positive rewards for arm 0, greedy should prefer arm 0."""
        np.random.seed(0)
        x = np.random.randn(DIM)
        for _ in range(50):
            agent.update(0, x, reward=1.0)
            agent.update(1, x, reward=-1.0)
            agent.update(2, x, reward=-1.0)
        assert agent.greedy_action(x) == 0


class TestPersistence:
    def test_save_load_roundtrip(self, agent, tmp_path):
        x = np.random.randn(DIM)
        agent.update(0, x, reward=0.9)
        path = str(tmp_path / "agent.pt")
        agent.save(path)

        loaded = DisjointLinUCB.load(path)
        assert loaded.num_actions == agent.num_actions
        assert loaded.feature_dim == agent.feature_dim
        assert np.allclose(loaded.A_inv[0], agent.A_inv[0])
        assert np.allclose(loaded.b[0], agent.b[0])
        assert loaded.select_action(x) == agent.select_action(x)


class TestLinUCBTrainer:
    def test_train_runs(self, env_setup):
        config, env = env_setup
        agent = DisjointLinUCB(num_actions=2, feature_dim=DIM, alpha=1.0)
        trainer = LinUCBTrainer(agent=agent, config=config)
        history = trainer.train(env, total_timesteps=16, eval_interval=8)
        assert len(history) == 2

    def test_metrics_logged(self, env_setup):
        config, env = env_setup
        agent = DisjointLinUCB(num_actions=2, feature_dim=DIM, alpha=1.0)
        trainer = LinUCBTrainer(agent=agent, config=config)
        trainer.train(env, total_timesteps=16, eval_interval=16)
        assert len(trainer.metrics.rewards) == 16

    def test_best_checkpoint_tracked(self, env_setup):
        config, env = env_setup
        agent = DisjointLinUCB(num_actions=2, feature_dim=DIM, alpha=1.0)
        trainer = LinUCBTrainer(agent=agent, config=config)
        trainer.train(env, total_timesteps=16, eval_interval=8)
        assert trainer.best_A_inv is not None
        assert trainer.best_reward > -float("inf")
