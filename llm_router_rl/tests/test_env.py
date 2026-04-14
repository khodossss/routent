"""Tests for LLMRouterEnv."""

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from llm_router_rl.env.feature_extractor import TfidfFeatureExtractor
from llm_router_rl.env.router_env import LLMRouterEnv
from llm_router_rl.models.llm_pool import BaseLLM


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
    env = LLMRouterEnv(
        benchmark=BENCHMARK, feature_extractor=fe, llm_pool=StubPool(), seed=42,
    )
    return env, fe.feature_dim


class TestEnvCreation:
    def test_env_instantiates(self, setup):
        env, dim = setup
        assert env is not None

    def test_observation_space_shape(self, setup):
        env, dim = setup
        assert env.observation_space.shape == (dim,)

    def test_action_space(self, setup):
        env, _ = setup
        assert env.action_space.n == 2


class TestEnvReset:
    def test_reset_returns_correct_shape(self, setup):
        env, dim = setup
        obs, info = env.reset()
        assert obs.shape == (dim,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)


class TestEnvStep:
    def test_step_returns_five_values(self, setup):
        env, _ = setup
        env.reset()
        assert len(env.step(0)) == 5

    def test_step_reward_is_float(self, setup):
        env, _ = setup
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert isinstance(reward, float)

    def test_step_terminated(self, setup):
        env, _ = setup
        env.reset()
        _, _, terminated, truncated, _ = env.step(0)
        assert terminated is True
        assert truncated is False

    def test_all_actions_valid(self, setup):
        env, dim = setup
        for action in range(env.action_space.n):
            env.reset()
            obs, reward, _, _, info = env.step(action)
            assert obs.shape == (dim,)
            assert isinstance(reward, float)


class TestEnvInfoDict:
    def test_required_keys(self, setup):
        env, _ = setup
        env.reset()
        _, _, _, _, info = env.step(0)
        for key in ("correct", "model_used", "latency_ms", "cost", "category", "question_id"):
            assert key in info

    def test_types(self, setup):
        env, _ = setup
        env.reset()
        _, _, _, _, info = env.step(0)
        assert isinstance(info["correct"], bool)
        assert isinstance(info["latency_ms"], float)


class TestEnvCumulativeStats:
    def test_total_steps(self, setup):
        env, _ = setup
        env.reset()
        for _ in range(5):
            env.step(0)
        assert env.total_steps == 5
