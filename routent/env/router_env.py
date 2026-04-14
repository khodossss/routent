"""Gymnasium environment for LLM routing as a contextual bandit."""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from routent.env.feature_extractor import BaseFeatureExtractor
from routent.evaluation.evaluator import Evaluator


class LLMRouterEnv(gym.Env):
    """Contextual bandit environment for LLM routing.

    Each step is one independent routing decision (horizon=1).
    Observation: feature vector of the current prompt.
    Action: discrete index of the model to use.
    Reward: K_accuracy * correct - K_latency * norm_latency - K_cost * norm_cost
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        benchmark: List[dict],
        feature_extractor: BaseFeatureExtractor,
        llm_pool,
        K_accuracy: float = 1.0,
        K_latency: float = 0.3,
        K_cost: float = 0.3,
        latency_range: Tuple[float, float] = (100.0, 5000.0),
        cost_range: Tuple[float, float] = (0.0, 0.001),
        eval_mode: str = "exact",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.benchmark = benchmark
        self.feature_extractor = feature_extractor
        self.llm_pool = llm_pool
        self.K_accuracy = K_accuracy
        self.K_latency = K_latency
        self.K_cost = K_cost
        self.eval_mode = eval_mode

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_extractor.feature_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(llm_pool.num_models)

        # Normalization ranges from config (user-defined)
        self._min_latency = latency_range[0]
        self._max_latency = latency_range[1]
        self._min_cost = cost_range[0]
        self._max_cost = cost_range[1]

        # Internal state
        self._rng = np.random.default_rng(seed)
        self._current_item: Optional[dict] = None
        self._current_obs: Optional[torch.Tensor] = None

        # Cumulative stats
        self.total_steps = 0
        self.total_correct = 0
        self.total_cost = 0.0
        self.model_usage_count = np.zeros(llm_pool.num_models, dtype=np.int64)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        idx = self._rng.integers(0, len(self.benchmark))
        self._current_item = self.benchmark[idx]
        self._current_obs = self.feature_extractor.transform(
            self._current_item["question"]
        )
        return self._current_obs.numpy(), {}

    def _evaluate_answer(self, predicted: str, ground_truth: str) -> bool:
        if self.eval_mode == "numeric":
            return Evaluator.numeric_match(predicted, ground_truth)
        elif self.eval_mode == "fuzzy":
            return Evaluator.fuzzy_match(predicted, ground_truth)
        else:
            return Evaluator.exact_match(predicted, ground_truth)

    def _compute_reward(self, correct: bool, latency_ms: float, cost: float) -> float:
        lat_range = self._max_latency - self._min_latency
        cost_range = self._max_cost - self._min_cost
        norm_latency = (latency_ms - self._min_latency) / lat_range if lat_range > 0 else 0.0
        norm_cost = (cost - self._min_cost) / cost_range if cost_range > 0 else 0.0
        norm_latency = max(0.0, min(1.0, norm_latency))
        norm_cost = max(0.0, min(1.0, norm_cost))

        return (
            self.K_accuracy * float(correct)
            - self.K_latency * norm_latency
            - self.K_cost * norm_cost
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        item = self._current_item
        model = self.llm_pool.get_model(action)

        predicted, latency_ms, cost = model.generate(
            prompt=item["question"],
            answer=item["answer"],
            difficulty=item.get("difficulty", ""),
            category=item.get("category", ""),
        )

        correct = self._evaluate_answer(predicted, item["answer"])
        reward = self._compute_reward(correct, latency_ms, cost)

        # Update cumulative stats
        self.total_steps += 1
        self.total_correct += int(correct)
        self.total_cost += cost
        self.model_usage_count[action] += 1

        info = {
            "correct": correct,
            "model_used": action,
            "latency_ms": latency_ms,
            "cost": cost,
            "category": item.get("category", ""),
            "question_id": item.get("id", 0),
        }

        # Contextual bandit: every step is terminal
        terminated = True
        truncated = False

        # Auto-reset: pick next question for the next step
        next_idx = self._rng.integers(0, len(self.benchmark))
        self._current_item = self.benchmark[next_idx]
        self._current_obs = self.feature_extractor.transform(
            self._current_item["question"]
        )
        next_obs = self._current_obs.numpy()

        return next_obs, reward, terminated, truncated, info
