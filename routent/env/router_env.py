"""Gymnasium environment for LLM routing as a contextual bandit."""

import json as _json
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
    Reward: K_quality * correct - K_latency * norm_latency - K_cost * norm_cost
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        benchmark: List[dict],
        feature_extractor: BaseFeatureExtractor,
        llm_pool,
        K_quality: float = 1.0,
        K_latency: float = 0.3,
        K_cost: float = 0.3,
        latency_range: Tuple[float, float] = (100.0, 5000.0),
        cost_range: Tuple[float, float] = (0.0, 0.001),
        eval_config: Optional[Dict[str, Any]] = None,
        judge=None,
        semantic_embedder=None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.benchmark = benchmark
        self.feature_extractor = feature_extractor
        self.llm_pool = llm_pool
        self.K_quality = K_quality
        self.K_latency = K_latency
        self.K_cost = K_cost
        self.eval_config = eval_config or {}
        self.eval_mode = self.eval_config.get("mode", "exact")
        self.judge = judge
        self.semantic_embedder = semantic_embedder

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
        self._last_eval_details: Dict[str, Any] = {}

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

    def _apply_threshold(self, score: float) -> float:
        """If binary_threshold is set in eval_config, binarize the score."""
        threshold = self.eval_config.get("binary_threshold")
        if threshold is not None:
            return 1.0 if score >= float(threshold) else 0.0
        return score

    def _evaluate_answer(self, predicted: str, ground_truth: str, item: Optional[dict] = None) -> float:
        """Return quality score in [0, 1].

        Also populates ``self._last_eval_details`` with mode-specific
        diagnostic info for transparency/debugging in saved JSON records.
        """
        item = item or self._current_item or {}
        mode = self.eval_mode
        cfg = self.eval_config
        details: Dict[str, Any] = {"eval_mode": mode}

        if mode == "numeric":
            score = float(Evaluator.numeric_match(predicted, ground_truth))

        elif mode == "fuzzy":
            raw = Evaluator.fuzzy_score(predicted, ground_truth)
            details["ratio"] = round(raw, 4)
            score = self._apply_threshold(raw)

        elif mode == "classification":
            if cfg.get("output_format") == "confidence_json":
                raw = Evaluator.classification_confidence_score(predicted, ground_truth)
                details["parsed_probs"] = Evaluator._parse_confidence_json(predicted)
                details["target_class"] = ground_truth.strip()
                details["target_confidence"] = round(raw, 4)
                score = self._apply_threshold(raw)
            else:
                score = float(Evaluator.classification_match(
                    predicted, ground_truth, choices=cfg.get("choices") or item.get("choices"),
                ))

        elif mode == "multilabel":
            delim = cfg.get("multilabel_delimiter", ",")
            raw = Evaluator.multilabel_score(predicted, ground_truth, delimiter=delim)
            pred_set = sorted(t.strip().lower() for t in predicted.split(delim) if t.strip())
            gt_set = sorted(t.strip().lower() for t in ground_truth.split(delim) if t.strip())
            details["predicted_labels"] = pred_set
            details["expected_labels"] = gt_set
            details["jaccard"] = round(raw, 4)
            score = self._apply_threshold(raw)

        elif mode == "regression":
            tolerance = item.get("tolerance", cfg.get("regression_tolerance", 0.1))
            raw = Evaluator.regression_score(predicted, ground_truth, tolerance=tolerance)
            details["tolerance"] = tolerance
            details["score"] = round(raw, 4)
            score = self._apply_threshold(raw)

        elif mode == "semantic":
            embedder = self.semantic_embedder
            if embedder is None:
                self._last_eval_details = details
                return 0.0
            raw = Evaluator.semantic_score(predicted, ground_truth, embedder)
            details["cosine_similarity"] = round(raw, 4)
            score = self._apply_threshold(raw)

        elif mode == "llm_judge":
            if self.judge is None:
                self._last_eval_details = details
                return 0.0
            criteria = cfg.get("judge_criteria")
            score = Evaluator.llm_judge_score(
                predicted, ground_truth, self.judge,
                question=item.get("question", ""), criteria=criteria,
            )
            details["judge_details"] = getattr(self.judge, "_last_details", {})

        else:  # "exact"
            score = float(Evaluator.exact_match(predicted, ground_truth))

        self._last_eval_details = details
        return score

    def _compute_reward(self, quality: float, latency_ms: float, cost: float) -> float:
        lat_range = self._max_latency - self._min_latency
        cost_range = self._max_cost - self._min_cost
        norm_latency = (latency_ms - self._min_latency) / lat_range if lat_range > 0 else 0.0
        norm_cost = (cost - self._min_cost) / cost_range if cost_range > 0 else 0.0
        norm_latency = max(0.0, min(1.0, norm_latency))
        norm_cost = max(0.0, min(1.0, norm_cost))

        return (
            self.K_quality * quality
            - self.K_latency * norm_latency
            - self.K_cost * norm_cost
        )

    def _build_prompt(self, item: dict) -> str:
        """Append eval-specific format instructions to the question.

        Priority: explicit prompt_suffix > auto-generated from mode + choices > raw question.
        """
        question = item["question"]
        cfg = self.eval_config
        suffix = cfg.get("prompt_suffix")

        if suffix is None:
            suffix = self._auto_suffix(item)

        if suffix:
            return f"{question}\n\n{suffix}"
        return question

    def _auto_suffix(self, item: dict) -> Optional[str]:
        """Generate format instructions based on eval mode."""
        cfg = self.eval_config
        mode = self.eval_mode

        if mode == "numeric":
            return "Respond with ONLY the final numeric answer. No explanation, no units, no words."

        if mode == "regression":
            return "Respond with ONLY a single number."

        if mode == "classification":
            choices = cfg.get("choices") or item.get("choices") or []
            if not choices:
                return None
            if cfg.get("output_format") == "confidence_json":
                example = {c: round(1.0 / len(choices), 2) for c in choices}
                return (
                    "Respond with ONLY a JSON object mapping each class to its probability (0-1).\n"
                    f"Classes: {', '.join(choices)}\n"
                    f"Example: {_json.dumps(example)}"
                )
            return f"Answer with ONLY one of: {', '.join(choices)}"

        if mode == "multilabel":
            delim = cfg.get("multilabel_delimiter", ",")
            return f"List all applicable labels, separated by '{delim}'."

        return None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        item = self._current_item
        model = self.llm_pool.get_model(action)

        prompt = self._build_prompt(item)
        predicted, latency_ms, cost = model.generate(
            prompt=prompt,
            answer=item["answer"],
            difficulty=item.get("difficulty", ""),
            category=item.get("category", ""),
        )

        correct = self._evaluate_answer(predicted, item["answer"], item=item)
        reward = self._compute_reward(correct, latency_ms, cost)

        # Update cumulative stats
        self.total_steps += 1
        self.total_correct += int(correct)
        self.total_cost += cost
        self.model_usage_count[action] += 1

        info = {
            "quality": correct,
            "model_used": action,
            "latency_ms": latency_ms,
            "cost": cost,
            "category": item.get("category", ""),
            "question_id": item.get("id", 0),
            "eval_details": self._last_eval_details,
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
