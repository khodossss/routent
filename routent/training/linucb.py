"""Disjoint LinUCB contextual bandit for LLM routing.

Replaces PPO for the contextual bandit setting (horizon=1).

Each arm corresponds to one LLM model. For arm a:
    theta_a  = A_a^{-1} b_a          (estimated reward weights)
    UCB score = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)

A_inv is maintained via Sherman-Morrison updates — O(d^2) per step vs O(d^3)
for a full matrix inversion.
"""

import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from routent.config import Config
from routent.evaluation.metrics import MetricsTracker


def _fmt_cost(cost: float) -> str:
    if cost == 0:
        return "0"
    if abs(cost) >= 1e-4:
        return f"{cost:.4f}"
    return f"{cost:.2e}"


class DisjointLinUCB:
    """Disjoint Linear Upper Confidence Bound contextual bandit.

    One independent linear model per arm. Updates are online (one sample at a
    time) but can be applied in any order since arms are disjoint.

    Args:
        num_actions: Number of arms (LLM models).
        feature_dim: Dimensionality of the context vector.
        alpha: Exploration parameter. Larger → more exploration.
    """

    def __init__(self, num_actions: int, feature_dim: int, alpha: float = 1.0) -> None:
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.alpha = alpha

        # A_inv[a] = (I + X_a^T X_a)^{-1}, initialised to identity.
        # b[a]     = sum of reward * x for arm a.
        self.A_inv: List[np.ndarray] = [
            np.eye(feature_dim, dtype=np.float64) for _ in range(num_actions)
        ]
        self.b: List[np.ndarray] = [
            np.zeros(feature_dim, dtype=np.float64) for _ in range(num_actions)
        ]

    # ------------------------------------------------------------------
    # Core bandit interface
    # ------------------------------------------------------------------

    def select_action(self, x: np.ndarray) -> int:
        """Return the arm with the highest UCB score.

        Ties are broken with a tiny uniform perturbation so that all arms are
        visited uniformly during cold-start (when b == 0 for every arm).
        """
        x = np.asarray(x, dtype=np.float64)
        scores = np.empty(self.num_actions)

        for a in range(self.num_actions):
            A_inv = self.A_inv[a]
            theta = A_inv @ self.b[a]
            variance = float(x @ A_inv @ x)
            bonus = self.alpha * np.sqrt(max(0.0, variance))
            scores[a] = theta @ x + bonus

        # Random tie-breaking (critical for uniform cold-start exploration)
        scores += np.random.uniform(0.0, 1e-9, self.num_actions)
        return int(np.argmax(scores))

    def update(self, action: int, x: np.ndarray, reward: float) -> None:
        """Update arm parameters with one observed (x, reward) pair.

        Uses the Sherman-Morrison rank-1 update to maintain A_inv directly:
            A_new     = A + x x^T
            A_inv_new = A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)
        """
        x = np.asarray(x, dtype=np.float64)
        A_inv = self.A_inv[action]
        Ax = A_inv @ x                    # (d,)
        denom = 1.0 + float(x @ Ax)      # scalar > 0
        self.A_inv[action] = A_inv - np.outer(Ax, Ax) / denom
        self.b[action] = self.b[action] + reward * x

    def greedy_action(self, x: np.ndarray) -> int:
        """Return the arm with the highest *expected* reward (no UCB bonus).

        Used at evaluation time so exploration noise doesn't distort metrics.
        """
        x = np.asarray(x, dtype=np.float64)
        scores = np.array([self.A_inv[a] @ self.b[a] @ x for a in range(self.num_actions)])
        return int(np.argmax(scores))

    # ------------------------------------------------------------------
    # Compatibility shim for evaluate.py (same call signature as
    # PolicyNetwork.get_action)
    # ------------------------------------------------------------------

    def get_action(self, obs) -> Tuple[int, None, None]:
        """Greedy action (no UCB bonus) — used during evaluation.

        Accepts torch.Tensor or np.ndarray.
        """
        if hasattr(obs, "numpy"):
            x = obs.numpy()
        else:
            x = np.asarray(obs)
        return self.greedy_action(x), None, None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "linucb_A_inv": self.A_inv,
                "linucb_b": self.b,
                "linucb_alpha": self.alpha,
                "num_actions": self.num_actions,
                "feature_dim": self.feature_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "DisjointLinUCB":
        data = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(
            num_actions=data["num_actions"],
            feature_dim=data["feature_dim"],
            alpha=data.get("linucb_alpha", data.get("alpha", 1.0)),
        )
        agent.A_inv = data["linucb_A_inv"]
        agent.b = data["linucb_b"]
        return agent


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LinUCBTrainer:
    """Online training loop for DisjointLinUCB.

    Supports two execution modes selected automatically:
    - Sequential: one API call per step (simulated / Ollama local models).
    - Batched:    groups calls by model using batch_generate (cloud APIs).
      Uses config.rollout_steps as the batch size.
    """

    def __init__(self, agent: DisjointLinUCB, config: Config) -> None:
        self.agent = agent
        self.config = config
        self.metrics = MetricsTracker()
        self.rollout_records: List[Dict] = []

        # Best checkpoint
        self.best_reward: float = -float("inf")
        self.best_step: int = 0
        self.best_A_inv: Optional[List[np.ndarray]] = None
        self.best_b: Optional[List[np.ndarray]] = None
        self.best_summary: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(
        self, env, total_timesteps: int, eval_interval: int = 200
    ) -> List[Dict]:
        if self._has_batch_support(env):
            return self._train_batched(env, total_timesteps, eval_interval)
        return self._train_sequential(env, total_timesteps, eval_interval)

    # ------------------------------------------------------------------
    # Sequential mode
    # ------------------------------------------------------------------

    def _train_sequential(
        self, env, total_timesteps: int, eval_interval: int
    ) -> List[Dict]:
        history: List[Dict] = []
        obs, _ = env.reset()

        for step in range(1, total_timesteps + 1):
            x = np.asarray(obs, dtype=np.float64)
            item = env._current_item
            action = self.agent.select_action(x)

            next_obs, reward, _terminated, _truncated, info = env.step(action)
            self.agent.update(action, x, reward)
            self.metrics.log_step(info, reward)

            model_name = (
                self.config.model_names[action]
                if action < len(self.config.model_names)
                else str(action)
            )
            self.rollout_records.append(
                {
                    "question": item["question"],
                    "correct_answer": item["answer"],
                    "model_index": action,
                    "model_name": model_name,
                    "correct": info["correct"],
                    "latency_ms": info["latency_ms"],
                    "cost": info["cost"],
                    "reward": reward,
                }
            )

            obs = next_obs

            if step % eval_interval == 0:
                summary = self.metrics.get_summary(last_n=eval_interval)
                summary["step"] = step
                history.append(summary)
                self._maybe_save_best(step, summary)
                self._log_progress(step, total_timesteps, summary)
                self._save_rollout_records(step)
                self.rollout_records = []

        return history

    # ------------------------------------------------------------------
    # Batched mode (cloud APIs)
    # ------------------------------------------------------------------

    def _train_batched(
        self, env, total_timesteps: int, eval_interval: int
    ) -> List[Dict]:
        history: List[Dict] = []
        steps_done = 0
        batch_size = self.config.rollout_steps

        while steps_done < total_timesteps:
            n = min(batch_size, total_timesteps - steps_done)
            rng = env._rng

            # 1. Sample questions
            indices = rng.integers(0, len(env.benchmark), size=n)
            items = [env.benchmark[idx] for idx in indices]

            # 2. Extract features + select actions (all offline, no API calls yet)
            obs_list: List[np.ndarray] = []
            actions: List[int] = []
            for item in items:
                x = env.feature_extractor.transform(item["question"]).numpy()
                obs_list.append(x)
                actions.append(self.agent.select_action(x.astype(np.float64)))

            # 3. Batch API calls grouped by chosen model
            model_groups: Dict[int, List[int]] = {}
            for i, action in enumerate(actions):
                model_groups.setdefault(action, []).append(i)

            responses: List[Optional[Tuple[str, float, float]]] = [None] * n
            for model_idx, step_indices in model_groups.items():
                model = env.llm_pool.get_model(model_idx)
                prompts = [items[i]["question"] for i in step_indices]
                batch_results = model.batch_generate(prompts)
                for local_i, step_i in enumerate(step_indices):
                    responses[step_i] = batch_results[local_i]

            # 4. Update agent + log
            for i in range(n):
                item = items[i]
                predicted, latency_ms, cost = responses[i]
                x = obs_list[i].astype(np.float64)

                correct = env._evaluate_answer(predicted, item["answer"], item=item)
                reward = env._compute_reward(correct, latency_ms, cost)

                self.agent.update(actions[i], x, reward)

                info = {
                    "correct": correct,
                    "model_used": actions[i],
                    "latency_ms": latency_ms,
                    "cost": cost,
                    "category": item.get("category", ""),
                    "question_id": item.get("id", i),
                }
                self.metrics.log_step(info, reward)

                model_name = (
                    self.config.model_names[actions[i]]
                    if actions[i] < len(self.config.model_names)
                    else str(actions[i])
                )
                self.rollout_records.append(
                    {
                        "question": item["question"],
                        "correct_answer": item["answer"],
                        "predicted_answer": predicted,
                        "model_index": actions[i],
                        "model_name": model_name,
                        "correct": correct,
                        "latency_ms": latency_ms,
                        "cost": cost,
                        "reward": reward,
                    }
                )

            steps_done += n

            if steps_done % eval_interval < batch_size:
                summary = self.metrics.get_summary(last_n=eval_interval)
                summary["step"] = steps_done
                history.append(summary)
                self._maybe_save_best(steps_done, summary)
                self._log_progress(steps_done, total_timesteps, summary)
                self._save_rollout_records(steps_done)
                self.rollout_records = []

        return history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_batch_support(self, env) -> bool:
        return hasattr(env.llm_pool.get_model(0), "batch_generate")

    def _maybe_save_best(self, step: int, summary: Dict) -> None:
        avg_reward = summary.get("avg_reward", -np.inf)
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.best_step = step
            self.best_A_inv = [A.copy() for A in self.agent.A_inv]
            self.best_b = [b.copy() for b in self.agent.b]
            self.best_summary = copy.copy(summary)

    def _log_progress(self, step: int, total: int, summary: Dict) -> None:
        usage = summary.get("per_model_usage", {})
        usage_str = ", ".join(
            f"{i}:{pct:.0%}" for i, pct in sorted(usage.items())
        )
        print(
            f"Step {step}/{total} | "
            f"Reward: {summary['avg_reward']:.3f} | "
            f"Acc: {summary['accuracy']:.3f} | "
            f"Cost: {_fmt_cost(summary['avg_cost'])} | "
            f"Latency: {summary['avg_latency']:.0f}ms | "
            f"Models: [{usage_str}]"
        )

    def _save_rollout_records(self, step: int) -> None:
        os.makedirs(self.config.results_dir, exist_ok=True)
        path = os.path.join(self.config.results_dir, f"eval_step_{step}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.rollout_records, f, indent=2, ensure_ascii=False, default=str)
        print(f"  → Saved {len(self.rollout_records)} records to {path}")
