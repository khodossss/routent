"""PPO (Proximal Policy Optimization) trainer for contextual bandit LLM routing."""

import copy
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np

from routent.config import Config
from routent.models.policy_network import PolicyNetwork
from routent.training.buffer import RolloutBuffer
from routent.evaluation.metrics import MetricsTracker


class PPOTrainer:
    """PPO algorithm implemented from scratch for contextual bandit setting."""

    def __init__(self, policy: PolicyNetwork, config: Config) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
        self.buffer = RolloutBuffer(
            capacity=config.rollout_steps,
            obs_dim=config.total_feature_dim,
        )
        self.metrics = MetricsTracker()

        # Best checkpoint tracking
        self.best_reward = -float("inf")
        self.best_step = 0
        self.best_state_dict: Optional[Dict] = None
        self.best_summary: Optional[Dict] = None

        # Detailed per-step records for current rollout (reset each collect)
        self.rollout_records: List[Dict] = []

    def _has_batch_support(self, env) -> bool:
        """Check if the LLM pool supports batched generation."""
        model = env.llm_pool.get_model(0)
        return hasattr(model, "batch_generate")

    def collect_rollout(self, env) -> Dict[str, float]:
        """Collect rollout_steps transitions using the current policy.

        Uses batched API calls when the LLM pool supports it (RealLLM).
        Falls back to sequential calls for simulated mode.

        Returns:
            Dict with summary stats from collection.
        """
        if self._has_batch_support(env):
            return self._collect_rollout_batched(env)
        return self._collect_rollout_sequential(env)

    def _collect_rollout_sequential(self, env) -> Dict[str, float]:
        """Sequential rollout collection."""
        self.buffer.clear()
        self.rollout_records = []
        obs, _ = env.reset()

        for _ in range(self.config.rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action, log_prob, value = self.policy.get_action(obs_tensor)

            item = env._current_item
            next_obs, reward, terminated, truncated, info = env.step(action)

            self.buffer.add(obs_tensor, action, log_prob, reward, value)
            self.metrics.log_step(info, reward)

            model_name = self.config.model_names[action] if action < len(self.config.model_names) else str(action)
            self.rollout_records.append({
                "question": item["question"],
                "correct_answer": item["answer"],
                "predicted_answer": "",
                "model_index": action,
                "model_name": model_name,
                "correct": info["correct"],
                "latency_ms": info["latency_ms"],
                "cost": info["cost"],
                "reward": reward,
            })

            obs = next_obs

        self.buffer.compute_returns_and_advantages()

        summary = self.metrics.get_summary(last_n=self.config.rollout_steps)
        return summary

    def _collect_rollout_batched(self, env) -> Dict[str, float]:
        """Batched rollout collection (real LLM mode).

        1. Sample all questions for the rollout
        2. Extract features, get policy actions for all
        3. Group by model, batch API calls
        4. Evaluate answers, compute rewards, fill buffer
        """
        from routent.evaluation.evaluator import Evaluator

        self.buffer.clear()
        self.rollout_records = []
        rng = env._rng
        n = self.config.rollout_steps

        # 1. Sample questions
        indices = rng.integers(0, len(env.benchmark), size=n)
        items = [env.benchmark[idx] for idx in indices]

        # 2. Get observations and policy decisions
        obs_list = []
        actions = []
        log_probs = []
        values = []

        for item in items:
            obs_tensor = env.feature_extractor.transform(item["question"])
            action, log_prob, value = self.policy.get_action(obs_tensor)
            obs_list.append(obs_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

        # 3. Group by model and batch call
        model_groups: Dict[int, List[int]] = {}  # model_idx -> list of step indices
        for step_idx, action in enumerate(actions):
            model_groups.setdefault(action, []).append(step_idx)

        responses = [None] * n  # (predicted, latency, cost)

        for model_idx, step_indices in model_groups.items():
            model = env.llm_pool.get_model(model_idx)
            prompts = [items[i]["question"] for i in step_indices]

            batch_results = model.batch_generate(prompts)

            for local_idx, step_idx in enumerate(step_indices):
                responses[step_idx] = batch_results[local_idx]

        # 4. Evaluate and fill buffer
        for i in range(n):
            item = items[i]
            predicted, latency_ms, cost = responses[i]

            correct = env._evaluate_answer(predicted, item["answer"])
            reward = env._compute_reward(correct, latency_ms, cost)

            self.buffer.add(obs_list[i], actions[i], log_probs[i], reward, values[i])

            info = {
                "correct": correct,
                "model_used": actions[i],
                "latency_ms": latency_ms,
                "cost": cost,
                "category": item.get("category", ""),
                "question_id": item.get("id", i),
            }
            self.metrics.log_step(info, reward)

            model_name = self.config.model_names[actions[i]] if actions[i] < len(self.config.model_names) else str(actions[i])
            self.rollout_records.append({
                "question": item["question"],
                "correct_answer": item["answer"],
                "predicted_answer": predicted if isinstance(predicted, str) else str(predicted),
                "model_index": actions[i],
                "model_name": model_name,
                "correct": correct,
                "latency_ms": latency_ms,
                "cost": cost,
                "reward": reward,
            })

        self.buffer.compute_returns_and_advantages()

        summary = self.metrics.get_summary(last_n=self.config.rollout_steps)
        return summary

    def update(self) -> Dict[str, float]:
        """Run PPO update on collected rollout data.

        Returns:
            Dict with loss statistics.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        num_batches = 0

        for _ in range(self.config.ppo_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                obs, actions, old_log_probs, returns, advantages = batch

                # Evaluate current policy on the batch
                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    obs, actions
                )

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)

                # Entropy bonus (for exploration)
                entropy_mean = entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                total_loss += loss.item()
                num_batches += 1

        stats = {
            "policy_loss": total_policy_loss / max(num_batches, 1),
            "value_loss": total_value_loss / max(num_batches, 1),
            "entropy": total_entropy / max(num_batches, 1),
            "total_loss": total_loss / max(num_batches, 1),
        }
        self.metrics.log_update(stats)
        return stats

    def train(
        self, env, total_timesteps: int, eval_interval: int = 1000
    ) -> List[Dict]:
        """Main training loop.

        Args:
            env: The LLMRouterEnv gymnasium environment.
            total_timesteps: Total number of environment steps.
            eval_interval: Print summary every this many steps.

        Returns:
            History of summary dicts per rollout collection.
        """
        history: List[Dict] = []
        steps_done = 0

        while steps_done < total_timesteps:
            # Collect rollout
            rollout_stats = self.collect_rollout(env)
            steps_done += self.config.rollout_steps

            # PPO update
            loss_stats = self.update()

            # Combine stats
            combined = {**rollout_stats, **loss_stats, "step": steps_done}
            history.append(combined)

            # Track best checkpoint by avg_reward
            avg_reward = rollout_stats["avg_reward"]
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.best_step = steps_done
                self.best_state_dict = copy.deepcopy(self.policy.state_dict())
                self.best_summary = rollout_stats.copy()

            # Logging + save detailed records
            if steps_done % eval_interval < self.config.rollout_steps:
                usage = rollout_stats.get("per_model_usage", {})
                usage_str = ", ".join(
                    f"{i}:{pct:.0%}" for i, pct in sorted(usage.items())
                )
                print(
                    f"Step {steps_done}/{total_timesteps} | "
                    f"Reward: {rollout_stats['avg_reward']:.3f} | "
                    f"Acc: {rollout_stats['accuracy']:.3f} | "
                    f"Cost: {rollout_stats['avg_cost']:.4f} | "
                    f"Latency: {rollout_stats['avg_latency']:.0f}ms | "
                    f"Models: [{usage_str}]"
                )

                # Save detailed records JSON
                self._save_rollout_records(steps_done)

        return history

    def _save_rollout_records(self, step: int) -> None:
        """Save detailed per-question records to JSON."""
        import json
        os.makedirs(self.config.results_dir, exist_ok=True)
        path = os.path.join(self.config.results_dir, f"rollout_step_{step}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.rollout_records, f, indent=2, ensure_ascii=False, default=str)
        print(f"  → Saved {len(self.rollout_records)} records to {path}")
