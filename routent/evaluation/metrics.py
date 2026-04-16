"""Training metrics tracker for LLM Router RL."""

from collections import defaultdict
from typing import Dict, List


class MetricsTracker:
    """Tracks and aggregates training metrics at step and update level."""

    def __init__(self) -> None:
        self.rewards: List[float] = []
        self.qualities: List[float] = []
        self.costs: List[float] = []
        self.latencies: List[float] = []
        self.models_used: List[int] = []
        self.categories: List[str] = []
        self.question_ids: List[int] = []

        self.update_logs: List[Dict[str, float]] = []

    def log_step(self, info: dict, reward: float) -> None:
        self.rewards.append(reward)
        self.qualities.append(info["quality"])
        self.costs.append(info["cost"])
        self.latencies.append(info["latency_ms"])
        self.models_used.append(info["model_used"])
        self.categories.append(info.get("category", ""))
        self.question_ids.append(info.get("question_id", 0))

    def log_update(self, loss_stats: dict) -> None:
        self.update_logs.append(loss_stats)

    def get_summary(self, last_n: int = 100) -> dict:
        n = min(last_n, len(self.rewards))
        if n == 0:
            return {
                "avg_reward": 0.0,
                "avg_quality": 0.0,
                "avg_cost": 0.0,
                "avg_latency": 0.0,
                "per_model_usage": {},
            }

        recent_rewards = self.rewards[-n:]
        recent_qualities = self.qualities[-n:]
        recent_costs = self.costs[-n:]
        recent_latencies = self.latencies[-n:]
        recent_models = self.models_used[-n:]

        avg_reward = sum(recent_rewards) / n
        avg_quality = sum(recent_qualities) / n
        avg_cost = sum(recent_costs) / n
        avg_latency = sum(recent_latencies) / n

        model_counts: Dict[int, int] = defaultdict(int)
        for m in recent_models:
            model_counts[m] += 1
        per_model_usage = {m: count / n for m, count in sorted(model_counts.items())}

        return {
            "avg_reward": avg_reward,
            "avg_quality": avg_quality,
            "avg_cost": avg_cost,
            "avg_latency": avg_latency,
            "per_model_usage": per_model_usage,
        }

    def get_full_history(self) -> dict:
        return {
            "steps": {
                "rewards": self.rewards,
                "qualities": self.qualities,
                "costs": self.costs,
                "latencies": self.latencies,
                "models_used": self.models_used,
                "categories": self.categories,
                "question_ids": self.question_ids,
            },
            "updates": self.update_logs,
        }
