from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ── LLM models ───────────────────────────────────────────────────
    # List of (provider, model_name) pairs. Supported providers:
    #   "openai", "google", "anthropic", "ollama"
    provider_models: List[Tuple[str, str]] = field(default_factory=list)

    # Cost per 1M tokens for each model (same order as provider_models)
    model_costs_per_1m_input: List[float] = field(default_factory=list)
    model_costs_per_1m_output: List[float] = field(default_factory=list)

    # Extra kwargs per model passed to LangChain constructor
    # E.g. [{"reasoning": "minimal"}, {}]
    model_kwargs: List[dict] = field(default_factory=list)

    # System prompt sent to all models
    system_prompt: str = "Answer with ONLY the final numeric answer. No explanation, no units, no words. Just the number."

    # ── Dataset ──────────────────────────────────────────────────────
    # HuggingFace dataset identifier, e.g. "openai/gsm8k"
    dataset: str = ""
    dataset_split: str = "train"
    train_size: int = 200
    test_size: int = 50

    # Evaluation mode: "exact", "numeric" (for math), "fuzzy"
    eval_mode: str = "exact"

    # ── Reward weights ───────────────────────────────────────────────
    # reward = K_accuracy * correct - K_latency * norm_latency - K_cost * norm_cost
    K_accuracy: float = 1.0
    K_latency: float = 0.3
    K_cost: float = 0.3

    # ── Normalization ranges ─────────────────────────────────────────
    latency_range: Tuple[float, float] = (100.0, 5000.0)   # (min_ms, max_ms)
    cost_range: Tuple[float, float] = (0.0, 0.001)          # (min_$, max_$)

    # ── PPO hyperparameters (contextual bandit) ──────────────────────
    lr: float = 3e-4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    rollout_steps: int = 200

    # ── Feature extraction ───────────────────────────────────────────
    tfidf_max_features: int = 100
    handcrafted_features: int = 10
    total_feature_dim: int = 110

    # ── Model pool (derived at runtime) ──────────────────────────────
    num_models: int = 0
    model_names: List[str] = field(default_factory=list)

    # ── Training ─────────────────────────────────────────────────────
    total_timesteps: int = 5000
    eval_interval: int = 200
    seed: int = 42

    # ── Paths ────────────────────────────────────────────────────────
    checkpoint_dir: str = "llm_router_rl/checkpoints"
    results_dir: str = "llm_router_rl/results"
