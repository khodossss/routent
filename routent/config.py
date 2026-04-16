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

    # Max concurrent calls per model (1 = sequential).
    # Use >1 for vendor APIs (OpenAI, Google), 1 for local CPU models.
    # E.g. [10, 10, 1, 1] for 2 APIs + 2 local models
    model_concurrency: List[int] = field(default_factory=list)

    # Candidate labels per model (used by hf_zero_shot).
    # Each element is a list of labels, or None for models that don't need them.
    # E.g. [null, ["positive", "negative"], null]
    model_labels: List = field(default_factory=list)

    # System prompt sent to all models (format instructions are auto-generated
    # from eval_config, so this should be generic or empty)
    system_prompt: str = ""

    # ── Dataset ──────────────────────────────────────────────────────
    # HuggingFace dataset identifier, e.g. "openai/gsm8k"
    dataset: str = ""
    dataset_split: str = "train"
    train_size: int = 200
    test_size: int = 50

    # ── Evaluation ────────────────────────────────────────────────────
    # All evaluation settings live in one dict. Required key:
    #   "mode": "exact"|"fuzzy"|"numeric"|"classification"|"multilabel"|
    #           "regression"|"semantic"|"llm_judge"
    #
    # Mode-specific keys:
    #   choices, output_format, multilabel_delimiter, regression_tolerance,
    #   binary_threshold, prompt_suffix, semantic_model,
    #   judge_criteria (list or dict), judge_provider, judge_model,
    #   judge_prompt_template, custom_criteria
    eval_config: dict = field(default_factory=dict)

    # ── Reward weights ───────────────────────────────────────────────
    # reward = K_quality * correct - K_latency * norm_latency - K_cost * norm_cost
    K_quality: float = 1.0
    K_latency: float = 0.3
    K_cost: float = 0.3

    # ── Normalization ranges ─────────────────────────────────────────
    latency_range: Tuple[float, float] = (100.0, 5000.0)   # (min_ms, max_ms)
    cost_range: Tuple[float, float] = (0.0, 0.001)          # (min_$, max_$)

    # ── LinUCB hyperparameters ───────────────────────────────────────
    # alpha controls exploration: larger → tries all models more aggressively.
    # Typical range: 0.1 (exploit) … 2.0 (explore). Default 1.0 is a safe start.
    linucb_alpha: float = 1.0

    # rollout_steps is reused as the batch size in batched (cloud API) mode.
    rollout_steps: int = 200

    # ── Feature extraction ───────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    total_feature_dim: int = 384  # auto-set after fit

    # ── Model pool (derived at runtime) ──────────────────────────────
    num_models: int = 0
    model_names: List[str] = field(default_factory=list)

    # ── Training ─────────────────────────────────────────────────────
    total_timesteps: int = 5000
    eval_interval: int = 200
    seed: int = 42

    # ── Paths (resolved at runtime based on config name + timestamp) ─
    # Base output directory; actual run goes into {output_root}/{config_name}_{timestamp}/
    output_root: str = "results"
    checkpoint_dir: str = ""  # auto-set: {output_root}/{run_name}/checkpoints
    results_dir: str = ""     # auto-set: {output_root}/{run_name}/evaluations
