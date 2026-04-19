"""Main training script for the LLM Router RL system."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Suppress noisy third-party warnings (must be FIRST, before transformers imports)
import routent.utils.silence  # noqa: F401

import argparse
import json

import numpy as np
import torch

from routent.config import Config
from routent.data.dataset_loader import load_benchmark
from routent.env.feature_extractor import SentenceEmbeddingFeatureExtractor
from routent.env.router_env import LLMRouterEnv
from routent.models.pool import LLMPool
from routent.training.linucb import DisjointLinUCB, LinUCBTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LLM Router policy")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--K_quality", type=float, default=None)
    parser.add_argument("--K_latency", type=float, default=None)
    parser.add_argument("--K_cost", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def load_config(args) -> Config:
    config = Config()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    for k, v in cfg_dict.items():
        if hasattr(config, k):
            if k == "provider_models":
                v = [tuple(x) for x in v]
            elif k in ("latency_range", "cost_range"):
                v = tuple(v)
            setattr(config, k, v)

    # Migrate legacy eval_mode / judge_config into eval_config
    if hasattr(config, "eval_mode") and config.eval_mode != "exact":
        ec = config.eval_config
        ec.setdefault("mode", config.eval_mode)
    if hasattr(config, "judge_config") and config.judge_config:
        ec = config.eval_config
        jc = config.judge_config
        ec.setdefault("judge_provider", jc.get("provider"))
        ec.setdefault("judge_model", jc.get("model_name"))
        if jc.get("prompt_template"):
            ec.setdefault("judge_prompt_template", jc["prompt_template"])

    # CLI overrides
    for field in ("total_timesteps", "seed", "K_quality", "K_latency", "K_cost", "lr"):
        val = getattr(args, field, None)
        if val is not None:
            setattr(config, field, val)

    return config


class _TeeWriter:
    """Duplicate stdout to both the terminal and a log file."""

    def __init__(self, log_path: str):
        self._terminal = sys.stdout
        self._log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)
        self._log.flush()

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def isatty(self):
        return False

    def close(self):
        self._log.close()


def main() -> None:
    args = parse_args()
    config = load_config(args)

    # Load .env (API keys + HF_TOKEN)
    from dotenv import load_dotenv
    load_dotenv()

    # Build run directory: {output_root}/{config_name}_{timestamp}/
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    run_dir = os.path.join(config.output_root, f"{timestamp}_{config_name}")
    config.results_dir = os.path.join(run_dir, "evaluations")
    config.checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Tee all console output to report.txt
    report_path = os.path.join(run_dir, "report.txt")
    tee = _TeeWriter(report_path)
    sys.stdout = tee

    print(f"Run dir: {run_dir}")

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load benchmark
    benchmark_train, _ = load_benchmark(config)
    print(f"Loaded {len(benchmark_train)} training questions")

    # Fit feature extractor
    feature_extractor = SentenceEmbeddingFeatureExtractor(
        model_name=config.embedding_model,
        device=config.embedding_device,
        pca_dim=config.pca_dim,
        prepend_bias=config.prepend_bias,
    )
    corpus = [item["question"] for item in benchmark_train]
    feature_extractor.fit(corpus)
    config.total_feature_dim = feature_extractor.feature_dim
    print(
        f"Feature dim: {config.total_feature_dim} "
        f"(raw {feature_extractor.raw_dim}, "
        f"pca_dim={config.pca_dim}, prepend_bias={config.prepend_bias})"
    )

    # Create LLM pool
    llm_pool = LLMPool(
        provider_models=[tuple(pm) for pm in config.provider_models],
        model_costs_input=config.model_costs_per_1m_input or None,
        model_costs_output=config.model_costs_per_1m_output or None,
        model_kwargs=config.model_kwargs or None,
        model_concurrency=config.model_concurrency or None,
        model_labels=config.model_labels or None,
        system_prompt=config.system_prompt,
    )
    config.num_models = llm_pool.num_models
    config.model_names = llm_pool.model_names
    print(f"Models ({config.num_models}): {config.model_names}")

    # Create judge/embedder if eval mode needs them
    eval_cfg = config.eval_config
    eval_mode = eval_cfg.get("mode", "exact")

    judge = None
    if eval_mode == "llm_judge":
        judge_provider = eval_cfg.get("judge_provider")
        judge_model = eval_cfg.get("judge_model")
        if judge_provider and judge_model:
            from routent.evaluation.judges import create_judge_from_config
            judge = create_judge_from_config({
                "provider": judge_provider,
                "model_name": judge_model,
                "prompt_template": eval_cfg.get("judge_prompt_template"),
                "custom_criteria": eval_cfg.get("custom_criteria"),
            })
            print(f"Judge: {judge_provider}/{judge_model}")

    semantic_embedder = None
    if eval_mode == "semantic":
        semantic_model_name = eval_cfg.get("semantic_model")
        if semantic_model_name and semantic_model_name != config.embedding_model:
            from sentence_transformers import SentenceTransformer
            _sem_model = SentenceTransformer(semantic_model_name, device=config.embedding_device)
            semantic_embedder = lambda texts: _sem_model.encode(
                texts, convert_to_tensor=True, show_progress_bar=False
            )
            print(f"Semantic eval embedder: {semantic_model_name}")
        else:
            semantic_embedder = lambda texts: feature_extractor._model.encode(
                texts, convert_to_tensor=True, show_progress_bar=False
            )

    # Create environment
    env = LLMRouterEnv(
        benchmark=benchmark_train,
        feature_extractor=feature_extractor,
        llm_pool=llm_pool,
        K_quality=config.K_quality,
        K_latency=config.K_latency,
        K_cost=config.K_cost,
        latency_range=config.latency_range,
        cost_range=config.cost_range,
        eval_config=eval_cfg,
        judge=judge,
        semantic_embedder=semantic_embedder,
        seed=config.seed,
    )

    # Create LinUCB agent
    agent = DisjointLinUCB(
        num_actions=config.num_models,
        feature_dim=config.total_feature_dim,
        alpha=config.linucb_alpha,
    )
    print(
        f"DisjointLinUCB: {config.num_models} arms × {config.total_feature_dim}-dim, "
        f"alpha={config.linucb_alpha}"
    )

    # Train
    trainer = LinUCBTrainer(agent=agent, config=config)

    print(f"\nTraining: {config.total_timesteps} steps, batch={config.rollout_steps}, alpha={config.linucb_alpha}")
    print(f"  K_quality={config.K_quality}, K_latency={config.K_latency}, K_cost={config.K_cost}")
    print(f"  dataset={config.dataset}, eval_mode={eval_mode}")
    print(f"  latency_range={config.latency_range}, cost_range={config.cost_range}")
    print()

    history = trainer.train(
        env=env,
        total_timesteps=config.total_timesteps,
        eval_interval=config.eval_interval,
    )

    # Save checkpoints
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ckpt_base = {
        "config": vars(config),
        "feature_dim": config.total_feature_dim,
        "num_actions": config.num_models,
        "fe_mean": feature_extractor._mean,
        "fe_std": feature_extractor._std,
        "fe_pca_components": feature_extractor._pca_components,
        "fe_pca_input_mean": feature_extractor._pca_input_mean,
        "fe_prepend_bias": feature_extractor._prepend_bias,
        "fe_pca_dim": feature_extractor._pca_dim,
    }

    final_path = os.path.join(config.checkpoint_dir, "policy_final.pt")
    torch.save(
        {
            **ckpt_base,
            "linucb_A_inv": agent.A_inv,
            "linucb_b": agent.b,
            "linucb_alpha": agent.alpha,
        },
        final_path,
    )
    print(f"\nSaved final checkpoint to {final_path}")

    if trainer.best_A_inv is not None:
        best_path = os.path.join(config.checkpoint_dir, "policy_best.pt")
        torch.save(
            {
                **ckpt_base,
                "linucb_A_inv": trainer.best_A_inv,
                "linucb_b": trainer.best_b,
                "linucb_alpha": agent.alpha,
                "best_reward": trainer.best_reward,
                "best_step": trainer.best_step,
            },
            best_path,
        )
        print(
            f"Saved best checkpoint to {best_path} "
            f"(reward={trainer.best_reward:.4f} at step {trainer.best_step})"
        )

    # Save training log
    os.makedirs(config.results_dir, exist_ok=True)
    log_path = os.path.join(config.results_dir, "training_log.json")
    full_history = trainer.metrics.get_full_history()
    full_history["rollout_summaries"] = history
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=2, default=str)
    print(f"Saved training log to {log_path}")

    # Print summaries
    def print_summary(title, summary):
        print(f"\n  {title}")
        print(f"  {'-' * len(title)}")
        print(f"  Avg reward:  {summary['avg_reward']:.4f}")
        print(f"  Avg quality: {summary['avg_quality']:.4f}")
        cost = summary['avg_cost']
        cost_str = f"{cost:.5f}" if abs(cost) >= 1e-4 or cost == 0 else f"{cost:.2e}"
        print(f"  Avg cost:    {cost_str}")
        print(f"  Avg latency: {summary['avg_latency']:.1f} ms")
        print(f"  Model usage:")
        for m, pct in sorted(summary.get("per_model_usage", {}).items()):
            name = config.model_names[m] if m < len(config.model_names) else str(m)
            print(f"    {name}: {pct:.1%}")

    final_summary = trainer.metrics.get_summary(last_n=1000)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print_summary("Final policy (last 1000 steps)", final_summary)

    if trainer.best_A_inv is not None:
        print_summary(
            f"Best policy (step {trainer.best_step}, reward={trainer.best_reward:.4f})",
            trainer.best_summary,
        )

    print(f"\nFull report saved to {report_path}")
    sys.stdout = tee._terminal
    tee.close()


if __name__ == "__main__":
    main()
