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
from routent.env.feature_extractor import TfidfFeatureExtractor, SentenceEmbeddingFeatureExtractor
from routent.env.router_env import LLMRouterEnv
from routent.models.real_llm import RealLLMPool
from routent.models.policy_network import PolicyNetwork
from routent.training.ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LLM Router policy")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--K_accuracy", type=float, default=None)
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

    # CLI overrides
    for field in ("total_timesteps", "seed", "K_accuracy", "K_latency", "K_cost", "lr"):
        val = getattr(args, field, None)
        if val is not None:
            setattr(config, field, val)

    return config


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
    run_dir = os.path.join(config.output_root, f"{config_name}_{timestamp}")
    config.results_dir = os.path.join(run_dir, "evaluations")
    config.checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load benchmark
    benchmark_train, _ = load_benchmark(config)
    print(f"Loaded {len(benchmark_train)} training questions")

    # Fit feature extractor
    if config.feature_extractor == "embeddings":
        feature_extractor = SentenceEmbeddingFeatureExtractor(
            model_name=config.embedding_model,
            device=config.embedding_device,
        )
    else:
        feature_extractor = TfidfFeatureExtractor(
            tfidf_max_features=config.tfidf_max_features
        )
    corpus = [item["question"] for item in benchmark_train]
    feature_extractor.fit(corpus)
    config.total_feature_dim = feature_extractor.feature_dim
    print(f"Feature dim: {config.total_feature_dim}")

    # Create LLM pool
    llm_pool = RealLLMPool(
        provider_models=[tuple(pm) for pm in config.provider_models],
        model_costs_input=config.model_costs_per_1m_input or None,
        model_costs_output=config.model_costs_per_1m_output or None,
        model_kwargs=config.model_kwargs or None,
        model_concurrency=config.model_concurrency or None,
        system_prompt=config.system_prompt,
    )
    config.num_models = llm_pool.num_models
    config.model_names = llm_pool.model_names
    print(f"Models ({config.num_models}): {config.model_names}")

    # Create environment
    env = LLMRouterEnv(
        benchmark=benchmark_train,
        feature_extractor=feature_extractor,
        llm_pool=llm_pool,
        K_accuracy=config.K_accuracy,
        K_latency=config.K_latency,
        K_cost=config.K_cost,
        latency_range=config.latency_range,
        cost_range=config.cost_range,
        eval_mode=config.eval_mode,
        seed=config.seed,
    )

    # Create policy network
    policy = PolicyNetwork(
        feature_dim=config.total_feature_dim,
        num_actions=config.num_models,
    )
    print(f"Policy network: {sum(p.numel() for p in policy.parameters())} parameters")

    # Train
    trainer = PPOTrainer(policy=policy, config=config)

    num_updates = config.total_timesteps // config.rollout_steps
    print(f"\nTraining: {config.total_timesteps} steps, {config.rollout_steps}/rollout, {num_updates} PPO updates")
    print(f"  K_accuracy={config.K_accuracy}, K_latency={config.K_latency}, K_cost={config.K_cost}")
    print(f"  dataset={config.dataset}, eval_mode={config.eval_mode}")
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
    }

    final_path = os.path.join(config.checkpoint_dir, "policy_final.pt")
    torch.save({**ckpt_base, "policy_state_dict": policy.state_dict()}, final_path)
    print(f"\nSaved final checkpoint to {final_path}")

    if trainer.best_state_dict is not None:
        best_path = os.path.join(config.checkpoint_dir, "policy_best.pt")
        torch.save(
            {**ckpt_base, "policy_state_dict": trainer.best_state_dict,
             "best_reward": trainer.best_reward, "best_step": trainer.best_step},
            best_path,
        )
        print(f"Saved best checkpoint to {best_path} (reward={trainer.best_reward:.4f} at step {trainer.best_step})")

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
        print(f"  Accuracy:    {summary['accuracy']:.4f}")
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

    if trainer.best_summary is not None:
        print_summary(
            f"Best policy (step {trainer.best_step}, reward={trainer.best_reward:.4f})",
            trainer.best_summary,
        )


if __name__ == "__main__":
    main()
