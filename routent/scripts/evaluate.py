"""Evaluate a trained LLM Router policy against baselines."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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
from routent.training.linucb import DisjointLinUCB


def evaluate_policy(policy, env, benchmark, feature_extractor, num_runs=1):
    """Evaluate a policy over the full benchmark."""
    total_correct = 0
    total_cost = 0.0
    total_latency = 0.0
    total_reward = 0.0
    model_counts = {}
    n = 0

    for _ in range(num_runs):
        for item in benchmark:
            obs = feature_extractor.transform(item["question"])
            obs_tensor = obs if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32)
            action, _, _ = policy.get_action(obs_tensor)

            model = env.llm_pool.get_model(action)
            prompt = env._build_prompt(item)
            predicted, latency, cost = model.generate(
                prompt, item["answer"],
                item.get("difficulty", ""), item.get("category", ""),
            )
            correct = env._evaluate_answer(predicted, item["answer"], item=item)
            reward = env._compute_reward(correct, latency, cost)

            total_correct += correct
            total_cost += cost
            total_latency += latency
            total_reward += reward
            model_counts[action] = model_counts.get(action, 0) + 1
            n += 1

    return {
        "avg_quality": total_correct / n,
        "avg_cost": total_cost / n,
        "avg_latency": total_latency / n,
        "avg_reward": total_reward / n,
        "model_usage": {k: v / n for k, v in sorted(model_counts.items())},
    }


def evaluate_baseline_fixed(model_idx, env, benchmark, num_runs=1):
    """Evaluate a fixed model selection baseline."""
    total_correct = 0
    total_cost = 0.0
    total_latency = 0.0
    total_reward = 0.0
    n = 0

    for _ in range(num_runs):
        for item in benchmark:
            model = env.llm_pool.get_model(model_idx)
            prompt = env._build_prompt(item)
            predicted, latency, cost = model.generate(
                prompt, item["answer"],
                item.get("difficulty", ""), item.get("category", ""),
            )
            correct = env._evaluate_answer(predicted, item["answer"], item=item)
            reward = env._compute_reward(correct, latency, cost)

            total_correct += correct
            total_cost += cost
            total_latency += latency
            total_reward += reward
            n += 1

    return {
        "avg_quality": total_correct / n,
        "avg_cost": total_cost / n,
        "avg_latency": total_latency / n,
        "avg_reward": total_reward / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LLM Router policy")
    parser.add_argument("--checkpoint", type=str, default="routent/checkpoints/policy_best.pt")
    parser.add_argument("--num_runs", type=int, default=1)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_dict = ckpt["config"]
    config = Config()
    for k, v in config_dict.items():
        if hasattr(config, k):
            if k == "provider_models":
                v = [tuple(x) for x in v]
            elif k in ("latency_range", "cost_range"):
                v = tuple(v)
            setattr(config, k, v)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    _, benchmark_test = load_benchmark(config)
    benchmark_train_for_fit, _ = load_benchmark(config)

    print(f"Test set: {len(benchmark_test)} questions")

    feature_extractor = SentenceEmbeddingFeatureExtractor(
        model_name=config.embedding_model,
        device=config.embedding_device,
    )
    feature_extractor.fit([item["question"] for item in benchmark_train_for_fit])

    llm_pool = LLMPool(
        provider_models=[tuple(pm) for pm in config.provider_models],
        model_costs_input=config.model_costs_per_1m_input or None,
        model_costs_output=getattr(config, "model_costs_per_1m_output", None) or None,
        model_kwargs=getattr(config, "model_kwargs", None) or None,
        model_concurrency=getattr(config, "model_concurrency", None) or None,
        model_labels=getattr(config, "model_labels", None) or None,
        system_prompt=getattr(config, "system_prompt", ""),
    )

    env = LLMRouterEnv(
        benchmark=benchmark_test,
        feature_extractor=feature_extractor,
        llm_pool=llm_pool,
        K_quality=config.K_quality,
        K_latency=config.K_latency,
        K_cost=config.K_cost,
        latency_range=config.latency_range,
        cost_range=config.cost_range,
        eval_config=config.eval_config,
        seed=config.seed,
    )

    agent = DisjointLinUCB(
        num_actions=ckpt["num_actions"],
        feature_dim=ckpt["feature_dim"],
        alpha=ckpt.get("linucb_alpha", 1.0),
    )
    agent.A_inv = ckpt["linucb_A_inv"]
    agent.b = ckpt["linucb_b"]

    print("\nEvaluating trained policy...")
    trained_results = evaluate_policy(
        agent, env, benchmark_test, feature_extractor, num_runs=args.num_runs
    )

    print("Evaluating baselines (each model individually)...")
    baselines = {}
    for i in range(llm_pool.num_models):
        name = llm_pool.model_names[i]
        baselines[name] = evaluate_baseline_fixed(i, env, benchmark_test, num_runs=args.num_runs)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    header = f"{'Strategy':<35} {'Accuracy':>10} {'Avg Cost':>10} {'Avg Latency':>12} {'Reward':>10}"
    print(header)
    print("-" * 70)

    def print_row(name, r):
        print(f"{name:<35} {r['accuracy']:>10.3f} {r['avg_cost']:>10.5f} {r['avg_latency']:>12.1f} {r['avg_reward']:>10.4f}")

    print_row("Trained Policy", trained_results)
    for name, result in baselines.items():
        print_row(f"Always {name}", result)

    print(f"\n  Trained policy model usage:")
    for m, pct in sorted(trained_results.get("model_usage", {}).items()):
        name = llm_pool.model_names[m] if m < len(llm_pool.model_names) else str(m)
        print(f"    {name}: {pct:.1%}")


if __name__ == "__main__":
    main()
