"""Route a single prompt through a trained LinUCB policy.

Usage:
    python routent/scripts/infer.py configs/inference/gsm8k_gpt-5-nano.json "What is 15% of 80?"

The inference config points to a checkpoint; everything else (models, embedding
model, feature dim) is read directly from the checkpoint so no training data is
needed at inference time.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import routent.utils.silence  # noqa: F401

import numpy as np
import torch

from dotenv import load_dotenv

from routent.config import Config
from routent.env.feature_extractor import SentenceEmbeddingFeatureExtractor
from routent.models.pool import LLMPool
from routent.training.linucb import DisjointLinUCB


# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt_cost(cost: float) -> str:
    if cost == 0:
        return "$0.000000"
    if abs(cost) >= 1e-3:
        return f"${cost:.5f}"
    return f"${cost:.2e}"


def _load_inference_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_config_from_checkpoint(ckpt: dict, overrides: dict) -> Config:
    """Reconstruct a Config from the saved dict, then apply inference overrides."""
    config = Config()
    for k, v in ckpt["config"].items():
        if hasattr(config, k):
            if k == "provider_models":
                v = [tuple(x) for x in v]
            elif k in ("latency_range", "cost_range"):
                v = tuple(v)
            setattr(config, k, v)
    # Inference config may override models (e.g. different API key endpoint)
    for k, v in overrides.items():
        if k == "checkpoint":
            continue
        if k == "provider_models":
            v = [tuple(x) for x in v]
        setattr(config, k, v)
    return config


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Route a prompt via a trained LinUCB policy")
    parser.add_argument("config", help="Path to inference config JSON")
    parser.add_argument("prompt", help="The question / task string to route")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show expected-reward scores for all models")
    args = parser.parse_args()

    load_dotenv()

    # ── Load inference config ────────────────────────────────────────────────
    inf_cfg = _load_inference_config(args.config)
    checkpoint_path = inf_cfg.get("checkpoint")
    if not checkpoint_path:
        sys.exit("Error: inference config must contain a 'checkpoint' key.")
    if not os.path.exists(checkpoint_path):
        sys.exit(f"Error: checkpoint not found: {checkpoint_path}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "linucb_A_inv" not in ckpt:
        sys.exit("Error: checkpoint does not contain a LinUCB policy. Re-train with the current code.")

    config = _build_config_from_checkpoint(ckpt, inf_cfg)

    # ── Feature extractor — restored from saved stats, no re-fitting needed ──
    fe = SentenceEmbeddingFeatureExtractor(
        model_name=config.embedding_model,
        device=getattr(config, "embedding_device", "cpu"),
    )
    if "fe_mean" not in ckpt or ckpt["fe_mean"] is None:
        sys.exit(
            "Error: checkpoint has no embedded fe_mean/fe_std.\n"
            "Re-train with the current version of train.py to get a compatible checkpoint."
        )
    fe.load_stats(ckpt["fe_mean"], ckpt["fe_std"])

    # ── LLM pool ─────────────────────────────────────────────────────────────
    llm_pool = LLMPool(
        provider_models=[tuple(pm) for pm in config.provider_models],
        model_costs_input=getattr(config, "model_costs_per_1m_input", None) or None,
        model_costs_output=getattr(config, "model_costs_per_1m_output", None) or None,
        model_kwargs=getattr(config, "model_kwargs", None) or None,
        model_concurrency=getattr(config, "model_concurrency", None) or None,
        model_labels=getattr(config, "model_labels", None) or None,
        system_prompt=getattr(config, "system_prompt", ""),
    )

    # ── LinUCB agent ─────────────────────────────────────────────────────────
    agent = DisjointLinUCB(
        num_actions=ckpt["num_actions"],
        feature_dim=ckpt["feature_dim"],
        alpha=ckpt.get("linucb_alpha", 1.0),
    )
    agent.A_inv = ckpt["linucb_A_inv"]
    agent.b = ckpt["linucb_b"]

    # ── Route ─────────────────────────────────────────────────────────────────
    prompt = args.prompt
    print(f'\nRouting: "{prompt}"\n')

    obs = fe.transform(prompt).numpy().astype(np.float64)

    # Expected reward per arm — greedy (no UCB bonus, pure exploitation)
    theta_scores = np.array([agent.A_inv[a] @ agent.b[a] @ obs for a in range(agent.num_actions)])
    selected = int(np.argmax(theta_scores))
    model_names = getattr(config, "model_names", [str(i) for i in range(agent.num_actions)])

    # Routing decision — always shown
    print("  Routing scores (expected reward):")
    for i, (name, score) in enumerate(zip(model_names, theta_scores)):
        marker = " ◀ selected" if i == selected else ""
        print(f"    [{i}] {score:+.4f}  {name}{marker}")
    print()

    # ── Call selected model ───────────────────────────────────────────────────
    print(f"  Calling [{selected}] {model_names[selected]} ...")
    model = llm_pool.get_model(selected)
    answer, latency_ms, cost = model.generate(
        prompt=prompt,
        answer="",      # unknown at inference time
        difficulty="",
        category="",
    )

    # ── Result ────────────────────────────────────────────────────────────────
    print()
    print(f"  Answer  : {answer}")
    print(f"  Latency : {latency_ms:.0f} ms")
    print(f"  Cost    : {_fmt_cost(cost)}")
    print()


if __name__ == "__main__":
    main()
