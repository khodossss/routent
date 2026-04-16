# Routent (Route Agent)

An adaptive LLM router that learns which model — or which compute budget — to use for each query, optimizing a balance of accuracy, latency, and cost that you define.

---

## Idea

Every LLM application makes the same implicit trade-off dozens of times per second: is this query worth the cost of the best model, or will a cheaper one do? Most teams answer this once, statically, and never revisit it.

**routent** learns the answer continuously. It trains a lightweight routing policy on your own queries and models, then at inference time dispatches each prompt to the model most likely to maximize your reward — where *your reward* is a weighted combination of three factors you control:

```
reward = K_quality · quality − K_latency · norm(latency) − K_cost · norm(cost)
```

Set `K_quality=1.0, K_cost=0.8, K_latency=0.2` and the router aggressively minimizes spend while keeping accuracy high. Flip the weights and it optimizes for speed. The policy adapts to your actual traffic, not a benchmark.

### Who is it for

- Teams running multiple LLM providers (OpenAI, Google, Anthropic, local HuggingFace) who want to cut costs without degrading quality.
- Applications where different query types need different models: math and code to a reasoning-heavy model, simple Q&A to a fast cheap one, sentiment classification to a local classifier.
- Developers experimenting with inference-time scaling — routing between `reasoning_effort=minimal/low/medium` on the same model to spend compute only where it matters.

### What it can route on

| Routing dimension | Example |
|---|---|
| **Cost** | Hard questions → GPT-5.4, easy ones → GPT-5-nano |
| **Latency** | Real-time chat → fast model, batch jobs → thorough model |
| **Domain / capability** | Math → reasoning model, sentiment → local classifier |
| **Compute budget** | Same model, different `reasoning_effort` or `temperature` |
| **Provider mix** | OpenAI + Google + local HuggingFace — all in one pool |

---

## Comparison

| | Routent | RouteLLM | LiteLLM | Martian | OpenRouter |
|---|---|---|---|---|---|
| **Learning** | Online (adapts per query) | Offline classifier | None | Offline (black box) | None |
| **Reward function** | User-defined (3 coefficients) | Binary strong/weak | — | Proprietary | — |
| **Providers** | Any mix | OpenAI-centric | Any | Any | Any |
| **Local models** | Yes (HuggingFace) | No | Yes | No | No |
| **Exploration** | LinUCB (principled UCB) | — | — | — | — |
| **Convergence guarantee** | Yes (sublinear regret) | No | No | No | No |
| **Open source** | Yes | Yes | Yes | No | No |

**RouteLLM** (LMSYS) is the closest prior work — it trains a binary classifier to choose between a strong and a weak model. routent generalizes this: N models, continuous reward, online learning, no pre-labeled data needed.

**LiteLLM / OpenRouter** are proxies with rule-based routing (cost tiers, fallbacks). They don't learn from traffic.

**Martian** is proprietary and treats routing as a black box. No visibility into what it optimizes for.

---

## How it works

### 1. Define your model pool

Any combination of providers and models becomes a set of *arms* in the bandit. You can mix cloud APIs and local HuggingFace models in the same pool.

```json
"provider_models": [
    ["openai",  "gpt-5-nano-2025-08-07"],
    ["openai",  "gpt-5-nano-2025-08-07"],
    ["google",  "gemini-3.1-flash-lite-preview"]
],
"model_kwargs": [
    {"reasoning_effort": "minimal"},
    {"reasoning_effort": "medium"},
    {}
]
```

### 2. Set your reward weights

Tell the router what you care about. Weights are relative — only their ratio matters.

```json
"K_quality": 1.0,
"K_latency":  0.25,
"K_cost":     0.10
```

Latency and cost are min-max normalized to `[0, 1]` using ranges you set, so the scale is always comparable to accuracy.

### 3. Train the routing policy

Run training on a dataset that matches your use case (GSM8K for math, MMLU for general knowledge, your own data via a custom loader). The router observes real model responses and costs — no labels needed beyond what the model itself returns.

```bash
python routent/scripts/train.py --config configs/train/gsm8k_gpt-5-nano.json
```

Internally:

- Each query is encoded into a 384-dimensional semantic embedding (MiniLM-L6-v2), zero-centered over the training corpus.
- **Disjoint LinUCB** maintains a separate linear reward model per arm. The UCB score — expected reward plus an uncertainty bonus — selects which model to call.
- After the call, the reward is observed and the arm's parameters update in one matrix operation (Sherman-Morrison, O(d²)).
- The uncertainty bonus shrinks as data accumulates for each arm, naturally shifting from exploration to exploitation.

### 4. Evaluate

```bash
python routent/scripts/evaluate.py --checkpoint results/.../checkpoints/policy_best.pt
```

Prints accuracy, average cost, average latency, and model usage breakdown — compared against always-using each individual model as a baseline.

### 5. Deploy

```bash
python routent/scripts/infer.py configs/inference/gsm8k_gpt-5-nano.json "What is 15% of 80?"
```

Output:

```
Routing: "What is 15% of 80?"

  Routing scores (expected reward):
    [0] +1.7534  openai/gpt-5-nano-2025-08-07(reasoning_effort=minimal) ◀ selected
    [1] +1.6157  openai/gpt-5-nano-2025-08-07(reasoning_effort=low)
    [2] -4.1286  openai/gpt-5-nano-2025-08-07(reasoning_effort=medium)

  Calling [0] openai/gpt-5-nano-2025-08-07(reasoning_effort=minimal) ...

  Answer  : 12
  Latency : 1142 ms
  Cost    : $1.90e-06
```

The checkpoint is self-contained: it stores the LinUCB parameters, the embedding centering stats, and the full model pool config. No training data needed at inference time.
