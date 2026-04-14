# LLM Router — Adaptive Model Selection via Contextual Bandit RL

## Project Overview

Build a **contextual bandit RL system** that learns to route incoming prompts to the optimal LLM model from a pool of N models, balancing **answer quality** vs **cost** (latency, price).

The system observes features of an incoming prompt, selects which model to call, receives an immediate reward based on correctness and cost, and updates its policy via PPO. Each routing decision is independent (horizon=1) — this is a contextual bandit formulation, which is a subclass of RL with single-step episodes.

**Input**: N LLM models + shared prompt/instruction + benchmark dataset (train/test split)
**Output**: Trained routing policy that learns model strengths — e.g., cheap models for easy tasks, math-strong models for math, etc.

This is a **standalone Python project**. No web UI, no API server. Pure training loop + evaluation + visualization.

---

## Architecture

```
Benchmark Dataset (questions + ground truth answers)
        ↓
    Gymnasium Environment (LLMRouterEnv)
        ↓
    observation = feature_extract(prompt)
        ↓
    Policy Network (MLP) → action = model_index
        ↓
    Simulated LLM Call (or real API call) → answer, latency, cost
        ↓
    Evaluator (exact match / fuzzy match) → correct: bool
        ↓
    Reward = α·correct - β·norm_latency - γ·norm_cost
        ↓
    PPO updates policy
```

---

## Project Structure

```
routent/
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters and settings
├── data/
│   └── benchmark.json         # Benchmark QA dataset
├── env/
│   ├── __init__.py
│   ├── router_env.py          # Gymnasium environment
│   └── feature_extractor.py   # TF-IDF + handcrafted features
├── models/
│   ├── __init__.py
│   ├── policy_network.py      # MLP policy (PyTorch)
│   └── llm_pool.py            # Simulated LLM model pool
├── training/
│   ├── __init__.py
│   ├── ppo.py                 # PPO algorithm (from scratch, PyTorch)
│   └── buffer.py              # Rollout buffer
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py           # Answer correctness checker
│   └── metrics.py             # Logging: reward, accuracy, cost over time
├── scripts/
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Evaluate trained policy
│   └── visualize.py           # Plot training curves
└── tests/
    ├── test_env.py
    ├── test_policy.py
    └── test_ppo.py
```

---

## Component Specifications

### 1. `config.py`

Dataclass or dict with all configurable parameters:

```python
@dataclass
class Config:
    # Reward weights
    alpha: float = 1.0          # weight for correctness
    beta: float = 0.3           # weight for latency penalty
    gamma: float = 0.3          # weight for cost penalty

    # PPO hyperparameters (contextual bandit: gamma=0, no GAE)
    lr: float = 3e-4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    rollout_steps: int = 256

    # Feature extraction
    tfidf_max_features: int = 100
    total_feature_dim: int = 110  # tfidf + handcrafted

    # Model pool
    num_models: int = 4

    # Training
    total_timesteps: int = 50_000
    eval_interval: int = 1000
    seed: int = 42
```

---

### 2. `data/benchmark.json`

Generate a synthetic benchmark with **varying difficulty levels**. This is critical — without difficulty variance, the agent has nothing to learn.

Format:
```json
[
  {
    "id": 1,
    "question": "What is the capital of France?",
    "answer": "Paris",
    "difficulty": "easy",
    "category": "geography"
  },
  {
    "id": 2,
    "question": "Solve: derivative of x^3 * sin(x)",
    "answer": "3x^2*sin(x) + x^3*cos(x)",
    "difficulty": "hard",
    "category": "math"
  }
]
```

Generate **500+ questions** across 3 difficulty levels (easy/medium/hard) and 4-5 categories (geography, math, science, coding, logic). Use a mix of:
- Simple factual (easy) — any model can answer
- Moderate reasoning (medium) — cheap models sometimes fail
- Complex multi-step (hard) — only expensive models get right

Include a script `data/generate_benchmark.py` that creates this dataset programmatically with templates.

---

### 3. `env/feature_extractor.py`

Extracts a fixed-size feature vector from a prompt string. Two parts:

**A. TF-IDF features (learned from benchmark corpus)**
- Fit `sklearn.feature_extraction.text.TfidfVectorizer` on all benchmark questions
- `max_features=100`
- Transform each new prompt into a 100-dim sparse vector → convert to dense

**B. Handcrafted features (~10 dims)**
- `num_tokens`: word count (normalized)
- `num_chars`: character count (normalized)
- `avg_word_length`: average word length
- `num_sentences`: sentence count
- `has_math_symbols`: bool (contains +, -, *, /, =, ^)
- `has_code_keywords`: bool (contains "function", "def", "class", "return", "print", etc.)
- `has_question_mark`: bool
- `num_numbers`: count of numeric tokens
- `vocabulary_richness`: unique_words / total_words
- `max_word_length`: length of the longest word

**Combined output**: `torch.Tensor` of shape `(total_feature_dim,)` = TF-IDF (100) + handcrafted (10) = 110 dims.

The extractor must have a `fit(corpus: List[str])` method and a `transform(prompt: str) -> torch.Tensor` method.

Normalization: all handcrafted features min-max normalized to [0, 1] based on stats computed during `fit()`.

---

### 4. `models/llm_pool.py`

**Simulated** LLM model pool. No real API calls — we simulate model behavior.

Define N=4 models with different accuracy/cost profiles. **Accuracy depends on (difficulty, category)** — not just difficulty. This makes the routing problem non-trivial: each model has strengths/weaknesses per domain.

Example accuracy profiles (4 models × 3 difficulties × 5 categories = 60 parameters):

| Model  | Latency (ms) | Cost per call | Strengths                    |
|--------|--------------|---------------|------------------------------|
| tiny   | 50           | 0.001         | Fast, ok on easy factual     |
| small  | 150          | 0.005         | Good at geography/science    |
| medium | 400          | 0.02          | Strong math, decent overall  |
| large  | 1000         | 0.06          | Best overall, especially hard|

Each model's `generate(question, difficulty, category) -> (answer: str, latency: float, cost: float)`:

- Sample correctness from Bernoulli based on accuracy table keyed by (difficulty, category)
- If correct: return ground truth answer
- If incorrect: return a plausible wrong answer
- Latency: sample from Normal(mean_latency, mean_latency * 0.1)
- Cost: fixed per call

This simulation allows training without burning API budget. The architecture supports swapping in real API calls later via a common interface:

```python
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, difficulty: str, category: str) -> Tuple[str, float, float]:
        """Returns (answer, latency_ms, cost)"""
        pass
```

---

### 5. `env/router_env.py`

Gymnasium environment. This is the core of the project.

```python
class LLMRouterEnv(gym.Env):
    """
    Observation: feature vector of current prompt (float32, shape=(feature_dim,))
    Action: discrete — index of model to use (0..N-1)
    Reward: α·correct - β·norm_latency - γ·norm_cost
    """
```

**Contextual bandit structure**: Each `step()` is ONE independent routing decision (horizon=1). Each `reset()` picks the next question from the benchmark (sequential or random). There are no multi-step episodes — each question is a self-contained decision with immediate reward.

**Implementation details**:
- `__init__`: receives Config, benchmark data, feature_extractor, llm_pool
- `reset()`: picks next question, returns observation = feature_extractor.transform(question)
- `step(action)`: calls llm_pool.models[action].generate(), evaluates answer, computes reward
- `observation_space`: `Box(low=-inf, high=inf, shape=(feature_dim,), dtype=float32)`
- `action_space`: `Discrete(num_models)`
- Normalizes latency and cost to [0, 1] range using known min/max from model pool
- Tracks cumulative stats: total_cost, total_correct, per_model_usage_count

**Info dict** returned by step():
```python
info = {
    "correct": bool,
    "model_used": int,
    "latency_ms": float,
    "cost": float,
    "difficulty": str,
    "category": str,
    "question_id": int
}
```

---

### 6. `models/policy_network.py`

Actor-Critic MLP in PyTorch.

```python
class PolicyNetwork(nn.Module):
    """
    Shared backbone + separate actor head (policy) and critic head (value).

    Architecture:
        Input (feature_dim=110)
            → Linear(110, 128) → ReLU → LayerNorm
            → Linear(128, 64) → ReLU → LayerNorm
            ↙                        ↘
        Actor Head                Critic Head
        Linear(64, num_actions)   Linear(64, 1)
        → Softmax (→ Categorical)   → Value scalar
    """
```

Methods:
- `forward(obs) -> (action_logits, value)`
- `get_action(obs) -> (action, log_prob, value)` — samples from Categorical
- `evaluate_actions(obs, actions) -> (log_probs, entropy, values)` — for PPO update

Use `torch.distributions.Categorical` for action sampling.

Weight initialization: orthogonal init for linear layers, bias = 0.

---

### 7. `training/buffer.py`

Rollout buffer that stores transitions for PPO (contextual bandit setting).

Stores: observations, actions, log_probs, rewards, values.

Since this is a contextual bandit (horizon=1), there are no done flags or temporal dependencies. Returns = rewards directly (no discounting, no GAE).

Methods:

- `add(obs, action, log_prob, reward, value)`
- `compute_returns_and_advantages()` — advantages = rewards - values (no GAE needed)
- `get_batches(batch_size)` — yields random mini-batches for PPO epochs
- `clear()` — reset buffer after PPO update

All tensors stored as PyTorch tensors on CPU.

---

### 8. `training/ppo.py`

PPO (Proximal Policy Optimization) implementation from scratch in PyTorch.

**Algorithm per update cycle:**
1. Collect `rollout_steps` transitions using current policy
2. Compute advantages (rewards - baseline values, no GAE needed for bandit)
3. For `ppo_epochs` epochs:
   a. Sample mini-batches from buffer
   b. Compute ratio = exp(new_log_prob - old_log_prob)
   c. Clipped surrogate loss: min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv)
   d. Value loss: MSE(predicted_value, returns)
   e. Entropy bonus
   f. Total loss = -policy_loss + value_coef * value_loss - entropy_coef * entropy
   g. Backprop + optimizer step (Adam)

```python
class PPOTrainer:
    def __init__(self, policy: PolicyNetwork, config: Config):
        self.policy = policy
        self.optimizer = Adam(policy.parameters(), lr=config.lr)
        self.buffer = RolloutBuffer(...)
        ...

    def collect_rollout(self, env) -> dict:
        """Run policy in env for rollout_steps, fill buffer. Return stats."""

    def update(self) -> dict:
        """Run PPO update on collected data. Return loss stats."""

    def train(self, env, total_timesteps) -> List[dict]:
        """Main training loop. Returns history of stats per update."""
```

Gradient clipping: `max_norm=0.5`.

---

### 9. `evaluation/evaluator.py`

Answer correctness checker.

```python
class Evaluator:
    @staticmethod
    def exact_match(predicted: str, ground_truth: str) -> bool:
        """Case-insensitive, stripped whitespace comparison."""

    @staticmethod
    def fuzzy_match(predicted: str, ground_truth: str, threshold: float = 0.85) -> bool:
        """Uses difflib.SequenceMatcher ratio. For partial credit."""
```

The environment uses `exact_match` by default, configurable.

---

### 10. `evaluation/metrics.py`

Track and log training metrics.

```python
class MetricsTracker:
    def log_step(self, info: dict, reward: float): ...
    def log_update(self, loss_stats: dict): ...
    def get_summary(self, last_n: int = 100) -> dict:
        """
        Returns:
        - avg_reward (last N steps)
        - accuracy (last N steps)
        - avg_cost (last N steps)
        - avg_latency (last N steps)
        - per_model_usage: Dict[int, float] — fraction of calls per model
        - per_difficulty_accuracy: Dict[str, float]
        - per_difficulty_model_preference: Dict[str, Dict[int, float]]
        """
```

---

### 11. `scripts/train.py`

Main entry point.

```
python scripts/train.py [--total_timesteps 50000] [--seed 42] [--alpha 1.0] [--beta 0.3] [--gamma 0.3]
```

Flow:
1. Load config (override from CLI args via argparse)
2. Load/generate benchmark data
3. Fit feature extractor on benchmark corpus
4. Create LLM pool
5. Create Gymnasium env
6. Create policy network
7. Create PPO trainer
8. Train
9. Save trained policy to `checkpoints/policy_final.pt`
10. Save training metrics to `results/training_log.json`
11. Print final summary table

**Logging during training** (every eval_interval steps):
```
Step 1000/50000 | Reward: 0.42 | Acc: 0.68 | Cost: 0.012 | Model usage: [tiny:45%, small:30%, med:20%, large:5%]
Step 2000/50000 | Reward: 0.58 | Acc: 0.75 | Cost: 0.010 | Model usage: [tiny:50%, small:25%, med:15%, large:10%]
...
```

---

### 12. `scripts/evaluate.py`

Evaluate a trained policy on the full benchmark.

```
python scripts/evaluate.py --checkpoint checkpoints/policy_final.pt
```

Output: detailed report showing:
- Overall accuracy, cost, latency
- Per-difficulty breakdown (easy/medium/hard): accuracy + which model was chosen
- Comparison with baselines:
  - **Always-cheap**: always use model 0 (tiny)
  - **Always-expensive**: always use model 3 (large)
  - **Random**: uniform random model selection
  - **Oracle**: always pick cheapest model that gets it right (computed from simulation)
- Table format

---

### 13. `scripts/visualize.py`

Generate matplotlib plots from training logs.

```
python scripts/visualize.py --log results/training_log.json --output results/plots/
```

Plots to generate:
1. **Reward curve** — moving average reward over training steps
2. **Accuracy curve** — moving average accuracy over training steps
3. **Cost curve** — moving average cost over training steps
4. **Model selection distribution over time** — stacked area chart showing how model usage evolves
5. **Per-difficulty routing heatmap** — which model is preferred for each difficulty level (final policy)
6. **Comparison bar chart** — trained agent vs baselines on accuracy, cost, reward

Save all as PNG files.

---

### 14. Tests

**`tests/test_env.py`**:
- Environment creates, resets, steps without errors
- Observation shape is correct
- Action space is correct
- Reward is within expected bounds
- Info dict has required keys

**`tests/test_policy.py`**:
- PolicyNetwork forward pass produces correct shapes
- get_action returns valid action in action space
- evaluate_actions returns correct shapes

**`tests/test_ppo.py`**:
- Buffer stores and retrieves correctly
- GAE computation produces reasonable values
- One PPO update step runs without error
- Loss decreases over multiple updates on fixed data

---

## Technical Requirements

- **Python 3.10+**
- **PyTorch** — policy network, training loop, all tensor operations
- **Gymnasium** — environment interface
- **scikit-learn** — TF-IDF vectorizer only
- **matplotlib** — visualization
- **numpy** — utilities
- No other dependencies. No Stable-Baselines3 — PPO is implemented from scratch.

`requirements.txt`:
```
torch>=2.0
gymnasium>=0.29
scikit-learn>=1.3
matplotlib>=3.7
numpy>=1.24
```

---

## Key Design Principles

1. **Simulation-first**: Everything runs locally with simulated LLM calls. No API keys needed. Real LLM integration is a future extension — the `BaseLLM` interface makes it a drop-in swap.

2. **PPO from scratch**: Do NOT use Stable-Baselines3 or any RL library. Implement PPO manually in PyTorch. This is the point — demonstrating RL engineering skill.

3. **Reproducibility**: All randomness seeded via config.seed. Set seeds for torch, numpy, and gymnasium.

4. **Clear separation**: env / models / training / evaluation are independent modules. Each can be tested and swapped independently.

5. **The agent should LEARN something visible**: After training, the policy should clearly route easy questions to cheap models and hard questions to expensive models. If it doesn't — the reward shaping or feature extraction needs debugging. The evaluate.py script should make this visible.

---

## Success Criteria

The project is successful when:
1. `python scripts/train.py` runs end-to-end without errors
2. Training curves show improving reward over time
3. `python scripts/evaluate.py` shows the trained agent **outperforms** random and always-cheap baselines on reward, while being **cheaper** than always-expensive baseline
4. The per-difficulty routing heatmap shows intelligent routing (cheap models for easy, expensive for hard)
5. All tests pass

---

## Future Extensions (NOT in scope now, but design for them)

- Swap simulated LLM pool for real API calls (LiteLLM) with real benchmarks (MMLU, GSM8K, ARC)
- Sentence embeddings (e.g., all-MiniLM-L6-v2) as feature extractor option
- Add cascade/fallback actions (try cheap first, escalate if low confidence)
- Confidence-based routing (use model's logprobs/confidence score)
- Online learning (continuously update policy in production)
- Pareto frontier sweep over reward weights (accuracy vs cost tradeoff visualization)
