"""LLM-based judges for answer evaluation.

Supports:
- Binary (YES/NO) — when no criteria specified.
- Criteria-based continuous — list (equal weights) or dict (custom weights).
- CustomCriteria class for reusable prompt templates.
"""

import re
from typing import Dict, List, Optional, Union


# ── Binary prompt (backward compat) ─────────────────────────────────────────

DEFAULT_PROMPT_TEMPLATE = """You are evaluating answer correctness.

Question: {question}

Reference answer: {ground_truth}

Submitted answer: {predicted}

Is the submitted answer correct? Consider semantic equivalence, not exact wording.
Reply with only "YES" or "NO"."""


# ── Criteria presets (1-10 scale) ────────────────────────────────────────────

CRITERIA_PROMPTS = {
    "correctness": (
        "Rate the factual correctness of the submitted answer compared to the reference answer.\n"
        "10 = perfectly correct, 1 = completely wrong."
    ),
    "completeness": (
        "Rate how completely the submitted answer covers all key points present in the reference answer.\n"
        "10 = covers everything, 1 = misses all key points."
    ),
    "relevance": (
        "Rate how relevant and on-topic the submitted answer is to the question asked.\n"
        "10 = perfectly relevant, 1 = completely off-topic."
    ),
    "conciseness": (
        "Rate how concise the submitted answer is — penalize unnecessary verbosity or filler.\n"
        "10 = optimally concise, 1 = extremely verbose or padded."
    ),
    "coherence": (
        "Rate the logical coherence and readability of the submitted answer.\n"
        "10 = perfectly clear and logical, 1 = incoherent."
    ),
}

_CRITERIA_WRAPPER = """You are an expert evaluator. Score the submitted answer on a specific criterion.

Question: {question}

Reference answer: {ground_truth}

Submitted answer: {predicted}

Criterion: {criterion_description}

Reply with ONLY a single integer from 1 to 10."""


# ── CustomCriteria ───────────────────────────────────────────────────────────

class CustomCriteria:
    """Reusable named criterion with a custom prompt template.

    Usage in config:
        {"judge_criteria": {"my_criterion": 0.5, "correctness": 0.5}}
    Then register via code:
        CRITERIA_PROMPTS["my_criterion"] = "Rate X on 1-10..."
    Or use CustomCriteria for programmatic creation:
        c = CustomCriteria("creativity", "Rate how creative...")
        c.register()  # adds to CRITERIA_PROMPTS
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    def register(self) -> None:
        """Register this criterion in the global CRITERIA_PROMPTS dict."""
        CRITERIA_PROMPTS[self.name] = self.description

    @classmethod
    def from_config(cls, criteria_config: list) -> None:
        """Register all custom criteria from a config list.

        Each item is either a preset name (str) or a dict:
        {"name": "my_criterion", "description": "Rate X on 1-10..."}
        """
        for item in criteria_config:
            if isinstance(item, dict) and "name" in item and "description" in item:
                cls(item["name"], item["description"]).register()


# ── Helpers ──────────────────────────────────────────────────────────────────

# Accepted criteria spec formats for judge_criteria:
#   list[str]           → equal weights, e.g. ["correctness", "completeness"]
#   dict[str, float]    → weighted,      e.g. {"correctness": 0.6, "completeness": 0.4}
CriteriaSpec = Union[List[str], Dict[str, float]]


def _normalize_criteria(criteria: CriteriaSpec) -> Dict[str, float]:
    """Convert list (equal weights) or dict (custom weights) to {name: weight}.

    Weights are normalized to sum to 1.0.
    """
    if isinstance(criteria, list):
        n = len(criteria)
        return {c: 1.0 / n for c in criteria} if n > 0 else {}

    # dict — normalize weights
    total = sum(criteria.values())
    if total <= 0:
        return {k: 1.0 / len(criteria) for k in criteria}
    return {k: v / total for k, v in criteria.items()}


def _parse_score(response: str) -> Optional[float]:
    """Extract a numeric score from a judge response and normalize to [0, 1]."""
    if not isinstance(response, str):
        return None
    text = response.strip()
    m = re.search(r"(\d+)\s*/\s*10", text)
    if m:
        return max(0.0, min(1.0, int(m.group(1)) / 10.0))
    m = re.search(r"\b(\d{1,2})\b", text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 10:
            return max(0.0, min(1.0, val / 10.0))
    return None


# ── LLMJudge ────────────────────────────────────────────────────────────────

class LLMJudge:
    """Uses an LLM to score predictions.

    - ``score()`` — binary YES/NO.
    - ``score_criteria()`` — weighted continuous [0, 1].
    """

    def __init__(
        self,
        judge_llm,
        prompt_template: Optional[str] = None,
    ):
        self.judge_llm = judge_llm
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self._last_details: Dict = {}

    def score(self, question: str, predicted: str, ground_truth: str) -> bool:
        """Binary YES/NO judgement."""
        try:
            filled = self.prompt_template.format(
                question=question, ground_truth=ground_truth, predicted=predicted,
            )
            result = self.judge_llm.generate(
                prompt=filled, answer="", difficulty="", category="",
            )
            response = result[0] if isinstance(result, tuple) else result
            if not isinstance(response, str):
                self._last_details = {"judge_response": str(response), "binary": False}
                return False
            binary = response.strip().upper().startswith("YES")
            self._last_details = {"judge_response": response.strip(), "binary": binary}
            return binary
        except Exception as e:
            self._last_details = {"error": str(e)}
            return False

    def score_criteria(
        self,
        question: str,
        predicted: str,
        ground_truth: str,
        criteria: CriteriaSpec,
    ) -> float:
        """Weighted continuous scoring over multiple criteria.

        Args:
            criteria: list[str] (equal weights) or dict[str, float] (weighted).
                Each key is a preset name from CRITERIA_PROMPTS or a custom
                prompt template with {question}, {predicted}, {ground_truth}.
        Returns:
            Weighted average of per-criterion scores in [0, 1].
        """
        if not criteria:
            return float(self.score(question, predicted, ground_truth))

        weighted = _normalize_criteria(criteria)
        criterion_details = {}
        total = 0.0
        for criterion, weight in weighted.items():
            s = self._score_one_criterion(question, predicted, ground_truth, criterion)
            criterion_details[criterion] = {
                "score": round(s, 3),
                "weight": round(weight, 3),
                "judge_response": self._last_criterion_response,
            }
            total += weight * s
        self._last_details = {"criteria": criterion_details, "weighted_score": round(total, 4)}
        return total

    # ── Batched scoring ────────────────────────────────────────────────

    def batch_score(
        self,
        items: List[tuple],
    ) -> List[float]:
        """Batch binary YES/NO scoring. items = [(question, predicted, ground_truth), ...]"""
        prompts = [
            self.prompt_template.format(question=q, ground_truth=g, predicted=p)
            for q, p, g in items
        ]
        results = self.judge_llm.batch_generate(prompts)
        scores = []
        self._last_batch_details = []
        for result in results:
            response = result[0] if isinstance(result, tuple) else result
            if isinstance(response, str):
                binary = response.strip().upper().startswith("YES")
                self._last_batch_details.append({"judge_response": response.strip(), "binary": binary})
                scores.append(float(binary))
            else:
                self._last_batch_details.append({"judge_response": str(response), "binary": False})
                scores.append(0.0)
        return scores

    def batch_score_criteria(
        self,
        items: List[tuple],
        criteria: CriteriaSpec,
    ) -> List[float]:
        """Batch criteria scoring — all prompts in one batch_generate call.

        items = [(question, predicted, ground_truth), ...]
        Returns one weighted score per item.
        """
        weighted = _normalize_criteria(criteria)
        crit_names = list(weighted.keys())
        n_criteria = len(crit_names)

        # Build all prompts: len(items) × len(criteria)
        all_prompts = []
        for q, p, g in items:
            for crit in crit_names:
                if crit in CRITERIA_PROMPTS:
                    prompt = _CRITERIA_WRAPPER.format(
                        question=q, ground_truth=g, predicted=p,
                        criterion_description=CRITERIA_PROMPTS[crit],
                    )
                else:
                    prompt = crit.format(question=q, ground_truth=g, predicted=p)
                all_prompts.append(prompt)

        results = self.judge_llm.batch_generate(all_prompts)

        # Regroup: each item gets n_criteria results
        scores = []
        self._last_batch_details = []
        for i in range(len(items)):
            chunk = results[i * n_criteria : (i + 1) * n_criteria]
            criterion_details = {}
            total = 0.0
            for j, crit in enumerate(crit_names):
                response = chunk[j][0] if isinstance(chunk[j], tuple) else chunk[j]
                resp_str = response.strip() if isinstance(response, str) else str(response)
                parsed = _parse_score(resp_str)
                s = parsed if parsed is not None else 0.0
                w = weighted[crit]
                criterion_details[crit] = {
                    "score": round(s, 3),
                    "weight": round(w, 3),
                    "judge_response": resp_str,
                }
                total += w * s
            scores.append(total)
            self._last_batch_details.append({"criteria": criterion_details, "weighted_score": round(total, 4)})
        return scores

    # ── Single-item helpers ──────────────────────────────────────────────

    def _score_one_criterion(
        self,
        question: str,
        predicted: str,
        ground_truth: str,
        criterion: str,
    ) -> float:
        """Score a single criterion, return [0, 1]."""
        try:
            if criterion in CRITERIA_PROMPTS:
                prompt = _CRITERIA_WRAPPER.format(
                    question=question,
                    ground_truth=ground_truth,
                    predicted=predicted,
                    criterion_description=CRITERIA_PROMPTS[criterion],
                )
            else:
                prompt = criterion.format(
                    question=question,
                    ground_truth=ground_truth,
                    predicted=predicted,
                )

            result = self.judge_llm.generate(
                prompt=prompt, answer="", difficulty="", category="",
            )
            response = result[0] if isinstance(result, tuple) else result
            self._last_criterion_response = response.strip() if isinstance(response, str) else str(response)
            parsed = _parse_score(response)
            return parsed if parsed is not None else 0.0
        except Exception as e:
            self._last_criterion_response = f"ERROR: {e}"
            return 0.0


def create_judge_from_config(judge_config: dict) -> LLMJudge:
    """Create an ``LLMJudge`` from a config dict."""
    from routent.models.generative import GenerativeLLM

    judge_llm = GenerativeLLM(
        provider=judge_config.get("provider"),
        model_name=judge_config.get("model_name"),
        max_concurrency=judge_config.get("concurrency", 10),
        system_prompt="",
    )

    # Register custom criteria if defined
    custom = judge_config.get("custom_criteria")
    if custom:
        CustomCriteria.from_config(custom)

    return LLMJudge(
        judge_llm=judge_llm,
        prompt_template=judge_config.get("prompt_template"),
    )
