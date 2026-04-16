"""LLM-based judges for answer correctness evaluation.

Used when exact-match or numeric-match is not applicable (e.g., free-form
answers where semantic equivalence matters).
"""

from typing import Optional


DEFAULT_PROMPT_TEMPLATE = """You are evaluating answer correctness.

Question: {question}

Reference answer: {ground_truth}

Submitted answer: {predicted}

Is the submitted answer correct? Consider semantic equivalence, not exact wording.
Reply with only "YES" or "NO"."""


class LLMJudge:
    """Uses an LLM to score predictions when exact/numeric match isn't applicable."""

    def __init__(
        self,
        judge_llm,  # BaseLLM instance
        prompt_template: Optional[str] = None,
    ):
        """Initialize the LLM judge.

        Args:
            judge_llm: A BaseLLM instance used to perform the judgement call.
            prompt_template: Optional override for the judge prompt. Must contain
                `{question}`, `{ground_truth}`, and `{predicted}` placeholders.
                If None, the default template is used.
        """
        self.judge_llm = judge_llm
        self.prompt_template = (
            prompt_template if prompt_template is not None else DEFAULT_PROMPT_TEMPLATE
        )

    def score(self, question: str, predicted: str, ground_truth: str) -> bool:
        """Return True if the judge considers the predicted answer correct.

        Args:
            question: The original question being answered.
            predicted: The submitted answer from a candidate model.
            ground_truth: The reference / gold answer.

        Returns:
            True if the judge replies with a "YES"-prefixed answer, False otherwise.
            Any exception (network error, malformed response, etc.) is swallowed
            and results in False.
        """
        try:
            filled = self.prompt_template.format(
                question=question,
                ground_truth=ground_truth,
                predicted=predicted,
            )
            result = self.judge_llm.generate(
                prompt=filled,
                answer="",
                difficulty="",
                category="",
            )
            # generate() returns a tuple; the first element is the response text
            if not result:
                return False
            response = result[0] if isinstance(result, tuple) else result
            if not isinstance(response, str):
                return False
            return response.strip().upper().startswith("YES")
        except Exception:
            return False


def create_judge_from_config(judge_config: dict) -> LLMJudge:
    """Create an ``LLMJudge`` from a config dict.

    Expected keys:
        - ``provider``: provider name passed to ``GenerativeLLM``.
        - ``model_name``: model identifier passed to ``GenerativeLLM``.
        - ``prompt_template`` (optional): override for the judge prompt.

    Additional keys in ``judge_config`` are forwarded to ``GenerativeLLM``
    where applicable. The judge LLM is constructed with concurrency of 1 and
    no system prompt, since the judging instructions live in the user message.
    """
    from routent.models.generative import GenerativeLLM

    provider = judge_config.get("provider")
    model_name = judge_config.get("model_name")
    prompt_template = judge_config.get("prompt_template")

    judge_llm = GenerativeLLM(
        provider=provider,
        model_name=model_name,
        max_concurrency=1,
        system_prompt=None,
    )

    return LLMJudge(judge_llm=judge_llm, prompt_template=prompt_template)
