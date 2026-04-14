"""Answer correctness checker for LLM Router RL."""

import difflib
import re


class Evaluator:
    """Evaluates predicted answers against ground truth."""

    @staticmethod
    def exact_match(predicted: str, ground_truth: str) -> bool:
        """Case-insensitive, stripped whitespace comparison."""
        return predicted.strip().lower() == ground_truth.strip().lower()

    @staticmethod
    def fuzzy_match(
        predicted: str, ground_truth: str, threshold: float = 0.85
    ) -> bool:
        """Uses difflib.SequenceMatcher ratio for partial credit.

        Args:
            predicted: The model's answer.
            ground_truth: The correct answer.
            threshold: Minimum similarity ratio to count as a match.

        Returns:
            True if the similarity ratio >= threshold.
        """
        ratio = difflib.SequenceMatcher(
            None,
            predicted.strip().lower(),
            ground_truth.strip().lower(),
        ).ratio()
        return ratio >= threshold

    @staticmethod
    def numeric_match(predicted: str, ground_truth: str) -> bool:
        """Extract and compare numeric answers.

        Useful for math benchmarks (GSM8K) where the LLM outputs
        reasoning text but the answer is a number.
        """
        def extract_number(text) -> str:
            if not isinstance(text, str):
                text = str(text)
            text = text.replace(",", "")
            # Look for #### marker first (GSM8K format)
            if "####" in text:
                return text.split("####")[-1].strip()
            # Look for "the answer is X" pattern
            match = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
            if match:
                return match.group(1)
            # Last number in the text
            numbers = re.findall(r"-?\d+\.?\d*", text)
            return numbers[-1] if numbers else text.strip()

        pred_num = extract_number(predicted if isinstance(predicted, str) else str(predicted))
        gt_num = extract_number(ground_truth if isinstance(ground_truth, str) else str(ground_truth))

        try:
            return float(pred_num) == float(gt_num)
        except ValueError:
            return pred_num.strip().lower() == gt_num.strip().lower()
