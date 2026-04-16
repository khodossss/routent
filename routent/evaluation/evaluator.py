"""Answer correctness checker for LLM Router RL."""

import difflib
import re
from typing import List, Optional


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

    @staticmethod
    def classification_match(
        predicted: str,
        ground_truth: str,
        choices: Optional[List[str]] = None,
    ) -> bool:
        """Extract a class label from predicted text and compare to ground_truth.

        - If ``choices`` is provided, try to find any choice that appears in
          ``predicted`` (case-insensitive).
        - If ``ground_truth`` looks like a single letter (A-Z) or a single
          short token, also try regex ``\\b([A-Z])\\b`` on predicted.
        - Match is case-insensitive and whitespace-stripped.
        """
        if not isinstance(predicted, str):
            predicted = str(predicted)
        if not isinstance(ground_truth, str):
            ground_truth = str(ground_truth)

        pred = predicted.strip()
        gt = ground_truth.strip()
        pred_lower = pred.lower()
        gt_lower = gt.lower()

        # Direct match
        if pred_lower == gt_lower:
            return True

        # Try choices-based extraction
        if choices:
            found = None
            for choice in choices:
                if not isinstance(choice, str):
                    continue
                c_lower = choice.strip().lower()
                if not c_lower:
                    continue
                # Word-boundary-ish containment
                pattern = r"\b" + re.escape(c_lower) + r"\b"
                if re.search(pattern, pred_lower):
                    found = c_lower
                    # Prefer an exact-tail match over arbitrary, but any hit works
                    if c_lower == gt_lower:
                        return True
            if found is not None:
                return found == gt_lower

        # Letter-style ground truth (e.g., "A", "B")
        if len(gt) <= 3 and re.fullmatch(r"[A-Za-z]", gt):
            match = re.search(r"\b([A-Z])\b", pred)
            if match:
                return match.group(1).lower() == gt_lower

        # Short-token ground truth: check word-boundary containment
        if " " not in gt and len(gt) > 0:
            pattern = r"\b" + re.escape(gt_lower) + r"\b"
            if re.search(pattern, pred_lower):
                return True

        return False

    @staticmethod
    def multilabel_match(
        predicted: str, ground_truth: str, delimiter: str = ","
    ) -> bool:
        """Compare predicted and ground_truth as sets of labels.

        Splits both strings by ``delimiter``, strips whitespace, lowercases,
        then checks for set equality. Empty tokens are excluded from sets.
        """
        if not isinstance(predicted, str):
            predicted = str(predicted)
        if not isinstance(ground_truth, str):
            ground_truth = str(ground_truth)

        pred_set = {
            t.strip().lower()
            for t in predicted.split(delimiter)
            if t.strip()
        }
        gt_set = {
            t.strip().lower()
            for t in ground_truth.split(delimiter)
            if t.strip()
        }
        return pred_set == gt_set

    @staticmethod
    def regression_match(
        predicted: str, ground_truth: str, tolerance: float = 0.01
    ) -> bool:
        """Parse both as floats, compare with absolute tolerance.

        Uses the same number-extraction logic as ``numeric_match``, but
        returns True only if ``|pred - gt| <= tolerance``.
        """
        def extract_number(text) -> str:
            if not isinstance(text, str):
                text = str(text)
            text = text.replace(",", "")
            if "####" in text:
                return text.split("####")[-1].strip()
            match = re.search(
                r"(?:answer|result)\s*(?:is|=|:)\s*(-?\d+\.?\d*)",
                text,
                re.IGNORECASE,
            )
            if match:
                return match.group(1)
            numbers = re.findall(r"-?\d+\.?\d*", text)
            return numbers[-1] if numbers else text.strip()

        try:
            pred_val = float(extract_number(predicted))
            gt_val = float(extract_number(ground_truth))
        except (ValueError, TypeError):
            return False

        return abs(pred_val - gt_val) <= tolerance

    @staticmethod
    def semantic_match(
        predicted: str,
        ground_truth: str,
        embedder,
        threshold: float = 0.85,
    ) -> bool:
        """Embedding-based semantic similarity.

        ``embedder`` is a callable that takes ``List[str]`` and returns a 2D
        torch tensor of shape ``(N, D)``. Returns True if cosine similarity
        between the two embeddings is >= threshold. On any error, returns
        False.
        """
        try:
            embeddings = embedder([predicted, ground_truth])
            try:
                from sentence_transformers.util import cos_sim  # type: ignore
                sim = cos_sim(embeddings[0], embeddings[1])
                # cos_sim returns a 2D tensor
                value = float(sim.reshape(-1)[0].item())
            except Exception:
                import torch
                a = embeddings[0]
                b = embeddings[1]
                a = a.reshape(-1).float()
                b = b.reshape(-1).float()
                denom = (torch.norm(a) * torch.norm(b)).item()
                if denom == 0:
                    return False
                value = float(torch.dot(a, b).item() / denom)
            return value >= threshold
        except Exception:
            return False

    @staticmethod
    def llm_judge_match(
        predicted: str,
        ground_truth: str,
        judge,
        question: str = "",
    ) -> bool:
        """Delegate scoring to an LLM judge.

        ``judge`` must expose ``.score(question, predicted, ground_truth) -> bool``.
        Returns False on any exception.
        """
        try:
            return bool(judge.score(question, predicted, ground_truth))
        except Exception:
            return False
