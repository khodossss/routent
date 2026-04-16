"""Answer correctness checker for LLM Router RL."""

import difflib
import json
import re
from typing import Dict, List, Optional, Union


class Evaluator:
    """Evaluates predicted answers against ground truth."""

    @staticmethod
    def exact_match(predicted: str, ground_truth: str) -> bool:
        """Case-insensitive, stripped whitespace comparison."""
        return predicted.strip().lower() == ground_truth.strip().lower()

    @staticmethod
    def fuzzy_score(predicted: str, ground_truth: str) -> float:
        """SequenceMatcher ratio — continuous [0, 1]."""
        return difflib.SequenceMatcher(
            None,
            predicted.strip().lower(),
            ground_truth.strip().lower(),
        ).ratio()

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
    def classification_confidence_score(
        predicted: str, ground_truth: str,
    ) -> float:
        """Extract probability of the correct class from a JSON response.

        Expects the model to return something like: {"positive": 0.92, "negative": 0.08}
        Falls back to binary 0/1 if JSON parsing fails.
        """
        gt_lower = ground_truth.strip().lower()

        # Try to parse JSON from the response
        text = predicted.strip()
        # Handle markdown code blocks
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)

        try:
            probs = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            # Try to find JSON-like substring
            match = re.search(r"\{[^}]+\}", text)
            if match:
                try:
                    probs = json.loads(match.group(0))
                except (json.JSONDecodeError, TypeError):
                    return 0.0
            else:
                return 0.0

        if not isinstance(probs, dict):
            return 0.0

        # Find the probability of the ground truth class (case-insensitive)
        for key, val in probs.items():
            if str(key).strip().lower() == gt_lower:
                try:
                    return max(0.0, min(1.0, float(val)))
                except (ValueError, TypeError):
                    return 0.0

        return 0.0

    @staticmethod
    def _parse_confidence_json(text: str) -> dict:
        """Parse JSON probabilities from a model response. Returns {} on failure."""
        text = text.strip()
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if m:
                text = m.group(1)
        try:
            probs = json.loads(text)
            if isinstance(probs, dict):
                return {str(k): v for k, v in probs.items()}
        except (json.JSONDecodeError, TypeError):
            m = re.search(r"\{[^}]+\}", text)
            if m:
                try:
                    probs = json.loads(m.group(0))
                    if isinstance(probs, dict):
                        return {str(k): v for k, v in probs.items()}
                except (json.JSONDecodeError, TypeError):
                    pass
        return {}

    @staticmethod
    def multilabel_score(
        predicted: str, ground_truth: str, delimiter: str = ","
    ) -> float:
        """Partial credit for multilabel: Jaccard similarity of label sets.

        Returns |intersection| / |union|, so each correct label contributes.
        Full match → 1.0, no overlap → 0.0.
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
        if not gt_set and not pred_set:
            return 1.0
        if not gt_set or not pred_set:
            return 0.0
        intersection = pred_set & gt_set
        union = pred_set | gt_set
        return len(intersection) / len(union)

    @staticmethod
    def regression_score(
        predicted: str, ground_truth: str, tolerance: float = 0.01
    ) -> float:
        """Continuous quality score for regression: 1.0 at exact match, 0.0 at/beyond tolerance.

        Formula: max(0, 1 - |pred - gt| / tolerance)
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
            return 0.0

        if tolerance <= 0:
            return 1.0 if pred_val == gt_val else 0.0
        return max(0.0, 1.0 - abs(pred_val - gt_val) / tolerance)

    @staticmethod
    def semantic_score(
        predicted: str,
        ground_truth: str,
        embedder,
    ) -> float:
        """Cosine similarity between embeddings — continuous [0, 1].

        ``embedder`` is a callable that takes ``List[str]`` and returns a 2D
        torch tensor of shape ``(N, D)``.
        """
        try:
            embeddings = embedder([predicted, ground_truth])
            try:
                from sentence_transformers.util import cos_sim  # type: ignore
                sim = cos_sim(embeddings[0], embeddings[1])
                return max(0.0, float(sim.reshape(-1)[0].item()))
            except Exception:
                import torch
                a = embeddings[0].reshape(-1).float()
                b = embeddings[1].reshape(-1).float()
                denom = (torch.norm(a) * torch.norm(b)).item()
                if denom == 0:
                    return 0.0
                return max(0.0, float(torch.dot(a, b).item() / denom))
        except Exception:
            return 0.0

    @staticmethod
    def llm_judge_score(
        predicted: str,
        ground_truth: str,
        judge,
        question: str = "",
        criteria: Optional[Union[List[str], Dict[str, float]]] = None,
    ) -> float:
        """Delegate scoring to an LLM judge — returns float [0, 1].

        criteria formats:
        - None           → binary YES/NO (0.0/1.0)
        - list[str]      → equal-weight average over criteria
        - dict[str,float] → weighted average (weights auto-normalized)
        """
        try:
            if criteria:
                return judge.score_criteria(question, predicted, ground_truth, criteria)
            return float(judge.score(question, predicted, ground_truth))
        except Exception:
            return 0.0
