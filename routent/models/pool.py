"""Unified model pool that dispatches to the correct wrapper per provider."""

from typing import Dict, List, Optional

from routent.models.base import BaseLLM


# Providers handled by GenerativeLLM (text generation)
_GENERATIVE_PROVIDERS = {
    "openai", "google", "gemini", "anthropic", "ollama",
    "huggingface", "hf", "huggingface_local", "hf_local",
}


def _create_model(
    provider: str,
    model_name: str,
    cost_per_1m_input: float = 0.0,
    cost_per_1m_output: float = 0.0,
    system_prompt: str = "",
    max_concurrency: int = 1,
    labels: Optional[List[str]] = None,
    label_map: Optional[Dict[int, str]] = None,
    multilabel: bool = False,
    threshold: float = 0.5,
    **kwargs,
) -> BaseLLM:
    """Factory: create the right BaseLLM implementation for the provider."""
    provider_lower = provider.lower()

    if provider_lower in _GENERATIVE_PROVIDERS:
        from routent.models.generative import GenerativeLLM, SYSTEM_PROMPT
        return GenerativeLLM(
            provider=provider,
            model_name=model_name,
            cost_per_1m_input=cost_per_1m_input,
            cost_per_1m_output=cost_per_1m_output,
            system_prompt=system_prompt or SYSTEM_PROMPT,
            max_concurrency=max_concurrency,
            **kwargs,
        )

    elif provider_lower == "hf_classifier":
        from routent.models.hf_classifier import HFClassifierLLM
        return HFClassifierLLM(
            model_name=model_name,
            label_map=label_map,
            multilabel=multilabel,
            threshold=threshold,
            cost_per_1m_input=cost_per_1m_input,
            **kwargs,
        )

    elif provider_lower == "hf_zero_shot":
        from routent.models.hf_zero_shot import HFZeroShotLLM
        if not labels:
            raise ValueError(
                f"hf_zero_shot provider requires 'labels' (candidate classes) — "
                f"set model_labels[i] in config for model {model_name}"
            )
        return HFZeroShotLLM(
            model_name=model_name,
            labels=labels,
            multilabel=multilabel,
            threshold=threshold,
            cost_per_1m_input=cost_per_1m_input,
            **kwargs,
        )

    elif provider_lower == "hf_regressor":
        from routent.models.hf_regressor import HFRegressorLLM
        return HFRegressorLLM(
            model_name=model_name,
            cost_per_1m_input=cost_per_1m_input,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: "
            f"{sorted(_GENERATIVE_PROVIDERS)} + "
            f"hf_classifier, hf_zero_shot, hf_regressor"
        )


class LLMPool:
    """Unified pool of models (generative + classifiers + regressors)."""

    def __init__(
        self,
        provider_models: list,
        model_costs_input: Optional[list] = None,
        model_costs_output: Optional[list] = None,
        model_kwargs: Optional[List[Dict]] = None,
        model_concurrency: Optional[List[int]] = None,
        model_labels: Optional[List[Optional[List[str]]]] = None,
        system_prompt: str = "",
    ) -> None:
        """
        Args:
            provider_models: List of (provider, model_name) tuples.
            model_costs_input: Cost per 1M input tokens per model.
            model_costs_output: Cost per 1M output tokens per model.
            model_kwargs: Per-model extra kwargs passed to wrapper constructor.
            model_concurrency: Max concurrent calls per model (1 = sequential).
            model_labels: Per-model candidate labels for zero-shot classifiers.
                          None for models that don't need them.
            system_prompt: Default system prompt for generative models.
        """
        n = len(provider_models)
        if model_costs_input is None:
            model_costs_input = [0.0] * n
        if model_costs_output is None:
            model_costs_output = [0.0] * n
        if model_kwargs is None:
            model_kwargs = [{}] * n
        if model_concurrency is None:
            model_concurrency = [1] * n
        if model_labels is None:
            model_labels = [None] * n

        self.models: List[BaseLLM] = []
        for i, (provider, model_name) in enumerate(provider_models):
            kw = dict(model_kwargs[i]) if i < len(model_kwargs) else {}
            # Extract special per-model flags from kwargs if present
            label_map = kw.pop("label_map", None)
            multilabel = kw.pop("multilabel", False)
            threshold = kw.pop("threshold", 0.5)

            model = _create_model(
                provider=provider,
                model_name=model_name,
                cost_per_1m_input=model_costs_input[i],
                cost_per_1m_output=model_costs_output[i],
                system_prompt=system_prompt,
                max_concurrency=model_concurrency[i],
                labels=model_labels[i],
                label_map=label_map,
                multilabel=multilabel,
                threshold=threshold,
                **kw,
            )
            self.models.append(model)

    def get_model(self, index: int) -> BaseLLM:
        return self.models[index]

    @property
    def num_models(self) -> int:
        return len(self.models)

    @property
    def model_names(self) -> list:
        return [m.name for m in self.models]
