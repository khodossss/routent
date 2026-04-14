"""Real LLM wrapper using LangChain for universal provider support."""

import asyncio
import time
from typing import Dict, List, Optional, Tuple

import tiktoken

from routent.models.llm_pool import BaseLLM

# Shared tokenizer for cost estimation (cl100k_base covers GPT-4/3.5, good approximation for others)
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    return len(_tokenizer.encode(text))

# LangChain imports are deferred to avoid hard dependency in simulated mode


# Known constructor params per provider (anything else goes into model_kwargs)
_KNOWN_PARAMS = {
    "openai": {"temperature", "max_tokens", "max_completion_tokens", "top_p", "timeout", "max_retries", "stop", "reasoning_effort"},
    "google": {"temperature", "max_tokens", "top_p", "top_k", "timeout", "max_retries"},
    "anthropic": {"temperature", "max_tokens", "top_p", "top_k", "timeout", "max_retries", "stop"},
    "ollama": {"temperature", "top_p", "top_k", "num_predict", "stop", "timeout"},
    "huggingface": {"temperature", "max_new_tokens", "top_p", "top_k", "repetition_penalty", "task"},
    "huggingface_local": {"temperature", "max_new_tokens", "top_p", "top_k", "repetition_penalty", "device", "task"},
}


def _is_hf_model_cached(model_id: str) -> bool:
    """Check if a HuggingFace model is already downloaded to local cache."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == model_id:
                return True
        return False
    except Exception:
        return False


def _print_download_notice(model_id: str) -> None:
    """Print 'downloading' notice if model isn't cached yet."""
    if not _is_hf_model_cached(model_id):
        print(f"  downloading {model_id}... (first-time download may take a while)")
    else:
        print(f"  loading {model_id} from cache")


def _create_langchain_llm(provider: str, model_name: str, **kwargs):
    """Create a LangChain chat model for the given provider.

    Supported providers: openai, google, anthropic, ollama.
    API keys are read from environment variables (loaded from .env).
    Known params are passed directly; unknown params go into model_kwargs.
    """
    provider = provider.lower()
    known = _KNOWN_PARAMS.get(provider, set())

    direct_params = {"temperature": 0}
    extra_model_kwargs = {}

    for k, v in kwargs.items():
        if k in known:
            direct_params[k] = v
        else:
            extra_model_kwargs[k] = v

    if extra_model_kwargs:
        direct_params["model_kwargs"] = extra_model_kwargs

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, **direct_params)

    elif provider in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, **direct_params)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, **direct_params)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, **direct_params)

    elif provider in ("huggingface", "hf"):
        # HuggingFace Serverless Inference API (remote, needs HF_TOKEN)
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        endpoint_kwargs = {k: v for k, v in direct_params.items() if k != "model_kwargs"}
        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            task=endpoint_kwargs.pop("task", "text-generation"),
            **endpoint_kwargs,
        )
        return ChatHuggingFace(llm=endpoint)

    elif provider in ("huggingface_local", "hf_local"):
        # Local inference via transformers (no API, runs on your machine)
        from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

        _print_download_notice(model_name)

        pipeline_kwargs = {k: v for k, v in direct_params.items() if k != "model_kwargs"}
        pipeline = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task=pipeline_kwargs.pop("task", "text-generation"),
            device=pipeline_kwargs.pop("device", -1),  # -1 = CPU, 0 = first GPU
            pipeline_kwargs={k: v for k, v in pipeline_kwargs.items() if k != "temperature" or v > 0},
        )
        return ChatHuggingFace(llm=pipeline)

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: openai, google, anthropic, ollama, huggingface, huggingface_local"
        )


SYSTEM_PROMPT = "Answer with ONLY the final numeric answer. No explanation, no units, no words. Just the number."


def _extract_text(result) -> str:
    """Extract plain text from a LangChain response.

    Handles different provider formats:
    - OpenAI/Anthropic: result.content is a string
    - Google: result.content can be a list of dicts with 'text' keys
    - HuggingFacePipeline: returns full prompt + completion with chat template tokens
    """
    content = result.content if hasattr(result, "content") else result

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        content = " ".join(parts)

    text = content if isinstance(content, str) else str(content)
    return _strip_chat_template(text)


def _strip_chat_template(text: str) -> str:
    """Remove chat template tokens and extract only the assistant's response.

    Handles common chat formats:
    - ChatML: <|im_start|>assistant\\n...<|im_end|>
    - Llama 3: <|start_header_id|>assistant<|end_header_id|>...
    - Mistral/Llama 2: [INST] ... [/INST]
    """
    # ChatML format (Qwen, modern Llama)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
        text = text.lstrip("\n").strip()
        if "<|im_end|>" in text:
            text = text.split("<|im_end|>")[0]
        return text.strip()

    # Llama 3 format
    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        text = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        if "<|eot_id|>" in text:
            text = text.split("<|eot_id|>")[0]
        return text.strip()

    # Mistral/Llama 2 format
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1]
        if "</s>" in text:
            text = text.split("</s>")[0]
        return text.strip()

    return text.strip()


class RealLLM(BaseLLM):
    """Real LLM that calls an API via LangChain.

    Measures latency from the actual API call.
    Cost is estimated from token count * cost_per_1m_input.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        cost_per_1m_input: float = 0.0,
        cost_per_1m_output: float = 0.0,
        system_prompt: str = SYSTEM_PROMPT,
        max_concurrency: int = 1,
        **kwargs,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        # Build unique name including non-default kwargs
        suffix = ",".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self.name = f"{provider}/{model_name}" + (f"({suffix})" if suffix else "")
        self.cost_per_1m_input = cost_per_1m_input
        self.cost_per_1m_output = cost_per_1m_output
        self.system_prompt = system_prompt
        self.max_concurrency = max(1, max_concurrency)
        self._llm = _create_langchain_llm(provider, model_name, **kwargs)

    def generate(
        self, prompt: str, answer: str, difficulty: str, category: str
    ) -> Tuple[str, float, float]:
        """Call the real LLM and return (response, latency_ms, cost).

        Args:
            prompt: The question to send.
            answer: Ground truth (not sent to model, used by env for eval).
            difficulty: Question difficulty (metadata, not sent to model).
            category: Question category (metadata, not sent to model).

        Returns:
            (response_text, latency_ms, estimated_cost)
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]

        input_text = self.system_prompt + " " + prompt

        start = time.perf_counter()
        try:
            result = self._llm.invoke(messages)
            elapsed_ms = (time.perf_counter() - start) * 1000

            response_text = _extract_text(result)
            cost = self._estimate_cost(result, input_text, response_text)

            return response_text, elapsed_ms, cost

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return f"ERROR: {e}", elapsed_ms, 0.0

    def _estimate_cost(self, result, input_text: str, output_text: str) -> float:
        """Estimate cost from API usage metadata (includes reasoning tokens).

        Falls back to tiktoken if usage metadata is unavailable.
        """
        input_tokens, output_tokens = self._extract_usage(result)

        if input_tokens == 0 and output_tokens == 0:
            # Fallback: tiktoken
            input_tokens = _count_tokens(input_text)
            output_tokens = _count_tokens(output_text)

        return (
            (input_tokens / 1_000_000) * self.cost_per_1m_input
            + (output_tokens / 1_000_000) * self.cost_per_1m_output
        )

    @staticmethod
    def _extract_usage(result) -> Tuple[int, int]:
        """Extract input and output token counts from LangChain result.

        Output tokens include reasoning tokens when available.
        """
        input_tokens = 0
        output_tokens = 0

        # Try usage_metadata (LangChain standard)
        if hasattr(result, "usage_metadata") and result.usage_metadata:
            um = result.usage_metadata
            input_tokens = getattr(um, "input_tokens", 0) or 0
            output_tokens = getattr(um, "output_tokens", 0) or 0
            return input_tokens, output_tokens

        # Try response_metadata (OpenAI raw)
        if hasattr(result, "response_metadata"):
            meta = result.response_metadata
            if "token_usage" in meta:
                tu = meta["token_usage"]
                input_tokens = tu.get("prompt_tokens", 0) or 0
                output_tokens = tu.get("completion_tokens", 0) or 0

        return input_tokens, output_tokens

    async def _ainvoke_one(self, prompt: str) -> Tuple[str, float, float]:
        """Invoke a single prompt async, measuring individual latency.

        Falls back to running sync invoke in a thread for providers that
        don't support async (e.g. HuggingFacePipeline local).
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]

        input_text = self.system_prompt + " " + prompt

        start = time.perf_counter()
        try:
            try:
                result = await self._llm.ainvoke(messages)
            except NotImplementedError:
                # Provider doesn't support async (e.g. HuggingFacePipeline local)
                result = await asyncio.to_thread(self._llm.invoke, messages)
            elapsed_ms = (time.perf_counter() - start) * 1000

            text = _extract_text(result)
            cost = self._estimate_cost(result, input_text, text)
            return text, elapsed_ms, cost

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Catch async-unsupported error message too
            if "async generation is not supported" in str(e).lower() or "does not support async" in str(e).lower():
                try:
                    start = time.perf_counter()
                    result = await asyncio.to_thread(self._llm.invoke, messages)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    text = _extract_text(result)
                    cost = self._estimate_cost(result, input_text, text)
                    return text, elapsed_ms, cost
                except Exception as e2:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    return f"ERROR: {e2}", elapsed_ms, 0.0
            return f"ERROR: {e}", elapsed_ms, 0.0

    def batch_generate(
        self, prompts: List[str]
    ) -> List[Tuple[str, float, float]]:
        """Send multiple prompts with concurrency controlled by max_concurrency.

        - max_concurrency=1: fully sequential (best for local CPU models,
          accurate latency).
        - max_concurrency>1: parallel via asyncio.Semaphore (best for
          vendor APIs that handle concurrent requests well).

        Returns:
            List of (response_text, latency_ms, cost) for each prompt.
        """
        if self.max_concurrency == 1:
            return [self.generate(p, "", "", "") for p in prompts]

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _one(p):
            async with sem:
                return await self._ainvoke_one(p)

        async def _run_all():
            return await asyncio.gather(*[_one(p) for p in prompts])

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(_run_all())
        return asyncio.run(_run_all())


class RealLLMPool:
    """Pool of real LLM models created from config."""

    def __init__(
        self,
        provider_models: list,
        model_costs_input: Optional[list] = None,
        model_costs_output: Optional[list] = None,
        model_kwargs: Optional[List[Dict]] = None,
        model_concurrency: Optional[List[int]] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        """
        Args:
            provider_models: List of (provider, model_name) tuples.
            model_costs_input: Cost per 1M input tokens per model.
            model_costs_output: Cost per 1M output tokens per model.
            model_kwargs: List of dicts with LangChain constructor kwargs per model.
            model_concurrency: Max concurrent requests per model (1 = sequential).
                               Use >1 for vendor APIs, 1 for local CPU models.
            system_prompt: System prompt sent with every request.
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

        self.models = []
        for (provider, model_name), ci, co, kw, conc in zip(
            provider_models, model_costs_input, model_costs_output, model_kwargs, model_concurrency
        ):
            self.models.append(
                RealLLM(provider, model_name, ci, co, system_prompt, max_concurrency=conc, **kw)
            )

    def get_model(self, index: int) -> RealLLM:
        return self.models[index]

    @property
    def num_models(self) -> int:
        return len(self.models)

    @property
    def model_names(self) -> list:
        return [m.name for m in self.models]
