"""Generative LLM wrapper using LangChain for universal provider support.

Supports: OpenAI, Google Gemini, Anthropic Claude, Ollama, HuggingFace
(both Inference API and local via transformers).
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple

from routent.models.base import BaseLLM


_tokenizer = None


def _count_tokens(text: str) -> int:
    global _tokenizer
    if _tokenizer is None:
        import tiktoken
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(_tokenizer.encode(text))


# Known constructor params per provider; anything else goes into model_kwargs
_KNOWN_PARAMS = {
    "openai": {"temperature", "max_tokens", "max_completion_tokens", "top_p",
               "timeout", "max_retries", "stop", "reasoning_effort"},
    "google": {"temperature", "max_tokens", "top_p", "top_k", "timeout", "max_retries"},
    "anthropic": {"temperature", "max_tokens", "top_p", "top_k", "timeout", "max_retries", "stop"},
    "ollama": {"temperature", "top_p", "top_k", "num_predict", "stop", "timeout"},
    "huggingface": {"temperature", "max_new_tokens", "top_p", "top_k", "repetition_penalty", "task"},
    "huggingface_local": {"temperature", "max_new_tokens", "top_p", "top_k",
                          "repetition_penalty", "device", "task"},
}


def is_hf_model_cached(model_id: str) -> bool:
    """Check if a HuggingFace model is already in local cache."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        return any(repo.repo_id == model_id for repo in cache.repos)
    except Exception:
        return False


def print_download_notice(model_id: str) -> None:
    """Print 'downloading' or 'loading from cache' notice."""
    if not is_hf_model_cached(model_id):
        print(f"  downloading {model_id}... (first-time download may take a while)")
    else:
        print(f"  loading {model_id} from cache")


def _create_langchain_llm(provider: str, model_name: str, **kwargs):
    """Create a LangChain chat model for the given provider."""
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
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        endpoint_kwargs = {k: v for k, v in direct_params.items() if k != "model_kwargs"}
        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            task=endpoint_kwargs.pop("task", "text-generation"),
            **endpoint_kwargs,
        )
        return ChatHuggingFace(llm=endpoint)

    elif provider in ("huggingface_local", "hf_local"):
        from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
        print_download_notice(model_name)
        pipeline_kwargs = {k: v for k, v in direct_params.items() if k != "model_kwargs"}
        pipeline = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task=pipeline_kwargs.pop("task", "text-generation"),
            device=pipeline_kwargs.pop("device", -1),
            pipeline_kwargs={k: v for k, v in pipeline_kwargs.items()
                             if k != "temperature" or v > 0},
        )
        return ChatHuggingFace(llm=pipeline)

    else:
        raise ValueError(
            f"Unknown generative provider '{provider}'. "
            f"Supported: openai, google, anthropic, ollama, huggingface, huggingface_local"
        )


SYSTEM_PROMPT = ""


def _extract_text(result) -> str:
    """Extract plain text from a LangChain response (all providers)."""
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
    """Remove chat template tokens and extract only the assistant's response."""
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
        text = text.lstrip("\n").strip()
        if "<|im_end|>" in text:
            text = text.split("<|im_end|>")[0]
        return text.strip()

    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        text = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        if "<|eot_id|>" in text:
            text = text.split("<|eot_id|>")[0]
        return text.strip()

    if "[/INST]" in text:
        text = text.split("[/INST]")[-1]
        if "</s>" in text:
            text = text.split("</s>")[0]
        return text.strip()

    return text.strip()


class GenerativeLLM(BaseLLM):
    """Text-generation LLM via LangChain.

    Works with OpenAI, Google Gemini, Anthropic, Ollama, and HuggingFace
    (both Inference API and local transformers).
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
        input_tokens, output_tokens = self._extract_usage(result)
        if input_tokens == 0 and output_tokens == 0:
            input_tokens = _count_tokens(input_text)
            output_tokens = _count_tokens(output_text)
        return (
            (input_tokens / 1_000_000) * self.cost_per_1m_input
            + (output_tokens / 1_000_000) * self.cost_per_1m_output
        )

    @staticmethod
    def _extract_usage(result) -> Tuple[int, int]:
        # usage_metadata on AIMessage is a TypedDict — use .get(), not getattr().
        # For reasoning models (OpenAI o-series, gpt-5-*) output_tokens already
        # includes reasoning_tokens per the OpenAI usage spec, so no extra
        # addition is needed here.
        um = getattr(result, "usage_metadata", None)
        if um:
            if isinstance(um, dict):
                in_tok = int(um.get("input_tokens", 0) or 0)
                out_tok = int(um.get("output_tokens", 0) or 0)
            else:
                in_tok = int(getattr(um, "input_tokens", 0) or 0)
                out_tok = int(getattr(um, "output_tokens", 0) or 0)
            if in_tok or out_tok:
                return in_tok, out_tok

        meta = getattr(result, "response_metadata", None)
        if meta and "token_usage" in meta:
            tu = meta["token_usage"]
            return (int(tu.get("prompt_tokens", 0) or 0),
                    int(tu.get("completion_tokens", 0) or 0))
        return 0, 0

    async def _ainvoke_one(self, prompt: str) -> Tuple[str, float, float]:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        input_text = self.system_prompt + " " + prompt

        start = time.perf_counter()
        try:
            try:
                result = await self._llm.ainvoke(messages)
            except NotImplementedError:
                result = await asyncio.to_thread(self._llm.invoke, messages)
            elapsed_ms = (time.perf_counter() - start) * 1000
            text = _extract_text(result)
            cost = self._estimate_cost(result, input_text, text)
            return text, elapsed_ms, cost
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            err = str(e).lower()
            if "async generation is not supported" in err or "does not support async" in err:
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

    def batch_generate(self, prompts: List[str]) -> List[Tuple[str, float, float]]:
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
