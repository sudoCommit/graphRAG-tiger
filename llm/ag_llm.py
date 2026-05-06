
import logging
from typing import Any, Optional

from openai import APIConnectionErrorm, RateLimitError
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)
LLM_ENDPOINT = "/large-language-models/{model_name}"


class AGOpenAIChat(ChatOpenAI):
    def __init__(self, model: str, **kwargs):
        kwargs.setdefault(
            "base_url", f"{AG_API_URL}/large-language-models-openai-compatible"
        )
        kwargs.setdefault("api_key", AG_TOKEN)
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("max_tokens", 4096)
        super().__init__(model=model, **kwargs)

    async def _get_model_price(self, model_name: str) -> dict:
        key = self.model_name.replace("-", "_").replace(".", "")
        return await self.make_get_request(LLM_ENDPOINT.format(model_name=key))

    async def _compute_cost(self, model_name: str, in_tokens: int, out_tokens: int) -> float:
        pricing = await self._get_model_price(model_name)
        return in_tokens * (pricing.get("in_token_cost", 0.0)) + out_tokens * (pricing.get("out_token_cost", 0.0))

    @retry(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_exponential_jitter(initial=2, max=30),
            stop=stop_after_attempt(3),
            before_sleep=before_sleep_log(logger, logger.warning),
            reraise=True,
    )
    async def ainvoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> BaseMessage:
        try:
            response = await super.ainvoke(input, config=config, **kwargs)
        except APIConnectionErrorm as e:
            logger.error(f"API connection error whe ncalling model {self.model_name}: {e}")

        if hasattr(response, "response_metadata"):
            token_usage = response.response_metadata.get("token_usage", {})
            in_token = token_usage.get("prompt_tokens", 0)
            out_token = token_usage.get("completion_tokens", 0)
            try:
                cost = await 
