# # Copyright (c) 2024 Microsoft Corporation.
# # Licensed under the MIT License

# """OpenAI Embedding model implementation."""
# in graphrag/query/llm/oai directory
# updated to run with lm-studio llm and embedding
# matrodge@cisco.com - 7/11/24
#
import logging
import numpy as np
from collections.abc import Callable
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential

from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.oai.base import OpenAILLMImpl
from graphrag.query.llm.text_utils import chunk_text
from graphrag.query.progress import StatusReporter
import asyncio
from collections.abc import Callable
from typing import Any
from typing import Union

import numpy as np
import tiktoken
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.oai.base import OpenAILLMImpl
from graphrag.query.llm.oai.typing import (
    OPENAI_RETRY_ERROR_TYPES,
    OpenaiApiType,
)
from graphrag.query.llm.text_utils import chunk_text
from graphrag.query.progress import StatusReporter
import logging
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenAIEmbedding(BaseTextEmbedding, OpenAILLMImpl):
    """Wrapper for OpenAI Embedding models."""

    def __init__(
        self,
        api_key: str | None = None,
        azure_ad_token_provider: Callable | None = None,
        model: str = "text-embedding-3-small",
        deployment_name: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        api_type: OpenaiApiType = OpenaiApiType.OpenAI,
        organization: str | None = None,
        encoding_name: str = "cl100k_base",
        max_tokens: int = 8191,
        max_retries: int = 10,
        request_timeout: float = 180.0,
        retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERROR_TYPES,  # type: ignore
        reporter: StatusReporter | None = None,
    ):
        OpenAILLMImpl.__init__(
            self=self,
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            deployment_name=deployment_name,
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,  # type: ignore
            organization=organization,
            max_retries=max_retries,
            request_timeout=request_timeout,
            reporter=reporter,
        )

        self.model = model
        self.encoding_name = encoding_name
        self.max_tokens = max_tokens
        self.token_encoder = tiktoken.get_encoding(self.encoding_name)
        self.retry_error_types = retry_error_types

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        logging.debug(f"Embedding text: {text[:50]}...")

        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )

        chunk_embeddings = []
        chunk_lens = []
        for chunk in token_chunks:
            if not isinstance(chunk, (str, list)) or (isinstance(chunk, list) and not all(isinstance(item, str) for item in chunk)):
                logging.error(f"Invalid chunk type: {type(chunk)}, chunk: {chunk}")
                self._reporter.error(
                    message="Error embedding chunk",
                    details={self.__class__.__name__: "Chunk is not a string or list of strings"},
                )
                continue
            try:
                logging.debug(f"Processing chunk: {chunk}")
                embedding, chunk_len = self._embed_with_retry(chunk, **kwargs)
                chunk_embeddings.append(embedding)
                chunk_lens.append(chunk_len)
            except Exception as e:  # noqa BLE001
                logging.error(f"Exception while embedding chunk: {str(e)}")
                self._reporter.error(
                    message="Error embedding chunk",
                    details={self.__class__.__name__: str(e)},
                )
                continue

        if not chunk_embeddings or not chunk_lens:
            logging.error("No valid chunks were processed. Returning empty embedding.")
            return []

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        logging.debug(f"Embedding text: {text[:50]}...")

        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )

        # Log the type and content of each chunk
        for chunk in token_chunks:
            logging.debug(f"Chunk type: {type(chunk)}, chunk: {chunk}")

        chunk_embeddings = []
        chunk_lens = []
        embedding_results = await asyncio.gather(*[
            self._aembed_with_retry(chunk, **kwargs) for chunk in token_chunks
        ])
        embedding_results = [result for result in embedding_results if result[0]]
        chunk_embeddings = [result[0] for result in embedding_results]
        chunk_lens = [result[1] for result in embedding_results]

        # Check if chunk_embeddings and chunk_lens are empty
        if not chunk_embeddings or not chunk_lens:
            logging.error("No valid chunks were processed. Returning empty embedding.")
            return []

        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)  # type: ignore
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        logging.debug(f"Final embedding: {chunk_embeddings.tolist()[:10]}...")
        return chunk_embeddings.tolist()
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _embed_with_retry(self, text: str | tuple, **kwargs: Any) -> tuple[list[float], int]:
        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    embedding = (
                        self.sync_client.embeddings.create(  # type: ignore
                            input=text,
                            model=self.model,
                            **kwargs,  # type: ignore
                        )
                        .data[0]
                        .embedding
                        or []
                    )
                    logging.debug(f"Generated embedding: {embedding} for text: {text}")
                    return (embedding, len(text))
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            logging.error(f"RetryError in _embed_with_retry: {str(e)}")
            return ([], 0)
        else:
            return ([], 0)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _aembed_with_retry(self, text: str | tuple, **kwargs: Any) -> tuple[list[float], int]:
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            async for attempt in retryer:
                with attempt:
                    embedding = (
                        await self.async_client.embeddings.create(  # type: ignore
                            input=text,
                            model=self.model,
                            **kwargs,  # type: ignore
                        )
                    ).data[0].embedding or []
                    logging.debug(f"Generated async embedding: {embedding} for text: {text}")
                    return (embedding, len(text))
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            logging.error(f"RetryError in _aembed_with_retry: {str(e)}")
            return ([], 0)
        else:
            return ([], 0)
