from __future__ import annotations

import importlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.rag.retriever import RAGEvidence

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


class AsyncChatModel(Protocol):
    async def ainvoke(self, input: object) -> object: ...


@dataclass(frozen=True, slots=True)
class GenerationResult:
    answer: str
    cited_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    max_chars_per_source: int = 4000
    max_total_context_chars: int = 16000
    no_evidence_answer: str = (
        "I could not find enough indexed evidence to answer this question."
    )

    def __post_init__(self) -> None:
        if self.max_chars_per_source < 1:
            raise ValueError("max_chars_per_source must be positive")
        if self.max_total_context_chars < self.max_chars_per_source:
            raise ValueError("max_total_context_chars must cover at least one source")


class LangChainAnswerGenerator:
    """Grounded answer generation over any LangChain async chat model."""

    def __init__(
        self,
        model: AsyncChatModel,
        *,
        config: GeneratorConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or GeneratorConfig()

    @classmethod
    def openai(
        cls,
        *,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        config: GeneratorConfig | None = None,
    ) -> LangChainAnswerGenerator:
        module = importlib.import_module("langchain_openai")
        chat_model = module.ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )
        return cls(chat_model, config=config)

    @classmethod
    def llama_ollama(
        cls,
        *,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        config: GeneratorConfig | None = None,
    ) -> LangChainAnswerGenerator:
        module = importlib.import_module("langchain_ollama")
        chat_model = module.ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        return cls(chat_model, config=config)

    async def generate(
        self,
        query: str,
        evidence: Sequence[RAGEvidence],
    ) -> GenerationResult:
        if not evidence:
            return GenerationResult(
                answer=self.config.no_evidence_answer,
                cited_ids=(),
            )
        prompt = self.build_prompt(query, evidence)
        response = await self.model.ainvoke(prompt)
        answer = _response_text(response).strip()
        if not answer:
            answer = self.config.no_evidence_answer
        valid_ids = {item.citation_id for item in evidence}
        cited_ids = tuple(
            dict.fromkeys(
                citation
                for citation in CITATION_PATTERN.findall(answer)
                if citation in valid_ids
            )
        )
        return GenerationResult(answer=answer, cited_ids=cited_ids)

    def build_prompt(
        self,
        query: str,
        evidence: Sequence[RAGEvidence],
    ) -> str:
        context_blocks: list[str] = []
        total_length = 0
        for item in evidence:
            block = item.prompt_block(max_chars=self.config.max_chars_per_source)
            additional = len(block) + (2 if context_blocks else 0)
            if (
                context_blocks
                and total_length + additional > self.config.max_total_context_chars
            ):
                break
            context_blocks.append(block)
            total_length += additional

        context = "\n\n".join(context_blocks)
        return (
            "You are a grounded research assistant. Answer the user using "
            "only the supplied evidence.\n"
            "Rules:\n"
            "1. Cite factual claims with source IDs exactly like [S1].\n"
            "2. Never invent a citation or use outside knowledge.\n"
            "3. If evidence is insufficient or conflicting, state that "
            "clearly.\n"
            "4. Keep the answer direct and distinguish uncertainty.\n\n"
            f"Question:\n{query.strip()}\n\n"
            f"Evidence:\n{context}\n\n"
            "Grounded answer:"
        )


def _response_text(response: object) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    raise TypeError("LangChain model returned unsupported response content")
