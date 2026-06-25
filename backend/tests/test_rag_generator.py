from dataclasses import dataclass
from uuid import uuid4

import pytest

from app.domain.models import RetrievalCandidate
from app.rag.generator import GeneratorConfig, LangChainAnswerGenerator
from app.rag.retriever import RAGEvidence


@dataclass
class FakeResponse:
    content: str


class FakeChatModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompt = ""

    async def ainvoke(self, input: object) -> object:
        self.prompt = str(input)
        return FakeResponse(self.answer)


def evidence(citation_id: str = "S1") -> RAGEvidence:
    return RAGEvidence(
        citation_id=citation_id,
        candidate=RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=uuid4(),
            title="Quantum retrieval",
            content="State overlap measures query-document relevance.",
        ),
        rank=1,
        hybrid_score=0.7,
        quantum_score=0.9,
        context_score=0.8,
        final_score=0.82,
    )


@pytest.mark.asyncio
async def test_generator_constructs_grounded_prompt_and_validates_citations() -> None:
    model = FakeChatModel("State overlap is used for relevance [S1]. Invalid [S9].")
    generator = LangChainAnswerGenerator(model)

    result = await generator.generate("How is relevance scored?", [evidence()])

    assert result.cited_ids == ("S1",)
    assert "[S1]" in model.prompt
    assert "only the supplied evidence" in model.prompt


@pytest.mark.asyncio
async def test_generator_does_not_call_model_without_evidence() -> None:
    model = FakeChatModel("should not be used")
    generator = LangChainAnswerGenerator(model)

    result = await generator.generate("Question", [])

    assert result.cited_ids == ()
    assert "could not find enough" in result.answer
    assert model.prompt == ""


def test_prompt_respects_context_budget() -> None:
    generator = LangChainAnswerGenerator(
        FakeChatModel("answer"),
        config=GeneratorConfig(
            max_chars_per_source=20,
            max_total_context_chars=80,
        ),
    )

    prompt = generator.build_prompt(
        "Question",
        [evidence("S1"), evidence("S2"), evidence("S3")],
    )

    assert "State overlap mea..." in prompt
    assert len(prompt) < 1000
