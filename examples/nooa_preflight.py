"""Minimal Predict-only NOOA integration with deterministic preflight."""

from __future__ import annotations

from typing import Any, Literal

from nooa import Agent, PredictStrategy, no_trace, strategy
from nooa.config import PredictConfig

from morgott.models.cascade import CascadeAssessment, CascadeScanner
from morgott.models.deepseek_nooa import refuse_nooa_tracing

InputChannel = Literal["direct_user", "untrusted_content"]


class AdvisoryAgent(Agent):
    """Example application agent whose learned assessment never grants authority."""

    def __init__(self, scanner: CascadeScanner, *, llm: Any) -> None:
        refuse_nooa_tracing()
        super().__init__(llm=llm)
        self._scanner = scanner

    @no_trace
    async def preflight(
        self,
        content: str,
        input_channel: InputChannel,
    ) -> CascadeAssessment:
        return await self._scanner.assess_text(
            content,
            input_channel=input_channel,
        )

    @no_trace
    @strategy(
        PredictStrategy(
            PredictConfig(
                max_retries=2,
                max_tokens=256,
                temperature=0,
            )
        )
    )
    async def handle(
        self,
        content: str,
        assessment: CascadeAssessment,
    ) -> str:
        """Respond to content while treating assessment only as advisory risk context."""
        ...


async def handle_with_preflight(
    agent: AdvisoryAgent,
    content: str,
    input_channel: InputChannel,
) -> str:
    assessment = await agent.preflight(content, input_channel)
    return await agent.handle(content, assessment)
