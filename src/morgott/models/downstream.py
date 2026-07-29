"""Advisory routing for the selected ``mmbert-lora-s42`` cascade."""

import math
from dataclasses import dataclass
from typing import Literal

MMBERT_LOW = 0.2
MMBERT_HIGH = 0.999
LLM_FLAG_PROBABILITY = 0.9


@dataclass(frozen=True)
class RouteResult:
    route: Literal["pass", "review", "restrict"]
    reason: str


def subversion_probability(logprob_0: float, logprob_1: float) -> float:
    """Return the stable sigmoid of the binary decision-token log odds."""
    for value in (logprob_0, logprob_1):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("decision-token log probabilities must be finite")
    margin = logprob_1 - logprob_0
    if not math.isfinite(margin):
        raise ValueError("decision-token log-odds margin must be finite")
    if margin >= 0:
        return 1.0 / (1.0 + math.exp(-margin))
    exponent = math.exp(margin)
    return exponent / (1.0 + exponent)


def route(
    mmbert_score: float,
    *,
    llm_probability: float | None = None,
    llm_failed: bool = False,
) -> RouteResult:
    """Return an advisory route; ``pass`` never grants authorization."""
    if (
        isinstance(mmbert_score, bool)
        or not isinstance(mmbert_score, (int, float))
        or not math.isfinite(mmbert_score)
        or not 0 <= mmbert_score <= 1
    ):
        raise ValueError("mmbert_score must be a finite probability")
    if not isinstance(llm_failed, bool):
        raise ValueError("llm_failed must be a boolean")
    if llm_failed and llm_probability is not None:
        raise ValueError("DeepSeek result and failure are mutually exclusive")
    if llm_probability is not None and (
        isinstance(llm_probability, bool)
        or not isinstance(llm_probability, (int, float))
        or not math.isfinite(llm_probability)
        or not 0 <= llm_probability <= 1
    ):
        raise ValueError("llm_probability must be a finite probability")
    if (llm_probability is not None or llm_failed) and not (
        MMBERT_LOW <= mmbert_score < MMBERT_HIGH
    ):
        raise ValueError("DeepSeek result is only valid in the middle zone")
    if mmbert_score < MMBERT_LOW:
        return RouteResult("pass", "mmbert_low")
    if mmbert_score >= MMBERT_HIGH:
        return RouteResult("restrict", "mmbert_high")
    if llm_failed:
        return RouteResult("restrict", "deepseek_failed")
    if llm_probability is not None:
        if llm_probability < LLM_FLAG_PROBABILITY:
            return RouteResult("pass", "deepseek_clear")
        return RouteResult("restrict", "deepseek_flag")
    return RouteResult("review", "deepseek_required")
