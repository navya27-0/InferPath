"""LLM Judge interface and mock/real implementations for InferPath evaluation."""

import time
from typing import Protocol

from pydantic import BaseModel, Field

from inferpath.evaluation.dataset import DatasetItem


class JudgeEvaluationResult(BaseModel):
    """Result returned by an LLM Judge evaluation."""

    score: float = Field(ge=0.0, le=1.0, description="Normalized quality score between 0.0 and 1.0")
    pass_threshold: bool = Field(description="Whether score satisfies the Q_min threshold")
    rationale: str = Field(description="Judge's stated explanation")
    judge_model_id: str = Field(description="Identifier of the judge model")
    judge_latency_ms: float = Field(description="Monotonic latency of judge call in ms")
    judge_cost_usd: float = Field(description="Calculated cost of judge invocation in USD")
    rubric_version: str = Field(default="1.0.0")


class LLMJudge(Protocol):
    """Protocol for LLM Judge implementations."""

    judge_model_id: str

    def evaluate(
        self,
        item: DatasetItem,
        output_text: str,
        q_min: float = 0.70,
    ) -> JudgeEvaluationResult:
        """Evaluates an output text against an item's prompt and quality criteria."""
        ...


class MockJudge:
    """
    Deterministic mock judge for offline evaluation.
    Evaluates responses based on criteria coverage, length, and format without network calls.
    """

    def __init__(self, judge_model_id: str = "mock-judge-v1"):
        self.judge_model_id = judge_model_id

    def evaluate(
        self,
        item: DatasetItem,
        output_text: str,
        q_min: float = 0.70,
    ) -> JudgeEvaluationResult:
        start_time = time.perf_counter()

        if not output_text or not output_text.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return JudgeEvaluationResult(
                score=0.0,
                pass_threshold=False,
                rationale="Output was empty or whitespace.",
                judge_model_id=self.judge_model_id,
                judge_latency_ms=round(elapsed_ms, 2),
                judge_cost_usd=0.0,
            )

        clean = output_text.lower()
        score = 0.85  # Baseline for coherent text

        # Check required criteria
        criteria_passed = 0
        for crit in item.quality_criteria:
            crit_words = [w.lower() for w in crit.split() if len(w) > 3]
            if any(w in clean for w in crit_words):
                criteria_passed += 1

        if item.quality_criteria:
            coverage = criteria_passed / len(item.quality_criteria)
            score = 0.50 + (coverage * 0.45)

        # Check if expected format was met
        fmt = item.task_parameters.get("expected_format")
        if fmt == "json":
            if not (output_text.strip().startswith("{") or output_text.strip().startswith("[")):
                score = min(score, 0.40)
        elif (
            fmt in ("markdown", "markdown_table")
            and "#" not in output_text
            and "|" not in output_text
        ):
            score = min(score, 0.50)

        # Ensure bounds
        score = max(0.0, min(1.0, score))
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return JudgeEvaluationResult(
            score=round(score, 3),
            pass_threshold=score >= q_min,
            rationale=f"Deterministic mock judge criteria coverage: {criteria_passed}/{len(item.quality_criteria)}",
            judge_model_id=self.judge_model_id,
            judge_latency_ms=round(elapsed_ms + 12.0, 2),  # Simulated evaluation overhead
            judge_cost_usd=0.0,  # Zero cost in mock mode
        )


class OpenRouterJudge:
    """
    Live LLM Judge using OpenRouter API.
    Used for opt-in real experiments with model-family isolation.
    """

    def __init__(
        self,
        judge_model_id: str = "openai/gpt-4o-mini",
        api_key: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.judge_model_id = judge_model_id
        self.api_key = api_key
        self.base_url = base_url

    def evaluate(
        self,
        item: DatasetItem,
        output_text: str,
        q_min: float = 0.70,
    ) -> JudgeEvaluationResult:
        # If no key, fallback cleanly to MockJudge
        if not self.api_key:
            return MockJudge(judge_model_id=f"fallback-{self.judge_model_id}").evaluate(
                item, output_text, q_min
            )

        import httpx

        prompt = f"""You are an impartial expert evaluator. Evaluate the candidate response on a scale of 0.0 to 1.0.
Prompt: {item.prompt}
Required Criteria: {", ".join(item.quality_criteria)}
Candidate Response:
{output_text}

Output JSON with keys "score" (float 0.0 to 1.0) and "rationale" (string)."""

        start = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.judge_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                import json

                cleaned = content.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                elif cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                parsed = json.loads(cleaned.strip())
                score = float(parsed.get("score", 0.7))
                rationale = str(parsed.get("rationale", "Judge evaluation"))
        except Exception as e:
            score = 0.5
            rationale = f"Judge evaluation fallback error: {str(e)[:100]}"

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return JudgeEvaluationResult(
            score=round(score, 3),
            pass_threshold=score >= q_min,
            rationale=rationale,
            judge_model_id=self.judge_model_id,
            judge_latency_ms=round(elapsed_ms, 2),
            judge_cost_usd=0.0005,  # Estimated mini judge call cost
        )
