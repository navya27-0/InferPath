"""FastAPI application exposing the InferPath inference routing service."""

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import PrivacyClass
from inferpath.pipeline import PipelineExecutionError, RequestPipeline
from inferpath.security.budget import BudgetExceededError
from inferpath.security.hashing import scrub_api_keys
from inferpath.security.privacy import PrivacyViolationError


class RespondRequest(BaseModel):
    """Client request schema for POST /v1/respond."""

    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    prompt: str
    context: str | None = None
    privacy_class: PrivacyClass = PrivacyClass.PUBLIC
    task_parameters: TaskParameters = Field(default_factory=TaskParameters)
    is_time_sensitive: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RespondResponse(BaseModel):
    """Sanitized response schema returned to clients."""

    request_id: str
    text: str
    route: str
    rule_id: str | None
    cache_hit: bool
    validated: bool
    fallback_used: bool
    needs_clarification: bool = False
    clarification_prompt: str | None = None
    cost_usd: float
    cost_provenance: str
    latency_ms: float
    latency_provenance: str
    route_trace: list[dict[str, Any]] = Field(default_factory=list)
    input_tokens: int | None = None
    input_tokens_provenance: str | None = None
    output_tokens: int | None = None
    output_tokens_provenance: str | None = None
    provider_reported_cost_usd: float | None = None
    energy_wh: float | None = None
    energy_provenance: str | None = None
    water_ml: float | None = None
    water_provenance: str | None = None


def create_app(pipeline: RequestPipeline | None = None) -> FastAPI:
    """Application factory supporting pipeline dependency injection."""
    app = FastAPI(
        title="InferPath API",
        version="0.1.0",
        description="Resource-aware AI inference routing service",
    )
    pipeline_instance = pipeline or RequestPipeline()

    @app.exception_handler(PrivacyViolationError)
    async def privacy_violation_handler(request: Request, exc: PrivacyViolationError):
        return JSONResponse(
            status_code=403,
            content={
                "error": "Privacy policy violation",
                "detail": scrub_api_keys(str(exc)),
                "code": "PRIVACY_BLOCKED",
            },
        )

    @app.exception_handler(BudgetExceededError)
    async def budget_exceeded_handler(request: Request, exc: BudgetExceededError):
        return JSONResponse(
            status_code=402,
            content={
                "error": "Budget cap exceeded",
                "detail": scrub_api_keys(str(exc)),
                "estimated_cost": exc.estimated_cost,
                "remaining_budget": exc.remaining_budget,
                "cap": exc.cap,
                "code": "BUDGET_EXCEEDED",
            },
        )

    @app.get("/health")
    @app.get("/v1/health")
    async def health_check():
        return {"status": "healthy", "service": "inferpath", "version": "0.1.0"}

    @app.post("/v1/respond", response_model=RespondResponse)
    async def respond(body: RespondRequest):
        # Validate prompt non-empty
        if not body.prompt or not body.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        user_req = UserRequest(
            request_id=body.request_id,
            prompt=body.prompt,
            context=body.context,
            privacy_class=body.privacy_class,
            task_parameters=body.task_parameters,
            is_time_sensitive=body.is_time_sensitive,
            metadata=body.metadata,
        )

        try:
            result = pipeline_instance.execute(user_req)
        except PrivacyViolationError:
            raise
        except BudgetExceededError:
            raise
        except PipelineExecutionError as err:
            raise HTTPException(status_code=400, detail=scrub_api_keys(str(err))) from err
        except Exception as err:
            # Never expose internal secrets or tracebacks to the client
            raise HTTPException(
                status_code=502,
                detail=f"Inference pipeline execution error: {scrub_api_keys(str(err))}",
            ) from err

        return RespondResponse(
            request_id=result.request_id,
            text=result.final_output,
            route=result.final_tier.value,
            rule_id=result.rule_id,
            cache_hit=result.was_cached,
            validated=result.validation_passed,
            fallback_used=result.fallback_occurred,
            needs_clarification=result.needs_clarification,
            clarification_prompt=result.clarification_prompt,
            cost_usd=result.cost_usd.value,
            cost_provenance=result.cost_usd.provenance.value,
            latency_ms=result.total_latency_ms.value,
            latency_provenance=result.total_latency_ms.provenance.value,
            route_trace=result.route_trace,
            input_tokens=result.input_tokens.value if result.input_tokens else None,
            input_tokens_provenance=result.input_tokens.provenance.value
            if result.input_tokens
            else None,
            output_tokens=result.output_tokens.value if result.output_tokens else None,
            output_tokens_provenance=result.output_tokens.provenance.value
            if result.output_tokens
            else None,
            provider_reported_cost_usd=result.provider_reported_cost_usd.value
            if result.provider_reported_cost_usd
            else None,
            energy_wh=result.energy_wh.value if result.energy_wh else None,
            energy_provenance=result.energy_wh.provenance.value if result.energy_wh else None,
            water_ml=result.water_ml.value if result.water_ml else None,
            water_provenance=result.water_ml.provenance.value if result.water_ml else None,
        )

    return app


app = create_app()
