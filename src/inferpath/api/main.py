"""FastAPI application exposing the InferPath inference routing service."""

from __future__ import annotations

import hmac
import os
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import PrivacyClass
from inferpath.pipeline import PipelineExecutionError, RequestPipeline
from inferpath.router.strategies import ConstraintsUnsatisfiableError
from inferpath.security.budget import BudgetExceededError
from inferpath.security.hashing import scrub_api_keys
from inferpath.security.privacy import PrivacyViolationError
from inferpath.telemetry.metrics import GLOBAL_METRICS

MAX_PROMPT_CHARS = 32_768


class RateLimiter:
    """In-memory rate limiter using a sliding 60-second window with bounded storage."""

    def __init__(self, requests_per_minute: int = 120, max_clients: int = 10_000):
        self.requests_per_minute = requests_per_minute
        self.max_clients = max_clients
        self._history: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            history = self._history.get(client_id, [])
            valid_history = [t for t in history if t > cutoff]

            if len(valid_history) >= self.requests_per_minute:
                self._history[client_id] = valid_history
                return False

            # Bounded capacity check
            if client_id not in self._history and len(self._history) >= self.max_clients:
                # Evict expired client histories
                expired_keys = [k for k, v in self._history.items() if not v or max(v) <= cutoff]
                for k in expired_keys:
                    del self._history[k]
                # If still at cap, evict oldest entry
                if len(self._history) >= self.max_clients:
                    oldest_k = next(iter(self._history))
                    del self._history[oldest_k]

            valid_history.append(now)
            self._history[client_id] = valid_history
            return True


class RespondRequest(BaseModel):
    """Client request schema for POST /v1/respond."""

    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    prompt: str
    context: str | None = None
    privacy_class: PrivacyClass = PrivacyClass.PUBLIC
    task_parameters: TaskParameters = Field(default_factory=TaskParameters)
    is_time_sensitive: bool | None = None
    minimum_quality: float | None = None
    maximum_cost_usd: float | None = None
    maximum_latency_ms: float | None = None
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


def _constant_time_compare(val_a: str | None, val_b: str | None) -> bool:
    """Performs constant-time comparison of two strings to prevent timing attacks."""
    if val_a is None or val_b is None:
        return False
    return hmac.compare_digest(val_a.encode("utf-8"), val_b.encode("utf-8"))


def create_app(
    pipeline: RequestPipeline | None = None,
    rate_limiter: RateLimiter | None = None,
    api_key: str | None = None,
    deployment_mode: str | None = None,
    trusted_proxies: set[str] | None = None,
) -> FastAPI:
    """Application factory supporting pipeline, rate limiting, and auth injection."""
    app = FastAPI(
        title="InferPath API",
        version="0.1.0",
        description="Resource-aware AI inference routing service",
    )
    pipeline_instance = pipeline or RequestPipeline()
    limiter = rate_limiter or RateLimiter(requests_per_minute=120)

    # Deployment mode configuration ("local-demo" vs "protected")
    dep_mode = (
        deployment_mode
        if deployment_mode is not None
        else os.getenv("INFERPATH_DEPLOYMENT_MODE", "local-demo").strip().lower()
    )
    required_api_key = api_key if api_key is not None else os.getenv("INFERPATH_API_KEY", "")

    # Trusted proxy network configuration
    configured_proxies: set[str] = set()
    if trusted_proxies is not None:
        configured_proxies = trusted_proxies
    else:
        proxy_env = os.getenv("INFERPATH_TRUSTED_PROXIES", "").strip()
        if proxy_env:
            configured_proxies = {p.strip() for p in proxy_env.split(",") if p.strip()}

    @app.exception_handler(PrivacyViolationError)
    async def privacy_violation_handler(request: Request, exc: PrivacyViolationError):
        GLOBAL_METRICS.record_privacy_block()
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
        GLOBAL_METRICS.record_budget_exceeded()
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

    @app.exception_handler(ConstraintsUnsatisfiableError)
    async def constraints_unsatisfiable_handler(
        request: Request, exc: ConstraintsUnsatisfiableError
    ):
        return JSONResponse(
            status_code=422,
            content={
                "error": "Constraints unsatisfiable",
                "detail": scrub_api_keys(str(exc)),
                "code": "CONSTRAINTS_UNSATISFIABLE",
                "unsatisfied_constraints": exc.unsatisfied_constraints,
                "eligible_tiers": [
                    t.value if hasattr(t, "value") else str(t) for t in exc.eligible_tiers
                ],
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        code = "HTTP_ERROR"
        error = "Request error"
        if isinstance(detail, dict):
            error = detail.get("error", error)
            code = detail.get("code", code)
            detail = detail.get("detail", str(detail))
        else:
            if exc.status_code == 400:
                code = "BAD_REQUEST"
                error = "Bad request"
            elif exc.status_code == 401:
                code = "UNAUTHORIZED"
                error = "Unauthorized"
            elif exc.status_code == 403:
                code = "FORBIDDEN"
                error = "Forbidden"
            elif exc.status_code == 404:
                code = "NOT_FOUND"
                error = "Not found"
            elif exc.status_code == 429:
                code = "RATE_LIMIT_EXCEEDED"
                error = "Rate limit exceeded"
            elif exc.status_code >= 500:
                code = "INTERNAL_ERROR"
                error = "Internal server error"

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": error,
                "detail": scrub_api_keys(str(detail)),
                "code": code,
            },
        )

    @app.get("/health")
    @app.get("/v1/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "inferpath",
            "version": "0.1.0",
            "deployment_mode": dep_mode,
        }

    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(
            content=GLOBAL_METRICS.export_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/v1/respond", response_model=RespondResponse)
    async def respond(body: RespondRequest, request: Request):
        # 1. Authentication check
        client_key = request.headers.get("X-API-Key")
        if not client_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                client_key = auth_header[7:].strip()

        if dep_mode == "protected":
            if not required_api_key or not _constant_time_compare(client_key, required_api_key):
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "Unauthorized",
                        "detail": "Protected deployment mode requires valid API key authentication.",
                        "code": "UNAUTHORIZED",
                    },
                )
        elif required_api_key and not _constant_time_compare(client_key, required_api_key):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Unauthorized",
                    "detail": "Invalid or missing X-API-Key header.",
                    "code": "UNAUTHORIZED",
                },
            )

        # 2. Rate limiting check (Strict client IP derivation)
        peer_ip = request.client.host if request.client else "127.0.0.1"
        if peer_ip in configured_proxies:
            forwarded = request.headers.get("X-Forwarded-For")
            client_ip = (forwarded.split(",")[0].strip() or peer_ip) if forwarded else peer_ip
        else:
            client_ip = peer_ip

        if not limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "detail": "Maximum 120 requests per minute allowed.",
                    "code": "RATE_LIMIT_EXCEEDED",
                },
            )

        # 3. Payload validation
        if not body.prompt or not body.prompt.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Bad request",
                    "detail": "Prompt cannot be empty.",
                    "code": "INVALID_PROMPT",
                },
            )

        if len(body.prompt) > MAX_PROMPT_CHARS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Payload too large",
                    "detail": f"Prompt length ({len(body.prompt)}) exceeds maximum limit of {MAX_PROMPT_CHARS} characters.",
                    "code": "PAYLOAD_TOO_LARGE",
                },
            )

        # Numeric constraints validation
        if body.minimum_quality is not None and not (0.0 <= body.minimum_quality <= 1.0):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid constraints",
                    "detail": f"minimum_quality must be between 0.0 and 1.0, got {body.minimum_quality}",
                    "code": "INVALID_CONSTRAINTS",
                },
            )
        if body.maximum_cost_usd is not None and body.maximum_cost_usd < 0.0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid constraints",
                    "detail": f"maximum_cost_usd cannot be negative, got {body.maximum_cost_usd}",
                    "code": "INVALID_CONSTRAINTS",
                },
            )
        if body.maximum_latency_ms is not None and body.maximum_latency_ms < 0.0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid constraints",
                    "detail": f"maximum_latency_ms cannot be negative, got {body.maximum_latency_ms}",
                    "code": "INVALID_CONSTRAINTS",
                },
            )

        task_params = body.task_parameters.model_copy()
        if body.minimum_quality is not None:
            task_params.minimum_quality = body.minimum_quality
        if body.maximum_cost_usd is not None:
            task_params.maximum_cost_usd = body.maximum_cost_usd
        if body.maximum_latency_ms is not None:
            task_params.maximum_latency_ms = body.maximum_latency_ms

        user_req = UserRequest(
            request_id=body.request_id,
            prompt=body.prompt,
            context=body.context,
            privacy_class=body.privacy_class,
            task_parameters=task_params,
            is_time_sensitive=body.is_time_sensitive,
            metadata=body.metadata,
        )

        try:
            result = await pipeline_instance.execute_async(user_req)
        except PrivacyViolationError:
            raise
        except BudgetExceededError:
            raise
        except ConstraintsUnsatisfiableError:
            raise
        except PipelineExecutionError as err:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Pipeline execution error",
                    "detail": scrub_api_keys(str(err)),
                    "code": "PIPELINE_ERROR",
                },
            ) from err
        except Exception as err:
            # Never expose internal secrets or tracebacks to the client
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "Inference backend error",
                    "detail": scrub_api_keys(str(err)),
                    "code": "BACKEND_ERROR",
                },
            ) from err

        # Record telemetry metrics
        GLOBAL_METRICS.record_request(
            tier=result.final_tier.value,
            status="success" if result.validation_passed else "validation_failed",
            latency_ms=result.total_latency_ms.value,
            cost_usd=result.cost_usd.value,
            was_cached=result.was_cached,
            fallback_used=result.fallback_occurred,
        )

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
