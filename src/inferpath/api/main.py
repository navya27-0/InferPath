"""FastAPI application exposing the InferPath inference routing service."""

from __future__ import annotations

import hmac
import math
import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from inferpath import __version__
from inferpath.config.deployment_profiles import (
    DeploymentProfileConfig,
    ProfileName,
    get_deployment_profile,
)
from inferpath.handlers.factory import AdapterFactory
from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import PrivacyClass
from inferpath.pipeline import PipelineExecutionError, RequestPipeline
from inferpath.router.strategies import ConstraintsUnsatisfiableError
from inferpath.security.budget import BudgetExceededError
from inferpath.security.hashing import scrub_api_keys
from inferpath.security.privacy import PrivacyViolationError
from inferpath.security.secret_manager import GLOBAL_SECRET_MANAGER
from inferpath.services.rate_limiter import BaseRateLimiter, DistributedRateLimiter, RateLimiter
from inferpath.telemetry.metrics import GLOBAL_METRICS
from inferpath.telemetry.structured_logging import configure_structured_logging, current_request_id
from inferpath.telemetry.tracing import GLOBAL_TRACER

MAX_PROMPT_CHARS = 32_768


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

    @field_validator("minimum_quality", "maximum_cost_usd", "maximum_latency_ms")
    @classmethod
    def validate_finite_and_bounds(cls, v: float | None, info) -> float | None:
        if v is None:
            return v
        if not math.isfinite(v):
            raise ValueError(f"{info.field_name} must be a finite number, got {v}")
        if info.field_name == "minimum_quality" and not (0.0 <= v <= 1.0):
            raise ValueError(f"minimum_quality must be between 0.0 and 1.0, got {v}")
        if info.field_name == "maximum_cost_usd" and (v < 0.0 or v > 1000.0):
            raise ValueError(f"maximum_cost_usd must be between 0.0 and 1000.0, got {v}")
        if info.field_name == "maximum_latency_ms" and (v < 0.0 or v > 300_000.0):
            raise ValueError(f"maximum_latency_ms must be between 0.0 and 300000.0, got {v}")
        return v


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
    constraints_satisfied: bool = True
    unsatisfied_constraints: list[str] = Field(default_factory=list)
    requested_minimum_quality: float | None = None
    requested_maximum_cost_usd: float | None = None
    requested_maximum_latency_ms: float | None = None
    estimated_call_cost_usd: float | None = None
    actual_calculated_cost_usd: float | None = None
    estimated_quality: float | None = None
    validated_correctness: bool = True
    empirical_quality: float | None = None


def _constant_time_compare(val_a: str | None, val_b: str | None) -> bool:
    """Performs constant-time comparison of two strings to prevent timing attacks."""
    if val_a is None or val_b is None:
        return False
    return hmac.compare_digest(val_a.encode("utf-8"), val_b.encode("utf-8"))


def create_app(
    pipeline: RequestPipeline | None = None,
    rate_limiter: BaseRateLimiter | None = None,
    api_key: str | None = None,
    deployment_mode: str | None = None,
    deployment_profile: DeploymentProfileConfig | str | None = None,
    trusted_proxies: set[str] | None = None,
    adapter_factory: type[AdapterFactory] | AdapterFactory | None = None,
) -> FastAPI:
    """Application factory supporting pipeline, rate limiting, and auth injection."""
    app = FastAPI(
        title="InferPath API",
        version=__version__,
        description="Resource-aware AI inference routing service",
    )
    # Resolve active deployment profile
    active_profile = (
        deployment_profile
        if isinstance(deployment_profile, DeploymentProfileConfig)
        else get_deployment_profile(
            str(deployment_profile) if deployment_profile else deployment_mode
        )
    )

    if pipeline is not None:
        pipeline_instance = pipeline
    else:
        if active_profile.adapter_mode == "real" and not active_profile.allow_mock_fallback:
            openrouter_key = GLOBAL_SECRET_MANAGER.get("OPENROUTER_API_KEY", "") or os.getenv(
                "OPENROUTER_API_KEY", ""
            )
            if not openrouter_key:
                raise RuntimeError(
                    f"Deployment profile '{active_profile.name.value}' enforces real adapter mode without mock fallback, "
                    "but OPENROUTER_API_KEY is not configured."
                )
        pipeline_instance = RequestPipeline(
            adapter_mode=active_profile.adapter_mode,
            allow_mock_fallback=active_profile.allow_mock_fallback,
            adapter_factory=adapter_factory,
        )

    # Store profile on app state for operational inspection
    app.state.deployment_profile = active_profile
    app.state.default_bind_host = active_profile.default_bind_host

    # Initialize rate limiter with injected instance, distributed backend, or profile defaults
    rate_limiter_status = "in_memory"
    if rate_limiter is not None:
        limiter: BaseRateLimiter = rate_limiter
        rate_limiter_status = (
            "distributed"
            if isinstance(rate_limiter, DistributedRateLimiter)
            else ("in_memory" if isinstance(rate_limiter, RateLimiter) else "custom")
        )
    else:
        redis_url = os.getenv("INFERPATH_REDIS_URL") or os.getenv("REDIS_URL")
        require_redis = os.getenv("INFERPATH_REQUIRE_REDIS", "").lower() in (
            "true",
            "1",
            "yes",
        ) or (
            active_profile.name in (ProfileName.PRODUCTION, ProfileName.STAGING)
            and redis_url is not None
        )
        if redis_url:
            try:
                import redis

                redis_client = redis.from_url(redis_url)
                limiter = DistributedRateLimiter(
                    redis_client=redis_client,
                    requests_per_minute=active_profile.rate_limit_rpm,
                )
                rate_limiter_status = "distributed"
            except Exception as e:
                if require_redis:
                    raise RuntimeError(
                        f"Failed to initialize distributed rate limiter with redis_url={redis_url}: {e}"
                    ) from e
                import logging

                logging.getLogger("inferpath.api").warning(
                    "Redis distributed rate limiter initialization failed (%s). Falling back to in-memory RateLimiter.",
                    e,
                )
                limiter = RateLimiter(requests_per_minute=active_profile.rate_limit_rpm)
                rate_limiter_status = "in_memory (degraded_fallback)"
        else:
            limiter = RateLimiter(requests_per_minute=active_profile.rate_limit_rpm)
            rate_limiter_status = "in_memory"

    app.state.rate_limiter = limiter
    app.state.rate_limiter_status = rate_limiter_status

    # Deployment mode configuration
    dep_mode = deployment_mode if deployment_mode is not None else active_profile.name.value
    # Resolve API key from argument, environment, or secret manager
    required_api_key = (
        api_key if api_key is not None else GLOBAL_SECRET_MANAGER.get("INFERPATH_API_KEY", "")
    )

    if (
        pipeline is None
        and active_profile.name == ProfileName.PRODUCTION
        and (not required_api_key or not required_api_key.strip())
    ):
        raise RuntimeError(
            "Deployment profile 'production' requires an explicit application API key, "
            "but INFERPATH_API_KEY is not configured. Startup halted to prevent unauthenticated exposure."
        )

    # Configure structured logging if enabled in profile
    if active_profile.structured_json_logging:
        configure_structured_logging(
            level=active_profile.log_level,
            json_format=True,
            service_name="inferpath",
            environment=dep_mode,
        )

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

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        err_msg = "; ".join(
            f"{'.'.join(str(loc) for loc in e.get('loc', []))}: {e.get('msg')}" for e in errors
        )
        is_constraint_error = any(
            any(
                k in str(e.get("loc", []))
                for k in (
                    "minimum_quality",
                    "maximum_cost_usd",
                    "maximum_latency_ms",
                    "quality",
                    "cost",
                    "latency",
                )
            )
            or "finite" in e.get("msg", "").lower()
            for e in errors
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid constraints" if is_constraint_error else "Bad request",
                "detail": scrub_api_keys(err_msg),
                "code": "INVALID_CONSTRAINTS" if is_constraint_error else "BAD_REQUEST",
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
            headers=exc.headers,
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
            "version": __version__,
            "deployment_mode": dep_mode,
            "profile": active_profile.name.value,
            "rate_limiter": app.state.rate_limiter_status,
        }

    @app.get("/health/live")
    @app.get("/v1/health/live")
    async def health_live():
        return {
            "status": "alive",
            "service": "inferpath",
            "version": __version__,
        }

    @app.get("/health/ready")
    @app.get("/v1/health/ready")
    async def health_ready():
        return {
            "status": "ready",
            "service": "inferpath",
            "version": __version__,
            "deployment_mode": dep_mode,
            "profile": active_profile.name.value,
            "rate_limiter": app.state.rate_limiter_status,
        }

    @app.get("/health/dependencies")
    @app.get("/v1/health/dependencies")
    async def health_dependencies():
        schema_version = 0
        db_status = "ok"
        try:
            if hasattr(pipeline_instance, "cache") and hasattr(
                pipeline_instance.cache, "get_schema_version"
            ):
                schema_version = pipeline_instance.cache.get_schema_version()
        except Exception as e:
            db_status = f"degraded: {e}"

        return {
            "status": "healthy" if db_status == "ok" else "degraded",
            "dependencies": {
                "cache_database": {
                    "status": db_status,
                    "schema_version": schema_version,
                },
                "rate_limiter": {
                    "backend": app.state.rate_limiter_status,
                    "rpm": limiter.requests_per_minute,
                },
                "pipeline": {
                    "adapter_mode": active_profile.adapter_mode,
                    "allow_mock_fallback": active_profile.allow_mock_fallback,
                },
            },
        }

    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(
            content=GLOBAL_METRICS.export_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/v1/respond", response_model=RespondResponse)
    async def respond(body: RespondRequest, request: Request, response: Response):
        # 1. Authentication check
        client_key = request.headers.get("X-API-Key")
        if not client_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                client_key = auth_header[7:].strip()

        if active_profile.require_api_key:
            if not required_api_key or not _constant_time_compare(client_key, required_api_key):
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "Unauthorized",
                        "detail": f"Deployment profile '{active_profile.name.value}' requires valid API key authentication.",
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
            headers = {
                "X-RateLimit-Limit": str(limiter.requests_per_minute),
                "X-RateLimit-Remaining": "0",
                "Retry-After": "60",
            }
            raise HTTPException(
                status_code=429,
                headers=headers,
                detail={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {limiter.requests_per_minute} requests per minute allowed.",
                    "code": "RATE_LIMIT_EXCEEDED",
                },
            )

        # Populate rate limit response headers
        remaining_quota = (
            limiter.get_remaining(client_ip) if hasattr(limiter, "get_remaining") else 0
        )
        response.headers["X-RateLimit-Limit"] = str(limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining_quota)

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
        for name, val in [
            ("minimum_quality", body.minimum_quality),
            ("maximum_cost_usd", body.maximum_cost_usd),
            ("maximum_latency_ms", body.maximum_latency_ms),
        ]:
            if val is not None and not math.isfinite(val):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid constraints",
                        "detail": f"{name} must be a finite number, got {val}",
                        "code": "INVALID_CONSTRAINTS",
                    },
                )
        if body.minimum_quality is not None and not (0.0 <= body.minimum_quality <= 1.0):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid constraints",
                    "detail": f"minimum_quality must be between 0.0 and 1.0, got {body.minimum_quality}",
                    "code": "INVALID_CONSTRAINTS",
                },
            )
        if body.maximum_cost_usd is not None and (
            body.maximum_cost_usd < 0.0 or body.maximum_cost_usd > 1000.0
        ):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid constraints",
                    "detail": f"maximum_cost_usd must be between 0.0 and 1000.0, got {body.maximum_cost_usd}",
                    "code": "INVALID_CONSTRAINTS",
                },
            )
        if body.maximum_latency_ms is not None and (
            body.maximum_latency_ms < 0.0 or body.maximum_latency_ms > 300_000.0
        ):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid constraints",
                    "detail": f"maximum_latency_ms must be between 0.0 and 300000.0, got {body.maximum_latency_ms}",
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

        # Trace context extraction from incoming request
        traceparent = request.headers.get("traceparent")
        parsed_trace = GLOBAL_TRACER.parse_w3c_traceparent(traceparent)
        in_trace_id, in_parent_span_id = parsed_trace if parsed_trace else (None, None)

        with GLOBAL_TRACER.start_as_current_span(
            name="inferpath.api.respond",
            trace_id=in_trace_id,
            parent_span_id=in_parent_span_id,
            attributes={
                "http.method": "POST",
                "http.route": "/v1/respond",
                "inferpath.request_id": body.request_id,
                "inferpath.client_ip": client_ip,
                "inferpath.deployment_mode": dep_mode,
            },
        ) as span:
            # Set context variable for structured logging
            tok_req = current_request_id.set(body.request_id)
            try:
                result = await pipeline_instance.execute_async(user_req)
            except PrivacyViolationError:
                span.set_attribute("error", True)
                span.set_attribute("error.type", "PrivacyViolationError")
                raise
            except BudgetExceededError:
                span.set_attribute("error", True)
                span.set_attribute("error.type", "BudgetExceededError")
                raise
            except ConstraintsUnsatisfiableError:
                span.set_attribute("error", True)
                span.set_attribute("error.type", "ConstraintsUnsatisfiableError")
                raise
            except PipelineExecutionError as err:
                span.set_attribute("error", True)
                span.set_attribute("error.type", "PipelineExecutionError")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Pipeline execution error",
                        "detail": scrub_api_keys(str(err)),
                        "code": "PIPELINE_ERROR",
                    },
                ) from err
            except Exception as err:
                span.set_attribute("error", True)
                span.set_attribute("error.type", "InternalException")
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": "Inference backend error",
                        "detail": scrub_api_keys(str(err)),
                        "code": "BACKEND_ERROR",
                    },
                ) from err
            finally:
                current_request_id.reset(tok_req)

            # Record telemetry metrics
            GLOBAL_METRICS.record_request(
                tier=result.final_tier.value,
                status="success" if result.validation_passed else "validation_failed",
                latency_ms=result.total_latency_ms.value,
                cost_usd=result.cost_usd.value,
                was_cached=result.was_cached,
                fallback_used=result.fallback_occurred,
            )

            # Decorate span with routing outcome
            span.set_attribute("inferpath.route", result.final_tier.value)
            span.set_attribute("inferpath.rule_id", result.rule_id or "none")
            span.set_attribute("inferpath.cache_hit", result.was_cached)
            span.set_attribute("inferpath.cost_usd", result.cost_usd.value)
            span.set_attribute("inferpath.latency_ms", result.total_latency_ms.value)
            span.set_attribute("inferpath.fallback_used", result.fallback_occurred)

            # Inject trace propagation headers into outgoing response if tracing is enabled
            if active_profile.enable_tracing:
                response.headers["X-Trace-Id"] = span.trace_id
                response.headers["traceparent"] = span.export_w3c_traceparent()

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
                constraints_satisfied=result.constraints_satisfied,
                unsatisfied_constraints=result.unsatisfied_constraints,
                requested_minimum_quality=result.requested_minimum_quality,
                requested_maximum_cost_usd=result.requested_maximum_cost_usd,
                requested_maximum_latency_ms=result.requested_maximum_latency_ms,
                estimated_call_cost_usd=result.estimated_call_cost_usd.value
                if result.estimated_call_cost_usd
                else None,
                actual_calculated_cost_usd=result.actual_calculated_cost_usd.value
                if result.actual_calculated_cost_usd
                else None,
                estimated_quality=result.estimated_quality,
                validated_correctness=result.validated_correctness,
                empirical_quality=result.empirical_quality,
            )

    return app


app = create_app()


def run(
    host: str | None = None,
    port: int = 8000,
    profile_str: str | None = None,
    **kwargs: Any,
) -> None:
    """Launch the InferPath API service honoring profile bind host defaults."""
    import uvicorn

    profile = get_deployment_profile(profile_str)
    bind_host = host if host is not None else profile.default_bind_host
    uvicorn.run("inferpath.api.main:app", host=bind_host, port=port, **kwargs)
