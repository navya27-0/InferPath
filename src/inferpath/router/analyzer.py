"""Deterministic, rule-based request analyzer for InferPath.

Extracts intent, task family, constraints, freshness, privacy, and complexity
without using an LLM or relying on prompt length as a difficulty proxy.
"""

import re
from typing import Any

from pydantic import BaseModel, Field

from inferpath.cache.freshness import FreshnessClassifier
from inferpath.config.settings import RoutingSettings
from inferpath.handlers.calculator import SafeCalculator
from inferpath.handlers.date_formatter import DateFormatter
from inferpath.handlers.unit_converter import UnitConverter
from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier, TaskType
from inferpath.templates.engine import REGISTRY, extract_template_fields_from_prompt


class RequestAnalysis(BaseModel):
    """Structured output of deterministic request analysis."""

    task_intent: str
    task_family: TaskType
    format: str = "text"
    language: str = "en"
    task_parameters: TaskParameters
    freshness_needed: bool = False
    freshness_reason: str | None = None
    privacy_class: PrivacyClass = PrivacyClass.PUBLIC
    missing_inputs: list[str] = Field(default_factory=list)
    template_id: str | None = None
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    suitable_tier: RouteTier = RouteTier.SMALL_MODEL_C
    complexity_indicators: list[str] = Field(default_factory=list)
    ambiguity_detected: bool = False
    confidence: float = 1.0
    rationale: str = ""


# Sensitive tokens and credential patterns triggering conservative privacy upgrade
# Note: Keyword/pattern heuristics are conservative defense-in-depth measures and
# cannot detect every sensitive request. Callers must supply explicit classification.
SENSITIVE_CONTENT_PATTERNS = [
    re.compile(
        r"\b(password|passwd|api[_\s]?key|secret[_\s]?key|private[_\s]?key|access[_\s]?token|auth[_\s]?token|bearer\s+[a-zA-Z0-9_\-\.]+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|xox[baprs]-[a-zA-Z0-9_\-]+)\b",
    ),
    re.compile(
        r"\b(ssn|social security number|credit card|cvv|confidential internal|strictly confidential|proprietary\s+and\s+confidential|trade\s+secret)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(payroll\s+data|employee\s+salary|patient\s+health\s+information|phi|hipaa\s+confidential)\b",
        re.IGNORECASE,
    ),
]

# Template intent keywords
TEMPLATE_TRIGGERS: dict[str, list[re.Pattern]] = {
    "leave_application": [
        re.compile(
            r"\b(leave application|apply for leave|sick leave letter|vacation leave request|annual leave request)\b",
            re.IGNORECASE,
        ),
    ],
    "formal_email": [
        re.compile(
            r"\b(formal email|draft (?:a )?formal email|professional email|business email)\b",
            re.IGNORECASE,
        ),
    ],
    "meeting_minutes": [
        re.compile(
            r"\b(meeting minutes|minutes of (?:the )?meeting|meeting notes template|formal meeting minutes)\b",
            re.IGNORECASE,
        ),
    ],
    "cover_letter": [
        re.compile(
            r"\b(cover letter|job application letter|cover letter for (?:the )?position)\b",
            re.IGNORECASE,
        ),
    ],
}


class RequestAnalyzer:
    """
    Deterministic rule-based analyzer inspecting requests.

    Invariants:
    1. Prompt length is strictly forbidden as a difficulty metric.
    2. Zero LLM calls are made during analysis.
    3. Privacy metadata is never silently downgraded; sensitive content is conservatively upgraded.
    4. Freshness requirements are identified using rule-based temporal heuristics.
    """

    def __init__(self, routing_settings: RoutingSettings | None = None):
        self.settings = routing_settings or RoutingSettings()
        self.complexity_keywords = [k.lower() for k in self.settings.complexity_keywords]

    def analyze(self, request: UserRequest) -> RequestAnalysis:
        prompt = request.prompt.strip()

        # 1. Privacy classification
        privacy_class = self._determine_privacy(request, prompt)

        # 2. Freshness evaluation
        freshness_needed, freshness_reason = self._determine_freshness(request, prompt)

        # 3. Format and language extraction
        fmt = self._extract_format(request, prompt)
        lang = self._extract_language(request, prompt)
        task_params = request.task_parameters.model_copy(
            update={"expected_format": fmt, "language": lang}
        )

        # 4. Complexity heuristic check (strictly config-driven keywords, not length)
        complexity_indicators = self._detect_complexity_keywords(prompt)

        # 5. Check Deterministic Intent (Tier A)
        # Arithmetic
        if self._is_arithmetic(prompt):
            return RequestAnalysis(
                task_intent="arithmetic",
                task_family=TaskType.ARITHMETIC_MATH,
                format=fmt,
                language=lang,
                task_parameters=task_params,
                freshness_needed=freshness_needed,
                freshness_reason=freshness_reason,
                privacy_class=privacy_class,
                suitable_tier=RouteTier.DETERMINISTIC_A,
                confidence=0.98,
                rationale="Matches deterministic arithmetic syntax.",
            )

        # Unit Conversion
        if self._is_unit_conversion(prompt):
            return RequestAnalysis(
                task_intent="unit_conversion",
                task_family=TaskType.UNIT_CONVERSION,
                format=fmt,
                language=lang,
                task_parameters=task_params,
                freshness_needed=freshness_needed,
                freshness_reason=freshness_reason,
                privacy_class=privacy_class,
                suitable_tier=RouteTier.DETERMINISTIC_A,
                confidence=0.98,
                rationale="Matches deterministic unit conversion pattern.",
            )

        # Date Formatting
        if self._is_date_formatting(prompt):
            return RequestAnalysis(
                task_intent="date_formatting",
                task_family=TaskType.STRING_MANIPULATION,
                format=fmt,
                language=lang,
                task_parameters=task_params,
                freshness_needed=freshness_needed,
                freshness_reason=freshness_reason,
                privacy_class=privacy_class,
                suitable_tier=RouteTier.DETERMINISTIC_A,
                confidence=0.95,
                rationale="Matches deterministic date formatting request.",
            )

        # JSON & Schema Validation
        if self._is_json_validation(prompt, request.metadata):
            return RequestAnalysis(
                task_intent="json_validation",
                task_family=TaskType.STRING_MANIPULATION,
                format="json",
                language=lang,
                task_parameters=task_params,
                freshness_needed=freshness_needed,
                freshness_reason=freshness_reason,
                privacy_class=privacy_class,
                suitable_tier=RouteTier.DETERMINISTIC_A,
                confidence=0.95,
                rationale="Matches deterministic JSON/schema validation intent.",
            )

        # 6. Check Template Intent (Tier B)
        template_id = self._match_template_intent(prompt)
        if template_id:
            extracted_fields = extract_template_fields_from_prompt(
                template_id, prompt, request.metadata
            )
            defn = REGISTRY[template_id]
            missing = [
                req
                for req in defn.required_fields
                if req not in extracted_fields or not extracted_fields[req]
            ]
            return RequestAnalysis(
                task_intent=f"template_{template_id}",
                task_family=TaskType.TEMPLATE_SLOT,
                format=fmt,
                language=lang,
                task_parameters=task_params,
                freshness_needed=freshness_needed,
                freshness_reason=freshness_reason,
                privacy_class=privacy_class,
                template_id=template_id,
                extracted_fields=extracted_fields,
                missing_inputs=sorted(missing),
                suitable_tier=RouteTier.TEMPLATE_B,
                confidence=0.95,
                rationale=f"Matches structured document template '{template_id}' with {len(missing)} missing fields.",
            )

        # 7. Check Ambiguity / Empty prompt
        if (
            not prompt
            or len(prompt.split()) <= 2
            and prompt.lower() in {"help", "hi", "do it", "convert", "calculate", "write"}
        ):
            return RequestAnalysis(
                task_intent="ambiguous_query",
                task_family=TaskType.UNKNOWN,
                format=fmt,
                language=lang,
                task_parameters=task_params,
                freshness_needed=freshness_needed,
                freshness_reason=freshness_reason,
                privacy_class=privacy_class,
                suitable_tier=RouteTier.LARGE_MODEL_D,
                ambiguity_detected=True,
                confidence=0.40,
                rationale="Prompt is underspecified or highly ambiguous.",
            )

        # 8. Check Complex Reasoning Indicators (Tier D)
        # Note: If complexity keywords are present, or quality preference is 'high'
        if complexity_indicators or request.task_parameters.quality_preference == "high":
            reason = "Complexity heuristic triggered: " + (
                f"matched keywords {complexity_indicators}"
                if complexity_indicators
                else "explicit high quality preference requested"
            )
            return RequestAnalysis(
                task_intent="complex_reasoning",
                task_family=TaskType.COMPLEX_REASONING,
                format=fmt,
                language=lang,
                task_parameters=task_params,
                freshness_needed=freshness_needed,
                freshness_reason=freshness_reason,
                privacy_class=privacy_class,
                suitable_tier=RouteTier.LARGE_MODEL_D,
                complexity_indicators=complexity_indicators,
                confidence=0.85,
                rationale=reason,
            )

        # 9. Default Standard / Low-Resource Tier (Tier C)
        return RequestAnalysis(
            task_intent="general_qa_or_drafting",
            task_family=TaskType.OPEN_ENDED_QA,
            format=fmt,
            language=lang,
            task_parameters=task_params,
            freshness_needed=freshness_needed,
            freshness_reason=freshness_reason,
            privacy_class=privacy_class,
            suitable_tier=RouteTier.SMALL_MODEL_C,
            confidence=0.80,
            rationale="Standard query suitable for small model tier without complexity triggers.",
        )

    def _determine_privacy(self, request: UserRequest, prompt: str) -> PrivacyClass:
        # Never downgrade explicit privacy
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            return request.privacy_class

        # Scan prompt for sensitive content patterns
        for pattern in SENSITIVE_CONTENT_PATTERNS:
            if pattern.search(prompt):
                return PrivacyClass.CONFIDENTIAL_PRIVATE

        # Scan context if provided
        if request.context:
            for pattern in SENSITIVE_CONTENT_PATTERNS:
                if pattern.search(request.context):
                    return PrivacyClass.CONFIDENTIAL_PRIVATE

        # Scan metadata keys and values
        if request.metadata:
            for k, v in request.metadata.items():
                str_entry = f"{k}: {v}"
                for pattern in SENSITIVE_CONTENT_PATTERNS:
                    if pattern.search(str_entry):
                        return PrivacyClass.CONFIDENTIAL_PRIVATE

        return request.privacy_class

    def _determine_freshness(self, request: UserRequest, prompt: str) -> tuple[bool, str | None]:
        if request.is_time_sensitive is not None:
            return request.is_time_sensitive, (
                "Explicit user freshness metadata override" if request.is_time_sensitive else None
            )
        return FreshnessClassifier.is_time_sensitive(prompt)

    def _extract_format(self, request: UserRequest, prompt: str) -> str:
        if request.task_parameters.expected_format != "text":
            return request.task_parameters.expected_format

        lower = prompt.lower()
        if "in json" in lower or "as json" in lower:
            return "json"
        if "in markdown" in lower or "as markdown" in lower:
            return "markdown"
        if "as a table" in lower or "in tabular format" in lower:
            return "table"
        return "text"

    def _extract_language(self, request: UserRequest, prompt: str) -> str:
        if request.task_parameters.language != "en":
            return request.task_parameters.language
        lower = prompt.lower()
        if "in spanish" in lower or "en español" in lower:
            return "es"
        if "in french" in lower or "en français" in lower:
            return "fr"
        if "in german" in lower or "auf deutsch" in lower:
            return "de"
        return "en"

    def _detect_complexity_keywords(self, prompt: str) -> list[str]:
        lower = prompt.lower()
        matched: list[str] = []
        for kw in self.complexity_keywords:
            # Word boundary search for multi-word and single-word keywords
            escaped = re.escape(kw)
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
            if pattern.search(lower):
                matched.append(kw)
        return sorted(list(set(matched)))

    def _is_arithmetic(self, prompt: str) -> bool:
        lower = prompt.lower()
        if any(w in lower for w in ["calculate", "compute", "what is", "solve", "evaluate"]):
            # Check if there is an arithmetic expression
            match = SafeCalculator.EXPR_EXTRACTION_PATTERN.search(prompt)
            if match:
                candidate = match.group(1).strip().rstrip("?. ")
                try:
                    SafeCalculator.evaluate(candidate)
                    return True
                except Exception:
                    pass
        # Try direct evaluation if the prompt itself is a pure math expression
        try:
            SafeCalculator.evaluate(prompt.rstrip("?. "))
            return True
        except Exception:
            return False

    def _is_unit_conversion(self, prompt: str) -> bool:
        match = UnitConverter.CONVERSION_PATTERN.search(prompt)
        if match:
            _, from_u, to_u = match.groups()
            try:
                dim_from = UnitConverter.get_dimension(from_u)
                dim_to = UnitConverter.get_dimension(to_u)
                if dim_from != "currency" and dim_to != "currency":
                    return True
            except Exception:
                return False
        return False

    def _is_date_formatting(self, prompt: str) -> bool:
        match = DateFormatter.DATE_FORMAT_PATTERN.search(prompt)
        if match:
            date_input, _ = match.groups()
            try:
                DateFormatter.parse_date(date_input)
                return True
            except Exception:
                return False
        return False

    def _is_json_validation(self, prompt: str, metadata: dict[str, Any]) -> bool:
        lower = prompt.lower()
        if (
            "validate json" in lower
            or "check if json is valid" in lower
            or "validate schema" in lower
        ):
            return True
        return bool("schema" in metadata and ("{" in prompt or "[" in prompt))

    def _match_template_intent(self, prompt: str) -> str | None:
        for template_id, patterns in TEMPLATE_TRIGGERS.items():
            for p in patterns:
                if p.search(prompt):
                    return template_id
        return None
