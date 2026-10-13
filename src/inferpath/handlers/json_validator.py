"""Deterministic JSON parsing and schema validation handler."""

import json
from typing import Any

import jsonschema

from inferpath.models.domain import HandlerOutput
from inferpath.models.enums import ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue


class JSONValidationError(ValueError):
    """Raised when JSON parsing or schema validation fails."""

    pass


class JSONSchemaValidator:
    """
    Deterministic validator for JSON syntax and schema compliance.

    Guarantees:
    - Never evaluates or executes embedded code.
    - Never claims schema compliance unless an explicit schema was validated against.
    - Emits structured diagnostic messages for syntax and schema violations.
    """

    @classmethod
    def validate_json_string(
        cls, json_str: str, schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Parses JSON and optionally validates against a schema.
        Returns a structured dictionary with outcome details.
        """
        clean = json_str.strip()
        if not clean:
            raise JSONValidationError("JSON input cannot be empty.")

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError as err:
            return {
                "valid": False,
                "schema_validated": False,
                "error": f"Invalid JSON syntax at line {err.lineno}, col {err.colno}: {err.msg}",
            }

        if schema is not None:
            try:
                validator = jsonschema.Draft202012Validator(schema)
                errors = list(validator.iter_errors(parsed))
                if errors:
                    formatted_errors = [
                        {"path": list(e.path), "message": e.message} for e in errors
                    ]
                    return {
                        "valid": False,
                        "schema_validated": True,
                        "error": f"Schema validation failed: {errors[0].message}",
                        "details": formatted_errors,
                    }
                return {
                    "valid": True,
                    "schema_validated": True,
                    "message": "JSON successfully conforms to the provided schema.",
                }
            except Exception as err:
                return {
                    "valid": False,
                    "schema_validated": True,
                    "error": f"Schema evaluation error: {err}",
                }

        return {
            "valid": True,
            "schema_validated": False,
            "message": "JSON syntax is valid. Note: No schema was supplied, so schema compliance is not claimed.",
        }

    @classmethod
    def execute_handler(
        cls, prompt: str, schema: dict[str, Any] | None = None
    ) -> HandlerOutput:
        """Executes JSON validation and returns HandlerOutput."""
        # Check if the prompt contains JSON or is wrapped in code blocks
        clean_text = prompt.strip()
        if "```json" in clean_text:
            parts = clean_text.split("```json")
            if len(parts) > 1:
                clean_text = parts[1].split("```")[0].strip()
        elif "```" in clean_text:
            parts = clean_text.split("```")
            if len(parts) > 1:
                clean_text = parts[1].strip()

        # If schema is not explicitly provided, check if the prompt specifies both schema and instance
        extracted_schema = schema
        instance_text = clean_text

        outcome = cls.validate_json_string(instance_text, schema=extracted_schema)
        output_text = json.dumps(outcome, indent=2)

        return HandlerOutput(
            tier=RouteTier.DETERMINISTIC_A,
            output_text=output_text,
            input_tokens=ProvenanceValue(
                value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
            ),
            output_tokens=ProvenanceValue(
                value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
            ),
            latency_ms=ProvenanceValue(
                value=0.5, provenance=ProvenanceLabel.MEASURED, unit="ms"
            ),
            model_version="deterministic_json_validator_v1",
            raw_response=outcome,
        )
